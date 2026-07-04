use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use tracing::Instrument;

use super::config::AnthropicProviderConfig;
use super::convert::{format_request, format_response, sse_byte_stream_to_events};
use super::types::{AnthropicResponse, ModelListResponse};
use crate::error::KovaError;
use crate::models::{
    ConversationMessage, InferenceConfig, ModelInfo, ModelResponse, StreamEvent, ToolDefinition,
};
use crate::provider::LlmProvider;
use crate::provider::http::{error_from_response, map_request_error};

/// Native Anthropic (Messages API) provider with automatic prompt caching
/// and adaptive-thinking support.
pub struct AnthropicProvider {
    client: Client,
    config: AnthropicProviderConfig,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicProviderConfig) -> Result<Self, KovaError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| KovaError::Connection(e.to_string()))?;
        Ok(Self { client, config })
    }

    fn apply_headers(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut req = req.header("anthropic-version", &self.config.api_version);
        if let Some(ref key) = self.config.api_key {
            req = req.header("x-api-key", key);
        }
        req
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        let model_name = config.model.as_deref().unwrap_or(&self.config.model);
        let span = tracing::info_span!(
            "llm.chat_completion",
            provider = "anthropic",
            model = %model_name,
            otel.status_code = tracing::field::Empty,
            llm.input_tokens = tracing::field::Empty,
            llm.output_tokens = tracing::field::Empty,
            llm.stop_reason = tracing::field::Empty,
        );
        async {
            let request = format_request(messages, tools, config, &self.config, false);
            let req = self
                .apply_headers(self.client.post(self.config.messages_url()))
                .json(&request);

            let start = std::time::Instant::now();
            let response = req.send().await.map_err(|e| {
                let err = map_request_error(e, self.config.timeout);
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "LLM request failed");
                err
            })?;

            if !response.status().is_success() {
                let err = error_from_response(response).await;
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "LLM provider returned error");
                return Err(err);
            }

            let anthropic_response: AnthropicResponse = response.json().await.map_err(|e| {
                let err =
                    KovaError::provider_invalid(format!("Failed to deserialize response: {e}"));
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "Failed to deserialize LLM response");
                err
            })?;

            let latency_ms = start.elapsed().as_millis() as u64;
            tracing::Span::current()
                .record("llm.input_tokens", anthropic_response.usage.input_tokens);
            tracing::Span::current()
                .record("llm.output_tokens", anthropic_response.usage.output_tokens);
            let result = format_response(anthropic_response)?;
            tracing::Span::current().record("llm.stop_reason", result.stop_reason.as_str());
            tracing::info!(latency_ms, "LLM chat completion succeeded");
            Ok(result)
        }
        .instrument(span)
        .await
    }

    async fn chat_completion_stream(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>, KovaError> {
        let model_name = config.model.as_deref().unwrap_or(&self.config.model);
        let span = tracing::info_span!(
            "llm.chat_completion_stream",
            provider = "anthropic",
            model = %model_name,
            otel.status_code = tracing::field::Empty,
        );
        async {
            let request = format_request(messages, tools, config, &self.config, true);
            let req = self
                .apply_headers(self.client.post(self.config.messages_url()))
                .json(&request);

            let response = req.send().await.map_err(|e| {
                let err = map_request_error(e, self.config.timeout);
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "LLM stream request failed");
                err
            })?;

            if !response.status().is_success() {
                let err = error_from_response(response).await;
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "LLM stream provider returned error");
                return Err(err);
            }

            Ok(sse_byte_stream_to_events(response.bytes_stream()))
        }
        .instrument(span)
        .await
    }

    async fn count_tokens(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
    ) -> Result<u32, KovaError> {
        // The count endpoint takes the same shape as /v1/messages minus the
        // generation-only fields.
        let request = format_request(
            messages,
            tools,
            &InferenceConfig::default(),
            &self.config,
            false,
        );
        let mut body = serde_json::to_value(&request)
            .map_err(|e| KovaError::provider_invalid(format!("serialize count request: {e}")))?;
        if let Some(obj) = body.as_object_mut() {
            obj.remove("max_tokens");
            obj.remove("stream");
            obj.remove("cache_control");
        }

        let response = self
            .apply_headers(self.client.post(self.config.count_tokens_url()))
            .json(&body)
            .send()
            .await
            .map_err(|e| map_request_error(e, self.config.timeout))?;
        if !response.status().is_success() {
            return Err(error_from_response(response).await);
        }

        #[derive(serde::Deserialize)]
        struct CountResponse {
            input_tokens: u32,
        }
        let count: CountResponse = response.json().await.map_err(|e| {
            KovaError::provider_invalid(format!("Failed to deserialize token count: {e}"))
        })?;
        Ok(count.input_tokens)
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        let span = tracing::info_span!(
            "llm.list_models",
            provider = "anthropic",
            otel.status_code = tracing::field::Empty,
        );
        async {
            let req = self.apply_headers(self.client.get(self.config.models_url()));
            let response = req.send().await.map_err(|e| {
                let err = map_request_error(e, self.config.timeout);
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "List models request failed");
                err
            })?;

            if !response.status().is_success() {
                let err = error_from_response(response).await;
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "List models provider returned error");
                return Err(err);
            }

            let list: ModelListResponse = response.json().await.map_err(|e| {
                let err =
                    KovaError::provider_invalid(format!("Failed to deserialize model list: {e}"));
                tracing::Span::current().record("otel.status_code", "ERROR");
                err
            })?;

            Ok(list
                .data
                .into_iter()
                .map(|m| ModelInfo {
                    id: m.id,
                    object: "model".to_string(),
                    created: 0,
                    owned_by: m.display_name.unwrap_or_else(|| "anthropic".to_string()),
                })
                .collect())
        }
        .instrument(span)
        .await
    }
}
