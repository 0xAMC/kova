use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use tracing::Instrument;

use super::config::GeminiProviderConfig;
use super::convert::{format_request, format_response, sse_byte_stream_to_events};
use super::types::{GeminiModelListResponse, GeminiResponse};
use crate::error::KovaError;
use crate::models::{
    ConversationMessage, InferenceConfig, ModelInfo, ModelResponse, StreamEvent, ToolDefinition,
};
use crate::provider::LlmProvider;
use crate::provider::http::{error_from_response, map_request_error};

pub struct GeminiProvider {
    client: Client,
    config: GeminiProviderConfig,
}

impl GeminiProvider {
    pub fn new(config: GeminiProviderConfig) -> Result<Self, KovaError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| KovaError::Connection(e.to_string()))?;
        Ok(Self { client, config })
    }

    fn effective_model<'a>(&'a self, config: &'a InferenceConfig) -> &'a str {
        config.model.as_deref().unwrap_or(&self.config.model)
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.config.api_key {
            req.header("x-goog-api-key", key)
        } else {
            req
        }
    }

    fn merge_config(&self, request_config: &InferenceConfig) -> InferenceConfig {
        InferenceConfig {
            model: request_config
                .model
                .clone()
                .or_else(|| Some(self.config.model.clone())),
            max_tokens: request_config.max_tokens,
            temperature: request_config.temperature,
            top_p: request_config.top_p,
            stop_sequences: request_config.stop_sequences.clone(),
            response_format: request_config.response_format.clone(),
        }
    }
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        let model_name = self.effective_model(config).to_string();
        let span = tracing::info_span!(
            "llm.chat_completion",
            provider = "gemini",
            model = %model_name,
            otel.status_code = tracing::field::Empty,
            llm.input_tokens = tracing::field::Empty,
            llm.output_tokens = tracing::field::Empty,
            llm.stop_reason = tracing::field::Empty,
        );
        async {
            let merged = self.merge_config(config);
            let gemini_request =
                format_request(messages, tools, &merged, self.config.thinking_budget);
            let url = self.config.generate_content_url(&model_name);
            let req = self.apply_auth(self.client.post(&url).json(&gemini_request));

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

            let gemini_response: GeminiResponse = response.json().await.map_err(|e| {
                let err =
                    KovaError::provider_invalid(format!("Failed to deserialize response: {e}"));
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "Failed to deserialize LLM response");
                err
            })?;

            let latency_ms = start.elapsed().as_millis() as u64;
            if let Some(ref usage) = gemini_response.usage_metadata {
                tracing::Span::current().record("llm.input_tokens", usage.prompt_token_count);
                tracing::Span::current().record("llm.output_tokens", usage.candidates_token_count);
            }
            let finish_reason = gemini_response
                .candidates
                .first()
                .and_then(|c| c.finish_reason.as_deref())
                .unwrap_or("STOP");
            tracing::Span::current().record("llm.stop_reason", finish_reason);
            tracing::info!(latency_ms, "LLM chat completion succeeded");

            format_response(gemini_response)
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
        let model_name = self.effective_model(config).to_string();
        let span = tracing::info_span!(
            "llm.chat_completion_stream",
            provider = "gemini",
            model = %model_name,
            otel.status_code = tracing::field::Empty,
        );
        async {
            let merged = self.merge_config(config);
            let gemini_request =
                format_request(messages, tools, &merged, self.config.thinking_budget);
            let url = self.config.stream_generate_content_url(&model_name);
            let req = self.apply_auth(self.client.post(&url).json(&gemini_request));

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

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        let span = tracing::info_span!(
            "llm.list_models",
            provider = "gemini",
            otel.status_code = tracing::field::Empty,
        );
        async {
            let url = self.config.models_url();
            let req = self.apply_auth(self.client.get(&url));

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

            let model_list: GeminiModelListResponse = response.json().await.map_err(|e| {
                let err =
                    KovaError::provider_invalid(format!("Failed to deserialize model list: {e}"));
                tracing::Span::current().record("otel.status_code", "ERROR");
                err
            })?;

            Ok(model_list
                .models
                .into_iter()
                .map(|m| {
                    // API returns names like "models/gemini-2.0-flash"; strip prefix.
                    let id = m
                        .name
                        .strip_prefix("models/")
                        .unwrap_or(&m.name)
                        .to_string();
                    ModelInfo {
                        id,
                        object: "model".to_string(),
                        created: 0,
                        owned_by: "google".to_string(),
                    }
                })
                .collect())
        }
        .instrument(span)
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ContentBlock, Role};
    use serde_json::json;
    use wiremock::matchers::{header, method, path, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_messages() -> Vec<ConversationMessage> {
        vec![ConversationMessage {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Hello".to_string(),
            }],
        }]
    }

    fn sample_config() -> InferenceConfig {
        InferenceConfig {
            model: Some("gemini-2.0-flash".to_string()),
            max_tokens: Some(100),
            temperature: Some(0.7),
            ..Default::default()
        }
    }

    fn sample_response_json() -> serde_json::Value {
        json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{ "text": "Hi there!" }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8
            }
        })
    }

    fn provider_for(server: &MockServer, api_key: Option<&str>) -> GeminiProvider {
        let mut cfg = GeminiProviderConfig::new("gemini-2.0-flash").with_base_url(server.uri());
        if let Some(k) = api_key {
            cfg = cfg.with_api_key(k);
        }
        GeminiProvider::new(cfg).unwrap()
    }

    #[tokio::test]
    async fn test_api_key_header_sent() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r".*:generateContent$"))
            .and(header("x-goog-api-key", "my-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(sample_response_json()))
            .mount(&server)
            .await;

        let provider = provider_for(&server, Some("my-key"));
        assert!(
            provider
                .chat_completion(&sample_messages(), &[], &sample_config())
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn test_no_api_key_still_sends_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r".*:generateContent$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(sample_response_json()))
            .mount(&server)
            .await;

        let provider = provider_for(&server, None);
        assert!(
            provider
                .chat_completion(&sample_messages(), &[], &sample_config())
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn test_http_400_returns_provider_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r".*:generateContent$"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .mount(&server)
            .await;

        let err = provider_for(&server, None)
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            KovaError::Provider {
                status_code: Some(400),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_http_429_returns_provider_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r".*:generateContent$"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let err = provider_for(&server, None)
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            KovaError::Provider {
                status_code: Some(429),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_http_500_returns_provider_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r".*:generateContent$"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let err = provider_for(&server, None)
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            KovaError::Provider {
                status_code: Some(500),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_list_models_strips_models_prefix() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [
                    { "name": "models/gemini-2.0-flash", "displayName": "Gemini 2.0 Flash" },
                    { "name": "models/gemini-1.5-pro", "displayName": "Gemini 1.5 Pro" }
                ]
            })))
            .mount(&server)
            .await;

        let provider = provider_for(&server, None);
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gemini-2.0-flash");
        assert_eq!(models[1].id, "gemini-1.5-pro");
        assert_eq!(models[0].owned_by, "google");
        assert_eq!(models[0].object, "model");
    }

    #[tokio::test]
    async fn test_list_models_api_key_sent() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .and(header("x-goog-api-key", "key-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "models": [] })))
            .mount(&server)
            .await;

        let provider = provider_for(&server, Some("key-123"));
        assert!(provider.list_models().await.is_ok());
    }

    #[tokio::test]
    async fn test_list_models_error_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;

        let err = provider_for(&server, None).list_models().await.unwrap_err();
        assert!(matches!(
            err,
            KovaError::Provider {
                status_code: Some(503),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_function_call_response_detected_as_tool_use() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r".*:generateContent$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "id": "fc-1",
                                "name": "search",
                                "args": { "query": "cats" }
                            }
                        }]
                    },
                    "finishReason": "STOP"
                }]
            })))
            .mount(&server)
            .await;

        let provider = provider_for(&server, None);
        let resp = provider
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await
            .unwrap();
        assert_eq!(resp.stop_reason, crate::models::StopReason::ToolUse);
        assert!(matches!(&resp.content[0], ContentBlock::ToolUse { name, .. } if name == "search"));
    }
}
