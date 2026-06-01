use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;

use super::config::OpenAiProviderConfig;
use super::convert::{format_request, format_response, sse_byte_stream_to_events};
use super::types::{OaiChatCompletionResponse, OaiModelListResponse, OaiStreamOptions};
use crate::error::KovaError;
use crate::models::{
    ConversationMessage, InferenceConfig, ModelInfo, ModelResponse, StreamEvent, ToolDefinition,
};
use crate::provider::LlmProvider;
use crate::provider::http::map_request_error;

pub struct OpenAiCompatibleProvider {
    client: Client,
    config: OpenAiProviderConfig,
}

impl OpenAiCompatibleProvider {
    pub fn new(config: OpenAiProviderConfig) -> Result<Self, KovaError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| KovaError::Connection(e.to_string()))?;
        Ok(Self { client, config })
    }

    fn merge_config(&self, request_config: &InferenceConfig) -> InferenceConfig {
        InferenceConfig {
            model: request_config
                .model
                .clone()
                .or_else(|| Some(self.config.model.clone())),
            max_tokens: request_config.max_tokens.or(self.config.max_tokens),
            temperature: request_config.temperature.or(self.config.temperature),
        }
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.config.api_key {
            req.header("Authorization", format!("Bearer {key}"))
        } else {
            req
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        let model_name = config.model.as_deref().unwrap_or(&self.config.model);
        let span = tracing::info_span!(
            "llm.chat_completion",
            provider = "openai",
            model = %model_name,
            otel.status_code = tracing::field::Empty,
            llm.input_tokens = tracing::field::Empty,
            llm.output_tokens = tracing::field::Empty,
            llm.stop_reason = tracing::field::Empty,
        );
        let _guard = span.enter();

        let merged = self.merge_config(config);
        let mut oai_request = format_request(messages, tools, &merged);
        oai_request.reasoning_effort = self.config.reasoning_effort.clone();
        let url = self.config.chat_completions_url();
        let req = self.apply_auth(self.client.post(&url).json(&oai_request));

        let start = std::time::Instant::now();
        let response = req.send().await.map_err(|e| {
            let err = map_request_error(e, self.config.timeout);
            tracing::Span::current().record("otel.status_code", "ERROR");
            tracing::warn!(error = %err, "LLM request failed");
            err
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let err = KovaError::Provider {
                message: body,
                status_code: Some(status.as_u16()),
            };
            tracing::Span::current().record("otel.status_code", "ERROR");
            tracing::warn!(error = %err, "LLM provider returned error");
            return Err(err);
        }

        let oai_response: OaiChatCompletionResponse = response.json().await.map_err(|e| {
            let err = KovaError::Provider {
                message: format!("Failed to deserialize response: {e}"),
                status_code: None,
            };
            tracing::Span::current().record("otel.status_code", "ERROR");
            tracing::warn!(error = %err, "Failed to deserialize LLM response");
            err
        })?;

        let latency_ms = start.elapsed().as_millis() as u64;
        if let Some(ref usage) = oai_response.usage {
            tracing::Span::current().record("llm.input_tokens", usage.prompt_tokens);
            tracing::Span::current().record("llm.output_tokens", usage.completion_tokens);
        }
        let finish_reason = oai_response
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
            .unwrap_or("unknown");
        tracing::Span::current().record("llm.stop_reason", finish_reason);
        tracing::info!(latency_ms, "LLM chat completion succeeded");

        format_response(oai_response)
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
            provider = "openai",
            model = %model_name,
            otel.status_code = tracing::field::Empty,
        );
        let _guard = span.enter();

        let merged = self.merge_config(config);
        let mut oai_request = format_request(messages, tools, &merged);
        oai_request.reasoning_effort = self.config.reasoning_effort.clone();
        oai_request.stream = Some(true);
        oai_request.stream_options = Some(OaiStreamOptions {
            include_usage: true,
        });

        let url = self.config.chat_completions_url();
        let req = self.apply_auth(self.client.post(&url).json(&oai_request));

        let response = req.send().await.map_err(|e| {
            let err = map_request_error(e, self.config.timeout);
            tracing::Span::current().record("otel.status_code", "ERROR");
            tracing::warn!(error = %err, "LLM stream request failed");
            err
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let err = KovaError::Provider {
                message: body,
                status_code: Some(status.as_u16()),
            };
            tracing::Span::current().record("otel.status_code", "ERROR");
            tracing::warn!(error = %err, "LLM stream provider returned error");
            return Err(err);
        }

        Ok(Box::pin(sse_byte_stream_to_events(response.bytes_stream())))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        let span = tracing::info_span!(
            "llm.list_models",
            provider = "openai",
            otel.status_code = tracing::field::Empty,
        );
        let _guard = span.enter();

        let url = self.config.models_url();
        let req = self.apply_auth(self.client.get(&url));

        let response = req.send().await.map_err(|e| {
            let err = map_request_error(e, self.config.timeout);
            tracing::Span::current().record("otel.status_code", "ERROR");
            tracing::warn!(error = %err, "List models request failed");
            err
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let err = KovaError::Provider {
                message: body,
                status_code: Some(status.as_u16()),
            };
            tracing::Span::current().record("otel.status_code", "ERROR");
            tracing::warn!(error = %err, "List models provider returned error");
            return Err(err);
        }

        let model_list: OaiModelListResponse = response.json().await.map_err(|e| {
            let err = KovaError::Provider {
                message: format!("Failed to deserialize model list: {e}"),
                status_code: None,
            };
            tracing::Span::current().record("otel.status_code", "ERROR");
            err
        })?;

        Ok(model_list
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                object: m.object,
                created: m.created,
                owned_by: m.owned_by,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ContentBlock, Role};
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
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
            model: Some("test-model".to_string()),
            max_tokens: Some(100),
            temperature: Some(0.7),
        }
    }

    fn sample_response_json() -> serde_json::Value {
        json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890u64,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8
            }
        })
    }

    fn provider_config(base_url: &str, api_key: Option<&str>) -> OpenAiProviderConfig {
        let mut cfg = OpenAiProviderConfig::new(base_url, "test-model");
        if let Some(key) = api_key {
            cfg = cfg.with_api_key(key);
        }
        cfg
    }

    #[tokio::test]
    async fn test_api_key_header_present_when_configured() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(sample_response_json()))
            .mount(&server)
            .await;

        let provider =
            OpenAiCompatibleProvider::new(provider_config(&server.uri(), Some("test-key")))
                .unwrap();
        let resp = provider
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn test_no_auth_header_when_api_key_absent() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(sample_response_json()))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(provider_config(&server.uri(), None)).unwrap();
        let resp = provider
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn test_http_400_returns_provider_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(provider_config(&server.uri(), None)).unwrap();
        let err = provider
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await
            .unwrap_err();
        match err {
            KovaError::Provider {
                status_code: Some(400),
                ..
            } => {}
            other => panic!("Expected Provider 400, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_http_429_returns_provider_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(provider_config(&server.uri(), None)).unwrap();
        let err = provider
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await
            .unwrap_err();
        match err {
            KovaError::Provider {
                status_code: Some(429),
                ..
            } => {}
            other => panic!("Expected Provider 429, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_http_500_returns_provider_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(provider_config(&server.uri(), None)).unwrap();
        let err = provider
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await
            .unwrap_err();
        match err {
            KovaError::Provider {
                status_code: Some(500),
                ..
            } => {}
            other => panic!("Expected Provider 500, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_list_models_with_api_key() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("Authorization", "Bearer my-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{
                    "id": "gpt-4",
                    "object": "model",
                    "created": 1000,
                    "owned_by": "openai"
                }]
            })))
            .mount(&server)
            .await;

        let provider =
            OpenAiCompatibleProvider::new(provider_config(&server.uri(), Some("my-key"))).unwrap();
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gpt-4");
        assert_eq!(models[0].owned_by, "openai");
    }

    #[tokio::test]
    async fn test_list_models_error_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(provider_config(&server.uri(), None)).unwrap();
        let err = provider.list_models().await.unwrap_err();
        match err {
            KovaError::Provider {
                status_code: Some(503),
                ..
            } => {}
            other => panic!("Expected Provider 503, got {:?}", other),
        }
    }
}
