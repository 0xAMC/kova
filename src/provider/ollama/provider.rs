use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use tracing::Instrument;

use super::config::OllamaProviderConfig;
use super::convert::{
    format_model_list, format_request, format_response, ndjson_byte_stream_to_events,
};
use super::types::{OllamaModelListResponse, OllamaResponse};
use crate::error::KovaError;
use crate::models::{
    ConversationMessage, InferenceConfig, ModelInfo, ModelResponse, StreamEvent, ToolDefinition,
};
use crate::provider::LlmProvider;
use crate::provider::http::map_request_error;

pub struct OllamaProvider {
    client: Client,
    config: OllamaProviderConfig,
}

impl OllamaProvider {
    pub fn new(config: OllamaProviderConfig) -> Result<Self, KovaError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| KovaError::Connection(e.to_string()))?;
        Ok(Self { client, config })
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        let model = config.model.as_deref().unwrap_or(&self.config.model);
        let span = tracing::info_span!(
            "llm.chat_completion",
            provider = "ollama",
            model = %model,
            otel.status_code = tracing::field::Empty,
            llm.input_tokens = tracing::field::Empty,
            llm.output_tokens = tracing::field::Empty,
            llm.stop_reason = tracing::field::Empty,
        );
        async {
            let request_body = format_request(messages, tools, config, &self.config, false);
            let url = self.config.chat_url();

            let start = std::time::Instant::now();
            let response = self
                .client
                .post(&url)
                .json(&request_body)
                .send()
                .await
                .map_err(|e| {
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

            let ollama_response: OllamaResponse = response.json().await.map_err(|e| {
                let err = KovaError::Provider {
                    message: format!("Failed to deserialize response: {e}"),
                    status_code: None,
                };
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "Failed to deserialize LLM response");
                err
            })?;

            let latency_ms = start.elapsed().as_millis() as u64;
            tracing::Span::current().record("llm.input_tokens", ollama_response.prompt_eval_count);
            tracing::Span::current().record("llm.output_tokens", ollama_response.eval_count);
            tracing::Span::current().record(
                "llm.stop_reason",
                ollama_response.done_reason.as_deref().unwrap_or("stop"),
            );
            tracing::info!(latency_ms, "LLM chat completion succeeded");

            format_response(ollama_response)
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
        let model = config.model.as_deref().unwrap_or(&self.config.model);
        let span = tracing::info_span!(
            "llm.chat_completion_stream",
            provider = "ollama",
            model = %model,
            otel.status_code = tracing::field::Empty,
        );
        async {
            let request_body = format_request(messages, tools, config, &self.config, true);
            let url = self.config.chat_url();

            let response = self
                .client
                .post(&url)
                .json(&request_body)
                .send()
                .await
                .map_err(|e| {
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

            Ok(ndjson_byte_stream_to_events(response.bytes_stream()))
        }
        .instrument(span)
        .await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        let span = tracing::info_span!(
            "llm.list_models",
            provider = "ollama",
            otel.status_code = tracing::field::Empty,
        );
        async {
            let url = self.config.tags_url();
            let response = self.client.get(&url).send().await.map_err(|e| {
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

            let model_list: OllamaModelListResponse = response.json().await.map_err(|e| {
                let err = KovaError::Provider {
                    message: format!("Failed to deserialize model list: {e}"),
                    status_code: None,
                };
                tracing::Span::current().record("otel.status_code", "ERROR");
                err
            })?;

            Ok(format_model_list(model_list))
        }
        .instrument(span)
        .await
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ContentBlock, Role};
    use serde_json::json;
    use wiremock::matchers::{method, path};
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
            model: Some("llama3.2".to_string()),
            max_tokens: Some(100),
            temperature: Some(0.7),
            ..Default::default()
        }
    }

    fn success_response_json() -> serde_json::Value {
        json!({
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hi there!"
            },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 3,
            "total_duration": 1000000000
        })
    }

    fn provider_for(server: &MockServer) -> OllamaProvider {
        let config = OllamaProviderConfig::new("llama3.2").with_base_url(server.uri());
        OllamaProvider::new(config).unwrap()
    }

    #[tokio::test]
    async fn test_chat_completion_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_response_json()))
            .mount(&server)
            .await;

        let provider = provider_for(&server);
        let resp = provider
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await
            .unwrap();
        assert_eq!(resp.stop_reason, crate::models::StopReason::EndTurn);
        assert!(matches!(&resp.content[0], ContentBlock::Text { text } if text == "Hi there!"));
    }

    #[tokio::test]
    async fn test_chat_completion_http_400() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .mount(&server)
            .await;

        let err = provider_for(&server)
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
    async fn test_chat_completion_http_500() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let err = provider_for(&server)
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
    async fn test_tool_call_response_detected() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "llama3.2",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"}
                        }
                    }]
                },
                "done": true,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
                "total_duration": 500000000
            })))
            .mount(&server)
            .await;

        let provider = provider_for(&server);
        let resp = provider
            .chat_completion(&sample_messages(), &[], &sample_config())
            .await
            .unwrap();
        assert_eq!(resp.stop_reason, crate::models::StopReason::ToolUse);
        match &resp.content[0] {
            ContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "London");
            }
            other => panic!("expected ToolUse, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_streaming_success() {
        let server = MockServer::start().await;
        let ndjson = concat!(
            "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"done\":false}\n",
            "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\" world\"},\"done\":false}\n",
            "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":5,\"eval_count\":3,\"total_duration\":1000000000}\n"
        );
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;

        use futures::StreamExt;
        let provider = provider_for(&server);
        let stream = provider
            .chat_completion_stream(&sample_messages(), &[], &sample_config())
            .await
            .unwrap();
        let events: Vec<_> = stream.collect().await;
        let events: Vec<StreamEvent> = events.into_iter().map(|r| r.unwrap()).collect();

        let text: String = events
            .iter()
            .filter_map(|e| {
                if let StreamEvent::ContentDelta { text } = e {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(text, "Hello world");
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::StopEvent {
                stop_reason: crate::models::StopReason::EndTurn
            }
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::UsageEvent {
                input_tokens: 5,
                output_tokens: 3,
                ..
            }
        )));
    }

    #[tokio::test]
    async fn test_streaming_tool_call() {
        let server = MockServer::start().await;
        let ndjson = concat!(
            "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"search\",\"arguments\":{\"q\":\"cats\"}}}]},\"done\":false}\n",
            "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":8,\"eval_count\":4,\"total_duration\":800000000}\n"
        );
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;

        use futures::StreamExt;
        let provider = provider_for(&server);
        let stream = provider
            .chat_completion_stream(&sample_messages(), &[], &sample_config())
            .await
            .unwrap();
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert!(events.iter().any(|e| matches!(e,
            StreamEvent::ToolUseDelta { name: Some(n), .. } if n == "search"
        )));
    }

    #[tokio::test]
    async fn test_streaming_thinking_delta() {
        let server = MockServer::start().await;
        let ndjson = concat!(
            "{\"model\":\"qwen3\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"thinking\":\"Let me think...\"},\"done\":false}\n",
            "{\"model\":\"qwen3\",\"message\":{\"role\":\"assistant\",\"content\":\"The answer is 42.\"},\"done\":false}\n",
            "{\"model\":\"qwen3\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":6,\"eval_count\":10,\"total_duration\":2000000000}\n"
        );
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_string(ndjson))
            .mount(&server)
            .await;

        use futures::StreamExt;
        let provider = provider_for(&server);
        let stream = provider
            .chat_completion_stream(&sample_messages(), &[], &sample_config())
            .await
            .unwrap();
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert!(events.iter().any(|e| matches!(e,
            StreamEvent::ThinkingDelta { text } if text == "Let me think..."
        )));
        assert!(events.iter().any(|e| matches!(e,
            StreamEvent::ContentDelta { text } if text == "The answer is 42."
        )));
    }

    #[tokio::test]
    async fn test_stream_http_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
            .mount(&server)
            .await;

        let err = provider_for(&server)
            .chat_completion_stream(&sample_messages(), &[], &sample_config())
            .await
            .err()
            .unwrap();
        assert!(matches!(
            err,
            KovaError::Provider {
                status_code: Some(503),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_list_models_success() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [
                    {"name": "llama3.2:latest", "size": 2000000000, "modified_at": "2024-01-01T00:00:00Z"},
                    {"name": "qwen3:8b", "size": 5_000_000_000_u64, "modified_at": "2024-01-02T00:00:00Z"}
                ]
            })))
            .mount(&server)
            .await;

        let models = provider_for(&server).list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "llama3.2:latest");
        assert_eq!(models[1].id, "qwen3:8b");
        assert_eq!(models[0].owned_by, "ollama");
        assert_eq!(models[0].object, "model");
    }

    #[tokio::test]
    async fn test_list_models_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(503).set_body_string("down"))
            .mount(&server)
            .await;

        let err = provider_for(&server).list_models().await.unwrap_err();
        assert!(matches!(
            err,
            KovaError::Provider {
                status_code: Some(503),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_list_models_empty() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": []})))
            .mount(&server)
            .await;

        let models = provider_for(&server).list_models().await.unwrap();
        assert!(models.is_empty());
    }
}
