//! Property tests for OpenAI-compatible provider behavior.
//!
//! Uses `wiremock` to capture and verify HTTP requests made by the provider.

use proptest::prelude::*;

use kova_sdk::error::KovaError;
use kova_sdk::models::*;
use kova_sdk::provider::LlmProvider;
use kova_sdk::provider::openai::{OpenAiCompatibleProvider, OpenAiProviderConfig};

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Helpers ────────────────────────────────────────────────────────

fn sample_messages() -> Vec<ConversationMessage> {
    vec![ConversationMessage {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "Hello".to_string(),
        }],
    }]
}

fn sample_oai_response_json() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
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

// ── Provider Config Propagation ────────────────────────
//
// For any ProviderConfig with a non-empty base_url and model name,
// all HTTP requests sent by OpenAiCompatibleProvider shall use the
// specified base_url as the URL prefix and include the specified model
// in the request body.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_provider_config_propagation(
        model_name in "[a-z][a-z0-9-]{1,20}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let server = MockServer::start().await;

            // Capture the request body to verify model name.
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(sample_oai_response_json()),
                )
                .mount(&server)
                .await;

            let config = OpenAiProviderConfig::new(server.uri(), &model_name);
            let provider = OpenAiCompatibleProvider::new(config).unwrap();

            let inference_config = InferenceConfig {
                model: Some(model_name.clone()),
                max_tokens: None,
                temperature: None,
            };

            let result = provider
                .chat_completion(&sample_messages(), &[], &inference_config)
                .await;

            // The request should succeed (server is at the configured base_url).
            prop_assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());

            // Verify the server received the request (base_url was used).
            let received = server.received_requests().await.unwrap();
            prop_assert!(
                !received.is_empty(),
                "Server should have received at least one request"
            );

            // Verify the model name is in the request body.
            let body: serde_json::Value =
                serde_json::from_slice(&received[0].body).unwrap();
            let sent_model = body["model"].as_str().unwrap_or("");
            prop_assert_eq!(
                sent_model, &model_name,
                "Request body model should match config model"
            );

            Ok(())
        })?;
    }
}

// ── API Key Authorization Header ───────────────────────
//
// For Some(api_key), Authorization header present; for None, absent.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_api_key_present_when_configured(
        api_key in "[a-zA-Z0-9]{8,32}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(sample_oai_response_json()),
                )
                .mount(&server)
                .await;

            let config = OpenAiProviderConfig::new(server.uri(), "test-model")
                .with_api_key(&api_key);
            let provider = OpenAiCompatibleProvider::new(config).unwrap();

            let inference_config = InferenceConfig {
                model: Some("test-model".to_string()),
                max_tokens: None,
                temperature: None,
            };

            let _ = provider
                .chat_completion(&sample_messages(), &[], &inference_config)
                .await;

            let received = server.received_requests().await.unwrap();
            prop_assert!(!received.is_empty());

            let auth_header = received[0]
                .headers
                .get("authorization")
                .map(|v| v.to_str().unwrap_or("").to_string());

            let expected = format!("Bearer {}", api_key);
            prop_assert_eq!(
                auth_header.as_deref(),
                Some(expected.as_str()),
                "Authorization header should be 'Bearer <api_key>'"
            );

            Ok(())
        })?;
    }

    #[test]
    fn prop_no_auth_header_when_api_key_absent(_seed in 0u32..100) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(sample_oai_response_json()),
                )
                .mount(&server)
                .await;

            let config = OpenAiProviderConfig::new(server.uri(), "test-model");
            // No api_key set.
            let provider = OpenAiCompatibleProvider::new(config).unwrap();

            let inference_config = InferenceConfig {
                model: Some("test-model".to_string()),
                max_tokens: None,
                temperature: None,
            };

            let _ = provider
                .chat_completion(&sample_messages(), &[], &inference_config)
                .await;

            let received = server.received_requests().await.unwrap();
            prop_assert!(!received.is_empty());

            let auth_header = received[0].headers.get("authorization");
            prop_assert!(
                auth_header.is_none(),
                "Authorization header should be absent when no API key is configured"
            );

            Ok(())
        })?;
    }
}

// ── HTTP Error Status Mapping ──────────────────────────
//
// For any HTTP response with a status code in the range 400–599,
// OpenAiCompatibleProvider shall return an KovaError::Provider
// containing that status code and the response body text.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_http_error_status_mapping(
        status_code in 400u16..=599,
        body_text in "[a-zA-Z0-9 .,!?]{1,100}",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(status_code)
                        .set_body_string(&body_text),
                )
                .mount(&server)
                .await;

            let config = OpenAiProviderConfig::new(server.uri(), "test-model");
            let provider = OpenAiCompatibleProvider::new(config).unwrap();

            let inference_config = InferenceConfig {
                model: Some("test-model".to_string()),
                max_tokens: None,
                temperature: None,
            };

            let result = provider
                .chat_completion(&sample_messages(), &[], &inference_config)
                .await;

            match result {
                Err(KovaError::Provider {
                    message,
                    status_code: Some(code),
                }) => {
                    prop_assert_eq!(
                        code, status_code,
                        "Error status code should match HTTP status"
                    );
                    prop_assert_eq!(
                        &message, &body_text,
                        "Error message should contain the response body"
                    );
                }
                Err(other) => {
                    prop_assert!(
                        false,
                        "Expected KovaError::Provider with status {}, got: {:?}",
                        status_code,
                        other
                    );
                }
                Ok(_) => {
                    prop_assert!(
                        false,
                        "Expected error for HTTP {}, got Ok",
                        status_code
                    );
                }
            }

            Ok(())
        })?;
    }
}
