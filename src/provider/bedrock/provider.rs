use std::pin::Pin;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_credential_types::provider::ProvideCredentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use futures::{Stream, StreamExt};
use tracing::Instrument;

/// Percent-encode a model identifier for use in a URL path segment.
///
/// Bedrock inference profile IDs (e.g. `us.anthropic.claude-haiku-4-5-20251001-v1:0`)
/// and ARNs contain characters like `:` and `/` that must be encoded in the URL path.
fn url_encode_model_id(model_id: &str) -> String {
    super::config::encode_path_segment(model_id)
}

use super::config::BedrockProviderConfig;
use super::convert::{format_request, format_response, format_stream_event};
use super::error::parse_bedrock_error;
use super::stream::parse_event_stream_frame;
use super::types::{BedrockConverseResponse, BedrockModelListResponse};
use crate::error::KovaError;
use crate::models::{
    ConversationMessage, InferenceConfig, ModelInfo, ModelResponse, StreamEvent, ToolDefinition,
};
use crate::provider::LlmProvider;
use crate::provider::http::map_request_error;

pub struct BedrockProvider {
    client: reqwest::Client,
    config: BedrockProviderConfig,
    credentials_provider: Arc<dyn ProvideCredentials>,
}

impl BedrockProvider {
    /// Create a new BedrockProvider.
    ///
    /// Credential resolution order:
    /// 1. Explicit credentials (access_key_id + secret_access_key)
    /// 2. Named profile
    /// 3. Default credential chain
    pub async fn new(config: BedrockProviderConfig) -> Result<Self, KovaError> {
        let credentials_provider: Arc<dyn ProvideCredentials> = if let (
            Some(access_key_id),
            Some(secret_access_key),
        ) =
            (&config.access_key_id, &config.secret_access_key)
        {
            let creds = Credentials::from_keys(
                access_key_id.clone(),
                secret_access_key.clone(),
                config.session_token.clone(),
            );
            Arc::new(creds)
        } else {
            let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
            if let Some(profile) = &config.profile {
                loader = loader.profile_name(profile);
            }
            let sdk_config = loader.load().await;
            let provider = sdk_config.credentials_provider().ok_or_else(|| {
                    KovaError::Provider {
                        message: "Failed to resolve AWS credentials: no credentials provider found in the default chain".to_string(),
                        status_code: None,
                    }
                })?;
            Arc::from(provider)
        };

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| KovaError::Provider {
                message: format!("Failed to build HTTP client: {e}"),
                status_code: None,
            })?;

        Ok(Self {
            client,
            config,
            credentials_provider,
        })
    }

    /// Sign an HTTP request with SigV4 for the Bedrock service.
    /// Returns signed headers to apply to the outgoing reqwest request.
    async fn sign_request(
        &self,
        method: &str,
        url: &str,
        body: &[u8],
    ) -> Result<Vec<(String, String)>, KovaError> {
        let creds = self
            .credentials_provider
            .provide_credentials()
            .await
            .map_err(|e| KovaError::Provider {
                message: format!("Failed to resolve AWS credentials: {e}"),
                status_code: None,
            })?;

        let identity = creds.into();
        let signing_settings = SigningSettings::default();
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.config.region)
            .name("bedrock")
            .time(SystemTime::now())
            .settings(signing_settings)
            .build()
            .map_err(|e| KovaError::Provider {
                message: format!("Failed to build SigV4 signing params: {e}"),
                status_code: None,
            })?
            .into();

        let headers = [("content-type", "application/json")];
        let signable_request =
            SignableRequest::new(method, url, headers.into_iter(), SignableBody::Bytes(body))
                .map_err(|e| KovaError::Provider {
                    message: format!("Failed to create signable request: {e}"),
                    status_code: None,
                })?;

        let (signing_instructions, _signature) = sign(signable_request, &signing_params)
            .map_err(|e| KovaError::Provider {
                message: format!("SigV4 signing failed: {e}"),
                status_code: None,
            })?
            .into_parts();

        Ok(signing_instructions
            .headers()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect())
    }

    fn resolve_base_url(&self, service: &str) -> String {
        self.config
            .endpoint_url
            .clone()
            .unwrap_or_else(|| format!("https://{}.{}.amazonaws.com", service, self.config.region))
    }
}

#[async_trait]
impl LlmProvider for BedrockProvider {
    async fn chat_completion(
        &self,
        messages: &[ConversationMessage],
        tools: &[ToolDefinition],
        config: &InferenceConfig,
    ) -> Result<ModelResponse, KovaError> {
        let span = tracing::info_span!(
            "llm.chat_completion",
            provider = "bedrock",
            model = %self.config.model_id,
            otel.status_code = tracing::field::Empty,
            llm.input_tokens = tracing::field::Empty,
            llm.output_tokens = tracing::field::Empty,
            llm.stop_reason = tracing::field::Empty,
        );
        async {
            let base = self.resolve_base_url("bedrock-runtime");
            let encoded_model = url_encode_model_id(&self.config.model_id);
            let url = format!("{}/model/{}/converse", base, encoded_model);

            let bedrock_request = format_request(
                messages,
                tools,
                config,
                self.config.additional_model_request_fields.clone(),
            );
            let body = serde_json::to_vec(&bedrock_request).map_err(|e| KovaError::Provider {
                message: format!("Failed to serialize request: {e}"),
                status_code: None,
            })?;
            tracing::debug!(body = %String::from_utf8_lossy(&body), "Bedrock request body");

            let signed_headers = self.sign_request("POST", &url, &body).await?;

            let mut req = self
                .client
                .post(&url)
                .header("content-type", "application/json")
                .body(body);
            for (name, value) in &signed_headers {
                req = req.header(name.as_str(), value.as_str());
            }

            let response = req.send().await.map_err(|e| {
                let err = map_request_error(e, self.config.timeout);
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "Bedrock chat_completion request failed");
                err
            })?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let err = parse_bedrock_error(status.as_u16(), &body);
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "Bedrock provider returned error");
                return Err(err);
            }

            let response_text = response.text().await.map_err(|e| {
                let err = KovaError::Provider {
                    message: format!("Failed to read response body: {e}"),
                    status_code: None,
                };
                tracing::Span::current().record("otel.status_code", "ERROR");
                err
            })?;
            tracing::debug!(body = %response_text, "Bedrock response body");
            let bedrock_response: BedrockConverseResponse = serde_json::from_str(&response_text)
                .map_err(|e| {
                    let err = KovaError::Provider {
                        message: format!("Failed to deserialize response: {e}"),
                        status_code: None,
                    };
                    tracing::Span::current().record("otel.status_code", "ERROR");
                    err
                })?;

            tracing::Span::current()
                .record("llm.input_tokens", bedrock_response.usage.input_tokens);
            tracing::Span::current()
                .record("llm.output_tokens", bedrock_response.usage.output_tokens);
            tracing::Span::current()
                .record("llm.stop_reason", bedrock_response.stop_reason.as_str());
            tracing::info!("Bedrock chat completion succeeded");

            format_response(bedrock_response)
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
        let span = tracing::info_span!(
            "llm.chat_completion_stream",
            provider = "bedrock",
            model = %self.config.model_id,
            otel.status_code = tracing::field::Empty,
        );
        async {
            let base = self.resolve_base_url("bedrock-runtime");
            let encoded_model = url_encode_model_id(&self.config.model_id);
            let url = format!("{}/model/{}/converse-stream", base, encoded_model);

            let bedrock_request = format_request(
                messages,
                tools,
                config,
                self.config.additional_model_request_fields.clone(),
            );
            let body = serde_json::to_vec(&bedrock_request).map_err(|e| KovaError::Provider {
                message: format!("Failed to serialize request: {e}"),
                status_code: None,
            })?;
            tracing::debug!(body = %String::from_utf8_lossy(&body), "Bedrock stream request body");

            let signed_headers = self.sign_request("POST", &url, &body).await?;

            let mut req = self
                .client
                .post(&url)
                .header("content-type", "application/json")
                .body(body);
            for (name, value) in &signed_headers {
                req = req.header(name.as_str(), value.as_str());
            }

            let response = req.send().await.map_err(|e| {
                let err = map_request_error(e, self.config.timeout);
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "Bedrock stream request failed");
                err
            })?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let err = parse_bedrock_error(status.as_u16(), &body);
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "Bedrock stream provider returned error");
                return Err(err);
            }

            let byte_stream = response.bytes_stream();

            let stream = futures::stream::try_unfold(
                (
                    byte_stream,
                    MessageFrameDecoder::new(),
                    bytes::BytesMut::new(),
                ),
                |(mut byte_stream, mut decoder, mut buffer)| async move {
                    loop {
                        match decoder.decode_frame(&mut buffer) {
                            Ok(DecodedFrame::Complete(message)) => {
                                match parse_event_stream_frame(&message) {
                                    Ok(Some(bedrock_event)) => {
                                        if let Some(stream_event) =
                                            format_stream_event(bedrock_event)
                                        {
                                            return Ok(Some((
                                                stream_event,
                                                (byte_stream, decoder, buffer),
                                            )));
                                        }
                                        continue;
                                    }
                                    Ok(None) => continue,
                                    Err(e) => return Err(e),
                                }
                            }
                            Ok(DecodedFrame::Incomplete) => {}
                            Err(e) => {
                                return Err(KovaError::Stream(format!(
                                    "Failed to decode event stream frame: {e}"
                                )));
                            }
                        }

                        match byte_stream.next().await {
                            Some(Ok(chunk)) => buffer.extend_from_slice(&chunk),
                            Some(Err(e)) => {
                                return Err(KovaError::Stream(format!(
                                    "Stream connection error: {e}"
                                )));
                            }
                            None => return Ok(None),
                        }
                    }
                },
            );

            Ok(Box::pin(stream)
                as Pin<
                    Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>,
                >)
        }
        .instrument(span)
        .await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, KovaError> {
        let span = tracing::info_span!(
            "llm.list_models",
            provider = "bedrock",
            otel.status_code = tracing::field::Empty,
        );
        async {
            let base = self.resolve_base_url("bedrock");
            let url = format!("{}/foundation-models", base);

            let signed_headers = self.sign_request("GET", &url, &[]).await?;

            let mut req = self.client.get(&url);
            for (name, value) in &signed_headers {
                req = req.header(name.as_str(), value.as_str());
            }

            let response = req.send().await.map_err(|e| {
                let err = map_request_error(e, self.config.timeout);
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "Bedrock list_models request failed");
                err
            })?;

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let err = parse_bedrock_error(status.as_u16(), &body);
                tracing::Span::current().record("otel.status_code", "ERROR");
                tracing::warn!(error = %err, "Bedrock list_models provider returned error");
                return Err(err);
            }

            let list_response: BedrockModelListResponse = response.json().await.map_err(|e| {
                let err = KovaError::Provider {
                    message: format!("Failed to deserialize model list response: {e}"),
                    status_code: None,
                };
                tracing::Span::current().record("otel.status_code", "ERROR");
                err
            })?;

            Ok(list_response
                .model_summaries
                .into_iter()
                .map(|s| ModelInfo {
                    id: s.model_id,
                    object: "model".to_string(),
                    created: 0,
                    owned_by: s.provider_name,
                })
                .collect())
        }
        .instrument(span)
        .await
    }
}

#[cfg(test)]
#[path = "provider_tests.rs"]
mod tests;
