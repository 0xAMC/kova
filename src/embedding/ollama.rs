//! Ollama `/api/embed` implementation.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::EmbeddingProvider;
use crate::error::KovaError;
use crate::provider::http::{error_from_response, map_request_error};

/// Embeddings via a local Ollama server (`POST /api/embed`).
pub struct OllamaEmbeddingProvider {
    client: Client,
    base_url: String,
    model: String,
    timeout: Duration,
}

impl OllamaEmbeddingProvider {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self, KovaError> {
        let timeout = Duration::from_secs(120);
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| KovaError::Connection(e.to_string()))?;
        Ok(Self {
            client,
            base_url: base_url.into(),
            model: model.into(),
            timeout,
        })
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl EmbeddingProvider for OllamaEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KovaError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .json(&EmbedRequest {
                model: &self.model,
                input: texts,
            })
            .send()
            .await
            .map_err(|e| map_request_error(e, self.timeout))?;
        if !response.status().is_success() {
            return Err(error_from_response(response).await);
        }
        let body: EmbedResponse = response.json().await.map_err(|e| {
            KovaError::provider_invalid(format!("Failed to deserialize embeddings: {e}"))
        })?;
        if body.embeddings.len() != texts.len() {
            return Err(KovaError::provider_invalid(format!(
                "embedding count mismatch: sent {} inputs, got {} vectors",
                texts.len(),
                body.embeddings.len()
            )));
        }
        Ok(body.embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_and_response_wire_shapes() {
        let texts = vec!["hello".to_string()];
        let req = EmbedRequest {
            model: "nomic-embed-text",
            input: &texts,
        };
        let body = serde_json::to_value(&req).unwrap();
        assert_eq!(body["model"], "nomic-embed-text");
        assert_eq!(body["input"][0], "hello");

        let resp: EmbedResponse =
            serde_json::from_value(serde_json::json!({"embeddings": [[0.5, -0.5]]})).unwrap();
        assert_eq!(resp.embeddings, vec![vec![0.5, -0.5]]);
    }
}
