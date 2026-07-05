//! OpenAI-compatible `/v1/embeddings` implementation.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::EmbeddingProvider;
use crate::error::KovaError;
use crate::provider::http::{error_from_response, map_request_error};

/// Embeddings via any OpenAI-compatible `/v1/embeddings` endpoint.
pub struct OpenAiEmbeddingProvider {
    client: Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    dimensions: Option<usize>,
    timeout: Duration,
}

impl OpenAiEmbeddingProvider {
    /// `base_url` is the API root (e.g. `https://api.openai.com`);
    /// `/v1/embeddings` is appended.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self, KovaError> {
        let timeout = Duration::from_secs(60);
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| KovaError::Connection(e.to_string()))?;
        Ok(Self {
            client,
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            dimensions: None,
            timeout,
        })
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Request reduced-dimension vectors (`dimensions` request field,
    /// supported by text-embedding-3 models).
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KovaError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'));
        let mut req = self.client.post(&url).json(&EmbeddingRequest {
            model: &self.model,
            input: texts,
            dimensions: self.dimensions,
        });
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }
        let response = req
            .send()
            .await
            .map_err(|e| map_request_error(e, self.timeout))?;
        if !response.status().is_success() {
            return Err(error_from_response(response).await);
        }
        let mut body: EmbeddingResponse = response.json().await.map_err(|e| {
            KovaError::provider_invalid(format!("Failed to deserialize embeddings: {e}"))
        })?;
        // The API documents order-preservation but carries an index — sort to
        // be safe, then verify completeness.
        body.data.sort_by_key(|d| d.index);
        if body.data.len() != texts.len() {
            return Err(KovaError::provider_invalid(format!(
                "embedding count mismatch: sent {} inputs, got {} vectors",
                texts.len(),
                body.data.len()
            )));
        }
        Ok(body.data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_and_response_wire_shapes() {
        let texts = vec!["a".to_string(), "b".to_string()];
        let req = EmbeddingRequest {
            model: "text-embedding-3-small",
            input: &texts,
            dimensions: Some(256),
        };
        let body = serde_json::to_value(&req).unwrap();
        assert_eq!(body["model"], "text-embedding-3-small");
        assert_eq!(body["input"].as_array().unwrap().len(), 2);
        assert_eq!(body["dimensions"], 256);

        // Out-of-order data is re-sorted by index.
        let resp: EmbeddingResponse = serde_json::from_value(serde_json::json!({
            "data": [
                {"index": 1, "embedding": [0.2]},
                {"index": 0, "embedding": [0.1]}
            ]
        }))
        .unwrap();
        let mut data = resp.data;
        data.sort_by_key(|d| d.index);
        assert_eq!(data[0].embedding, vec![0.1]);
        assert_eq!(data[1].embedding, vec![0.2]);
    }
}
