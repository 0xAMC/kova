use super::*;
use proptest::prelude::*;
use std::time::Duration;

fn arb_identifier() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_]{0,19}"
}

proptest! {
    #[test]
    fn prop_config_field_storage(
        region in arb_identifier(),
        model_id in arb_identifier(),
        profile in prop::option::of(arb_identifier()),
        access_key_id in prop::option::of(arb_identifier()),
        secret_access_key in prop::option::of(arb_identifier()),
        session_token in prop::option::of(arb_identifier()),
        timeout_secs in 1u64..300u64,
    ) {
        let mut config = BedrockProviderConfig::new(region.clone(), model_id.clone());
        if let Some(ref p) = profile {
            config = config.with_profile(p.clone());
        }
        if let (Some(ak), Some(sk)) = (&access_key_id, &secret_access_key) {
            config = config.with_credentials(ak.clone(), sk.clone(), session_token.clone());
        }
        config = config.with_timeout(Duration::from_secs(timeout_secs));

        prop_assert_eq!(&config.region, &region);
        prop_assert_eq!(&config.model_id, &model_id);
        prop_assert_eq!(&config.profile, &profile);
        if access_key_id.is_some() && secret_access_key.is_some() {
            prop_assert_eq!(&config.access_key_id, &access_key_id);
            prop_assert_eq!(&config.secret_access_key, &secret_access_key);
            prop_assert_eq!(&config.session_token, &session_token);
        }
        prop_assert_eq!(config.timeout, Duration::from_secs(timeout_secs));
    }

    #[test]
    fn prop_provider_config_url_construction(
        region in arb_identifier(),
        model_id in arb_identifier(),
    ) {
        use crate::provider::config::ProviderConfig;
        let config = BedrockProviderConfig::new(region.clone(), model_id.clone());
        let expected_url = format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/converse",
            region, model_id
        );
        prop_assert_eq!(config.base_url(), expected_url.as_str());
        prop_assert_eq!(config.model(), model_id.as_str());
    }

    #[test]
    fn prop_sigv4_authorization_header(
        access_key in "[A-Z0-9]{20}",
        secret_key in "[a-zA-Z0-9]{40}",
        region in "[a-z]{2}-[a-z]+-[0-9]",
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = BedrockProviderConfig::new(region.clone(), "test-model".to_string())
                .with_credentials(access_key.clone(), secret_key.clone(), None);
            let provider = BedrockProvider::new(config).await.unwrap();
            let url = format!(
                "https://bedrock-runtime.{}.amazonaws.com/model/test-model/converse",
                region
            );
            let headers = provider.sign_request("POST", &url, b"{}").await.unwrap();
            let auth_header = headers
                .iter()
                .find(|(k, _)| k.to_lowercase() == "authorization")
                .map(|(_, v)| v.clone());
            prop_assert!(auth_header.is_some(), "Authorization header should be present");
            let auth = auth_header.unwrap();
            prop_assert!(auth.starts_with("AWS4-HMAC-SHA256"), "Should use AWS4-HMAC-SHA256");
            prop_assert!(auth.contains("bedrock"), "Should contain service name 'bedrock'");
            prop_assert!(auth.contains(&region), "Should contain region");
            Ok(())
        })?;
    }

    #[test]
    fn prop_session_token_header(
        access_key in "[A-Z0-9]{20}",
        secret_key in "[a-zA-Z0-9]{40}",
        session_token in prop::option::of("[a-zA-Z0-9]{20,50}"),
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = BedrockProviderConfig::new("us-east-1".to_string(), "test-model".to_string())
                .with_credentials(access_key, secret_key, session_token.clone());
            let provider = BedrockProvider::new(config).await.unwrap();
            let headers = provider
                .sign_request(
                    "POST",
                    "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/converse",
                    b"{}",
                )
                .await
                .unwrap();
            let token_header = headers
                .iter()
                .find(|(k, _)| k.to_lowercase() == "x-amz-security-token")
                .map(|(_, v)| v.clone());
            match session_token {
                Some(ref token) => {
                    prop_assert!(token_header.is_some(), "X-Amz-Security-Token should be present");
                    prop_assert_eq!(token_header.unwrap(), token.clone());
                }
                None => {
                    prop_assert!(token_header.is_none(), "X-Amz-Security-Token should be absent");
                }
            }
            Ok(())
        })?;
    }

    #[test]
    fn prop_model_summary_conversion(
        model_id in arb_identifier(),
        _model_name in "[a-zA-Z0-9 ]{1,50}",
        provider_name in "[a-zA-Z0-9 ]{1,50}",
    ) {
        let info = ModelInfo {
            id: model_id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: provider_name.clone(),
        };
        prop_assert_eq!(&info.id, &model_id);
        prop_assert_eq!(&info.object, "model");
        prop_assert_eq!(info.created, 0);
        prop_assert_eq!(&info.owned_by, &provider_name);
    }
}

#[tokio::test]
async fn test_sigv4_signing_produces_valid_authorization_header() {
    let config = BedrockProviderConfig::new("us-east-1", "anthropic.claude-v2")
        .with_credentials(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
        );
    let provider = BedrockProvider::new(config).await.unwrap();
    let headers = provider
        .sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/converse",
            b"{}",
        )
        .await
        .unwrap();

    let auth = headers
        .iter()
        .find(|(k, _)| k.to_lowercase() == "authorization")
        .expect("should have Authorization header");
    assert!(auth.1.starts_with("AWS4-HMAC-SHA256"));
    assert!(auth.1.contains("us-east-1"));
    assert!(auth.1.contains("bedrock"));
    let date = headers.iter().find(|(k, _)| k.to_lowercase() == "x-amz-date");
    assert!(date.is_some(), "should have X-Amz-Date header");
}

#[tokio::test]
async fn test_sigv4_session_token_included() {
    let config = BedrockProviderConfig::new("us-west-2", "anthropic.claude-v2")
        .with_credentials(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            Some("FwoGZXIvYXdzEBYaDHqa0AP".to_string()),
        );
    let provider = BedrockProvider::new(config).await.unwrap();
    let headers = provider
        .sign_request(
            "POST",
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-v2/converse",
            b"{}",
        )
        .await
        .unwrap();
    let token = headers
        .iter()
        .find(|(k, _)| k.to_lowercase() == "x-amz-security-token");
    assert!(token.is_some(), "should have X-Amz-Security-Token header");
    assert_eq!(token.unwrap().1, "FwoGZXIvYXdzEBYaDHqa0AP");
}

#[tokio::test]
async fn test_credential_resolution_failure_returns_provider_error() {
    let config = BedrockProviderConfig::new("us-east-1", "test-model")
        .with_profile("nonexistent-profile-that-does-not-exist-12345");
    match BedrockProvider::new(config).await {
        Ok(_) => {}
        Err(KovaError::Provider { message, status_code }) => {
            assert!(
                message.contains("credentials") || message.contains("credential") || message.contains("provider"),
                "Error message should mention credentials, got: {}",
                message
            );
            assert_eq!(status_code, None);
        }
        Err(other) => panic!("Expected KovaError::Provider, got: {:?}", other),
    }
}

#[test]
fn test_default_timeout_is_60_seconds() {
    let config = BedrockProviderConfig::new("us-east-1", "test-model");
    use crate::provider::config::ProviderConfig;
    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.timeout(), Duration::from_secs(60));
}
