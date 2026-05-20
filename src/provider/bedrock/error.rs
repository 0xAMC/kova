use serde::Deserialize;

use crate::error::KovaError;

#[derive(Debug, Deserialize)]
pub(super) struct BedrockErrorResponse {
    #[serde(default)]
    message: Option<String>,
    #[serde(rename = "__type", default)]
    exception_type: Option<String>,
}

pub(super) fn parse_bedrock_error(status_code: u16, body: &str) -> KovaError {
    match serde_json::from_str::<BedrockErrorResponse>(body) {
        Ok(err_resp) => {
            let message = err_resp.message.unwrap_or_else(|| body.to_string());
            let mapped_status = err_resp
                .exception_type
                .as_deref()
                .and_then(map_exception_status)
                .unwrap_or(status_code);
            KovaError::Provider {
                message,
                status_code: Some(mapped_status),
            }
        }
        Err(_) => KovaError::Provider {
            message: body.to_string(),
            status_code: Some(status_code),
        },
    }
}

pub(super) fn map_exception_status(exception_type: &str) -> Option<u16> {
    match exception_type {
        "ThrottlingException" => Some(429),
        "AccessDeniedException" => Some(403),
        "ValidationException" => Some(400),
        "ModelNotReadyException" => Some(503),
        "ModelTimeoutException" => Some(504),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_throttling_exception() {
        let body = r#"{"__type":"ThrottlingException","message":"Rate exceeded"}"#;
        let err = parse_bedrock_error(500, body);
        match err {
            KovaError::Provider { message, status_code } => {
                assert_eq!(message, "Rate exceeded");
                assert_eq!(status_code, Some(429));
            }
            other => panic!("Expected KovaError::Provider, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_access_denied_exception() {
        let body = r#"{"__type":"AccessDeniedException","message":"Access denied"}"#;
        let err = parse_bedrock_error(500, body);
        match err {
            KovaError::Provider { message, status_code } => {
                assert_eq!(message, "Access denied");
                assert_eq!(status_code, Some(403));
            }
            other => panic!("Expected KovaError::Provider, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_validation_exception() {
        let body = r#"{"__type":"ValidationException","message":"Invalid input"}"#;
        let err = parse_bedrock_error(500, body);
        match err {
            KovaError::Provider { message, status_code } => {
                assert_eq!(message, "Invalid input");
                assert_eq!(status_code, Some(400));
            }
            other => panic!("Expected KovaError::Provider, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_model_not_ready_exception() {
        let body = r#"{"__type":"ModelNotReadyException","message":"Model not ready"}"#;
        let err = parse_bedrock_error(500, body);
        match err {
            KovaError::Provider { message, status_code } => {
                assert_eq!(message, "Model not ready");
                assert_eq!(status_code, Some(503));
            }
            other => panic!("Expected KovaError::Provider, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_model_timeout_exception() {
        let body = r#"{"__type":"ModelTimeoutException","message":"Model timeout"}"#;
        let err = parse_bedrock_error(500, body);
        match err {
            KovaError::Provider { message, status_code } => {
                assert_eq!(message, "Model timeout");
                assert_eq!(status_code, Some(504));
            }
            other => panic!("Expected KovaError::Provider, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_unknown_exception_falls_back_to_original_status() {
        let body = r#"{"__type":"SomeUnknownException","message":"Something went wrong"}"#;
        let err = parse_bedrock_error(418, body);
        match err {
            KovaError::Provider { message, status_code } => {
                assert_eq!(message, "Something went wrong");
                assert_eq!(status_code, Some(418));
            }
            other => panic!("Expected KovaError::Provider, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_non_json_body_falls_back_to_raw_body() {
        let body = "This is not JSON at all";
        let err = parse_bedrock_error(502, body);
        match err {
            KovaError::Provider { message, status_code } => {
                assert_eq!(message, "This is not JSON at all");
                assert_eq!(status_code, Some(502));
            }
            other => panic!("Expected KovaError::Provider, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_missing_message_field_falls_back_to_raw_body() {
        let body = r#"{"__type":"ThrottlingException"}"#;
        let err = parse_bedrock_error(429, body);
        match err {
            KovaError::Provider { message, status_code } => {
                assert_eq!(message, body);
                assert_eq!(status_code, Some(429));
            }
            other => panic!("Expected KovaError::Provider, got: {:?}", other),
        }
    }
}
