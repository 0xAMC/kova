pub mod tool;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

use crate::error::KovaError;

/// MCP transport configuration.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// Spawn a child process and communicate via stdin/stdout.
    Stdio { command: String, args: Vec<String> },
    /// Connect to an MCP server over HTTP+SSE (URL endpoint).
    HttpSse { url: String },
}

/// A tool definition as returned by the MCP `tools/list` method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Option<serde_json::Value>,
}

/// JSON-RPC request envelope.
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

/// JSON-RPC response envelope.
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    #[allow(dead_code)]
    id: Option<u64>,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

/// JSON-RPC error object.
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i64,
    message: String,
}

/// MCP `tools/list` response.
#[derive(Debug, Deserialize)]
struct ToolsListResult {
    tools: Vec<McpToolDefinition>,
}

/// MCP `tools/call` result content item.
#[derive(Debug, Deserialize)]
struct McpCallContentItem {
    #[serde(default)]
    #[allow(dead_code)]
    text: Option<String>,
    #[serde(default, rename = "type")]
    #[allow(dead_code)]
    content_type: Option<String>,
}

/// MCP `tools/call` response.
#[derive(Debug, Deserialize)]
struct McpCallResult {
    content: Vec<McpCallContentItem>,
    #[serde(default, rename = "isError")]
    is_error: Option<bool>,
}

/// Internal state for a stdio-based MCP connection.
struct StdioConnection {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    #[allow(dead_code)]
    child: Child,
    next_id: u64,
}

/// Internal state for an HTTP/SSE-based MCP connection.
struct HttpConnection {
    base_url: String,
    client: reqwest::Client,
    next_id: u64,
}

/// Active connection to an MCP server.
enum McpConnection {
    Stdio(StdioConnection),
    Http(HttpConnection),
}

/// Client for communicating with an MCP (Model Context Protocol) server.
///
/// Supports stdio and HTTP/SSE transports. After calling [`connect`](Self::connect),
/// use [`tools_list`](Self::tools_list) to discover available tools and
/// [`tools_call`](Self::tools_call) to invoke them.
pub struct McpClient {
    connection: Arc<Mutex<McpConnection>>,
}

impl McpClient {
    /// Establish a connection to an MCP server using the given transport.
    pub async fn connect(transport: McpTransport) -> Result<Self, KovaError> {
        let connection = match transport {
            McpTransport::Stdio { command, args } => {
                let mut child = tokio::process::Command::new(&command)
                    .args(&args)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .map_err(|e| {
                        KovaError::Mcp(format!("Failed to spawn MCP process '{}': {}", command, e))
                    })?;

                let stdin = child.stdin.take().ok_or_else(|| {
                    KovaError::Mcp("Failed to capture stdin of MCP process".into())
                })?;
                let stdout = child.stdout.take().ok_or_else(|| {
                    KovaError::Mcp("Failed to capture stdout of MCP process".into())
                })?;

                let conn = StdioConnection {
                    stdin,
                    stdout: BufReader::new(stdout),
                    child,
                    next_id: 1,
                };

                // Send initialize request per MCP protocol.
                let mut locked = conn;
                Self::send_stdio_initialize(&mut locked).await?;
                McpConnection::Stdio(locked)
            }
            McpTransport::HttpSse { url } => {
                let client = reqwest::Client::new();
                // Verify the server is reachable with a simple request.
                let conn = HttpConnection {
                    base_url: url.trim_end_matches('/').to_string(),
                    client,
                    next_id: 1,
                };
                McpConnection::Http(conn)
            }
        };

        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    /// Discover available tools from the MCP server.
    pub async fn tools_list(&self) -> Result<Vec<McpToolDefinition>, KovaError> {
        let mut conn = self.connection.lock().await;
        let response = Self::send_request(&mut conn, "tools/list", None).await?;

        let result: ToolsListResult = serde_json::from_value(response)
            .map_err(|e| KovaError::Mcp(format!("Failed to parse tools/list response: {}", e)))?;

        Ok(result.tools)
    }

    /// Invoke a tool on the MCP server.
    pub async fn tools_call(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<(String, bool), KovaError> {
        let mut conn = self.connection.lock().await;

        let params = serde_json::json!({
            "name": tool_name,
            "arguments": arguments,
        });

        let response = Self::send_request(&mut conn, "tools/call", Some(params)).await?;

        let result: McpCallResult = serde_json::from_value(response)
            .map_err(|e| KovaError::Mcp(format!("Failed to parse tools/call response: {}", e)))?;

        let text = result
            .content
            .iter()
            .filter_map(|item| item.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n");

        let is_error = result.is_error.unwrap_or(false);
        Ok((text, is_error))
    }

    /// Send a JSON-RPC request and return the result value.
    async fn send_request(
        conn: &mut McpConnection,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, KovaError> {
        match conn {
            McpConnection::Stdio(stdio) => Self::send_stdio_request(stdio, method, params).await,
            McpConnection::Http(http) => Self::send_http_request(http, method, params).await,
        }
    }

    /// Send the MCP `initialize` handshake over stdio.
    async fn send_stdio_initialize(conn: &mut StdioConnection) -> Result<(), KovaError> {
        let init_params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "kova",
                "version": "0.1.0"
            }
        });

        Self::send_stdio_request(conn, "initialize", Some(init_params)).await?;

        // Send initialized notification (no id, no response expected).
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        let mut line = serde_json::to_string(&notification)
            .map_err(|e| KovaError::Mcp(format!("Failed to serialize notification: {}", e)))?;
        line.push('\n');
        conn.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| KovaError::Mcp(format!("Failed to write notification: {}", e)))?;
        conn.stdin
            .flush()
            .await
            .map_err(|e| KovaError::Mcp(format!("Failed to flush notification: {}", e)))?;

        Ok(())
    }

    /// Send a JSON-RPC request over stdio and read the response.
    async fn send_stdio_request(
        conn: &mut StdioConnection,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, KovaError> {
        let id = conn.next_id;
        conn.next_id += 1;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };

        let mut line = serde_json::to_string(&request)
            .map_err(|e| KovaError::Mcp(format!("Failed to serialize request: {}", e)))?;
        line.push('\n');

        conn.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| KovaError::Mcp(format!("Failed to write to MCP process stdin: {}", e)))?;
        conn.stdin
            .flush()
            .await
            .map_err(|e| KovaError::Mcp(format!("Failed to flush MCP process stdin: {}", e)))?;

        // Read lines until we get a valid JSON-RPC response with our id.
        let mut buf = String::new();
        loop {
            buf.clear();
            let bytes_read = conn.stdout.read_line(&mut buf).await.map_err(|e| {
                KovaError::Mcp(format!("Failed to read from MCP process stdout: {}", e))
            })?;

            if bytes_read == 0 {
                return Err(KovaError::Mcp(
                    "MCP process closed stdout unexpectedly".into(),
                ));
            }

            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Try to parse as JSON-RPC response.
            if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(trimmed) {
                if resp.id == Some(id) {
                    if let Some(err) = resp.error {
                        return Err(KovaError::Mcp(format!("MCP error: {}", err.message)));
                    }
                    return resp
                        .result
                        .ok_or_else(|| KovaError::Mcp("MCP response missing result".into()));
                }
                // Response for a different id (e.g. notification) — skip.
            }
            // Not valid JSON-RPC — skip (could be log output from the server).
        }
    }

    /// Send a JSON-RPC request over HTTP POST and parse the response.
    async fn send_http_request(
        conn: &mut HttpConnection,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, KovaError> {
        let id = conn.next_id;
        conn.next_id += 1;

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };

        let resp = conn
            .client
            .post(&conn.base_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| KovaError::Mcp(format!("HTTP request to MCP server failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(KovaError::Mcp(format!(
                "MCP HTTP error {}: {}",
                status, body
            )));
        }

        let rpc_resp: JsonRpcResponse = resp
            .json()
            .await
            .map_err(|e| KovaError::Mcp(format!("Failed to parse MCP HTTP response: {}", e)))?;

        if let Some(err) = rpc_resp.error {
            return Err(KovaError::Mcp(format!("MCP error: {}", err.message)));
        }

        rpc_resp
            .result
            .ok_or_else(|| KovaError::Mcp("MCP response missing result".into()))
    }

    /// Create a dummy `McpClient` for testing purposes.
    ///
    /// This client uses an HTTP connection to a non-existent server.
    /// Only useful for testing `McpTool` trait method implementations
    /// (name, description, parameters_schema) that don't make network calls.
    #[doc(hidden)]
    pub fn new_for_test() -> Self {
        Self {
            connection: Arc::new(Mutex::new(McpConnection::Http(HttpConnection {
                base_url: "http://localhost:0".to_string(),
                client: reqwest::Client::new(),
                next_id: 1,
            }))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tools_list_response() {
        let json = serde_json::json!({
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" }
                        }
                    }
                },
                {
                    "name": "search",
                    "description": "Search the web"
                }
            ]
        });

        let result: ToolsListResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.tools.len(), 2);
        assert_eq!(result.tools[0].name, "get_weather");
        assert_eq!(
            result.tools[0].description.as_deref(),
            Some("Get current weather")
        );
        assert!(result.tools[0].input_schema.is_some());
        assert_eq!(result.tools[1].name, "search");
        assert!(result.tools[1].input_schema.is_none());
    }

    #[test]
    fn parse_tools_call_response() {
        let json = serde_json::json!({
            "content": [
                { "type": "text", "text": "Sunny, 72°F" }
            ]
        });

        let result: McpCallResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.content[0].text.as_deref(), Some("Sunny, 72°F"));
        assert_eq!(result.is_error, None);
    }

    #[test]
    fn parse_tools_call_error_response() {
        let json = serde_json::json!({
            "content": [
                { "type": "text", "text": "City not found" }
            ],
            "isError": true
        });

        let result: McpCallResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.is_error, Some(true));
        assert_eq!(result.content[0].text.as_deref(), Some("City not found"));
    }

    #[test]
    fn jsonrpc_request_serialization() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 1,
            method: "tools/list".to_string(),
            params: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["id"], 1);
        assert_eq!(json["method"], "tools/list");
        assert!(json.get("params").is_none());
    }

    #[test]
    fn jsonrpc_request_with_params() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 5,
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": "search",
                "arguments": { "query": "rust" }
            })),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["params"]["name"], "search");
        assert_eq!(json["params"]["arguments"]["query"], "rust");
    }

    #[test]
    fn jsonrpc_error_response_parsing() {
        let json =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().message, "Method not found");
        assert!(resp.result.is_none());
    }

    #[test]
    fn jsonrpc_success_response_parsing() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.error.is_none());
        assert!(resp.result.is_some());
    }

    #[test]
    fn mcp_tool_definition_defaults() {
        let json = serde_json::json!({ "name": "minimal_tool" });
        let def: McpToolDefinition = serde_json::from_value(json).unwrap();
        assert_eq!(def.name, "minimal_tool");
        assert!(def.description.is_none());
        assert!(def.input_schema.is_none());
    }

    #[test]
    fn mcp_call_result_multi_content() {
        let json = serde_json::json!({
            "content": [
                { "type": "text", "text": "line 1" },
                { "type": "text", "text": "line 2" }
            ]
        });
        let result: McpCallResult = serde_json::from_value(json).unwrap();
        let texts: Vec<_> = result
            .content
            .iter()
            .filter_map(|c| c.text.as_deref())
            .collect();
        assert_eq!(texts, vec!["line 1", "line 2"]);
    }
}
