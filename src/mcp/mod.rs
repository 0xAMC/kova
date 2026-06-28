pub mod tool;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

use crate::error::KovaError;

/// Default per-request timeout for MCP calls (handshake, tools/list, tools/call).
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// MCP transport configuration.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// Spawn a child process and communicate via stdin/stdout.
    Stdio {
        command: String,
        args: Vec<String>,
        /// Extra environment variables set on the spawned process, on top of
        /// the parent environment. Empty for servers that need no configuration.
        env: HashMap<String, String>,
    },
    /// Connect to an MCP server over HTTP+SSE (URL endpoint).
    HttpSse {
        url: String,
        /// Extra HTTP headers sent with every request (e.g. `Authorization`).
        /// Empty for servers that need no headers.
        headers: HashMap<String, String>,
    },
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
    /// Extra headers applied to every request to this server.
    headers: HashMap<String, String>,
    next_id: u64,
}

/// Active connection to an MCP server.
enum McpConnection {
    Stdio(Box<StdioConnection>),
    Http(HttpConnection),
}

/// Client for communicating with an MCP (Model Context Protocol) server.
///
/// Supports stdio and HTTP/SSE transports. After calling [`connect`](Self::connect),
/// use [`tools_list`](Self::tools_list) to discover available tools and
/// [`tools_call`](Self::tools_call) to invoke them.
pub struct McpClient {
    connection: Arc<Mutex<McpConnection>>,
    request_timeout: Duration,
}

impl McpClient {
    /// Establish a connection to an MCP server using the given transport.
    ///
    /// Uses a 30-second per-request timeout; see
    /// [`connect_with_timeout`](Self::connect_with_timeout) to customise.
    pub async fn connect(transport: McpTransport) -> Result<Self, KovaError> {
        Self::connect_with_timeout(transport, DEFAULT_REQUEST_TIMEOUT).await
    }

    /// Establish a connection with a custom per-request timeout.
    ///
    /// The timeout bounds every JSON-RPC round-trip (the `initialize`
    /// handshake, `tools/list`, and each `tools/call`), so a wedged server
    /// cannot hang the agent loop indefinitely.
    pub async fn connect_with_timeout(
        transport: McpTransport,
        request_timeout: Duration,
    ) -> Result<Self, KovaError> {
        let connection = match transport {
            McpTransport::Stdio { command, args, env } => {
                let mut child = tokio::process::Command::new(&command)
                    .args(&args)
                    .envs(&env)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .kill_on_drop(true)
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

                // Surface server diagnostics instead of discarding them.
                if let Some(stderr) = child.stderr.take() {
                    let server = command.clone();
                    tokio::spawn(async move {
                        let mut lines = BufReader::new(stderr).lines();
                        while let Ok(Some(line)) = lines.next_line().await {
                            tracing::debug!(mcp.server = %server, "stderr: {line}");
                        }
                    });
                }

                let conn = StdioConnection {
                    stdin,
                    stdout: BufReader::new(stdout),
                    child,
                    next_id: 1,
                };

                // Send initialize request per MCP protocol.
                let mut locked = conn;
                tokio::time::timeout(request_timeout, Self::send_stdio_initialize(&mut locked))
                    .await
                    .map_err(|_| KovaError::Timeout(request_timeout))??;
                McpConnection::Stdio(Box::new(locked))
            }
            McpTransport::HttpSse { url, headers } => {
                let client = reqwest::Client::builder()
                    .timeout(request_timeout)
                    .build()
                    .map_err(|e| KovaError::Mcp(format!("Failed to build MCP HTTP client: {e}")))?;
                let conn = HttpConnection {
                    base_url: url.trim_end_matches('/').to_string(),
                    client,
                    headers,
                    next_id: 1,
                };
                McpConnection::Http(conn)
            }
        };

        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
            request_timeout,
        })
    }

    /// Discover available tools from the MCP server.
    pub async fn tools_list(&self) -> Result<Vec<McpToolDefinition>, KovaError> {
        let mut conn = self.connection.lock().await;
        let response = tokio::time::timeout(
            self.request_timeout,
            Self::send_request(&mut conn, "tools/list", None),
        )
        .await
        .map_err(|_| KovaError::Timeout(self.request_timeout))??;

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

        let response = tokio::time::timeout(
            self.request_timeout,
            Self::send_request(&mut conn, "tools/call", Some(params)),
        )
        .await
        .map_err(|_| KovaError::Timeout(self.request_timeout))??;

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
            if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(trimmed)
                && resp.id == Some(id)
            {
                if let Some(err) = resp.error {
                    return Err(KovaError::Mcp(format!("MCP error: {}", err.message)));
                }
                return resp
                    .result
                    .ok_or_else(|| KovaError::Mcp("MCP response missing result".into()));
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

        let mut builder = conn.client.post(&conn.base_url);
        for (key, value) in &conn.headers {
            builder = builder.header(key, value);
        }
        let resp = builder
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
                headers: HashMap::new(),
                next_id: 1,
            }))),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
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

    /// HTTP transport headers are attached to every JSON-RPC request, so bearer
    /// tokens and similar credentials reach the server.
    #[tokio::test]
    async fn http_transport_sends_configured_headers() {
        use wiremock::matchers::{header, method};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header("authorization", "Bearer secret-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": { "tools": [{ "name": "ping" }] }
            })))
            .mount(&server)
            .await;

        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            "Bearer secret-token".to_string(),
        );
        let client = McpClient::connect(McpTransport::HttpSse {
            url: server.uri(),
            headers,
        })
        .await
        .expect("connect");

        // tools/list only succeeds if the Authorization header matched the mock.
        let tools = client.tools_list().await.expect("tools_list");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "ping");
    }

    /// stdio transport environment variables reach the spawned child process.
    /// Uses an unbuffered python MCP stub that echoes an env var as a tool name;
    /// skipped when python3 is unavailable so CI without it stays green.
    #[cfg(unix)]
    #[tokio::test]
    async fn stdio_transport_passes_env_to_child() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: python3 not available");
            return;
        }

        // Minimal MCP stdio server: reply to `initialize` and `tools/list`,
        // naming the single tool after $MCP_TEST_TOOL so the test can prove the
        // env var was inherited by the child.
        let stub = r#"
import sys, json, os
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialize":
        print(json.dumps({"jsonrpc":"2.0","id":msg["id"],"result":{}}), flush=True)
    elif method == "tools/list":
        name = os.environ.get("MCP_TEST_TOOL", "missing")
        print(json.dumps({"jsonrpc":"2.0","id":msg["id"],"result":{"tools":[{"name":name}]}}), flush=True)
"#;

        let mut env = HashMap::new();
        env.insert("MCP_TEST_TOOL".to_string(), "from_env".to_string());
        let client = McpClient::connect(McpTransport::Stdio {
            command: "python3".to_string(),
            args: vec!["-u".to_string(), "-c".to_string(), stub.to_string()],
            env,
        })
        .await
        .expect("connect");

        let tools = client.tools_list().await.expect("tools_list");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "from_env");
    }
}
