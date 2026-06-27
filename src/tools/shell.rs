//! The `shell` tool: run a command via `sh -c` in the workspace directory.
//!
//! Output is head+tail truncated and byte-capped so a runaway command cannot
//! exhaust memory or the next prompt's context window. The child is killed if it
//! exceeds [`ToolPolicy::shell_timeout`](super::ToolPolicy).

use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use super::policy::ToolPolicy;
use super::{req_str, tool_error, tool_result};
use crate::error::KovaError;
use crate::models::ToolResult;
use crate::tool::Tool;

const OUTPUT_HEAD_LINES: usize = 100;
const OUTPUT_TAIL_LINES: usize = 100;
const MAX_OUTPUT_BYTES: usize = 100_000;

fn format_output(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let code = output.status.code().unwrap_or(-1);
    format!("exit_code: {code}\nstdout:\n{stdout}\nstderr:\n{stderr}")
}

/// Head+tail truncate `s` by lines, then hard-cap the total byte length on a
/// UTF-8 char boundary.
fn truncate_output(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let mut out = if lines.len() > OUTPUT_HEAD_LINES + OUTPUT_TAIL_LINES {
        let head = lines[..OUTPUT_HEAD_LINES].join("\n");
        let tail = lines[lines.len() - OUTPUT_TAIL_LINES..].join("\n");
        let skipped = lines.len() - OUTPUT_HEAD_LINES - OUTPUT_TAIL_LINES;
        format!("{head}\n[... {skipped} lines truncated ...]\n{tail}")
    } else {
        s.to_string()
    };
    if out.len() > MAX_OUTPUT_BYTES {
        let mut end = MAX_OUTPUT_BYTES;
        while end > 0 && !out.is_char_boundary(end) {
            end -= 1;
        }
        out.truncate(end);
        out.push_str("\n[... output truncated ...]");
    }
    out
}

/// Executes a shell command via `sh -c` and returns combined output.
pub struct ShellTool {
    policy: Arc<ToolPolicy>,
}

impl ShellTool {
    pub fn new(policy: Arc<ToolPolicy>) -> ShellTool {
        ShellTool { policy }
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }
    fn description(&self) -> &str {
        "Execute a shell command via `sh -c` in the workspace directory. \
         Output is truncated when very long; long-running commands are killed on timeout."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to execute" }
            },
            "required": ["command"]
        })
    }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let command = match req_str(&args, "command") {
            Some(c) => c.to_string(),
            None => return Ok(tool_error("Missing required parameter 'command'")),
        };
        let mut cmd = Command::new("sh");
        cmd.args(["-c", &command])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(root) = &self.policy.workspace_root {
            cmd.current_dir(root);
        }
        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return Ok(tool_error(format!("Cannot run shell: {e}"))),
        };
        match tokio::time::timeout(self.policy.shell_timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => Ok(tool_result(truncate_output(&format_output(&output)))),
            Ok(Err(e)) => Ok(tool_error(format!("Cannot run shell: {e}"))),
            // Dropping the timed-out future kills the child (kill_on_drop).
            Err(_) => Ok(tool_error(format!(
                "Command timed out after {}s and was killed",
                self.policy.shell_timeout.as_secs()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn shell_echo_and_exit_code() {
        let tool = ShellTool::new(Arc::new(ToolPolicy::default()));
        let ok = tool
            .execute(json!({ "command": "echo hello" }))
            .await
            .unwrap();
        assert!(!ok.is_error);
        assert!(ok.content.contains("hello"));
        assert!(ok.content.contains("exit_code: 0"));

        let nonzero = tool.execute(json!({ "command": "exit 3" })).await.unwrap();
        assert!(nonzero.content.contains("exit_code: 3"));
    }

    #[tokio::test]
    async fn shell_runs_in_workspace_root() {
        let tmp = tempfile::TempDir::new().unwrap();
        let canonical = tmp.path().canonicalize().unwrap();
        let policy = Arc::new(ToolPolicy {
            workspace_root: Some(canonical.clone()),
            ..ToolPolicy::default()
        });
        let tool = ShellTool::new(policy);
        let result = tool.execute(json!({ "command": "pwd" })).await.unwrap();
        assert!(result.content.contains(&*canonical.to_string_lossy()));
    }

    #[tokio::test]
    async fn shell_times_out_and_kills_command() {
        let policy = Arc::new(ToolPolicy {
            shell_timeout: Duration::from_millis(200),
            ..ToolPolicy::default()
        });
        let tool = ShellTool::new(policy);
        let result = tool.execute(json!({ "command": "sleep 5" })).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
    }

    #[test]
    fn truncate_output_caps_bytes_and_lines() {
        assert_eq!(truncate_output("short"), "short");
        let big = "a".repeat(MAX_OUTPUT_BYTES + 5000);
        let out = truncate_output(&big);
        assert!(out.len() <= MAX_OUTPUT_BYTES + 64);
        assert!(out.contains("output truncated"));
    }
}
