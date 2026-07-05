//! Batteries-included [`Tool`] implementations for filesystem, shell, and web
//! access.
//!
//! These are generic, ready-to-register tools that cover the common needs of a
//! coding/research agent. They are deliberately decoupled from any host
//! application: every runtime constraint is expressed through [`ToolPolicy`],
//! which the host constructs and injects. The tools themselves know nothing
//! about where that policy comes from.
//!
//! # Features
//!
//! - `tools` — the filesystem and shell tools ([`ReadFileTool`], [`ListDirTool`],
//!   [`SearchTool`], [`EditFileTool`], [`WriteFileTool`], [`PatchFileTool`],
//!   [`ShellTool`]). Light dependencies only.
//! - `web-tools` — additionally the [`FetchWebpageTool`] and the [`fetch_text`]
//!   SSRF-guarded HTTP helper for building further network tools. Pulls in an
//!   HTML parser and readability extraction.
//!
//! # Security
//!
//! The tools enforce, but do not *replace*, an approval layer. [`ToolPolicy`]
//! confines filesystem reads/writes to a workspace root and blocks
//! caller-designated protected directories; the web tools guard against SSRF by
//! refusing private/internal addresses and pinning the connection to the
//! validated IP. Pair them with a [`ToolApprovalHandler`](crate::ToolApprovalHandler)
//! for interactive gating.
//!
//! ```no_run
//! # #[cfg(feature = "tools")]
//! # {
//! use std::sync::Arc;
//! use kova_sdk::tools::{ToolPolicy, register_all_tools_with_policy};
//!
//! let policy = Arc::new(ToolPolicy {
//!     workspace_root: Some("/srv/project".into()),
//!     ..ToolPolicy::default()
//! });
//! let tools = register_all_tools_with_policy(policy);
//! # }
//! ```

mod fs;
mod policy;
mod shell;
#[cfg(feature = "web-tools")]
mod web;

use std::sync::Arc;

use serde_json::Value;

use crate::models::ToolResult;
use crate::tool::Tool;

pub use fs::{EditFileTool, ListDirTool, PatchFileTool, ReadFileTool, SearchTool, WriteFileTool};
pub use policy::{
    DEFAULT_USER_AGENT, ToolPolicy, WebFormat, WebPolicy, normalize_path, resolve_for_containment,
};
pub use shell::ShellTool;
#[cfg(feature = "web-tools")]
pub use web::{FetchWebpageTool, fetch_text, fetch_text_with_headers};

/// Wrap `content` as a successful tool result.
pub fn tool_result(content: impl Into<String>) -> ToolResult {
    ToolResult {
        content: content.into(),
        is_error: false,
    }
}

/// Wrap `msg` as a failed tool result. Tool failures are reported to the model
/// in-band (as an error result) rather than as a transport [`KovaError`], so the
/// model can read the message and recover.
///
/// [`KovaError`]: crate::error::KovaError
pub fn tool_error(msg: impl Into<String>) -> ToolResult {
    ToolResult {
        content: msg.into(),
        is_error: true,
    }
}

/// Read a required string argument, returning `None` when absent or non-string.
pub(crate) fn req_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args[key].as_str()
}

/// Build all built-in tools with a default, unconfined [`ToolPolicy`].
///
/// Convenience for quick starts; production callers should inject a policy via
/// [`register_all_tools_with_policy`] to confine filesystem and network access.
pub fn register_all_tools() -> Vec<Arc<dyn Tool>> {
    register_all_tools_with_policy(Arc::new(ToolPolicy::default()))
}

/// Build all built-in tools bound to `policy`.
///
/// The web tools are included only when the `web-tools` feature is enabled.
pub fn register_all_tools_with_policy(policy: Arc<ToolPolicy>) -> Vec<Arc<dyn Tool>> {
    // `mut` is used only when the web tools are appended below.
    #[cfg_attr(not(feature = "web-tools"), allow(unused_mut))]
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool::new(Arc::clone(&policy))),
        Arc::new(ListDirTool::new(Arc::clone(&policy))),
        Arc::new(SearchTool::new(Arc::clone(&policy))),
        Arc::new(EditFileTool::new(Arc::clone(&policy))),
        Arc::new(WriteFileTool::new(Arc::clone(&policy))),
        Arc::new(PatchFileTool::new(Arc::clone(&policy))),
        Arc::new(ShellTool::new(Arc::clone(&policy))),
    ];
    #[cfg(feature = "web-tools")]
    tools.push(Arc::new(FetchWebpageTool::new(Arc::clone(&policy))));
    tools
}
