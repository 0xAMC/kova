//! Runtime constraints for the built-in tools.
//!
//! [`ToolPolicy`] is the single injection point through which a host application
//! confines tool execution. It carries no notion of *where* its values come
//! from — the host maps its own configuration onto these fields and constructs
//! the policy directly (e.g. `ToolPolicy { workspace_root, ..Default::default() }`).

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

/// A realistic desktop-browser User-Agent used by the web tools.
///
/// The default `reqwest` agent is rejected by a large share of sites
/// (Cloudflare, news sites, …), the most common reason a fetch "fails on a JS
/// site". Presenting as a normal browser avoids that.
pub const DEFAULT_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

/// Default cap on bytes downloaded by the web tools before truncation (2 MiB).
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
/// Default cap on characters returned by the web tools after extraction.
pub const DEFAULT_MAX_CONTENT_CHARS: usize = 50_000;

/// Runtime constraints applied to the built-in tools.
///
/// These limits apply only to tool execution driven by the model. A host's own
/// first-party code is not routed through this policy.
#[derive(Debug, Clone)]
pub struct ToolPolicy {
    /// If set, file tools may only read/write inside this directory, and `shell`
    /// runs with it as the working directory. `None` leaves filesystem access
    /// unconfined (suitable only for trusted, non-interactive use).
    pub workspace_root: Option<PathBuf>,
    /// Directories the file tools may never touch regardless of approval state.
    /// Use this to fence off the host's own configuration and secrets so the
    /// model cannot, for instance, edit a config file to widen its own approvals.
    pub protected_paths: Vec<PathBuf>,
    /// Wall-clock limit for one `shell` invocation; the child is killed on expiry.
    pub shell_timeout: Duration,
    /// Guardrails for the network-facing tools (`fetch_webpage`, `web_search`).
    pub web: WebPolicy,
}

impl Default for ToolPolicy {
    fn default() -> ToolPolicy {
        ToolPolicy {
            workspace_root: None,
            protected_paths: Vec::new(),
            shell_timeout: Duration::from_secs(120),
            web: WebPolicy::default(),
        }
    }
}

impl ToolPolicy {
    /// Reject writes to protected directories or outside the workspace root.
    pub(crate) fn check_write_path(&self, path: &Path) -> Result<(), String> {
        self.check_path(path, "Writes")
    }

    /// Reject reads from protected directories or outside the workspace root.
    ///
    /// Reads are confined like writes: protected directories may hold secrets or
    /// transcripts, and read tools are commonly auto-approved, so containment is
    /// the only barrier against exfiltrating files into the model's context.
    pub(crate) fn check_read_path(&self, path: &Path) -> Result<(), String> {
        self.check_path(path, "Reads")
    }

    fn check_path(&self, path: &Path, action: &str) -> Result<(), String> {
        let resolved = resolve_for_containment(path);
        for protected in &self.protected_paths {
            let protected = resolve_for_containment(protected);
            if resolved.starts_with(&protected) {
                return Err(format!(
                    "{action} under '{}' are not permitted: it is a protected directory.",
                    protected.display()
                ));
            }
        }
        if let Some(root) = &self.workspace_root {
            let root = resolve_for_containment(root);
            if !resolved.starts_with(&root) {
                return Err(format!(
                    "Path '{}' is outside the workspace root '{}'.",
                    resolved.display(),
                    root.display()
                ));
            }
        }
        Ok(())
    }

    /// The directory a relative tool path is resolved against.
    fn base_dir(&self) -> PathBuf {
        self.workspace_root
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    /// Resolve a path argument: relative paths are anchored at the workspace
    /// root so the model can use repo-relative paths naturally.
    pub(crate) fn resolve_arg_path(&self, raw: &str) -> PathBuf {
        let p = Path::new(raw);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.base_dir().join(p)
        }
    }
}

/// Output format for `fetch_webpage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebFormat {
    Markdown,
    Text,
}

impl WebFormat {
    /// Parse a format string, defaulting to `fallback` for unknown/empty input.
    pub fn parse(s: Option<&str>, fallback: WebFormat) -> WebFormat {
        match s {
            Some("text") => WebFormat::Text,
            Some("markdown") => WebFormat::Markdown,
            _ => fallback,
        }
    }
}

/// Resolved, ready-to-enforce guardrails for the network-facing tools. The glob
/// lists are pre-compiled so each request just matches against them.
///
/// Construct from [`WebPolicy::default`] and override the fields you care about;
/// the defaults are safe (HTTPS-only, no private hosts).
#[derive(Debug, Clone)]
pub struct WebPolicy {
    /// Allow reaching loopback/private/link-local addresses (off by default to
    /// block SSRF against cloud metadata and internal services).
    pub allow_private_hosts: bool,
    /// Reject plain `http://`, requiring `https://`.
    pub https_only: bool,
    /// When non-empty, a URL must match one of these globs to be fetched.
    pub allowed_urls: Vec<glob::Pattern>,
    /// Deny-by-default egress: require every URL to match `allowed_urls` even
    /// when that list is empty (so an empty allowlist blocks *all* web access).
    /// Off by default (empty `allowed_urls` allows any URL, subject to the
    /// scheme/host checks).
    pub allowlist_only: bool,
    /// A URL matching any of these globs is always rejected (wins over allow).
    pub denied_urls: Vec<glob::Pattern>,
    /// Cap on bytes downloaded before the body is truncated.
    pub max_response_bytes: usize,
    /// Cap on characters returned after extraction.
    pub max_content_chars: usize,
    /// Format used when a `fetch_webpage` call omits the `format` argument.
    pub default_format: WebFormat,
    /// User-Agent header sent by the web tools.
    pub user_agent: String,
}

impl Default for WebPolicy {
    fn default() -> WebPolicy {
        WebPolicy {
            allow_private_hosts: false,
            https_only: true,
            allowed_urls: Vec::new(),
            allowlist_only: false,
            denied_urls: Vec::new(),
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            max_content_chars: DEFAULT_MAX_CONTENT_CHARS,
            default_format: WebFormat::Markdown,
            user_agent: DEFAULT_USER_AGENT.to_string(),
        }
    }
}

#[cfg(feature = "web-tools")]
impl WebPolicy {
    /// Enforce the scheme and allow/deny URL globs against a parsed URL.
    /// Returns a user-facing message on rejection. The private-host check is
    /// performed separately during DNS resolution (see the web module).
    pub(crate) fn check_url(&self, url: &reqwest::Url) -> Result<(), String> {
        let as_str = url.as_str();
        if self.https_only && url.scheme() != "https" {
            return Err(format!(
                "Refusing to fetch '{as_str}': only https:// URLs are allowed \
                 (https_only is enabled)."
            ));
        }
        if let Some(deny) = self.denied_urls.iter().find(|p| p.matches(as_str)) {
            return Err(format!(
                "Refusing to fetch '{as_str}': it matches a denied URL pattern ('{deny}')."
            ));
        }
        if (self.allowlist_only || !self.allowed_urls.is_empty())
            && !self.allowed_urls.iter().any(|p| p.matches(as_str))
        {
            return Err(if self.allowed_urls.is_empty() {
                format!(
                    "Refusing to fetch '{as_str}': web egress is locked down \
                     (allowlist-only, and the allowlist is empty)."
                )
            } else {
                format!("Refusing to fetch '{as_str}': it is not in the allowlist.")
            });
        }
        Ok(())
    }
}

/// Lexically normalize `path` to an absolute path, resolving `.` and `..`
/// without touching the filesystem.
///
/// Relative paths are anchored at the current working directory. Because `..`
/// is collapsed lexically, this does not follow symlinks — pair it with
/// [`resolve_for_containment`] when the result guards a security boundary.
pub fn normalize_path(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(path)
    };
    let mut out = PathBuf::new();
    for comp in abs.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// Resolve `path` to a canonical, symlink-free absolute path suitable for
/// containment checks (`starts_with` against a root or protected directory).
///
/// The longest existing prefix is canonicalized (following symlinks), then any
/// not-yet-existing trailing components are appended. This defeats both `..`
/// traversal and symlink escapes: a path that canonicalizes outside the
/// workspace root will fail a `starts_with` test even if its textual form
/// looked contained.
pub fn resolve_for_containment(path: &Path) -> PathBuf {
    let norm = normalize_path(path);
    let mut existing = norm.as_path();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    while !existing.exists() {
        match (existing.file_name(), existing.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name.to_os_string());
                existing = parent;
            }
            _ => break,
        }
    }
    let canon = existing
        .canonicalize()
        .unwrap_or_else(|_| existing.to_path_buf());
    tail.into_iter().rev().fold(canon, |acc, c| acc.join(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_path_resolves_dots() {
        assert_eq!(
            normalize_path(Path::new("/a/b/../c/./d")),
            PathBuf::from("/a/c/d")
        );
        assert_eq!(normalize_path(Path::new("/a/../../..")), PathBuf::from("/"));
    }

    #[test]
    fn web_format_parse_falls_back() {
        assert_eq!(
            WebFormat::parse(Some("text"), WebFormat::Markdown),
            WebFormat::Text
        );
        assert_eq!(
            WebFormat::parse(Some("bogus"), WebFormat::Text),
            WebFormat::Text
        );
        assert_eq!(
            WebFormat::parse(None, WebFormat::Markdown),
            WebFormat::Markdown
        );
    }

    #[cfg(feature = "web-tools")]
    #[test]
    fn check_url_enforces_https_only() {
        let policy = WebPolicy::default();
        let http = reqwest::Url::parse("http://example.com").unwrap();
        assert!(policy.check_url(&http).is_err());
        let https = reqwest::Url::parse("https://example.com").unwrap();
        assert!(policy.check_url(&https).is_ok());
    }

    #[cfg(feature = "web-tools")]
    #[test]
    fn check_url_denylist_wins_over_allowlist() {
        let policy = WebPolicy {
            allowed_urls: vec![glob::Pattern::new("https://example.com/*").unwrap()],
            denied_urls: vec![glob::Pattern::new("*/secret*").unwrap()],
            ..WebPolicy::default()
        };
        assert!(
            policy
                .check_url(&reqwest::Url::parse("https://example.com/ok").unwrap())
                .is_ok()
        );
        assert!(
            policy
                .check_url(&reqwest::Url::parse("https://example.com/secret").unwrap())
                .is_err()
        );
        assert!(
            policy
                .check_url(&reqwest::Url::parse("https://other.com/ok").unwrap())
                .is_err()
        );
    }

    #[cfg(feature = "web-tools")]
    #[test]
    fn allowlist_only_with_empty_list_denies_all() {
        let policy = WebPolicy {
            allowlist_only: true,
            ..WebPolicy::default()
        };
        let err = policy
            .check_url(&reqwest::Url::parse("https://example.com/ok").unwrap())
            .unwrap_err();
        assert!(err.contains("locked down"));

        // With a matching allowlist entry, the same mode permits it.
        let policy = WebPolicy {
            allowlist_only: true,
            allowed_urls: vec![glob::Pattern::new("https://example.com/*").unwrap()],
            ..WebPolicy::default()
        };
        assert!(
            policy
                .check_url(&reqwest::Url::parse("https://example.com/ok").unwrap())
                .is_ok()
        );
    }
}
