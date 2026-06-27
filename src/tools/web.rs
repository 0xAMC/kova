//! Network tools: `fetch_webpage` and `web_search`.
//!
//! # SSRF defense
//!
//! Both tools resolve the target host themselves and reject any hostname that
//! resolves to a loopback, private, link-local, or otherwise internal address
//! (unless [`WebPolicy::allow_private_hosts`] is set). The HTTP client is then
//! *pinned* to the already-validated IP, so a second DNS lookup cannot be
//! rebound to an internal address between the check and the connection (a DNS
//! rebinding attack). Redirects are followed manually, re-running every
//! guardrail on each hop.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::policy::{ToolPolicy, WebFormat, WebPolicy};
use super::{req_str, tool_error, tool_result};
use crate::error::KovaError;
use crate::models::ToolResult;
use crate::tool::Tool;
use std::sync::Arc;

const MAX_REDIRECTS: usize = 5;
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);

// ── fetch_webpage ─────────────────────────────────────────────────────────────

/// Fetches a URL, extracts the main article content, and returns it as Markdown
/// (default) or plain text.
pub struct FetchWebpageTool {
    policy: Arc<ToolPolicy>,
}

impl FetchWebpageTool {
    pub fn new(policy: Arc<ToolPolicy>) -> FetchWebpageTool {
        FetchWebpageTool { policy }
    }
}

#[async_trait]
impl Tool for FetchWebpageTool {
    fn name(&self) -> &str {
        "fetch_webpage"
    }
    fn description(&self) -> &str {
        "Fetch an http(s) URL and return its readable content. HTML pages are run \
         through readability extraction (navigation, ads, and boilerplate removed) \
         and returned as Markdown by default, or plain text with format=\"text\". \
         JSON and plain-text responses are returned as-is; non-textual responses \
         are rejected. Set raw=true to convert the full page without readability. \
         20-second timeout; large responses are truncated."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The http(s) URL to fetch." },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "text"],
                    "description": "Output format. Defaults to markdown (or the configured default)."
                },
                "raw": {
                    "type": "boolean",
                    "description": "Skip readability extraction and convert/strip the whole page. \
                                    Use when extraction drops content you need. Default false."
                }
            },
            "required": ["url"]
        })
    }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let url = match req_str(&args, "url") {
            Some(u) => u.to_string(),
            None => return Ok(tool_error("Missing required parameter 'url'")),
        };
        let format = WebFormat::parse(req_str(&args, "format"), self.policy.web.default_format);
        let raw = args["raw"].as_bool().unwrap_or(false);

        match fetch_webpage(&url, &self.policy.web, format, raw).await {
            Ok(text) => Ok(tool_result(text)),
            Err(msg) => Ok(tool_error(msg)),
        }
    }
}

/// Fetched response body plus the metadata needed to interpret it.
struct FetchedBody {
    bytes: Vec<u8>,
    content_type: Option<String>,
    final_url: reqwest::Url,
}

/// Fetch `url` and render it according to `format`/`raw`, applying every
/// guardrail in `web`.
async fn fetch_webpage(
    url: &str,
    web: &WebPolicy,
    format: WebFormat,
    raw: bool,
) -> Result<String, String> {
    let fetched = fetch_body(url, web).await?;
    let rendered = render_body(&fetched, format, raw)?;
    Ok(truncate_chars(rendered, web.max_content_chars))
}

/// Issue a single guardrailed HTTP GET and return the response body as text.
///
/// This is the building block for higher-level network tools (such as a search
/// tool) that drive a fixed, trusted endpoint and want the raw body rather than
/// readability-extracted content. It applies `web`'s **SSRF protection** — the
/// host is resolved and rejected if it maps to a private/internal address
/// (unless `allow_private_hosts`), and the client is pinned to the validated IP
/// to defeat DNS rebinding — and sends `web`'s `user_agent`.
///
/// It deliberately does *not* enforce the `allowed_urls`/`denied_urls` globs:
/// those govern model-supplied fetch targets, whereas this helper is for
/// endpoints the tool author chose. Redirects are not followed.
pub async fn fetch_text(url: &reqwest::Url, web: &WebPolicy) -> Result<String, String> {
    let addr = validate_target(url, web.allow_private_hosts).await?;
    let host = url
        .host_str()
        .ok_or_else(|| format!("URL '{url}' has no host"))?
        .to_string();
    let client = build_pinned_client(&host, addr, &web.user_agent)?;
    let response = client
        .get(url.clone())
        .send()
        .await
        .map_err(|e| format!("Request to '{url}' failed: {e}"))?;
    response
        .text()
        .await
        .map_err(|e| format!("Cannot read response from '{url}': {e}"))
}

/// Run the manual redirect loop, re-validating scheme/allowlist/private-address
/// guardrails on every hop, and return the raw body of the final response.
async fn fetch_body(url: &str, web: &WebPolicy) -> Result<FetchedBody, String> {
    let mut current = reqwest::Url::parse(url).map_err(|e| format!("Invalid URL '{url}': {e}"))?;

    for _ in 0..=MAX_REDIRECTS {
        web.check_url(&current)?;
        let addr = validate_target(&current, web.allow_private_hosts).await?;
        let host = current
            .host_str()
            .ok_or_else(|| format!("URL '{current}' has no host"))?
            .to_string();

        let client = build_pinned_client(&host, addr, &web.user_agent)?;
        let mut response = client
            .get(current.clone())
            .send()
            .await
            .map_err(|e| format!("Cannot fetch '{current}': {e}"))?;

        if response.status().is_redirection() {
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("Redirect from '{current}' without a Location header"))?;
            current = current
                .join(location)
                .map_err(|e| format!("Invalid redirect target '{location}': {e}"))?;
            continue;
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let mut bytes: Vec<u8> = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| format!("Cannot read response from '{current}': {e}"))?
        {
            bytes.extend_from_slice(&chunk);
            if bytes.len() >= web.max_response_bytes {
                break;
            }
        }

        return Ok(FetchedBody {
            bytes,
            content_type,
            final_url: current,
        });
    }

    Err(format!("Too many redirects fetching '{url}'"))
}

/// Build an HTTP client pinned to `addr` (so a second DNS lookup cannot be
/// rebound to an internal address) and presenting `user_agent`.
fn build_pinned_client(
    host: &str,
    addr: SocketAddr,
    user_agent: &str,
) -> Result<reqwest::Client, String> {
    use reqwest::header::{ACCEPT, ACCEPT_LANGUAGE, HeaderMap, HeaderValue};
    let mut headers = HeaderMap::new();
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
    );
    headers.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("en-US,en;q=0.9"));
    reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(user_agent)
        .default_headers(headers)
        .resolve(host, addr)
        .build()
        .map_err(|e| format!("Cannot build HTTP client: {e}"))
}

/// Interpret a fetched body by content type and render it to text/markdown.
/// All HTML parsing happens here, synchronously, so the `!Send` DOM types from
/// `dom_smoothie`/`htmd` never cross an `.await` in the async fetch path.
fn render_body(fetched: &FetchedBody, format: WebFormat, raw: bool) -> Result<String, String> {
    let kind = fetched
        .content_type
        .as_deref()
        .and_then(|ct| ct.split(';').next())
        .map(|s| s.trim().to_ascii_lowercase());

    match kind.as_deref() {
        // HTML — the common case: extract readable content.
        Some("text/html") | Some("application/xhtml+xml") | None => {
            let html = String::from_utf8_lossy(&fetched.bytes);
            Ok(render_html(&html, fetched.final_url.as_str(), format, raw))
        }
        // JSON — pretty-print when parseable, else pass through.
        Some("application/json") | Some("text/json") => {
            let text = String::from_utf8_lossy(&fetched.bytes);
            Ok(match serde_json::from_str::<Value>(&text) {
                Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| text.into_owned()),
                Err(_) => text.into_owned(),
            })
        }
        // Other textual content (plain, markdown, csv, xml, …) — pass through.
        Some(ct) if ct.starts_with("text/") || ct == "application/xml" => {
            Ok(String::from_utf8_lossy(&fetched.bytes).into_owned())
        }
        // Anything else is binary; don't dump bytes into the model's context.
        Some(ct) => Err(format!(
            "Response from '{}' has non-textual Content-Type '{ct}' ({} bytes); not returned.",
            fetched.final_url,
            fetched.bytes.len()
        )),
    }
}

/// Minimum extracted body length below which we treat readability as having
/// failed and fall back to whole-page conversion.
const MIN_EXTRACTED_CHARS: usize = 200;

/// Render an HTML document: readability extraction by default, or whole-page
/// conversion when `raw` is set or extraction yields too little.
fn render_html(html: &str, url: &str, format: WebFormat, raw: bool) -> String {
    if !raw && let Some(article) = extract_article(html, url, format) {
        return article;
    }
    match format {
        WebFormat::Markdown => html_to_markdown(html).unwrap_or_else(|_| strip_html(html)),
        WebFormat::Text => strip_html(html),
    }
}

/// Convert HTML to Markdown, dropping non-content tags so whole-page fallbacks
/// don't leak CSS/JS or chrome.
fn html_to_markdown(html: &str) -> std::io::Result<String> {
    htmd::HtmlToMarkdown::builder()
        .skip_tags(vec!["script", "style", "head", "nav", "noscript", "iframe", "svg"])
        .build()
        .convert(html)
}

/// Run readability over `html`. Returns `None` (so the caller falls back to
/// whole-page conversion) when extraction fails or produces too little content.
fn extract_article(html: &str, url: &str, format: WebFormat) -> Option<String> {
    use dom_smoothie::{Config, Readability};

    let cfg = Config::default();
    let mut readability = Readability::new(html, Some(url), Some(cfg)).ok()?;
    let article = readability.parse().ok()?;

    let body = match format {
        WebFormat::Markdown => htmd::convert(&article.content).ok()?,
        WebFormat::Text => article.text_content.trim().to_string(),
    };
    if body.trim().chars().count() < MIN_EXTRACTED_CHARS {
        return None;
    }

    // Prepend a compact header so the model keeps the title and source URL.
    let mut out = String::new();
    if !article.title.trim().is_empty() {
        match format {
            WebFormat::Markdown => out.push_str(&format!("# {}\n\n", article.title.trim())),
            WebFormat::Text => out.push_str(&format!("{}\n\n", article.title.trim())),
        }
    }
    if let Some(byline) = article.byline.as_deref().filter(|b| !b.trim().is_empty()) {
        out.push_str(&format!("By {}\n\n", byline.trim()));
    }
    out.push_str(&format!("Source: {url}\n\n"));
    out.push_str(body.trim());
    Some(out)
}

/// Truncate `text` to at most `max_chars` characters, appending a marker.
fn truncate_chars(mut text: String, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        text = text.chars().take(max_chars).collect();
        text.push_str("\n\n[... content truncated ...]");
    }
    text
}

fn strip_html(html: &str) -> String {
    use scraper::Html;
    let document = Html::parse_document(html);
    let text: String = document
        .root_element()
        .text()
        .collect::<Vec<&str>>()
        .join(" ");
    text.split_whitespace().collect::<Vec<&str>>().join(" ")
}

// ── SSRF guards ───────────────────────────────────────────────────────────────

/// Validate scheme and resolve the host, rejecting private/internal addresses
/// unless explicitly allowed. Returns the validated address to connect to.
async fn validate_target(
    url: &reqwest::Url,
    allow_private_hosts: bool,
) -> Result<SocketAddr, String> {
    match url.scheme() {
        "http" | "https" => {}
        s => return Err(format!("URL scheme '{s}' is not allowed (http/https only)")),
    }
    let host = url
        .host_str()
        .ok_or_else(|| format!("URL '{url}' has no host"))?;
    let port = url.port_or_known_default().unwrap_or(443);

    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| format!("Cannot resolve host '{host}': {e}"))?
        .collect();
    let first = *addrs
        .first()
        .ok_or_else(|| format!("Host '{host}' did not resolve to any address"))?;

    if !allow_private_hosts
        && let Some(private) = addrs.iter().find(|a| is_private_addr(&a.ip()))
    {
        return Err(format!(
            "Refusing to fetch '{host}': it resolves to a private/internal address ({}). \
             Set allow_private_hosts for local development.",
            private.ip()
        ));
    }

    Ok(first)
}

fn is_private_addr(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_private_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_private_v4(&mapped);
            }
            let seg0 = v6.segments()[0];
            v6.is_loopback()
                || v6.is_unspecified()
                || (seg0 & 0xfe00) == 0xfc00 // unique-local fc00::/7
                || (seg0 & 0xffc0) == 0xfe80 // link-local fe80::/10
        }
    }
}

fn is_private_v4(v4: &Ipv4Addr) -> bool {
    let octets = v4.octets();
    v4.is_private()
        || v4.is_loopback()
        || v4.is_link_local()
        || v4.is_unspecified()
        || v4.is_broadcast()
        || octets[0] == 0
        || (octets[0] == 100 && (octets[1] & 0xc0) == 64) // CGNAT 100.64.0.0/10
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fetched(bytes: &[u8], content_type: Option<&str>) -> FetchedBody {
        FetchedBody {
            bytes: bytes.to_vec(),
            content_type: content_type.map(|s| s.to_string()),
            final_url: reqwest::Url::parse("https://example.com/x").unwrap(),
        }
    }

    #[tokio::test]
    async fn fetch_rejects_non_http_scheme() {
        let web = WebPolicy {
            https_only: false,
            ..WebPolicy::default()
        };
        let err = fetch_webpage("file:///etc/passwd", &web, WebFormat::Markdown, false)
            .await
            .unwrap_err();
        assert!(err.contains("not allowed") || err.contains("https"));
    }

    #[tokio::test]
    async fn fetch_rejects_loopback_address() {
        let web = WebPolicy {
            https_only: false,
            ..WebPolicy::default()
        };
        let err = fetch_webpage("http://127.0.0.1/", &web, WebFormat::Markdown, false)
            .await
            .unwrap_err();
        assert!(err.contains("private/internal"));
    }

    #[tokio::test]
    async fn fetch_rejects_cloud_metadata_address() {
        let web = WebPolicy {
            https_only: false,
            ..WebPolicy::default()
        };
        let err = fetch_webpage("http://169.254.169.254/latest/meta-data/", &web, WebFormat::Markdown, false)
            .await
            .unwrap_err();
        assert!(err.contains("private/internal"));
    }

    #[tokio::test]
    async fn fetch_rejects_denied_url_before_network() {
        let web = WebPolicy {
            denied_urls: vec![glob::Pattern::new("*evil.com*").unwrap()],
            ..WebPolicy::default()
        };
        let err = fetch_webpage("https://evil.com/", &web, WebFormat::Markdown, false)
            .await
            .unwrap_err();
        assert!(err.contains("denied URL pattern"));
    }

    #[test]
    fn private_addr_classification() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "::1",
            "fc00::1",
            "fe80::1",
        ] {
            assert!(is_private_addr(&ip.parse().unwrap()), "{ip} should be private");
        }
        for ip in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            assert!(!is_private_addr(&ip.parse().unwrap()), "{ip} should be public");
        }
    }

    #[test]
    fn render_body_handles_json_text_and_binary() {
        let json = render_body(&fetched(br#"{"a":1}"#, Some("application/json")), WebFormat::Markdown, false).unwrap();
        assert!(json.contains("\"a\": 1"));

        let text = render_body(&fetched(b"plain text", Some("text/plain")), WebFormat::Markdown, false).unwrap();
        assert_eq!(text, "plain text");

        let binary = render_body(&fetched(&[0, 1, 2], Some("image/png")), WebFormat::Markdown, false);
        assert!(binary.is_err());
    }

    #[test]
    fn render_html_text_mode_has_no_markup() {
        let html = "<html><body><h1>Hi</h1><p>Some text here.</p></body></html>";
        let out = render_html(html, "https://x.com", WebFormat::Text, true);
        assert!(!out.contains('<'));
        assert!(out.contains("Hi"));
    }

    #[test]
    fn truncate_chars_appends_marker() {
        let out = truncate_chars("abcdef".to_string(), 3);
        assert!(out.starts_with("abc"));
        assert!(out.contains("content truncated"));
    }
}
