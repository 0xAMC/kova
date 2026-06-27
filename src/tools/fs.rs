//! Filesystem tools: `read_file`, `list_dir`, `search`, `edit_file`,
//! `write_file`, `patch_file`.
//!
//! All paths are resolved and confined through the shared [`ToolPolicy`], so a
//! relative path is anchored at the workspace root and any path escaping the
//! root (or hitting a protected directory) is rejected before I/O happens.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::{req_str, tool_error, tool_result};
use super::policy::ToolPolicy;
use crate::error::KovaError;
use crate::models::ToolResult;
use crate::tool::Tool;

/// Heuristic binary check: a NUL byte in the first 8 KiB.
fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|b| *b == 0)
}

// ── read_file ───────────────────────────────────────────────────────────────

const READ_MAX_LINES: usize = 2000;
const READ_MAX_BYTES: u64 = 10 * 1024 * 1024;

/// Reads a text file, optionally from a line offset with a line limit.
pub struct ReadFileTool {
    policy: Arc<ToolPolicy>,
}

impl ReadFileTool {
    pub fn new(policy: Arc<ToolPolicy>) -> ReadFileTool {
        ReadFileTool { policy }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }
    fn description(&self) -> &str {
        "Read a text file and return its content. Use `offset` (1-based line number) and \
         `limit` (number of lines) for large files; at most 2000 lines are returned per call."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path (relative paths resolve against the workspace root)" },
                "offset": { "type": "integer", "description": "1-based line number to start reading from" },
                "limit": { "type": "integer", "description": "Maximum number of lines to return" }
            },
            "required": ["path"]
        })
    }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let path = match req_str(&args, "path") {
            Some(p) => p.to_string(),
            None => return Ok(tool_error("Missing required parameter 'path'")),
        };
        let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
        let limit =
            (args["limit"].as_u64().unwrap_or(READ_MAX_LINES as u64) as usize).min(READ_MAX_LINES);

        let resolved = self.policy.resolve_arg_path(&path);
        if let Err(reason) = self.policy.check_read_path(&resolved) {
            return Ok(tool_error(reason));
        }
        match std::fs::metadata(&resolved) {
            Ok(meta) if meta.is_dir() => {
                return Ok(tool_error(format!(
                    "'{path}' is a directory — use list_dir instead"
                )));
            }
            Ok(meta) if meta.len() > READ_MAX_BYTES => {
                return Ok(tool_error(format!(
                    "'{path}' is too large to read ({} bytes; limit {READ_MAX_BYTES})",
                    meta.len()
                )));
            }
            Err(e) => return Ok(tool_error(format!("Cannot read '{path}': {e}"))),
            _ => {}
        }
        let bytes = match std::fs::read(&resolved) {
            Ok(b) => b,
            Err(e) => return Ok(tool_error(format!("Cannot read '{path}': {e}"))),
        };
        if looks_binary(&bytes) {
            return Ok(tool_error(format!("'{path}' appears to be a binary file")));
        }
        let content = String::from_utf8_lossy(&bytes);
        let total_lines = content.lines().count();
        if offset > total_lines && total_lines > 0 {
            return Ok(tool_error(format!(
                "Offset {offset} is past the end of '{path}' ({total_lines} lines)"
            )));
        }
        let selected: Vec<&str> = content.lines().skip(offset - 1).take(limit).collect();
        let mut out = selected.join("\n");
        let shown = selected.len();
        if offset > 1 || shown < total_lines.saturating_sub(offset - 1) {
            out.push_str(&format!(
                "\n[showing lines {offset}–{} of {total_lines}]",
                offset + shown.saturating_sub(1)
            ));
        }
        Ok(tool_result(out))
    }
}

// ── list_dir ────────────────────────────────────────────────────────────────

const LIST_MAX_ENTRIES: usize = 500;

/// Lists the entries of a directory (directories get a `/` suffix).
pub struct ListDirTool {
    policy: Arc<ToolPolicy>,
}

impl ListDirTool {
    pub fn new(policy: Arc<ToolPolicy>) -> ListDirTool {
        ListDirTool { policy }
    }
}

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }
    fn description(&self) -> &str {
        "List the entries of a directory. Directories are suffixed with '/'. \
         Defaults to the workspace root when no path is given."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path (defaults to the workspace root)" }
            }
        })
    }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let raw = req_str(&args, "path").unwrap_or(".").to_string();
        let resolved = self.policy.resolve_arg_path(&raw);
        if let Err(reason) = self.policy.check_read_path(&resolved) {
            return Ok(tool_error(reason));
        }
        let entries = match std::fs::read_dir(&resolved) {
            Ok(e) => e,
            Err(e) => return Ok(tool_error(format!("Cannot list '{raw}': {e}"))),
        };
        let mut names: Vec<String> = Vec::new();
        for entry in entries.flatten() {
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                name.push('/');
            }
            names.push(name);
        }
        names.sort();
        let total = names.len();
        if total == 0 {
            return Ok(tool_result(format!("'{raw}' is empty")));
        }
        let mut out = names
            .iter()
            .take(LIST_MAX_ENTRIES)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if total > LIST_MAX_ENTRIES {
            out.push_str(&format!(
                "\n[... {} more entries not shown ...]",
                total - LIST_MAX_ENTRIES
            ));
        }
        Ok(tool_result(out))
    }
}

// ── search ──────────────────────────────────────────────────────────────────

const SEARCH_MAX_RESULTS: usize = 200;
const SEARCH_MAX_FILES: usize = 20_000;
const SEARCH_MAX_FILE_BYTES: u64 = 1024 * 1024;
const SEARCH_MAX_LINE_CHARS: usize = 400;

/// ripgrep-style recursive content search (regex) with an optional filename glob.
pub struct SearchTool {
    policy: Arc<ToolPolicy>,
}

impl SearchTool {
    pub fn new(policy: Arc<ToolPolicy>) -> SearchTool {
        SearchTool { policy }
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }
    fn description(&self) -> &str {
        "Search file contents recursively with a regular expression. Optional `glob` filters \
         file names (e.g. '*.rs'); `path` restricts the search to a subdirectory. Hidden \
         files and binary files are skipped. Results are 'path:line: text' lines."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regular expression to search for" },
                "path": { "type": "string", "description": "Directory to search (defaults to the workspace root)" },
                "glob": { "type": "string", "description": "Filename glob filter, e.g. '*.rs'" }
            },
            "required": ["pattern"]
        })
    }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let pattern = match req_str(&args, "pattern") {
            Some(p) => p.to_string(),
            None => return Ok(tool_error("Missing required parameter 'pattern'")),
        };
        let raw_root = req_str(&args, "path").unwrap_or(".").to_string();
        let glob_filter = match req_str(&args, "glob") {
            Some(g) => match glob::Pattern::new(g) {
                Ok(p) => Some(p),
                Err(e) => return Ok(tool_error(format!("Invalid glob '{g}': {e}"))),
            },
            None => None,
        };
        let re = match regex::Regex::new(&pattern) {
            Ok(r) => r,
            Err(e) => return Ok(tool_error(format!("Invalid regex '{pattern}': {e}"))),
        };

        let root = self.policy.resolve_arg_path(&raw_root);
        if let Err(reason) = self.policy.check_read_path(&root) {
            return Ok(tool_error(reason));
        }
        if !root.is_dir() {
            return Ok(tool_error(format!("'{raw_root}' is not a directory")));
        }

        // The walk is synchronous filesystem work; move it off the async thread.
        let policy = Arc::clone(&self.policy);
        let result = tokio::task::spawn_blocking(move || {
            search_dir(&root, &re, glob_filter.as_ref(), &policy)
        })
        .await
        .map_err(|e| KovaError::ToolExecution {
            tool_name: "search".into(),
            message: format!("search task failed: {e}"),
        })?;

        Ok(tool_result(result))
    }
}

/// Walk `root` depth-first, matching `re` against each line of every regular
/// text file that passes the glob filter. Hidden entries (dot-prefixed),
/// protected paths, files over 1 MiB, and binary files are skipped.
fn search_dir(
    root: &Path,
    re: &regex::Regex,
    glob_filter: Option<&glob::Pattern>,
    policy: &ToolPolicy,
) -> String {
    let mut matches: Vec<String> = Vec::new();
    let mut files_scanned = 0usize;
    let mut truncated = false;

    let mut stack = vec![root.to_path_buf()];
    'walk: while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                if policy.check_read_path(&path).is_ok() {
                    stack.push(path);
                }
                continue;
            }
            if let Some(g) = glob_filter
                && !g.matches(&name)
            {
                continue;
            }
            if files_scanned >= SEARCH_MAX_FILES {
                truncated = true;
                break 'walk;
            }
            files_scanned += 1;
            if std::fs::metadata(&path)
                .map(|m| m.len() > SEARCH_MAX_FILE_BYTES)
                .unwrap_or(true)
            {
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            if looks_binary(&bytes) {
                continue;
            }
            let content = String::from_utf8_lossy(&bytes);
            let display = path.strip_prefix(root).unwrap_or(&path).display().to_string();
            for (i, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    let mut text = line.trim_end().to_string();
                    if text.chars().count() > SEARCH_MAX_LINE_CHARS {
                        text = text.chars().take(SEARCH_MAX_LINE_CHARS).collect();
                        text.push('…');
                    }
                    matches.push(format!("{display}:{}: {text}", i + 1));
                    if matches.len() >= SEARCH_MAX_RESULTS {
                        truncated = true;
                        break 'walk;
                    }
                }
            }
        }
    }

    if matches.is_empty() {
        return "No matches found".to_string();
    }
    let mut out = matches.join("\n");
    if truncated {
        out.push_str(&format!(
            "\n[results truncated at {SEARCH_MAX_RESULTS} matches / {SEARCH_MAX_FILES} files — narrow the pattern, path, or glob]"
        ));
    }
    out
}

// ── edit_file ───────────────────────────────────────────────────────────────

/// Targeted string replacement in an existing file (`old_string` → `new_string`).
pub struct EditFileTool {
    policy: Arc<ToolPolicy>,
}

impl EditFileTool {
    pub fn new(policy: Arc<ToolPolicy>) -> EditFileTool {
        EditFileTool { policy }
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }
    fn description(&self) -> &str {
        "Replace an exact string in an existing file. `old_string` must match the file \
         content exactly and unambiguously (include surrounding lines to disambiguate); \
         set `replace_all` to true to replace every occurrence. Use write_file to create \
         or overwrite whole files."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "old_string": { "type": "string", "description": "Exact text to replace" },
                "new_string": { "type": "string", "description": "Replacement text" },
                "replace_all": { "type": "boolean", "description": "Replace every occurrence (default: false)" }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let path = match req_str(&args, "path") {
            Some(p) => p.to_string(),
            None => return Ok(tool_error("Missing required parameter 'path'")),
        };
        let old_string = match req_str(&args, "old_string") {
            Some(s) => s.to_string(),
            None => return Ok(tool_error("Missing required parameter 'old_string'")),
        };
        let new_string = match req_str(&args, "new_string") {
            Some(s) => s.to_string(),
            None => return Ok(tool_error("Missing required parameter 'new_string'")),
        };
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);

        if old_string.is_empty() {
            return Ok(tool_error("'old_string' must not be empty"));
        }
        if old_string == new_string {
            return Ok(tool_error("'old_string' and 'new_string' are identical"));
        }

        let resolved = self.policy.resolve_arg_path(&path);
        if let Err(reason) = self.policy.check_write_path(&resolved) {
            return Ok(tool_error(reason));
        }
        let original = match std::fs::read_to_string(&resolved) {
            Ok(s) => s,
            Err(e) => {
                return Ok(tool_error(format!(
                    "Cannot read '{path}': {e}. edit_file only modifies existing files — \
                     use write_file to create one."
                )));
            }
        };
        let occurrences = original.matches(&old_string).count();
        match occurrences {
            0 => Ok(tool_error(format!(
                "old_string not found in '{path}'. Make sure it matches the file content \
                 exactly, including whitespace and indentation."
            ))),
            1 => {
                let updated = original.replacen(&old_string, &new_string, 1);
                match std::fs::write(&resolved, updated) {
                    Ok(()) => Ok(tool_result(format!("Edited '{path}' (1 replacement)"))),
                    Err(e) => Ok(tool_error(format!("Cannot write '{path}': {e}"))),
                }
            }
            n if replace_all => {
                let updated = original.replace(&old_string, &new_string);
                match std::fs::write(&resolved, updated) {
                    Ok(()) => Ok(tool_result(format!("Edited '{path}' ({n} replacements)"))),
                    Err(e) => Ok(tool_error(format!("Cannot write '{path}': {e}"))),
                }
            }
            n => Ok(tool_error(format!(
                "old_string matches {n} locations in '{path}'. Add surrounding context to \
                 make it unique, or set replace_all to true."
            ))),
        }
    }
}

// ── write_file ──────────────────────────────────────────────────────────────

/// Overwrites a file with new content (creating it and parents if missing).
pub struct WriteFileTool {
    policy: Arc<ToolPolicy>,
}

impl WriteFileTool {
    pub fn new(policy: Arc<ToolPolicy>) -> WriteFileTool {
        WriteFileTool { policy }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }
    fn description(&self) -> &str {
        "Write a file with the provided content, creating it (and parent directories) if \
         missing and overwriting it otherwise. Prefer edit_file for targeted changes to \
         existing files."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        })
    }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let path = match req_str(&args, "path") {
            Some(p) => p.to_string(),
            None => return Ok(tool_error("Missing required parameter 'path'")),
        };
        let content = match req_str(&args, "content") {
            Some(c) => c.to_string(),
            None => return Ok(tool_error("Missing required parameter 'content'")),
        };
        let resolved = self.policy.resolve_arg_path(&path);
        if let Err(reason) = self.policy.check_write_path(&resolved) {
            return Ok(tool_error(reason));
        }
        if let Some(parent) = resolved.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(&resolved, content) {
            Ok(()) => Ok(tool_result(format!("Wrote '{path}'"))),
            Err(e) => Ok(tool_error(format!("Cannot write '{path}': {e}"))),
        }
    }
}

// ── patch_file ──────────────────────────────────────────────────────────────

/// Applies a unified diff patch to an existing file.
pub struct PatchFileTool {
    policy: Arc<ToolPolicy>,
}

impl PatchFileTool {
    pub fn new(policy: Arc<ToolPolicy>) -> PatchFileTool {
        PatchFileTool { policy }
    }
}

#[async_trait]
impl Tool for PatchFileTool {
    fn name(&self) -> &str {
        "patch_file"
    }
    fn description(&self) -> &str {
        "Apply a unified diff patch to an existing file."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "patch": { "type": "string", "description": "Unified diff patch text" }
            },
            "required": ["path", "patch"]
        })
    }
    async fn execute(&self, args: Value) -> Result<ToolResult, KovaError> {
        let path = match req_str(&args, "path") {
            Some(p) => p.to_string(),
            None => return Ok(tool_error("Missing required parameter 'path'")),
        };
        let patch_text = match req_str(&args, "patch") {
            Some(p) => p.to_string(),
            None => return Ok(tool_error("Missing required parameter 'patch'")),
        };
        let resolved = self.policy.resolve_arg_path(&path);
        if let Err(reason) = self.policy.check_write_path(&resolved) {
            return Ok(tool_error(reason));
        }
        let original = match std::fs::read_to_string(&resolved) {
            Ok(s) => s,
            Err(e) => return Ok(tool_error(format!("Cannot read '{path}': {e}"))),
        };
        let patch = match diffy::Patch::from_str(&patch_text) {
            Ok(p) => p,
            Err(e) => return Ok(tool_error(format!("Invalid patch: {e}"))),
        };
        match diffy::apply(&original, &patch) {
            Ok(patched) => match std::fs::write(&resolved, patched) {
                Ok(()) => Ok(tool_result(format!("Patched '{path}'"))),
                Err(e) => Ok(tool_error(format!("Cannot write '{path}': {e}"))),
            },
            Err(e) => Ok(tool_error(format!("Patch failed for '{path}': {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_policy() -> Arc<ToolPolicy> {
        Arc::new(ToolPolicy::default())
    }

    fn confined_policy(root: &Path) -> Arc<ToolPolicy> {
        Arc::new(ToolPolicy {
            workspace_root: Some(root.to_path_buf()),
            ..ToolPolicy::default()
        })
    }

    #[tokio::test]
    async fn write_file_overwrites_content() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("edit.txt");
        std::fs::write(&path, "old").unwrap();

        let tool = WriteFileTool::new(open_policy());
        let result = tool
            .execute(json!({ "path": path.to_str().unwrap(), "content": "new" }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
    }

    #[tokio::test]
    async fn write_file_denies_protected_path() {
        let tmp = TempDir::new().unwrap();
        let protected = tmp.path().join("config");
        std::fs::create_dir_all(&protected).unwrap();
        let policy = Arc::new(ToolPolicy {
            protected_paths: vec![protected.clone()],
            ..ToolPolicy::default()
        });

        let target = protected.join("config.yaml");
        let tool = WriteFileTool::new(policy);
        let result = tool
            .execute(json!({ "path": target.to_str().unwrap(), "content": "x" }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not permitted"));
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn write_file_denies_protected_path_via_traversal() {
        let tmp = TempDir::new().unwrap();
        let protected = tmp.path().join("config");
        std::fs::create_dir_all(&protected).unwrap();
        let policy = Arc::new(ToolPolicy {
            protected_paths: vec![protected.clone()],
            ..ToolPolicy::default()
        });

        let sneaky = tmp.path().join("ok").join("..").join("config").join("config.yaml");
        let tool = WriteFileTool::new(policy);
        let result = tool
            .execute(json!({ "path": sneaky.to_str().unwrap(), "content": "x" }))
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn write_file_denies_outside_workspace_root() {
        let tmp = TempDir::new().unwrap();
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let tool = WriteFileTool::new(confined_policy(&workspace));

        let inside = workspace.join("file.txt");
        let ok = tool
            .execute(json!({ "path": inside.to_str().unwrap(), "content": "hi" }))
            .await
            .unwrap();
        assert!(!ok.is_error, "inside workspace should be allowed: {}", ok.content);

        let outside = tmp.path().join("outside.txt");
        let denied = tool
            .execute(json!({ "path": outside.to_str().unwrap(), "content": "hi" }))
            .await
            .unwrap();
        assert!(denied.is_error);
        assert!(denied.content.contains("workspace root"));
        assert!(!outside.exists());
    }

    #[tokio::test]
    async fn write_file_resolves_relative_path_against_workspace_root() {
        let tmp = TempDir::new().unwrap();
        let tool = WriteFileTool::new(confined_policy(tmp.path()));
        let result = tool
            .execute(json!({ "path": "sub/rel.txt", "content": "hello" }))
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("sub/rel.txt")).unwrap(),
            "hello"
        );
    }

    #[tokio::test]
    async fn edit_file_replaces_unique_string() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("code.rs");
        std::fs::write(&path, "fn main() {\n    old();\n}\n").unwrap();

        let tool = EditFileTool::new(open_policy());
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "    old();",
                "new_string": "    new();"
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "fn main() {\n    new();\n}\n"
        );
    }

    #[tokio::test]
    async fn edit_file_rejects_ambiguous_match_without_replace_all() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("f.txt");
        std::fs::write(&path, "x\nx\n").unwrap();

        let tool = EditFileTool::new(open_policy());
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "x",
                "new_string": "y"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("2 locations"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "x\nx\n");
    }

    #[tokio::test]
    async fn edit_file_refuses_to_create_files() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("missing.txt");
        let tool = EditFileTool::new(open_policy());
        let result = tool
            .execute(json!({
                "path": path.to_str().unwrap(),
                "old_string": "a",
                "new_string": "b"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn read_file_returns_content_and_respects_offset_limit() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("f.txt");
        std::fs::write(&path, "l1\nl2\nl3\nl4\n").unwrap();
        let tool = ReadFileTool::new(open_policy());

        let all = tool
            .execute(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert!(!all.is_error);
        assert!(all.content.contains("l1") && all.content.contains("l4"));

        let windowed = tool
            .execute(json!({ "path": path.to_str().unwrap(), "offset": 2, "limit": 2 }))
            .await
            .unwrap();
        assert!(windowed.content.contains("l2") && windowed.content.contains("l3"));
        assert!(!windowed.content.contains("l1"));
    }

    #[tokio::test]
    async fn read_file_rejects_binary() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bin");
        std::fs::write(&path, [0u8, 1, 2, 3]).unwrap();
        let tool = ReadFileTool::new(open_policy());
        let result = tool
            .execute(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("binary"));
    }

    #[tokio::test]
    async fn list_dir_marks_directories() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join("sub")).unwrap();
        std::fs::write(tmp.path().join("file.txt"), "x").unwrap();
        let tool = ListDirTool::new(open_policy());
        let result = tool
            .execute(json!({ "path": tmp.path().to_str().unwrap() }))
            .await
            .unwrap();
        assert!(result.content.contains("sub/"));
        assert!(result.content.contains("file.txt"));
    }

    #[tokio::test]
    async fn search_finds_matches_and_filters_by_glob() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "fn alpha() {}\n").unwrap();
        let tool = SearchTool::new(open_policy());

        let result = tool
            .execute(json!({ "pattern": "fn alpha", "path": tmp.path().to_str().unwrap(), "glob": "*.rs" }))
            .await
            .unwrap();
        assert!(result.content.contains("a.rs:1: fn alpha() {}"));
        assert!(!result.content.contains("b.txt"));
    }

    #[tokio::test]
    async fn search_rejects_invalid_regex() {
        let tmp = TempDir::new().unwrap();
        let tool = SearchTool::new(open_policy());
        let result = tool
            .execute(json!({ "pattern": "(", "path": tmp.path().to_str().unwrap() }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Invalid regex"));
    }

    #[tokio::test]
    async fn patch_file_applies_and_rejects_mismatch() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("f.txt");
        std::fs::write(&path, "a\nb\nc\n").unwrap();
        let tool = PatchFileTool::new(open_policy());

        let good = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n";
        let ok = tool
            .execute(json!({ "path": path.to_str().unwrap(), "patch": good }))
            .await
            .unwrap();
        assert!(!ok.is_error, "{}", ok.content);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\nB\nc\n");

        let bad = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n a\n-Z\n+Y\n c\n";
        let err = tool
            .execute(json!({ "path": path.to_str().unwrap(), "patch": bad }))
            .await
            .unwrap();
        assert!(err.is_error);
    }
}
