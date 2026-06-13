//! Shared line-oriented byte-stream adapter for provider streaming responses.
//!
//! OpenAI and Gemini stream SSE lines; Ollama streams NDJSON. All three are
//! newline-framed, so one adapter handles the byte buffering and each
//! provider supplies a line parser.

use std::collections::VecDeque;
use std::pin::Pin;

use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};

use crate::error::KovaError;
use crate::models::StreamEvent;

/// Result of parsing one complete line from a provider stream.
pub(crate) enum LineOutcome {
    /// Zero or more events decoded from the line (empty for blank/comment lines).
    Events(Vec<StreamEvent>),
    /// End-of-stream sentinel (e.g. SSE `data: [DONE]`).
    /// Only SSE providers construct this; NDJSON streams end at EOF.
    #[cfg_attr(not(any(feature = "openai", feature = "gemini")), allow(dead_code))]
    Done,
    /// Malformed line.
    Fail(KovaError),
}

struct State<F> {
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, KovaError>> + Send>>,
    buffer: BytesMut,
    pending: VecDeque<StreamEvent>,
    parse_line: F,
    eof: bool,
    finished: bool,
}

/// Convert a raw byte stream into `StreamEvent`s by framing on newlines.
///
/// Bytes are buffered raw (`BytesMut`) and only complete lines are decoded,
/// so a multi-byte UTF-8 character split across network chunks is never
/// corrupted, and consuming a line is O(1) in the remaining buffer. At EOF
/// any final unterminated line is parsed too (NDJSON responses are not
/// guaranteed a trailing newline).
pub(crate) fn line_stream_to_events<F>(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    parse_line: F,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>
where
    F: FnMut(&str) -> LineOutcome + Send + 'static,
{
    let mapped = byte_stream
        .map(|chunk| chunk.map_err(|e| KovaError::Stream(format!("Connection error: {e}"))));

    let state = State {
        stream: Box::pin(mapped),
        buffer: BytesMut::new(),
        pending: VecDeque::new(),
        parse_line,
        eof: false,
        finished: false,
    };

    Box::pin(futures::stream::unfold(state, |mut st| async move {
        loop {
            if let Some(event) = st.pending.pop_front() {
                return Some((Ok(event), st));
            }
            if st.finished {
                return None;
            }

            if let Some(nl) = st.buffer.iter().position(|&b| b == b'\n') {
                let line_bytes = st.buffer.split_to(nl + 1);
                // Lossy decode is safe here: a complete line cannot end
                // mid-character ('\n' never appears inside a UTF-8 sequence).
                let line = String::from_utf8_lossy(&line_bytes[..nl]);
                match (st.parse_line)(line.as_ref()) {
                    LineOutcome::Events(events) => {
                        st.pending.extend(events);
                        continue;
                    }
                    LineOutcome::Done => return None,
                    LineOutcome::Fail(e) => return Some((Err(e), st)),
                }
            }

            if st.eof {
                st.finished = true;
                if st.buffer.is_empty() {
                    return None;
                }
                let rest = std::mem::take(&mut st.buffer);
                let line = String::from_utf8_lossy(&rest);
                match (st.parse_line)(line.as_ref()) {
                    LineOutcome::Events(events) => {
                        st.pending.extend(events);
                        continue;
                    }
                    LineOutcome::Done => return None,
                    LineOutcome::Fail(e) => return Some((Err(e), st)),
                }
            }

            match st.stream.next().await {
                Some(Ok(bytes)) => st.buffer.extend_from_slice(&bytes),
                Some(Err(e)) => return Some((Err(e), st)),
                None => st.eof = true,
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::StreamEvent;

    fn byte_chunks(
        chunks: Vec<Vec<u8>>,
    ) -> impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static {
        futures::stream::iter(chunks.into_iter().map(|c| Ok(Bytes::from(c))))
    }

    fn text_events(line: &str) -> LineOutcome {
        if line.trim().is_empty() {
            return LineOutcome::Events(vec![]);
        }
        LineOutcome::Events(vec![StreamEvent::ContentDelta {
            text: line.to_string(),
        }])
    }

    async fn collect_texts(
        stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, KovaError>> + Send>>,
    ) -> Vec<String> {
        stream
            .map(|r| match r.unwrap() {
                StreamEvent::ContentDelta { text } => text,
                other => panic!("unexpected event {other:?}"),
            })
            .collect()
            .await
    }

    #[tokio::test]
    async fn multibyte_char_split_across_chunks_survives() {
        // "héllo\n" with 'é' (0xC3 0xA9) split across two network chunks.
        let stream = line_stream_to_events(
            byte_chunks(vec![vec![b'h', 0xC3], vec![0xA9, b'l', b'l', b'o', b'\n']]),
            text_events,
        );
        assert_eq!(collect_texts(stream).await, vec!["héllo"]);
    }

    #[tokio::test]
    async fn lines_split_across_many_chunks_reassemble() {
        let stream = line_stream_to_events(
            byte_chunks(vec![
                b"first ".to_vec(),
                b"line\nsecond".to_vec(),
                b" line\n".to_vec(),
            ]),
            text_events,
        );
        assert_eq!(
            collect_texts(stream).await,
            vec!["first line", "second line"]
        );
    }

    #[tokio::test]
    async fn trailing_line_without_newline_is_parsed_at_eof() {
        let stream = line_stream_to_events(byte_chunks(vec![b"a\nb".to_vec()]), text_events);
        assert_eq!(collect_texts(stream).await, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn done_outcome_ends_stream() {
        let stream = line_stream_to_events(byte_chunks(vec![b"a\nDONE\nb\n".to_vec()]), |line| {
            if line == "DONE" {
                LineOutcome::Done
            } else {
                text_events(line)
            }
        });
        assert_eq!(collect_texts(stream).await, vec!["a"]);
    }

    #[tokio::test]
    async fn fail_outcome_surfaces_error() {
        let mut stream = line_stream_to_events(byte_chunks(vec![b"bad\n".to_vec()]), |_line| {
            LineOutcome::Fail(KovaError::Stream("boom".into()))
        });
        let first = stream.next().await.unwrap();
        assert!(matches!(first, Err(KovaError::Stream(m)) if m == "boom"));
    }
}
