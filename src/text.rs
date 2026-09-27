//! Classifiable-text extraction (SPEC §7.1).
//!
//! Only topical signal is read: the decoded `Subject` and the inline body
//! parts. `mail-parser` already implements RFC 2047 decoding and
//! `multipart/alternative` semantics — its `text_body` list contains every
//! `text/plain` part, plus `text/html` parts *only* where no plain
//! alternative exists, converting those through its own HTML stripper
//! (`decoders/html.rs`: drops script/style/template content and comments,
//! decodes entities). Protocol headers, MIME boundaries and non-text
//! attachment bodies never enter the token stream.

use mail_parser::{Message, MessageParser, PartType};

/// Parse `raw` and extract the classifiable text. `None` means the input
/// could not be parsed as a message at all (SPEC §2: callers fail open).
pub fn extract_bytes(raw: &[u8]) -> Option<String> {
    let msg = MessageParser::default().parse(raw)?;
    Some(extract(&msg))
}

/// Extract classifiable text from a parsed message: Subject + every inline
/// text body, per §7.1.
pub fn extract(msg: &Message) -> String {
    let mut out = String::new();
    if let Some(subject) = msg.subject() {
        out.push_str(subject);
        out.push('\n');
    }
    collect_bodies(msg, &mut out);
    out
}

fn collect_bodies(msg: &Message, out: &mut String) {
    for pos in 0..msg.text_body_count() {
        if let Some(text) = msg.body_text(pos) {
            out.push_str(&text);
            out.push('\n');
        }
    }
    // Forwarded messages (message/rfc822) carry topical text of their own;
    // their Subject and bodies are inline signal, not protocol noise.
    for part in &msg.parts {
        if let PartType::Message(nested) = &part.body {
            if let Some(subject) = nested.subject() {
                out.push_str(subject);
                out.push('\n');
            }
            collect_bodies(nested, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_message_yields_subject_and_body() {
        let raw = b"Subject: =?utf-8?q?Quarterly_report?=\n\
                    Message-ID: <a@b>\n\
                    Date: Sat, 27 Sep 2025 10:00:00 +0100\n\
                    MIME-Version: 1.0\n\
                    Content-Type: text/plain; charset=utf-8\n\
                    \n\
                    Payment terms discussed yesterday.\n";
        let text = extract_bytes(raw).unwrap();
        assert_eq!(
            text.trim(),
            "Quarterly report\nPayment terms discussed yesterday."
        );
    }

    #[test]
    fn html_alternative_is_ignored_when_plain_exists() {
        let raw = b"Subject: mixed\n\
                    MIME-Version: 1.0\n\
                    Content-Type: multipart/alternative; boundary=BOUND\n\
                    \n\
                    --BOUND\n\
                    Content-Type: text/plain; charset=utf-8\n\
                    \n\
                    plain body words here\n\
                    --BOUND\n\
                    Content-Type: text/html; charset=utf-8\n\
                    \n\
                    <html><body><p>html-only markup noise</p></body></html>\n\
                    --BOUND--\n";
        let text = extract_bytes(raw).unwrap();
        assert!(text.contains("plain body words here"));
        assert!(!text.contains("markup"));
    }

    #[test]
    fn html_only_message_is_stripped_and_entities_decoded() {
        let raw = b"Subject: newsletter\n\
                    MIME-Version: 1.0\n\
                    Content-Type: text/html; charset=utf-8\n\
                    \n\
                    <html><head><style>.a { color: red; }</style></head>\n\
                    <body><p>Weekly&nbsp;digest &amp; rumours</p>\n\
                    <script>alert('tracker');</script>\n\
                    <p>Second line</p></body></html>\n";
        let text = extract_bytes(raw).unwrap();
        // nbsp decodes to U+00A0, which is fine for tokenization purposes.
        assert!(text.contains("digest & rumours"));
        assert!(text.contains("Second line"));
        assert!(!text.contains("color"));
        assert!(!text.contains("alert"));
        // No raw HTML entities survive.
        assert!(!text.contains("&amp;"));
        assert!(!text.contains("&nbsp;"));
    }

    #[test]
    fn attachment_bodies_and_headers_are_not_read() {
        let raw = b"Subject: report attached\n\
                    Message-ID: <xyz boundary--token>\n\
                    MIME-Version: 1.0\n\
                    Content-Type: multipart/mixed; boundary=BOUND\n\
                    \n\
                    --BOUND\n\
                    Content-Type: text/plain; charset=utf-8\n\
                    \n\
                    see attached quarterly numbers\n\
                    --BOUND\n\
                    Content-Type: application/pdf; name=\"report.pdf\"\n\
                    Content-Disposition: attachment; filename=\"report.pdf\"\n\
                    Content-Transfer-Encoding: base64\n\
                    \n\
                    JVBERi0xLjQKJcOkw7zDtsOf\n\
                    --BOUND--\n";
        let text = extract_bytes(raw).unwrap();
        assert!(text.contains("see attached quarterly numbers"));
        assert!(!text.contains("JVBERi"));
        assert!(!text.contains("report.pdf"));
        assert!(!text.contains("xyz"));
    }

    #[test]
    fn nested_forwarded_message_is_included() {
        let raw = b"Subject: fwd: plan\n\
                    MIME-Version: 1.0\n\
                    Content-Type: multipart/mixed; boundary=BOUND\n\
                    \n\
                    --BOUND\n\
                    Content-Type: text/plain; charset=utf-8\n\
                    \n\
                    thoughts below\n\
                    --BOUND\n\
                    Content-Type: message/rfc822\n\
                    \n\
                    Subject: original budget thread\n\
                    Content-Type: text/plain; charset=utf-8\n\
                    \n\
                    budget numbers inside\n\
                    --BOUND--\n";
        let text = extract_bytes(raw).unwrap();
        assert!(text.contains("thoughts below"));
        assert!(text.contains("original budget thread"));
        assert!(text.contains("budget numbers inside"));
    }

    #[test]
    fn message_with_no_text_yields_empty_string() {
        let raw = b"Subject: \n\n\n";
        let text = extract_bytes(raw).unwrap();
        assert!(text.trim().is_empty());
    }
}
