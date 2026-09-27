//! The `infer` subcommand — the procmail hot path (SPEC §6, §8, §9).
//!
//! The index is always opened in mmap `view()` mode here; mutable `load()`
//! is reserved for `learn`. Classification is inverse-distance-weighted
//! k-NN voting with a confidence threshold. Every failure path fails open
//! (SPEC §2): the message is passed through untouched rather than lost.
//! Non-zero exit is reserved for genuine misuse (unreadable input, missing
//! state), never for "couldn't classify this particular email".

use crate::{config, embed, store, text};
use std::io::{Read, Write};
use std::path::Path;
use usearch::Index;

/// Reserved category for below-threshold results (SPEC §6).
pub const UNSURE: &str = "unsure";

const OUR_HEADER_PREFIX: &[u8] = b"x-postmaster-k-";

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Print exactly one word: the winning category (or `unsure`).
    Word,
    /// Print one `X-Postmaster-K-Category:` header line.
    Header,
    /// Insert classification headers and emit the message untouched otherwise.
    Full,
}

#[derive(Debug, Clone)]
pub struct Verdict {
    pub category: String,
    /// Winner's share of the total vote weight, 0.0..=1.0.
    pub share: f64,
}

#[derive(Debug)]
pub enum InferError {
    /// Missing/invalid persistent state — genuine misuse: `word`/`header`
    /// exit 2, `full` still fail-opens (§8).
    Misuse(String),
    /// Per-message internal failure (corrupt index, dims mismatch, search
    /// error) — always fail-opens (§2).
    Soft(String),
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferError::Misuse(e) | InferError::Soft(e) => f.write_str(e),
        }
    }
}

/// Classify one message and emit it per `mode`. Returns the process exit code.
pub fn run(home: &Path, mode: Mode, file: Option<&Path>) -> i32 {
    // Without the input bytes there is nothing to fail open with, so a read
    // error is the one failure that exits non-zero in every mode.
    let raw = match read_input(file) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!("postmaster-k: {e}");
            return 2;
        }
    };

    match classify(home, &raw) {
        Ok(v) => {
            match mode {
                Mode::Word => print_line(&v.category),
                Mode::Header => print_line(&format!("X-Postmaster-K-Category: {}", v.category)),
                Mode::Full => write_stdout(&tag_message(&raw, &v.category, v.share)),
            }
            0
        }
        Err(InferError::Soft(e)) => {
            eprintln!("postmaster-k: {e}");
            match mode {
                Mode::Word => print_line(UNSURE),
                Mode::Header => {}
                Mode::Full => write_stdout(&raw),
            }
            0
        }
        Err(InferError::Misuse(e)) => {
            eprintln!("postmaster-k: {e}");
            if mode == Mode::Full {
                // Even misuse must not break delivery in pipe mode (§2).
                write_stdout(&raw);
                0
            } else {
                2
            }
        }
    }
}

/// Full-mode output for one message: tagged on success, the original bytes
/// on any failure — the fail-open contract of §2/§8.
pub fn full_output(home: &Path, raw: &[u8]) -> Vec<u8> {
    match classify(home, raw) {
        Ok(v) => tag_message(raw, &v.category, v.share),
        Err(e) => {
            eprintln!("postmaster-k: {e}");
            raw.to_vec()
        }
    }
}

/// Parse, extract, embed and k-NN-classify one message (SPEC §6).
pub fn classify(home: &Path, raw: &[u8]) -> Result<Verdict, InferError> {
    let settings = config::settings(home)
        .map_err(|e| InferError::Misuse(format!("cannot read config: {e}")))?;

    // §7.1: only classifiable text is read; anything unparseable or
    // token-less classifies as `unsure` rather than erroring.
    let txt = text::extract_bytes(raw).unwrap_or_default();
    let counts = embed::token_counts(embed::tokenize(&txt).into_iter());
    if counts.is_empty() {
        return Ok(Verdict {
            category: UNSURE.into(),
            share: 0.0,
        });
    }
    let vec = embed::embed_counts(&counts, settings.dims);

    let index_path = config::index_path(home);
    if !index_path.exists() {
        return Err(InferError::Misuse(
            "no index yet — run `postmaster-k learn <category> <directory>` first".into(),
        ));
    }
    let index = open_view(&index_path, settings.dims)?;

    let labels = store::load_labels(&config::labels_path(home))
        .map_err(|e| InferError::Misuse(format!("cannot read labels.db: {e}")))?;

    let size = index.size();
    if size == 0 {
        return Ok(Verdict {
            category: UNSURE.into(),
            share: 0.0,
        });
    }
    let k = settings.k.clamp(1, size);
    let matches = index
        .search(&vec, k)
        .map_err(|e| InferError::Soft(format!("index search failed: {e}")))?;

    Ok(vote(
        &labels,
        &matches.keys,
        &matches.distances,
        settings.threshold,
        settings.threshold_factor,
    ))
}

/// mmap-only index open (§2: cold start must not load the corpus into RAM);
/// validates that the on-disk dimensionality matches the config.
fn open_view(index_path: &Path, dims: usize) -> Result<Index, InferError> {
    let ps = index_path.to_str().ok_or_else(|| {
        InferError::Soft(format!("path {} is not valid UTF-8", index_path.display()))
    })?;
    let meta = Index::metadata(ps)
        .map_err(|e| InferError::Soft(format!("cannot read index header: {e}")))?;
    if meta.dimensions != dims as u64 {
        return Err(InferError::Soft(format!(
            "index was built with dims={} but config says {dims}",
            meta.dimensions
        )));
    }
    Index::restore_view(ps).map_err(|e| InferError::Soft(format!("cannot mmap-view index: {e}")))
}

/// Inverse-distance-weighted vote; ties go to the earlier match in search
/// order so the result is deterministic for a given index state (§6 steps 5–6).
/// The winner must also have a weight at least `threshold_factor` times the
/// second-greatest, enforcing a hockey-stick drop-off.
fn vote(
    labels: &[store::LabelEntry],
    keys: &[u64],
    distances: &[f32],
    threshold: f64,
    threshold_factor: f64,
) -> Verdict {
    let mut votes: Vec<(String, f64)> = Vec::new();
    let mut total = 0.0;
    for (key, dist) in keys.iter().zip(distances.iter()) {
        let Some(cat) = labels
            .iter()
            .find(|l| l.key == *key)
            .map(|l| l.category.clone())
        else {
            // Possible after a partially failed side-table write; never guess.
            eprintln!("postmaster-k: warning: vector key {key} is unlabelled; ignored");
            continue;
        };
        let weight = 1.0 / ((*dist as f64).max(0.0) + 1e-6);
        total += weight;
        match votes.iter_mut().find(|(c, _)| *c == cat) {
            Some(v) => v.1 += weight,
            None => votes.push((cat, weight)),
        }
    }

    if votes.is_empty() || total <= f64::EPSILON {
        return Verdict {
            category: UNSURE.into(),
            share: 0.0,
        };
    }
    // Sort descending by weight so best and second-best are adjacent.
    votes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    votes.reverse();
    let best_w = votes[0].1;
    let second_w = votes[1].1;
    let share = best_w / total;
    if share < threshold || best_w < second_w * threshold_factor {
        Verdict {
            category: UNSURE.into(),
            share,
        }
    } else {
        Verdict {
            category: votes[0].0.clone(),
            share,
        }
    }
}

fn read_input(file: Option<&Path>) -> Result<Vec<u8>, String> {
    match file {
        Some(p) if p.to_str() != Some("-") => {
            std::fs::read(p).map_err(|e| format!("cannot read {}: {e}", p.display()))
        }
        _ => {
            let mut buf = Vec::new();
            std::io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| format!("cannot read stdin: {e}"))?;
            Ok(buf)
        }
    }
}

// Panic-free stdout: with `panic = "abort"` a failed println! would abort
// mid-delivery, so output goes through explicit, error-ignored writes.
fn print_line(line: &str) {
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(line.as_bytes());
    let _ = out.write_all(b"\n");
    let _ = out.flush();
}

fn write_stdout(bytes: &[u8]) {
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(bytes);
    let _ = out.flush();
}

// ---------------------------------------------------------------------------
// Full-mode header injection (§8 `--mode=full`)
// ---------------------------------------------------------------------------

enum Split {
    /// headers, blank line, body
    Separated,
    /// headers only, no trailing blank line
    HeadersOnly,
    /// no recognizable headers; the payload is treated as the body
    BodyOnly,
}

/// Insert `X-Postmaster-K-*` headers at the top of the header block,
/// replacing any we added on a previous pass, leaving every other byte of
/// the message — mbox envelope line, headers, body, line endings — intact.
pub fn tag_message(raw: &[u8], category: &str, share: f64) -> Vec<u8> {
    let (envelope, rest) = split_envelope(raw);
    let eol: &[u8] = if message_uses_crlf(rest) {
        b"\r\n"
    } else {
        b"\n"
    };

    let (split, orig_headers, body) = match find_separator(rest) {
        Some((headers_end, body_start)) => {
            (Split::Separated, &rest[..headers_end], &rest[body_start..])
        }
        None if first_line_is_header(rest) => (Split::HeadersOnly, rest, &rest[rest.len()..]),
        None => (Split::BodyOnly, &rest[..0], rest),
    };

    let score = format!("{share:.3}");
    let mut out = Vec::with_capacity(raw.len() + 128);
    out.extend_from_slice(envelope);
    out.extend_from_slice(b"X-Postmaster-K-Category: ");
    out.extend_from_slice(category.as_bytes());
    out.extend_from_slice(eol);
    out.extend_from_slice(b"X-Postmaster-K-Score: ");
    out.extend_from_slice(score.as_bytes());
    out.extend_from_slice(eol);
    out.extend_from_slice(&strip_our_headers(orig_headers));
    if !matches!(split, Split::HeadersOnly) {
        out.extend_from_slice(eol);
    }
    out.extend_from_slice(body);
    out
}

/// Split a leading mbox `From ` envelope line off, if one is present. To
/// avoid eating body text that merely begins with "From ...", the following
/// line must look like a header for the first line to count as an envelope.
fn split_envelope(raw: &[u8]) -> (&[u8], &[u8]) {
    if !raw.starts_with(b"From ") {
        return (&[], raw);
    }
    let Some(nl) = raw.iter().position(|&b| b == b'\n') else {
        return (&[], raw);
    };
    let rest = &raw[nl + 1..];
    if first_line_is_header(rest) {
        (&raw[..=nl], rest)
    } else {
        (&[], raw)
    }
}

/// Locate the header/body separator: the first `\n\n` or `\n\r\n`.
/// Returns (offset of the separator's first byte, offset of the body).
fn find_separator(rest: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < rest.len() {
        if rest[i] == b'\n' {
            if rest.get(i + 1) == Some(&b'\n') {
                return Some((i + 1, i + 2));
            }
            if rest.get(i + 1) == Some(&b'\r') && rest.get(i + 2) == Some(&b'\n') {
                return Some((i + 1, i + 3));
            }
        }
        i += 1;
    }
    None
}

fn first_line_is_header(bytes: &[u8]) -> bool {
    let end = bytes
        .iter()
        .position(|&b| b == b'\n')
        .unwrap_or(bytes.len());
    is_header_line(trim_eol(&bytes[..end]))
}

/// `field-name:` with a plausible RFC 5322 name (printable ASCII, no space,
/// no colon inside the name).
fn is_header_line(line: &[u8]) -> bool {
    let Some(colon) = line.iter().position(|&b| b == b':') else {
        return false;
    };
    colon > 0
        && line[..colon]
            .iter()
            .all(|&b| b.is_ascii_graphic() && b != b':')
}

/// Detect the message's line-ending style from its first newline so the
/// inserted headers match the host message.
fn message_uses_crlf(bytes: &[u8]) -> bool {
    match bytes.iter().position(|&b| b == b'\n') {
        Some(i) => i > 0 && bytes[i - 1] == b'\r',
        None => false,
    }
}

/// Drop previously-inserted `X-Postmaster-K-*` headers (any case) so a
/// message that passes through twice doesn't accumulate stale verdicts.
/// Folded continuation lines of a dropped header go with it.
fn strip_our_headers(block: &[u8]) -> Vec<u8> {
    let mut kept = Vec::with_capacity(block.len());
    let mut skipping = false;
    let mut rest = block;
    while !rest.is_empty() {
        let nl = rest
            .iter()
            .position(|&b| b == b'\n')
            .map_or(rest.len(), |i| i + 1);
        let (line, tail) = rest.split_at(nl);
        rest = tail;
        let content = trim_eol(line);
        if is_our_header_line(content) {
            skipping = true;
            continue;
        }
        if skipping && matches!(content.first(), Some(b' ') | Some(b'\t')) {
            continue;
        }
        skipping = false;
        kept.extend_from_slice(line);
    }
    kept
}

fn trim_eol(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    if end > 0 && line[end - 1] == b'\n' {
        end -= 1;
    }
    if end > 0 && line[end - 1] == b'\r' {
        end -= 1;
    }
    &line[..end]
}

fn is_our_header_line(content: &[u8]) -> bool {
    content.len() > OUR_HEADER_PREFIX.len()
        && content[..OUR_HEADER_PREFIX.len()].eq_ignore_ascii_case(OUR_HEADER_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAT: &str = "invoices";

    #[test]
    fn tags_lf_message_at_top_of_header_block() {
        let raw = b"Subject: hi\n\nbody\n";
        let out = tag_message(raw, CAT, 0.875);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "X-Postmaster-K-Category: invoices\nX-Postmaster-K-Score: 0.875\nSubject: hi\n\nbody\n"
        );
    }

    #[test]
    fn tags_crlf_message_with_crlf_headers() {
        let raw = b"Subject: hi\r\n\r\nbody\r\n";
        let out = tag_message(raw, CAT, 0.5);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "X-Postmaster-K-Category: invoices\r\nX-Postmaster-K-Score: 0.500\r\nSubject: hi\r\n\r\nbody\r\n"
        );
    }

    #[test]
    fn retag_replaces_previous_headers_including_folds() {
        let raw = b"X-Postmaster-K-Category: old\nX-Postmaster-K-Score: 0.100\n\tfolded\nSubject: hi\n\nb\n";
        let out = tag_message(raw, CAT, 0.9);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "X-Postmaster-K-Category: invoices\nX-Postmaster-K-Score: 0.900\nSubject: hi\n\nb\n"
        );
    }

    #[test]
    fn mbox_envelope_stays_above_inserted_headers() {
        let raw = b"From bob@example.com Fri Sep 26 10:00:00 2026\nSubject: hi\n\nb\n";
        let out = tag_message(raw, CAT, 0.5);
        assert!(String::from_utf8(out).unwrap().starts_with(
            "From bob@example.com Fri Sep 26 10:00:00 2026\nX-Postmaster-K-Category: invoices\n"
        ));
    }

    #[test]
    fn body_starting_with_from_is_not_treated_as_envelope() {
        let raw = b"From my point of view this is the body\n";
        let out = tag_message(raw, CAT, 0.5);
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("X-Postmaster-K-Category: invoices\n"));
        assert!(s.ends_with("From my point of view this is the body\n"));
    }

    #[test]
    fn header_only_message_gets_no_extra_blank_line() {
        let raw = b"Subject: hi\n";
        let out = tag_message(raw, CAT, 0.5);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "X-Postmaster-K-Category: invoices\nX-Postmaster-K-Score: 0.500\nSubject: hi\n"
        );
    }

    #[test]
    fn body_only_message_gets_a_header_block() {
        let raw = b"just some text\n";
        let out = tag_message(raw, CAT, 0.5);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "X-Postmaster-K-Category: invoices\nX-Postmaster-K-Score: 0.500\n\njust some text\n"
        );
    }

    #[test]
    fn own_headers_stripped_regardless_of_case() {
        let raw = b"x-postmaster-k-category: old\nSubject: hi\n\nb\n";
        let out = tag_message(raw, CAT, 0.5);
        let s = String::from_utf8(out).unwrap();
        assert!(!s.contains("old"));
        assert_eq!(s.matches("X-Postmaster-K-Category").count(), 1);
    }

    #[test]
    fn vote_clear_winner() {
        let labels = vec![
            store::LabelEntry {
                key: 1,
                category: "a".into(),
                source: None,
            },
            store::LabelEntry {
                key: 2,
                category: "a".into(),
                source: None,
            },
            store::LabelEntry {
                key: 3,
                category: "b".into(),
                source: None,
            },
        ];
        let v = vote(&labels, &[1, 2, 3], &[0.1, 0.2, 0.9], 0.4, 2.0);
        assert_eq!(v.category, "a");
        assert!(v.share > 0.4);
    }

    #[test]
    fn vote_below_threshold_is_unsure() {
        let labels = vec![
            store::LabelEntry {
                key: 1,
                category: "a".into(),
                source: None,
            },
            store::LabelEntry {
                key: 3,
                category: "b".into(),
                source: None,
            },
        ];
        // one neighbour of each, equally distant → 0.5 share
        let v = vote(&labels, &[1, 3], &[0.5, 0.5], 0.6, 2.0);
        assert_eq!(v.category, UNSURE);
        assert!((v.share - 0.5).abs() < 1e-9);
    }

    #[test]
    fn vote_skips_unlabelled_keys() {
        let labels = vec![store::LabelEntry {
            key: 1,
            category: "a".into(),
            source: None,
        }];
        let v = vote(&labels, &[7, 7], &[0.1, 0.1], 0.4, 2.0);
        assert_eq!(v.category, UNSURE);
        assert_eq!(v.share, 0.0);
    }

    #[test]
    fn vote_below_threshold_factor_is_unsure() {
        // "a" wins share but its weight is only 1.5× the second place — below factor 2.0.
        let labels = vec![
            store::LabelEntry {
                key: 1,
                category: "a".into(),
                source: None,
            },
            store::LabelEntry {
                key: 2,
                category: "a".into(),
                source: None,
            },
            store::LabelEntry {
                key: 3,
                category: "b".into(),
                source: None,
            },
        ];
        // weights: a≈1/0.1=10, a≈1/0.2=5 → total_a=15; b≈1/0.9=1.11 → total_b=1.11
        // share ≈ 15/16.11 ≈ 0.93 (>0.4) but ratio 15/1.11 ≈ 13.5 — above factor here, so pick closer distances:
        let v = vote(&labels, &[1, 2, 3], &[0.1, 0.15, 0.12], 0.4, 2.0);
        // a weight: 1/0.1 + 1/0.15 = 10 + 6.67 = 16.67; b: 1/0.12 = 8.33; ratio ≈ 2.0
        assert_eq!(v.category, UNSURE);
    }

    #[test]
    fn vote_hockey_stick_winner_passes() {
        let labels = vec![
            store::LabelEntry {
                key: 1,
                category: "a".into(),
                source: None,
            },
            store::LabelEntry {
                key: 2,
                category: "b".into(),
                source: None,
            },
        ];
        // a weight 10, b weight 1 → ratio 10 ≥ 2.0
        let v = vote(&labels, &[1, 2], &[0.1, 0.9], 0.4, 2.0);
        assert_eq!(v.category, "a");
    }
}
