//! Tokenization and the feature-hashing embedder (SPEC §7).
//!
//! No model, no runtime: lowercase word tokens are folded into a
//! fixed-dimension vector with VW-style *signed* hashing (two independent
//! FNV-1a hashes — one picks the bucket, one's low bit picks the sign),
//! then the vector is L2-normalized so cosine/dot-product metrics are
//! length-independent. Pure arithmetic: deterministic across runs and
//! platforms, essentially free per message.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// Second hash salt so bucket and sign come from independent hashes.
const SIGN_SALT: u64 = 0x9ee3_779b_97f4_a7c1;

fn fnv1a(data: &str, basis: u64) -> u64 {
    data.bytes()
        .fold(basis, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn fnv1a_with_salt(data: &str) -> (u64, u64) {
    (fnv1a(data, FNV_OFFSET), fnv1a(data, FNV_OFFSET ^ SIGN_SALT))
}

/// The usual short high-frequency function words, plus reply-chain
/// boilerplate (`re`, `fwd`, `fw`) that carries no topical signal.
const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "you", "your", "re", "fwd", "fw",
];

fn stopwords() -> &'static HashSet<&'static str> {
    static STOP: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP.get_or_init(|| STOPWORDS.iter().copied().collect())
}

/// Lowercased alphanumeric word tokens, stopwords removed.
/// Single-character tokens are dropped; the stopword list handles the
/// short function words.
pub fn tokenize(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    lowered
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .filter(|t| !stopwords().contains(t))
        .map(str::to_string)
        .collect()
}

/// Feature-hash `tokens` into a dims-vector and L2-normalize it.
/// Returns a zero vector (norm 0) when there are no tokens.
pub fn embed_tokens(tokens: impl Iterator<Item = String>, dims: usize) -> Vec<f32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    for t in tokens {
        *counts.entry(t).or_insert(0) += 1;
    }
    let mut v = vec![0f32; dims];
    for (tok, count) in &counts {
        let (h_bucket, h_sign) = fnv1a_with_salt(tok);
        let bucket = (h_bucket % dims as u64) as usize;
        let sign = if h_sign & 1 == 0 { -1.0 } else { 1.0 };
        v[bucket] += sign * *count as f32;
    }
    l2_normalize(&mut v);
    v
}

/// Convenience: tokenize then embed in one call.
pub fn embed_text(text: &str, dims: usize) -> Vec<f32> {
    embed_tokens(tokenize(text).into_iter(), dims)
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_splits_folds_and_stops() {
        let toks = tokenize("Hello, World! The-AND of invoicing; naïve café 2025 I re:");
        assert_eq!(
            toks,
            vec!["hello", "world", "invoicing", "naïve", "café", "2025"]
        );
    }

    #[test]
    fn embed_is_deterministic_and_l2_normalized() {
        let a = embed_text("invoice payment due amount", 2048);
        let b = embed_text("due amount payment invoice", 2048);
        let c = embed_text("invoice payment due amount", 2048);
        assert_eq!(a, b); // bag of words: order-independent
        assert_eq!(a, c); // deterministic across calls
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
    }

    #[test]
    fn embed_respects_dims_and_signs_exist() {
        let v = embed_text("alpha beta gamma delta epsilon zeta", 64);
        assert_eq!(v.len(), 64);
        // With 6 distinct tokens and signed hashing both + and - buckets
        // should appear (a collision with all-same sign is 2^-6 likely).
        assert!(v.iter().any(|x| *x > 0.0));
        assert!(v.iter().any(|x| *x < 0.0));
    }

    #[test]
    fn empty_text_embeds_to_zero_vector() {
        let v = embed_text("!!! ...", 128);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn buckets_stay_in_range_for_many_tokens() {
        let joined = (0..500)
            .map(|i| format!("tok{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let v = embed_text(&joined, 97);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_eq!(v.len(), 97);
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
