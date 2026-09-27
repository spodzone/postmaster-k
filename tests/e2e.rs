//! End-to-end tests: learn → infer round trips against a throwaway state
//! root, exercising the SPEC §6/§8/§2 semantics (separation, threshold,
//! fail-open, byte-exact full-mode passthrough).

use postmaster_k::{config, infer, learn};
use std::fs;
use std::path::Path;

/// `Subject, body` fixtures per category with deliberately disjoint topical
/// vocabularies so nearest-neighbour separation is deterministic.
const INVOICES: &[(&str, &str)] = &[
    (
        "Invoice 10011 due",
        "Please remit payment of the amount due on invoice 10011 by the due date. \
         Overdue accounts incur interest on the outstanding total.",
    ),
    (
        "Payment reminder for invoice",
        "Your invoice total is overdue. The billing department asks for \
         remittance advice with your payment.",
    ),
    (
        "Purchase order billing",
        "The billing amount on this invoice covers purchase order 4471. \
         Payment is due within 30 days of the invoice date.",
    ),
];

const NEWSLETTERS: &[(&str, &str)] = &[
    (
        "Weekly digest",
        "This edition of the newsletter features stories about technology, \
         culture and science. Subscribe or unsubscribe at any time.",
    ),
    (
        "Headlines for you",
        "Top headlines and featured stories in this week's newsletter \
         edition. Browse the archive of past editions.",
    ),
    (
        "Subscriber news",
        "Our newsletter brings you a curated digest of stories weekly. \
         Subscribe and browse the featured archive.",
    ),
];

const INVOICE_QUERY: &[u8] =
    b"Subject: invoice 9999 final notice\n\nFinal notice: the amount on invoice 9999 is overdue. \
      Please send your payment immediately to the billing department.\n";

const NEWSLETTER_QUERY: &[u8] =
    b"Subject: special edition digest\n\nA special edition of the newsletter with featured stories \
      and top headlines. Subscribe today to the weekly digest.\n";

fn temp_home(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("pmk-e2e-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

fn learn_dir(home: &Path, category: &str, label: &str, msgs: &[(&str, &str)]) {
    let dir = home.join(format!("fixtures-{label}"));
    fs::create_dir_all(&dir).unwrap();
    for (i, (subject, body)) in msgs.iter().enumerate() {
        fs::write(
            dir.join(format!("msg{i}.eml")),
            format!("{subject}\n\n{body}\n"),
        )
        .unwrap();
    }
    let out = learn::run(home, category, Some(dir.to_str().unwrap())).unwrap();
    assert!(
        out.contains(&format!("learned {}", msgs.len())),
        "learn said: {out}"
    );
}

fn seed(home: &Path) {
    learn_dir(home, "invoices", "seed-invoices", INVOICES);
    learn_dir(home, "newsletters", "seed-newsletters", NEWSLETTERS);
}

/// Rewrite one scalar in the generated config.toml (no toml round-trip
/// needed for tests; the flat `key = value` lines are stable).
fn set_config_value(home: &Path, key: &str, value: &str) {
    let path = config::config_path(home);
    let s = fs::read_to_string(&path).unwrap();
    let mut out = String::new();
    for line in s.lines() {
        if line.trim_start().starts_with(key) && line.contains('=') {
            out.push_str(&format!("{key} = {value}\n"));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    fs::write(path, out).unwrap();
}

#[test]
fn learn_then_infer_separates_categories() {
    let home = temp_home("separation");
    config::init(&home).unwrap();
    seed(&home);

    let v = infer::classify(&home, INVOICE_QUERY).unwrap();
    assert_eq!(v.category, "invoices", "share was {}", v.share);
    assert!(v.share >= 0.4, "share was {}", v.share);

    let v = infer::classify(&home, NEWSLETTER_QUERY).unwrap();
    assert_eq!(v.category, "newsletters", "share was {}", v.share);
    assert!(v.share >= 0.4, "share was {}", v.share);

    // Incremental correction: extra example for an existing category lands
    // with fresh keys and inference keeps working (labels.db alignment).
    learn_dir(
        &home,
        "invoices",
        "extra-invoices",
        &[(
            "Invoice overdue",
            "Overdue invoice amount: please arrange payment of the invoice \
             total this week.",
        )],
    );
    let v = infer::classify(&home, INVOICE_QUERY).unwrap();
    assert_eq!(v.category, "invoices", "share was {}", v.share);
}

#[test]
fn below_threshold_reports_unsure() {
    let home = temp_home("unsure");
    config::init(&home).unwrap();
    seed(&home);

    // k is clamped to the index size, so all six vectors vote and both
    // categories get weight; any winner share is strictly below 1.0.
    set_config_value(&home, "threshold", "1.0");
    let v = infer::classify(&home, INVOICE_QUERY).unwrap();
    assert_eq!(v.category, "unsure");

    // ... and full mode tags `unsure` like any other category.
    let raw = b"Subject: x\n\ny\n";
    let out = infer::full_output(&home, raw);
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("X-Postmaster-K-Category: unsure\n"), "{s}");
}

#[test]
fn full_mode_tags_and_preserves_everything_else() {
    let home = temp_home("full");
    config::init(&home).unwrap();
    seed(&home);

    let raw = b"From alice@example.com Fri Sep 26 10:00:00 2026\nSubject: invoice 9999 final notice\n\nFinal notice: the amount on invoice 9999 is overdue. Please send your payment immediately.\n";
    let out = infer::full_output(&home, raw);
    let s = String::from_utf8(out).unwrap();

    assert!(s.starts_with("From alice@example.com Fri Sep 26 10:00:00 2026\n"));
    assert!(s.contains("\nX-Postmaster-K-Category: invoices\n"), "{s}");
    let body = s.split_once("\n\n").map(|(_, b)| b).unwrap();
    assert_eq!(
        body,
        "Final notice: the amount on invoice 9999 is overdue. Please send your payment immediately.\n"
    );
}

#[test]
fn missing_index_is_misuse_and_full_mode_fails_open() {
    let home = temp_home("no-index");
    config::init(&home).unwrap();

    let raw: &[u8] = b"Subject: hello\n\nworld\n";
    match infer::classify(&home, raw) {
        Err(infer::InferError::Misuse(_)) => {}
        other => panic!("expected Misuse, got {other:?}"),
    }
    // `full` must still pass the message through untouched, exit 0 (§2).
    assert_eq!(infer::full_output(&home, raw), raw.to_vec());
}

#[test]
fn dims_mismatch_is_soft_failure_and_fails_open() {
    let home = temp_home("dims");
    config::init(&home).unwrap();
    seed(&home);
    set_config_value(&home, "dims", "512");

    let raw: &[u8] = b"Subject: invoice 9999\n\nOverdue amount, please pay.\n";
    match infer::classify(&home, raw) {
        Err(infer::InferError::Soft(_)) => {}
        other => panic!("expected Soft, got {other:?}"),
    }
    assert_eq!(infer::full_output(&home, raw), raw.to_vec());
}
