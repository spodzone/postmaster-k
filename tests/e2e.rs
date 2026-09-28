//! End-to-end tests: learn → infer round trips against a throwaway state
//! root, exercising the SPEC §6/§8/§2 semantics (separation, threshold,
//! fail-open, byte-exact full-mode passthrough).

use postmaster_k::{config, infer, learn, store};
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
    (
        "Overdue balance notice",
        "Your account shows an overdue balance. Please remit payment immediately \
         to avoid suspension. Invoice references and remittance advice are required.",
    ),
    (
        "Billing statement enclosed",
        "Enclosed is your monthly billing statement. The total due includes \
         invoice charges and any applicable interest on past-due amounts.",
    ),
    (
        "Remittance advice request",
        "Kindly provide remittance advice together with your payment for the \
         outstanding invoice. Past-due accounts will be subject to interest charges.",
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
    (
        "Editorial update",
        "A new editorial update is now available in your newsletter dashboard. \
         Unsubscribe from future editions via the link below.",
    ),
    (
        "Monthly roundup",
        "This month's newsletter roundup covers culture, science and tech stories. \
         Archive access and subscription management are available online.",
    ),
    (
        "Featured articles digest",
        "Browse the featured articles in this month's editorial digest. \
         Subscribe to receive weekly editions or unsubscribe from future issues.",
    ),
];

const INVOICE_QUERY: &[u8] =
    b"Subject: invoice 9999 final notice\n\nFinal notice: the amount on invoice 9999 is overdue. \
      Please send your payment immediately to the billing department with remittance advice.\n";

const NEWSLETTER_QUERY: &[u8] =
    b"Subject: special edition digest\n\nA special edition of the newsletter with featured stories \
      and top headlines. Subscribe today to the weekly editorial digest and browse the archive.\n";

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
        // Match only lines whose trimmed content starts with `key ` (key followed
        // by whitespace) and contains '=', so setting `threshold` doesn't also
        // match the `threshold_factor` key.
        let trimmed = line.trim_start();
        if trimmed.starts_with(&format!("{key} ")) && trimmed.contains('=') {
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
    // Use a low threshold_factor so the test exercises separation rather than
    // the hockey-stick criterion (which is covered by unit tests).
    set_config_value(&home, "threshold_factor", "1.1");
    seed(&home);

    let v = infer::classify(&home, INVOICE_QUERY).unwrap();
    assert_eq!(v.category, "invoices", "share was {}", v.share);

    let v = infer::classify(&home, NEWSLETTER_QUERY).unwrap();
    assert_eq!(v.category, "newsletters", "share was {}", v.share);

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
    // Low threshold_factor so the test exercises full-mode tagging, not the
    // hockey-stick criterion.
    set_config_value(&home, "threshold_factor", "1.1");
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

#[test]
fn infer_learn_reinforces_winning_category() {
    let home = temp_home("infer-learn");
    config::init(&home).unwrap();
    // Low threshold_factor so the verdict is a clean `invoices`.
    set_config_value(&home, "threshold_factor", "1.1");
    seed(&home);

    let before = store::load_labels(&config::labels_path(&home))
        .unwrap()
        .len();

    let msg = home.join("query.eml");
    fs::write(&msg, INVOICE_QUERY).unwrap();
    assert_eq!(infer::run(&home, infer::Mode::Word, true, Some(&msg)), 0);

    // One new label entry, tagged with the category infer just chose.
    let labels = store::load_labels(&config::labels_path(&home)).unwrap();
    assert_eq!(labels.len(), before + 1, "--learn must add one vector");
    assert_eq!(labels.last().unwrap().category, "invoices");

    // And the index still classifies correctly afterwards.
    assert_eq!(
        infer::classify(&home, INVOICE_QUERY).unwrap().category,
        "invoices"
    );
}

#[test]
fn infer_learn_skips_unsure() {
    let home = temp_home("infer-learn-unsure");
    config::init(&home).unwrap();
    seed(&home);
    // Force every verdict to `unsure`.
    set_config_value(&home, "threshold", "1.0");

    let before = store::load_labels(&config::labels_path(&home))
        .unwrap()
        .len();

    let msg = home.join("query.eml");
    fs::write(&msg, INVOICE_QUERY).unwrap();
    assert_eq!(infer::run(&home, infer::Mode::Word, true, Some(&msg)), 0);

    let labels = store::load_labels(&config::labels_path(&home)).unwrap();
    assert_eq!(
        labels.len(),
        before,
        "an `unsure` verdict must not be learned"
    );
}
