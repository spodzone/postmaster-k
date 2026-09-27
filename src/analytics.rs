//! The `analytics` subcommand (SPEC §8): top-N terms for a category,
//! sourced from the unhashed per-category term tables (§7.2) — not from the
//! hashed vector index, since hash buckets aren't reversible into words.
//! A diagnostic tool for spotting categories that separated on real
//! vocabulary versus absorbed boilerplate.

use crate::{config, store};
use std::path::Path;

pub fn run(home: &Path, category: Option<&str>, n: usize, all: bool) -> Result<String, String> {
    let settings = config::settings(home)?;

    let cats: Vec<String> = if all {
        let mut c = settings.categories.clone();
        c.sort();
        c
    } else {
        // clap guarantees exactly one of category/--all.
        let cat = config::normalize_category(category.expect("clap enforces category or --all"))?;
        if !settings.categories.contains(&cat) {
            let known = if settings.categories.is_empty() {
                "none".to_string()
            } else {
                settings.categories.join(", ")
            };
            return Err(format!("unknown category `{cat}` (configured: {known})"));
        }
        vec![cat]
    };

    if cats.is_empty() {
        return Err(
            "no categories configured yet — run `postmaster-k learn <category> <directory>` first"
                .into(),
        );
    }

    let mut sections = Vec::with_capacity(cats.len());
    for cat in &cats {
        let path = config::terms_path(home, cat);
        let terms = match store::load_terms(&path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => store::TermMap::new(),
            Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
        };
        sections.push(render(cat, &terms, n, cats.len() > 1));
    }
    Ok(sections.join("\n\n"))
}

fn render(cat: &str, terms: &store::TermMap, n: usize, header: bool) -> String {
    // Count descending; ties alphabetical so the report is deterministic.
    let mut ranked: Vec<(&String, &store::TermStat)> = terms.iter().collect();
    ranked.sort_by(|a, b| b.1.count.cmp(&a.1.count).then_with(|| a.0.cmp(b.0)));
    ranked.truncate(n);

    let mut out = String::new();
    if header {
        out.push_str(&format!("{cat}:\n"));
    }
    if ranked.is_empty() {
        out.push_str(&format!("(no terms recorded for `{cat}` yet)"));
        return out;
    }
    let width = ranked
        .iter()
        .map(|(t, _)| t.len())
        .max()
        .unwrap_or(0)
        .min(28);
    for (i, (term, stat)) in ranked.iter().enumerate() {
        out.push_str(&format!(
            "{:>2}. {:<width$}   ({})\n",
            i + 1,
            term,
            stat.count
        ));
    }
    // `run`'s caller adds the final newline; don't double up.
    while out.ends_with('\n') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn temp_home(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("pmk-ana-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn term(count: u64) -> store::TermStat {
        store::TermStat { count, docs: 1 }
    }

    #[test]
    fn ranks_by_count_then_alphabetically() {
        let mut terms: store::TermMap = BTreeMap::new();
        terms.insert("invoice".into(), term(412));
        terms.insert("payment".into(), term(388));
        terms.insert("due".into(), term(301));
        terms.insert("zebra".into(), term(50));
        terms.insert("apple".into(), term(50));

        let out = render("invoices", &terms, 10, false);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], " 1. invoice   (412)");
        assert_eq!(lines[1], " 2. payment   (388)");
        assert_eq!(lines[3], " 4. apple     (50)", "tie broken alphabetically");
        assert_eq!(lines[4], " 5. zebra     (50)");
    }

    #[test]
    fn honours_n_limit() {
        let mut terms: store::TermMap = BTreeMap::new();
        for i in 0..30 {
            terms.insert(format!("term{i}"), term(i as u64 + 1));
        }
        let out = render("c", &terms, 5, false);
        assert_eq!(out.lines().count(), 5);
    }

    #[test]
    fn empty_table_reports_placeholder() {
        let out = render("ghost", &store::TermMap::new(), 20, false);
        assert_eq!(out, "(no terms recorded for `ghost` yet)");
    }

    #[test]
    fn all_mode_renders_one_section_per_category() {
        let home = temp_home("all");
        config::init(&home).unwrap();
        config::add_category(&home, "invoices").unwrap();
        config::add_category(&home, "newsletters").unwrap();

        let mut inv: store::TermMap = BTreeMap::new();
        inv.insert("invoice".into(), term(9));
        store::store_terms(&config::terms_path(&home, "invoices"), &inv).unwrap();

        let out = run(&home, None, 20, true).unwrap();
        assert!(out.contains("invoices:\n 1. invoice   (9)"), "{out}");
        assert!(
            out.contains("newsletters:\n(no terms recorded for `newsletters` yet)"),
            "{out}"
        );
    }

    #[test]
    fn unknown_category_is_an_error() {
        let home = temp_home("unknown");
        config::init(&home).unwrap();
        config::add_category(&home, "invoices").unwrap();
        let err = run(&home, Some("nonesuch"), 20, false).unwrap_err();
        assert!(err.contains("unknown category `nonesuch`"), "{err}");
    }
}
