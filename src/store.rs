//! Side-tables: `labels.db` (usearch key -> category) and the per-category
//! unhashed term tables `terms/<category>.db` (SPEC §5, §7.2).
//!
//! Both are plain tab-separated text, rewritten atomically (tmp + fsync +
//! rename) on `learn`. `labels.db` rows: `key<TAB>category<TAB>source`
//! (source may be empty; it records where an example came from so a bad
//! example can be pruned/relearned later). Term rows:
//! `token<TAB>count<TAB>docs` — the `docs` column is the document
//! frequency, pre-staged for optional IDF weighting (SPEC §7 step 3).

use crate::config::atomic_write;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// One labels.db row.
#[derive(Debug, Clone)]
pub struct LabelEntry {
    pub key: u64,
    pub category: String,
    pub source: Option<String>,
}

/// Load labels.db; a missing file is an empty table (fresh state root).
/// Malformed lines are skipped so a hand-edited file can't wedge learning.
pub fn load_labels(path: &Path) -> io::Result<Vec<LabelEntry>> {
    let mut entries = Vec::new();
    match File::open(path) {
        Ok(f) => {
            for line in BufReader::new(f).lines() {
                let line = line?;
                let mut it = line.split('\t');
                let (Some(k), Some(cat)) = (it.next(), it.next()) else {
                    continue;
                };
                let Ok(key) = k.parse::<u64>() else {
                    continue;
                };
                if cat.is_empty() {
                    continue;
                }
                let source = it.next().filter(|s| !s.is_empty()).map(str::to_string);
                entries.push(LabelEntry {
                    key,
                    category: cat.to_string(),
                    source,
                });
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    Ok(entries)
}

/// Atomically rewrite labels.db.
pub fn store_labels(path: &Path, entries: &[LabelEntry]) -> io::Result<()> {
    let mut out = String::new();
    for e in entries {
        let src = e.source.as_deref().unwrap_or("").replace(['\t', '\n'], " ");
        writeln!(out, "{}\t{}\t{}", e.key, e.category, src).map_err(io::Error::other)?;
    }
    atomic_write(path, out.as_bytes())
}

/// Next free usearch key (keys are dense, but this is robust to gaps).
pub fn next_key(entries: &[LabelEntry]) -> u64 {
    entries.iter().map(|e| e.key).max().unwrap_or(0) + 1
}

/// token -> (total occurrences, number of messages it appeared in)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TermStat {
    pub count: u64,
    pub docs: u64,
}

pub type TermMap = std::collections::BTreeMap<String, TermStat>;

pub fn load_terms(path: &Path) -> io::Result<TermMap> {
    let mut map = TermMap::new();
    match File::open(path) {
        Ok(f) => {
            for line in BufReader::new(f).lines() {
                let line = line?;
                let mut it = line.split('\t');
                let (Some(tok), Some(cnt)) = (it.next(), it.next()) else {
                    continue;
                };
                if tok.is_empty() {
                    continue;
                }
                let Ok(count) = cnt.parse::<u64>() else {
                    continue;
                };
                let docs = it.next().and_then(|d| d.parse::<u64>().ok()).unwrap_or(1);
                map.insert(tok.to_string(), TermStat { count, docs });
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    Ok(map)
}

/// Fold one message's token counts into a category's term table.
pub fn record_terms(map: &mut TermMap, message_counts: &HashMap<String, u32>) {
    for (tok, n) in message_counts {
        let st = map.entry(tok.clone()).or_default();
        st.count += *n as u64;
        st.docs += 1;
    }
}

/// Atomically rewrite a terms/<category>.db, rows sorted by token
/// (BTreeMap order) so the files stay diffable.
pub fn store_terms(path: &Path, map: &TermMap) -> io::Result<()> {
    let mut out = String::with_capacity(map.len() * 24);
    for (tok, st) in map {
        writeln!(out, "{}\t{}\t{}", tok, st.count, st.docs).map_err(io::Error::other)?;
    }
    atomic_write(path, out.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "pmk-store-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn labels_round_trip_and_skip_malformed() {
        let dir = tempdir("labels");
        let path = dir.join("labels.db");
        fs::write(
            &path,
            "1\tinvoices\t/home/me/mail/1\n\
             2\tnewsletters\t\n\
             not-a-key\tbogus\n\
             4\tspam\n\
             \n",
        )
        .unwrap();

        let entries = load_labels(&path).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key, 1);
        assert_eq!(entries[0].source.as_deref(), Some("/home/me/mail/1"));
        assert_eq!(entries[1].source, None);
        assert_eq!(entries[2].key, 4);

        let mut updated = entries.clone();
        updated.push(LabelEntry {
            key: next_key(&updated),
            category: "unsure-free-zone".into(),
            source: None,
        });
        store_labels(&path, &updated).unwrap();

        let reloaded = load_labels(&path).unwrap();
        assert_eq!(reloaded.len(), 4);
        assert_eq!(reloaded.last().unwrap().key, 5);
        assert_eq!(
            reloaded
                .iter()
                .map(|e| e.category.as_str())
                .collect::<Vec<_>>(),
            vec!["invoices", "newsletters", "spam", "unsure-free-zone"]
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn labels_missing_file_is_empty() {
        let dir = tempdir("labels-missing");
        let entries = load_labels(&dir.join("nope.db")).unwrap();
        assert!(entries.is_empty());
        assert_eq!(next_key(&entries), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn terms_round_trip_and_record() {
        let dir = tempdir("terms");
        let path = dir.join("invoices.db");
        let mut map = load_terms(&path).unwrap();
        assert!(map.is_empty());

        let mut counts = HashMap::new();
        counts.insert("invoice".to_string(), 3u32);
        counts.insert("due".to_string(), 1u32);
        record_terms(&mut map, &counts);
        record_terms(&mut map, &counts); // second message

        let st = &map["due"];
        assert_eq!((st.count, st.docs), (2, 2));
        let st = &map["invoice"];
        assert_eq!((st.count, st.docs), (6, 2));

        store_terms(&path, &map).unwrap();
        let reloaded = load_terms(&path).unwrap();
        assert_eq!(reloaded, map);

        let text = fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("due\t2\t2\ninvoice\t6\t2\n")); // sorted
        let _ = fs::remove_dir_all(&dir);
    }
}
