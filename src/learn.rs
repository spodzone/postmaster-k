//! The `learn` subcommand (SPEC §8).
//!
//! Same code path for bootstrap batches and single-message corrections:
//! extract text (§7.1), tokenize + feature-hash (§7), insert into the
//! usearch index with a fresh key, tag the key in `labels.db`, and fold
//! the tokens into `terms/<category>.db`. The index is opened in mutable
//! `load()` mode — this is the ONLY subcommand allowed to do that; `infer`
//! stays mmap-view-only. Nothing is persisted until the whole batch
//! parsed+embedded cleanly, so a failed `learn` leaves the previous state
//! intact.

use crate::{config, embed, store, text};
use std::fs;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use usearch::{new_index, Index, IndexOptions, MetricKind, ScalarKind};

/// Spare index capacity so a run of small incremental `learn`s doesn't
/// grow the file on every invocation.
const SLACK: u64 = 256;

pub fn run(home: &Path, category: &str, source: Option<&str>) -> Result<String, String> {
    let cat = config::normalize_category(category)?;
    // SPEC §8: `learn <cat> --` leaves clap's escape as the last arg, i.e.
    // no source — treat "no source" the same as an explicit `-`: stdin.
    let source = source.unwrap_or("-");

    let dims = config::settings(home)?.dims;

    let mut st = LearnState {
        labels: store::load_labels(&config::labels_path(home))
            .map_err(|e| format!("cannot read {}: {e}", config::labels_path(home).display()))?,
        terms: store::load_terms(&config::terms_path(home, &cat))
            .map_err(|e| format!("cannot read term table: {e}"))?,
        next_key: 0,
        learned: 0,
        skipped: 0,
    };
    st.next_key = store::next_key(&st.labels);

    let (files, from_stdin) = match source {
        "-" => {
            if std::io::stdin().is_terminal() {
                return Err(
                    "stdin is a terminal: give a directory of messages, or pipe one through `-`"
                        .into(),
                );
            }
            (Vec::new(), true)
        }
        dir => (list_message_files(Path::new(dir))?, false),
    };

    // Only load()-mode user in the whole tool (SPEC §8).
    let index = open_for_learning(
        &config::index_path(home),
        dims,
        files.len() as u64 + 1 + SLACK,
    )?;

    if from_stdin {
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .map_err(|e| format!("cannot read stdin: {e}"))?;
        ingest(&index, &buf, None, &cat, dims, &mut st)?;
    } else {
        for file in &files {
            let bytes =
                fs::read(file).map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            if bytes.is_empty() {
                st.skipped += 1;
                continue;
            }
            ingest(
                &index,
                &bytes,
                Some(file.display().to_string()),
                &cat,
                dims,
                &mut st,
            )?;
        }
    }

    if st.learned == 0 {
        return Err(format!(
            "no usable messages found for category `{cat}`; nothing was written"
        ));
    }

    // Register the category only once learning actually produced something.
    if config::add_category(home, &cat)? {
        eprintln!("postmaster-k: created category `{cat}`");
    }

    // Persist: index first (tmp+rename), then side-tables. If a side-table
    // write fails, the index may briefly contain unlabelled vectors;
    // `infer` skips unknown keys with a warning instead of guessing.
    save_index(&index, &config::index_path(home))?;
    store::store_labels(&config::labels_path(home), &st.labels)
        .map_err(|e| format!("cannot write {}: {e}", config::labels_path(home).display()))?;
    store::store_terms(&config::terms_path(home, &cat), &st.terms)
        .map_err(|e| format!("cannot write term table: {e}"))?;

    Ok(format!(
        "learned {} message{} into `{cat}` (index now {} vector{}, skipped {})",
        st.learned,
        if st.learned == 1 { "" } else { "s" },
        index.size(),
        if index.size() == 1 { "" } else { "s" },
        st.skipped
    ))
}

struct LearnState {
    labels: Vec<store::LabelEntry>,
    terms: store::TermMap,
    next_key: u64,
    learned: u64,
    skipped: u64,
}

fn ingest(
    index: &Index,
    raw: &[u8],
    source: Option<String>,
    category: &str,
    dims: usize,
    st: &mut LearnState,
) -> Result<(), String> {
    match learn_one(index, raw, st.next_key, dims, &mut st.terms) {
        Ok(true) => {
            st.labels.push(store::LabelEntry {
                key: st.next_key,
                category: category.to_string(),
                source,
            });
            st.next_key += 1;
            st.learned += 1;
            Ok(())
        }
        Ok(false) => {
            st.skipped += 1;
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Parse, extract, embed and insert one message. `Ok(false)` = nothing
/// classifiable (caller counts it as skipped).
fn learn_one(
    index: &Index,
    raw: &[u8],
    key: u64,
    dims: usize,
    terms: &mut store::TermMap,
) -> Result<bool, String> {
    let Some(txt) = text::extract_bytes(raw) else {
        return Ok(false);
    };
    let counts = embed::token_counts(embed::tokenize(&txt).into_iter());
    if counts.is_empty() {
        return Ok(false);
    }
    let vector = embed::embed_counts(&counts, dims);
    index
        .add(key, &vector)
        .map_err(|e| format!("usearch add(key={key}): {e}"))?;
    store::record_terms(terms, &counts);
    Ok(true)
}

fn list_message_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.is_dir() {
        return Err(format!(
            "`{}` is not a directory (or `-` for stdin)",
            dir.display()
        ));
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
        // metadata() follows symlinks; unreadable entries are skipped.
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        files.push(entry.path());
    }
    files.sort();
    Ok(files)
}

fn index_options(dims: usize) -> IndexOptions {
    IndexOptions {
        dimensions: dims,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: 0,     // auto
        expansion_add: 0,    // auto
        expansion_search: 0, // auto
        multi: false,
    }
}

fn path_str(p: &Path) -> Result<&str, String> {
    p.to_str()
        .ok_or_else(|| format!("path {} is not valid UTF-8", p.display()))
}

/// Open the index in mutable mode: `Index::restore` = read metadata +
/// construct with matching options + `load()`. Validates that the on-disk
/// index was built with the same `dims` the config asks for.
fn open_for_learning(index_path: &Path, dims: usize, extra: u64) -> Result<Index, String> {
    if !index_path.exists() {
        let index =
            new_index(&index_options(dims)).map_err(|e| format!("cannot create index: {e}"))?;
        index
            .reserve(extra as usize)
            .map_err(|e| format!("cannot size index: {e}"))?;
        return Ok(index);
    }
    let ps = path_str(index_path)?;
    let meta = Index::metadata(ps).map_err(|e| format!("cannot read index header: {e}"))?;
    if meta.dimensions != dims as u64 {
        return Err(format!(
            "config `dims`={dims} but {} was built with dims={}; re-learn from scratch or remove it",
            index_path.display(),
            meta.dimensions
        ));
    }
    let index = Index::restore(ps).map_err(|e| format!("cannot load index: {e}"))?;
    let needed = index.size() as u64 + extra;
    if (index.capacity() as u64) < needed {
        index
            .reserve(needed as usize)
            .map_err(|e| format!("cannot grow index: {e}"))?;
    }
    Ok(index)
}

/// Save via tmp file + rename so a crash mid-`learn` can't leave a
/// half-written index.usearch.
fn save_index(index: &Index, path: &Path) -> Result<(), String> {
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    index
        .save(path_str(&tmp)?)
        .map_err(|e| format!("cannot save index: {e}"))?;
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(format!("cannot replace {}: {e}", path.display()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_one_skips_empty_and_learns_real_text() {
        let dims = 256;
        let index = new_index(&index_options(dims)).unwrap();
        index.reserve(10).unwrap();
        let mut terms = store::TermMap::new();

        assert!(!learn_one(&index, b"", 1, dims, &mut terms).unwrap());
        assert!(!learn_one(&index, b"Subject: \n\n...", 1, dims, &mut terms).unwrap());
        assert_eq!(index.size(), 0);

        let raw = b"Subject: invoice due\n\nPlease pay the amount on the invoice.\n";
        assert!(learn_one(&index, raw, 1, dims, &mut terms).unwrap());
        assert_eq!(index.size(), 1);
        assert!(terms.contains_key("invoice"));
        assert!(terms.contains_key("amount"));
        assert!(!terms.contains_key("the")); // stopword
    }
}
