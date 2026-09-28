//! The `autocalibrate` subcommand — sample a Maildir, score every example
//! under a grid of `(threshold, threshold_factor)` pairs, and write the best
//! combination back into config.toml.
//!
//! Sampling is max(20%, 200) per category with uniform random ordering so
//! large categories aren't fully scored on every run while small ones still
//! get enough data to be representative.

use crate::{config, embed, infer, store, text};
use rand::rng;
use rand::seq::SliceRandom;
use std::fs;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

/// Minimum sample size per category so small Maildirs still get enough data.
const MIN_SAMPLES: usize = 200;

/// Default fraction of each category to sample when the category is large.
const SAMPLE_FRACTION: f64 = 0.20;

/// Grid step for `threshold`: 0.0, 0.1, …, 0.6.
const THRESHOLD_STEP: f64 = 0.1;
/// Grid step for `threshold_factor`: 1.0, 1.25, …, 5.0.
const FACTOR_STEP: f64 = 0.25;

#[derive(Clone)]
pub(crate) struct LabeledPoint {
    category: String,
    vec: Vec<f32>,
}

struct CategorySample {
    category: String,
    messages: Vec<(Vec<u8>, PathBuf)>,
}

struct GridResult {
    threshold: f64,
    factor: f64,
    correct: usize,
    total: usize,
    pct: f64,
}

/// Run autocalibration over `maildir`, which must contain one subdirectory
/// per category of message files. Returns the chosen (threshold, factor) pair.
pub fn run(maildir: &Path, home: &Path) -> Result<(f64, f64), String> {
    if !maildir.is_dir() {
        return Err(format!("`{}` is not a directory", maildir.display()));
    }

    let settings = config::settings(home)?;
    let dims = settings.dims;
    let k = settings.k;

    let samples = discover_samples(maildir)?;
    if samples.is_empty() {
        return Err(format!(
            "no category subdirectories found under {}",
            maildir.display()
        ));
    }

    let labels = store::load_labels(&config::labels_path(home))
        .map_err(|e| format!("cannot read {}: {e}", config::labels_path(home).display()))?;

    let index_path = config::index_path(home);
    if !index_path.exists() {
        return Err("no index yet — run `postmaster-k learn <category> <directory>` first".into());
    }
    let index = open_view(&index_path)?;
    let size = index.size();
    if size == 0 {
        return Err("index is empty — run `postmaster-k learn` first".into());
    }
    let k = k.clamp(1, size as usize);

    // Embed every sampled message into its vector. Keep vectors to avoid
    // re-embedding across grid points.
    let mut points: Vec<LabeledPoint> = Vec::new();
    for sample in &samples {
        for (raw, _path) in &sample.messages {
            let txt = text::extract_bytes(raw).unwrap_or_default();
            let counts = embed::token_counts(embed::tokenize(&txt).into_iter());
            if counts.is_empty() {
                continue;
            }
            let vec = embed::embed_counts(&counts, dims);
            points.push(LabeledPoint {
                category: sample.category.clone(),
                vec,
            });
        }
    }
    if points.is_empty() {
        return Err("no classifiable messages found in any category".into());
    }

    // Grid search.
    let thresholds: Vec<f64> = (0..=6).map(|i| i as f64 * THRESHOLD_STEP).collect();
    let factors: Vec<f64> = std::iter::successors(Some(1.0f64), move |v| {
        let next = v + FACTOR_STEP;
        if next <= 5.0 + 1e-9 {
            Some(next)
        } else {
            None
        }
    })
    .collect();

    eprintln!(
        "postmaster-k: autocalibrate: scoring {} (threshold x factor) combinations",
        thresholds.len() * factors.len()
    );

    let mut results: Vec<GridResult> = Vec::new();
    for (&threshold, ti) in thresholds.iter().zip(0..) {
        for (&factor, fi) in factors.iter().zip(0..) {
            let (correct, total) = score_grid_point(&points, &labels, &index, k, threshold, factor);
            results.push(GridResult {
                threshold,
                factor,
                correct,
                total,
                pct: if total > 0 {
                    correct as f64 / total as f64 * 100.0
                } else {
                    0.0
                },
            });
            // Update progress bar inline.
            let outer_done = ti + 1;
            let inner_done = fi + 1;
            let outer_total = thresholds.len();
            let inner_total = factors.len();
            let bar_w = 10;
            let ob = (inner_done as f64 / inner_total as f64 * bar_w as f64)
                .round()
                .clamp(0.0, bar_w as f64) as usize;
            let ib = (outer_done as f64 / outer_total as f64 * bar_w as f64)
                .round()
                .clamp(0.0, bar_w as f64) as usize;
            eprint!(
                "\r  [{}{}] [{}{}] ({:.2}, {:.2}, {}/{}, {:.2}%)",
                "#".repeat(ob),
                ".".repeat(bar_w - ob),
                "#".repeat(ib),
                ".".repeat(bar_w - ib),
                threshold,
                factor,
                correct,
                total,
                results.last().unwrap().pct
            );
        }
    }
    eprintln!(); // newline after progress bar

    // Pick best: max accuracy, tie-break by second-highest threshold then lowest factor.
    // We want the highest accuracy; among those with equal accuracy we prefer the
    // threshold that is *second highest* (i.e. just below the peak) because lower
    // thresholds are more permissive and still achieve the same top performance.
    let max_correct = results.iter().map(|r| r.correct).max().unwrap_or(0);
    let top_results: Vec<&GridResult> = results
        .iter()
        .filter(|r| r.correct == max_correct)
        .collect();
    // Sort by threshold descending, then factor ascending; pick the second if available.
    let mut sorted: Vec<&GridResult> = top_results.clone();
    sorted.sort_by(|a, b| {
        a.threshold
            .partial_cmp(&b.threshold)
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
            .then_with(|| {
                a.factor
                    .partial_cmp(&b.factor)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let best = if sorted.len() > 1 {
        sorted[1]
    } else {
        sorted[0]
    };

    if std::io::stdout().is_terminal() {
        print_table(&results)?;
    } else {
        // Non-terminal: print a compact markdown table so it can be piped.
        // Compute widths for alignment.
        let tw = results
            .iter()
            .map(|r| fmt_f(r.threshold).len())
            .max()
            .unwrap_or(7);
        let fw = results
            .iter()
            .map(|r| fmt_f(r.factor).len())
            .max()
            .unwrap_or(6);
        let pw = results
            .iter()
            .map(|r| format!("{:.1}", r.pct).len())
            .max()
            .unwrap_or(8);
        println!(
            "| {threshold:>tw$} | {factor:>fw$} | {:>pw$.1} |",
            0.0,
            threshold = "threshold",
            factor = "factor",
            tw = tw,
            fw = fw,
            pw = pw
        );
        println!("|{}|{}|{}|", "-".repeat(tw), "-".repeat(fw), "-".repeat(pw));
        for r in &results {
            println!(
                "| {:>tw$} | {:>fw$} | {:>pw$.1} |",
                fmt_f(r.threshold),
                fmt_f(r.factor),
                r.pct,
                tw = tw,
                fw = fw,
                pw = pw
            );
        }
        println!();
        println!(
            "**recommended: threshold = {}, factor = {}** ({}/{})",
            fmt_f(best.threshold),
            fmt_f(best.factor),
            best.correct,
            best.total
        );
    }

    config::update_float(home, "threshold", best.threshold)
        .map_err(|e| format!("cannot write config: {e}"))?;
    config::update_float(home, "threshold_factor", best.factor)
        .map_err(|e| format!("cannot write config: {e}"))?;
    eprintln!(
        "postmaster-k: wrote threshold={} factor={} into {}",
        fmt_f(best.threshold),
        fmt_f(best.factor),
        config::config_path(home).display()
    );

    Ok((best.threshold, best.factor))
}

fn fmt_f(v: f64) -> String {
    // Render clean: strip unnecessary trailing zeros but keep at least one
    // decimal place for readability (e.g. "2.0" not "2").
    let s = format!("{v:.2}");
    if s.contains('.') {
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        if trimmed.is_empty() || trimmed == "-" {
            "0.0".to_string()
        } else {
            trimmed.to_string()
        }
    } else {
        format!("{v}.0")
    }
}

fn print_table(results: &[GridResult]) -> Result<(), String> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();

    // Compute column widths from all rows using formatted values.
    let tw = results
        .iter()
        .map(|r| fmt_f(r.threshold).len())
        .max()
        .unwrap_or(7);
    let fw = results
        .iter()
        .map(|r| fmt_f(r.factor).len())
        .max()
        .unwrap_or(6);
    let pw = results
        .iter()
        .map(|r| format!("{:.1}", r.pct).len())
        .max()
        .unwrap_or(8);

    // Header cells padded to column widths.
    let hdr_thresh = format!("{:>width$}", "threshold", width = tw);
    let hdr_factor = format!("{:>fw$}", "factor", fw = fw);
    let hdr_pct = format!("{:>pw$.1}", 0.0, pw = pw);

    let _ = writeln!(handle, "| {hdr_thresh} | {hdr_factor} | {hdr_pct} |");
    let sep = format!("|{}|{}|{}|", "-".repeat(tw), "-".repeat(fw), "-".repeat(pw));
    let _ = writeln!(handle, "{sep}");
    for r in results {
        let ts = fmt_f(r.threshold);
        let fs = fmt_f(r.factor);
        let _ = writeln!(
            handle,
            "| {:>width$} | {:>fw$} | {:>pw$.1} |",
            ts,
            fs,
            r.pct,
            width = tw,
            fw = fw,
            pw = pw
        );
    }
    let _ = writeln!(handle);
    let _ = writeln!(
        handle,
        "**recommended: threshold = {} factor = {}** ({}/{})",
        fmt_f(results[0].threshold),
        fmt_f(results[0].factor),
        results[0].correct,
        results[0].total
    );
    handle
        .flush()
        .map_err(|e| format!("cannot flush stdout: {e}"))
}

fn discover_samples(maildir: &Path) -> Result<Vec<CategorySample>, String> {
    let mut samples = Vec::new();
    for entry in
        fs::read_dir(maildir).map_err(|e| format!("cannot read {}: {e}", maildir.display()))?
    {
        let entry = entry.map_err(|e| format!("cannot read {}: {e}", maildir.display()))?;
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let category = dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("non-UTF-8 category directory: {}", dir.display()))?
            .to_string();
        let files = list_message_files(&dir)?;
        if files.is_empty() {
            continue;
        }
        let n_total = files.len();
        let want = (n_total as f64 * SAMPLE_FRACTION).ceil() as usize;
        let sample_n = want.max(MIN_SAMPLES).min(n_total);

        let mut rng = rng();
        let mut shuffled = files;
        shuffled.shuffle(&mut rng);
        let chosen: Vec<PathBuf> = shuffled.into_iter().take(sample_n).collect();

        let messages: Vec<(Vec<u8>, PathBuf)> = chosen
            .into_iter()
            .map(|p| {
                let bytes =
                    fs::read(&p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
                Ok((bytes, p))
            })
            .collect::<Result<Vec<_>, String>>()?;

        samples.push(CategorySample {
            category: category.clone(),
            messages,
        });
    }
    // Deterministic order (alphabetical by category) so output is stable.
    samples.sort_by(|a, b| a.category.cmp(&b.category));
    let total = samples.iter().map(|s| s.messages.len()).sum::<usize>();
    eprintln!(
        "postmaster-k: autocalibrate: sampled {} message{} across {} category{}",
        total,
        if total == 1 { "" } else { "s" },
        samples.len(),
        if samples.len() == 1 { "" } else { "s" },
    );
    for s in &samples {
        eprintln!("  {}: {} messages", s.category, s.messages.len());
    }
    Ok(samples)
}

fn list_message_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        files.push(entry.path());
    }
    files.sort();
    Ok(files)
}

fn score_grid_point(
    points: &[LabeledPoint],
    labels: &[store::LabelEntry],
    index: &usearch::Index,
    k: usize,
    threshold: f64,
    factor: f64,
) -> (usize, usize) {
    let mut correct = 0;
    let mut total = 0;
    for p in points {
        let matches = match index.search(&p.vec, k) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if matches.keys.is_empty() {
            continue;
        }
        let verdict = infer::vote(labels, &matches.keys, &matches.distances, threshold, factor);
        total += 1;
        if verdict.category == p.category {
            correct += 1;
        }
    }
    (correct, total)
}

/// Open the index in mmap view mode. Mirrors `infer::open_view` logic so we
/// don't duplicate error handling here.
fn open_view(index_path: &Path) -> Result<usearch::Index, String> {
    let ps = index_path
        .to_str()
        .ok_or_else(|| format!("path {} is not valid UTF-8", index_path.display()))?;
    let _meta =
        usearch::Index::metadata(ps).map_err(|e| format!("cannot read index header: {e}"))?;
    usearch::Index::restore_view(ps).map_err(|e| format!("cannot mmap-view index: {e}"))
}
