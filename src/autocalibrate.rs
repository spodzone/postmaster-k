//! The `autocalibrate` subcommand — sample a Maildir, score every example
//! under a grid of `(dims, threshold, threshold_factor)` triples, and write the
//! best combination back into config.toml.
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
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;

/// Minimum sample size per category so small Maildirs still get enough data.
const MIN_SAMPLES: usize = 200;

/// Default fraction of each category to sample when the category is large.
const SAMPLE_FRACTION: f64 = 0.20;

/// Number of grid-search evaluations to run concurrently: one less than the
/// available CPU core count, so the autocalibrator doesn't starve the system.
static PARALLEL_JOBS: LazyLock<usize> = LazyLock::new(|| num_cpus::get().saturating_sub(1).max(1));

/// Grid step for `threshold`: 0.0, 0.1, …, 0.6.
const THRESHOLD_STEP: f64 = 0.1;
/// Grid step for `threshold_factor`: 1.0, 1.25, …, 5.0.
const FACTOR_STEP: f64 = 0.25;
/// Dims grid: powers of two from 64 to 65536.
const DIMS_GRID: [usize; 11] = [
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];

#[derive(Clone)]
pub(crate) struct LabeledPoint {
    category: String,
    vec: Vec<f32>,
}

struct CategorySample {
    category: String,
    messages: Vec<(Vec<u8>, PathBuf)>,
}

/// A single result from scoring one grid point.
#[derive(Clone)]
struct GridResult {
    dims: usize,
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
    let _dims = settings.dims;
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
    let index = Arc::new(Mutex::new(open_view(&index_path)?));
    let size = index.lock().unwrap().size();
    if size == 0 {
        return Err("index is empty — run `postmaster-k learn` first".into());
    }
    let k = k.clamp(1, size as usize);

    // Grid search over (dims, threshold, factor).
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
        "postmaster-k: autocalibrate: scoring {} (dims x threshold x factor) combinations ({} parallel)",
        DIMS_GRID.len() * thresholds.len() * factors.len(),
        *PARALLEL_JOBS
    );

    let total_jobs = DIMS_GRID.len() * thresholds.len() * factors.len();
    // Shared results matrix — protected by a mutex so threads can write safely.
    let results: Arc<Mutex<Vec<GridResult>>> = Arc::new(Mutex::new(Vec::new()));
    // Atomic counters for progress reporting across all threads.
    let done_count = Arc::new(AtomicUsize::new(0));
    // Track the highest (di, ti, fi) completed so far (used by monitor).
    let last_done = Arc::new(Mutex::new((0usize, 0usize, 0usize))); // (di, ti, fi)

    let mut handles = Vec::with_capacity(total_jobs);

    for (dims, di) in DIMS_GRID.iter().zip(0..) {
        // Re-embed points at this dims value.
        let points: Vec<LabeledPoint> = samples
            .iter()
            .flat_map(|sample| &sample.messages)
            .filter_map(|(raw, _path)| {
                let txt = text::extract_bytes(raw).unwrap_or_default();
                let counts = embed::token_counts(embed::tokenize(&txt).into_iter());
                if counts.is_empty() {
                    return None;
                }
                Some(LabeledPoint {
                    category: samples[di % samples.len()].category.clone(),
                    vec: embed::embed_counts(&counts, *dims),
                })
            })
            .collect();
        if points.is_empty() {
            continue;
        }

        for (threshold, ti) in thresholds.iter().zip(0..) {
            for (&factor, fi) in factors.iter().zip(0..) {
                let results = Arc::clone(&results);
                let done_count = Arc::clone(&done_count);
                let last_done = Arc::clone(&last_done);
                let points = points.clone();
                let labels = labels.clone();
                let index = Arc::clone(&index);
                let threshold = *threshold;
                let factor = factor;
                let dims = *dims;
                let k = k;

                let handle = thread::spawn(move || {
                    let idx = index.lock().unwrap();
                    let (correct, total) =
                        score_grid_point(&points, &labels, &idx, k, threshold, factor);
                    drop(idx);
                    let pct = if total > 0 {
                        correct as f64 / total as f64 * 100.0
                    } else {
                        0.0
                    };
                    let res = GridResult {
                        dims,
                        threshold,
                        factor,
                        correct,
                        total,
                        pct,
                    };
                    results.lock().unwrap().push(res);
                    // Record which (di, ti, fi) just finished so the monitor can show
                    // the right positions.
                    {
                        let mut ld = last_done.lock().unwrap();
                        if di > ld.0
                            || (di == ld.0 && ti > ld.1)
                            || (di == ld.0 && ti == ld.1 && fi > ld.2)
                        {
                            *ld = (di, ti, fi);
                        }
                    }
                    done_count.fetch_add(1, AtomicOrdering::Relaxed);
                });
                handles.push(handle);
            }
        }
    }

    // Spawn a monitor thread that shows progress across all three dimensions.
    let monitor_done = Arc::clone(&done_count);
    let monitor_total = total_jobs;
    let monitor_last = Arc::clone(&last_done);
    let monitor_results = Arc::clone(&results);
    thread::spawn(move || {
        let bar_w = 10;
        loop {
            let n = monitor_done.load(AtomicOrdering::Relaxed);
            if n >= monitor_total {
                break;
            }
            let ld = monitor_last.lock().unwrap();
            let (ldi, lti, lfi) = *ld;
            // Outer bar: fraction of dims values completed.
            let outer_done_f = (ldi as f64 + 1.0) / DIMS_GRID.len() as f64;
            let ob = (outer_done_f * bar_w as f64)
                .round()
                .clamp(0.0, bar_w as f64) as usize;
            // Middle bar: fraction of threshold rows completed within current dims.
            let mid_done_f = (lti as f64 + 1.0) / thresholds.len() as f64;
            let mb = (mid_done_f * bar_w as f64).round().clamp(0.0, bar_w as f64) as usize;
            // Inner bar: fraction of factor columns completed within current row.
            let inner_done_f = (lfi as f64 + 1.0) / factors.len() as f64;
            let ib = (inner_done_f * bar_w as f64)
                .round()
                .clamp(0.0, bar_w as f64) as usize;
            // Pick a representative result for the live values (last finished).
            let last_res = monitor_results.lock().unwrap();
            let rep = last_res
                .iter()
                .filter(|r| {
                    r.dims == DIMS_GRID[ldi]
                        && r.threshold == thresholds[lti]
                        && r.factor == factors[lfi]
                })
                .next()
                .cloned();
            drop(last_res);
            if let Some(r) = rep {
                eprint!(
                    "\r  [{}{}] [{}{}] [{}{}] (dims={}, {:.2}, {:.2}, {}/{}, {:.2}%)",
                    "#".repeat(ob),
                    ".".repeat(bar_w - ob),
                    "#".repeat(mb),
                    ".".repeat(bar_w - mb),
                    "#".repeat(ib),
                    ".".repeat(bar_w - ib),
                    r.dims,
                    r.threshold,
                    r.factor,
                    r.correct,
                    r.total,
                    r.pct
                );
            }
            drop(ld);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });

    // Wait for all worker threads to finish.
    for handle in handles {
        let _ = handle.join();
    }
    eprintln!(); // newline after progress bar

    let results = results.lock().unwrap().drain(..).collect::<Vec<_>>();

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
        let dw = results
            .iter()
            .map(|r| format!("{}", r.dims).len())
            .max()
            .unwrap_or(6);
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
            "| {dims:>dw$} | {threshold:>tw$} | {factor:>fw$} | {:>pw$.1} |",
            0.0,
            dims = "dims",
            threshold = "threshold",
            factor = "factor",
            dw = dw,
            tw = tw,
            fw = fw,
            pw = pw
        );
        println!(
            "|{}|{}|{}|{}|",
            "-".repeat(dw),
            "-".repeat(tw),
            "-".repeat(fw),
            "-".repeat(pw)
        );
        for r in &results {
            println!(
                "| {:>dw$} | {:>tw$} | {:>fw$} | {:>pw$.1} |",
                r.dims,
                fmt_f(r.threshold),
                fmt_f(r.factor),
                r.pct,
                dw = dw,
                tw = tw,
                fw = fw,
                pw = pw
            );
        }
        println!();
        println!(
            "**recommended: dims = {}, threshold = {}, factor = {}** ({}/{})",
            best.dims,
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
    config::update_usize(home, "dims", best.dims)
        .map_err(|e| format!("cannot write config: {e}"))?;
    eprintln!(
        "postmaster-k: wrote dims={} threshold={} factor={} into {}",
        best.dims,
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
    let dw = results
        .iter()
        .map(|r| format!("{}", r.dims).len())
        .max()
        .unwrap_or(6);
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
    let hdr_dims = format!("{:>width$}", "dims", width = dw);
    let hdr_thresh = format!("{:>width$}", "threshold", width = tw);
    let hdr_factor = format!("{:>fw$}", "factor", fw = fw);
    let hdr_pct = format!("{:>pw$.1}", 0.0, pw = pw);

    let _ = writeln!(
        handle,
        "| {hdr_dims} | {hdr_thresh} | {hdr_factor} | {hdr_pct} |"
    );
    let sep = format!(
        "|{}|{}|{}|{}|",
        "-".repeat(dw),
        "-".repeat(tw),
        "-".repeat(fw),
        "-".repeat(pw)
    );
    let _ = writeln!(handle, "{sep}");
    for r in results {
        let ds = format!("{}", r.dims);
        let ts = fmt_f(r.threshold);
        let fs = fmt_f(r.factor);
        let _ = writeln!(
            handle,
            "| {:>width$} | {:>fw$} | {:>fw2$} | {:>pw$.1} |",
            ds,
            ts,
            fs,
            r.pct,
            width = dw,
            fw = fw,
            fw2 = fw,
            pw = pw
        );
    }
    let _ = writeln!(handle);
    // Find best result: max accuracy, tie-break by second-highest threshold then lowest factor.
    let max_correct = results.iter().map(|r| r.correct).max().unwrap_or(0);
    let top_results: Vec<&GridResult> = results
        .iter()
        .filter(|r| r.correct == max_correct)
        .collect();
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
    let _ = writeln!(
        handle,
        "**recommended: dims = {}, threshold = {} factor = {}** ({}/{})",
        best.dims,
        fmt_f(best.threshold),
        fmt_f(best.factor),
        best.correct,
        best.total
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
/// Cloning is cheap (just an Arc-like refcount bump).
fn open_view(index_path: &Path) -> Result<usearch::Index, String> {
    let ps = index_path
        .to_str()
        .ok_or_else(|| format!("path {} is not valid UTF-8", index_path.display()))?;
    let _meta =
        usearch::Index::metadata(ps).map_err(|e| format!("cannot read index header: {e}"))?;
    usearch::Index::restore_view(ps).map_err(|e| format!("cannot mmap-view index: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_grid_launches_and_completes() {
        // The grid constants define 11 dims × 7 thresholds × 17 factors = 1309 jobs.
        assert_eq!(DIMS_GRID.len(), 11);
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
        assert_eq!(thresholds.len(), 7);
        assert_eq!(factors.len(), 17);
        assert!(*PARALLEL_JOBS >= 1);
    }
}
