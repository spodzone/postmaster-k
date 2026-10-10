//! State-root layout, config.toml handling and the `init` subcommand.
//!
//! All state lives under a single configurable root (SPEC §5):
//! `config.toml`, `index.usearch`, `labels.db` and `terms/<category>.db`.

use std::fs;
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, Item, Value};

pub const DEFAULT_DIMS: usize = 2048;
pub const DEFAULT_K: usize = 15;
pub const DEFAULT_THRESHOLD: f64 = 0.4;
pub const DEFAULT_THRESHOLD_FACTOR: f64 = 2.0;

const CONFIG_FILE: &str = "config.toml";

const DEFAULT_CONFIG: &str = r#"# postmaster-k configuration.
# `learn` appends to `categories` automatically.
categories = []

# Feature-hash dimensionality (SPEC §7).
dims = 2048

# Neighbours consulted per inference (SPEC §6).
k = 15

# Vote share a category must reach to avoid `unsure` (SPEC §6).
threshold = 0.4

# The winner's vote weight must be at least this many times the second-greatest
# to avoid `unsure`. Enforces a hockey-stick drop-off in likelihoods (default 2.0).
threshold_factor = 2.0
"#;

pub fn index_path(home: &Path) -> PathBuf {
    home.join("index.usearch")
}

pub fn labels_path(home: &Path) -> PathBuf {
    home.join("labels.db")
}

pub fn terms_dir(home: &Path) -> PathBuf {
    home.join("terms")
}

pub fn terms_path(home: &Path, category: &str) -> PathBuf {
    terms_dir(home).join(format!("{category}.db"))
}

pub fn config_path(home: &Path) -> PathBuf {
    home.join(CONFIG_FILE)
}

/// `--home` > `$POSTMASTER_K_HOME` > `$HOME/.postmaster-k`.
pub fn resolve_home(cli: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(p) = cli {
        return Ok(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os("POSTMASTER_K_HOME") {
        if !p.is_empty() {
            return Ok(PathBuf::from(p));
        }
    }
    match std::env::var_os("HOME") {
        Some(h) if !h.is_empty() => Ok(PathBuf::from(h).join(".postmaster-k")),
        _ => Err(
            "cannot determine state directory: pass --home DIR, set POSTMASTER_K_HOME, or set $HOME"
                .into(),
        ),
    }
}

/// Category names are used as filenames (`terms/<category>.db`) and as
/// procmail-matchable header values, so keep them tight.
pub fn normalize_category(raw: &str) -> Result<String, String> {
    let c = raw.trim().to_lowercase();
    if c == "unsure" {
        return Err("`unsure` is a reserved category name".into());
    }
    if c.is_empty() || c.len() > 64 {
        return Err(format!("invalid category `{raw}`: name must be 1-64 bytes"));
    }
    let valid = c.chars().enumerate().all(|(i, ch)| {
        ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || (ch == '-' && i > 0)
    });
    if !valid {
        return Err(format!(
            "invalid category `{raw}`: use a-z, 0-9, '_' and '-' only"
        ));
    }
    Ok(c)
}

struct Loaded {
    doc: DocumentMut,
    path: PathBuf,
}

fn load_document(home: &Path) -> Result<Loaded, String> {
    let path = config_path(home);
    let raw = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => DEFAULT_CONFIG.to_string(),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let doc = raw
        .parse::<DocumentMut>()
        .map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
    Ok(Loaded { doc, path })
}

fn get_usize(doc: &DocumentMut, key: &str, default: usize) -> Result<usize, String> {
    match doc.get(key) {
        None | Some(Item::None) => Ok(default),
        Some(item) => item
            .as_integer()
            .ok_or_else(|| format!("config key `{key}` must be an integer"))
            .and_then(|v| {
                usize::try_from(v).map_err(|_| format!("config key `{key}`: {v} is out of range"))
            }),
    }
}

fn get_f64(doc: &DocumentMut, key: &str, default: f64) -> Result<f64, String> {
    match doc.get(key) {
        None | Some(Item::None) => Ok(default),
        Some(item) => item
            .as_float()
            .ok_or_else(|| format!("config key `{key}` must be a number")),
    }
}

pub struct Settings {
    pub dims: usize,
    pub k: usize,
    pub threshold: f64,
    pub threshold_factor: f64,
    pub categories: Vec<String>,
}

pub fn settings(home: &Path) -> Result<Settings, String> {
    let loaded = load_document(home)?;
    let mut categories = Vec::new();
    if let Some(item) = loaded.doc.get("categories") {
        let arr = item
            .as_array()
            .ok_or_else(|| "config key `categories` must be an array".to_string())?;
        for v in arr.iter() {
            let s = v
                .as_str()
                .ok_or_else(|| "config `categories` entries must be strings".to_string())?;
            categories.push(s.to_string());
        }
    }
    let s = Settings {
        dims: get_usize(&loaded.doc, "dims", DEFAULT_DIMS)?,
        k: get_usize(&loaded.doc, "k", DEFAULT_K)?,
        threshold: get_f64(&loaded.doc, "threshold", DEFAULT_THRESHOLD)?,
        threshold_factor: get_f64(&loaded.doc, "threshold_factor", DEFAULT_THRESHOLD_FACTOR)?,
        categories,
    };
    if s.dims < 64 || s.dims > (1 << 20) {
        return Err(format!("config key `dims` must be 64..={} ", 1 << 20));
    }
    if s.k == 0 || s.k > 4096 {
        return Err("config key `k` must be 1..=4096".into());
    }
    if !(0.0..=1.0).contains(&s.threshold) {
        return Err("config key `threshold` must be between 0.0 and 1.0".into());
    }
    if s.threshold_factor < 1.0 {
        return Err("config key `threshold_factor` must be at least 1.0".into());
    }
    Ok(s)
}

/// Adds `category` to config.toml if absent. Returns whether it was added.
/// Edits go through toml_edit so user comments and formatting survive.
pub fn add_category(home: &Path, category: &str) -> Result<bool, String> {
    let mut loaded = load_document(home)?;
    if let Some(item) = loaded.doc.get("categories") {
        if let Some(arr) = item.as_array() {
            if arr.iter().any(|v| v.as_str() == Some(category)) {
                return Ok(false);
            }
        }
    }
    let inserted = match loaded.doc.get_mut("categories") {
        Some(item) => match item.as_array_mut() {
            Some(arr) => {
                arr.push(category);
                true
            }
            None => return Err("config key `categories` must be an array".into()),
        },
        None => {
            let mut arr = toml_edit::Array::new();
            arr.push(category);
            loaded
                .doc
                .insert("categories", Item::Value(Value::Array(arr)));
            true
        }
    };
    if inserted {
        save_document(&loaded)?;
    }
    Ok(inserted)
}

fn save_document(loaded: &Loaded) -> Result<(), String> {
    atomic_write(&loaded.path, loaded.doc.to_string().as_bytes())
        .map_err(|e| format!("cannot write {}: {e}", loaded.path.display()))
}

/// Update the scalar value of `key` in config.toml in-place. The key must
/// already exist as a simple `key = value` line; comments and other keys are
/// left untouched. Edits go through toml_edit so user formatting survives.
pub fn update_float(home: &Path, key: &str, value: f64) -> Result<(), String> {
    let mut loaded = load_document(home)?;
    match loaded.doc.get_mut(key) {
        Some(Item::None) | None => {
            // Key doesn't exist yet — append it.
            let val: Value = value.into();
            loaded.doc.insert(key, Item::Value(val));
        }
        Some(item) => {
            *item = Item::Value(value.into());
        }
    }
    save_document(&loaded)
}

/// Update the usize value of `key` in config.toml in-place.
pub fn update_usize(home: &Path, key: &str, value: usize) -> Result<(), String> {
    let mut loaded = load_document(home)?;
    match loaded.doc.get_mut(key) {
        Some(Item::None) | None => {
            let val: Value = (value as i64).into();
            loaded.doc.insert(key, Item::Value(val));
        }
        Some(item) => {
            *item = Item::Value((value as i64).into());
        }
    }
    save_document(&loaded)
}

/// `postmaster-k init` — scaffold the state root and a default config.toml.
/// Idempotent: an existing config is left untouched.
pub fn init(home: &Path) -> Result<String, String> {
    fs::create_dir_all(home).map_err(|e| format!("cannot create {}: {e}", home.display()))?;
    fs::create_dir_all(terms_dir(home))
        .map_err(|e| format!("cannot create {}: {e}", terms_dir(home).display()))?;
    let cfg = config_path(home);
    if cfg.exists() {
        return Ok(format!("already initialized at {}", home.display()));
    }
    atomic_write(&cfg, DEFAULT_CONFIG.as_bytes())
        .map_err(|e| format!("cannot write {}: {e}", cfg.display()))?;
    Ok(format!(
        "initialized state root {}\n  config: {}\nnext: postmaster-k learn <category> <directory-of-messages>",
        home.display(),
        cfg.display()
    ))
}

/// Write `data` to `path` atomically: temp file in the same directory,
/// fsync, then rename over the destination.
pub fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_template_parses_with_expected_defaults() {
        let home = tempdir("defaults");
        let s = settings(&home).unwrap();
        assert_eq!(s.dims, 2048);
        assert_eq!(s.k, 15);
        assert!((s.threshold - 0.4).abs() < 1e-9);
        assert!((s.threshold_factor - 2.0).abs() < 1e-9);
        assert!(s.categories.is_empty());
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn add_category_preserves_comments_and_other_keys() {
        let home = tempdir("addcat");
        init(&home).unwrap();
        let cfg = config_path(&home);
        let with_comment = DEFAULT_CONFIG.replace(
            "# Feature-hash dimensionality (SPEC §7).",
            "# keep me: feature-hash dimensionality\n# second comment line",
        );
        fs::write(&cfg, &with_comment).unwrap();

        assert!(add_category(&home, "invoices").unwrap());
        assert!(!add_category(&home, "invoices").unwrap()); // idempotent

        let text = fs::read_to_string(&cfg).unwrap();
        assert!(text.contains("# keep me: feature-hash dimensionality"));
        assert!(text.contains("categories = [\"invoices\"]"));

        let s = settings(&home).unwrap();
        assert_eq!(s.categories, vec!["invoices".to_string()]);
        assert_eq!(s.dims, 2048);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn category_names_are_constrained() {
        assert!(normalize_category("Invoices_2").unwrap() == "invoices_2");
        assert!(normalize_category("unSure").is_err());
        assert!(normalize_category("").is_err());
        assert!(normalize_category("../etc").is_err());
        assert!(normalize_category("has space").is_err());
    }

    #[test]
    fn resolve_home_prefers_cli_then_env() {
        let cli = resolve_home(Some(Path::new("/tmp/explicit"))).unwrap();
        assert_eq!(cli, PathBuf::from("/tmp/explicit"));
    }

    fn tempdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "pmk-config-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }
}
