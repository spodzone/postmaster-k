# postmaster-k

Multi-category, dynamically-learning email classifier for procmail. Single
Rust binary; invoked once per incoming message from `.procmailrc`; embeds
each message with the feature-hashing trick and classifies it by
k-nearest-neighbour vote against a disk-backed (mmap) vector index. Unlike
Bogofilter-style filters it isn't limited to spam/ham — classify into any
categories you like (`work`, `invoices`, `newsletters`, `spam`, …) and
correct it incrementally with no retraining step.

Design goals: never break mail delivery (fail open, always), fast enough
for the procmail hot path (~ms per message, mmap view of the index — cold
start does not scale with corpus size), and no runtime dependencies beyond
what's statically linked in.

## Build

```sh
cargo build            # → ./postmaster-k  (dev profile mirrors release: opt-level 3, lto, strip)
cargo build --release  # → target/release/postmaster-k, additionally with panic=abort
```

No installer: drop the binary somewhere on `$PATH`. Statically-linked musl
builds for Linux:

```sh
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

## First run

```sh
postmaster-k init    # scaffolds ~/.postmaster-k/config.toml
```

State lives under one root (default `~/.postmaster-k/`, override with
`--home DIR` or `$POSTMASTER_K_HOME`):

```
~/.postmaster-k/
  config.toml          # categories, dims (2048), k (15), threshold (0.4)
  index.usearch        # combined vector index, one tagged vector per learned message
  labels.db            # usearch key -> category -> source path
  terms/<category>.db  # unhashed per-category term frequencies (analytics / future IDF)
```

## Usage

**Learn** (bootstrap a category, or correct it later — same code path):

```sh
postmaster-k learn invoices ~/mail-examples/invoices/   # one message per file
postmaster-k learn invoices - < some-message.eml        # single message on stdin
```

**Classify** (the procmail hot path; reads `<file>` or stdin):

```sh
postmaster-k infer --mode=word   < msg.eml   # → one word: `invoices`, `unsure`, …
postmaster-k infer --mode=header < msg.eml   # → X-Postmaster-K-Category: invoices
postmaster-k infer --mode=full   < msg.eml   # → message with the headers inserted at top
```

**Inspect what a category actually separated on:**

```sh
postmaster-k analytics invoices -n 20
postmaster-k analytics --all
```

## .procmailrc integration

Tag and file by category:

```procmail
:0fw
| postmaster-k infer --mode=full

:0:
* ^X-Postmaster-K-Category: invoices
Mail/invoices/

:0:
* ^X-Postmaster-K-Category: unsure
Mail/needs-sorting/
```

Or drive recipes directly, without a header round-trip:

```procmail
CATEGORY=`postmaster-k infer --mode=word`

:0:
* CATEGORY ?? ^invoices$
Mail/invoices/
```

Bootstrap categories from an existing mbox archive (`formail -ds` splits it
and pipes each message to stdin):

```sh
formail -ds postmaster-k learn invoices -- < ~/Mail/archive/invoices.mbox
```

## Failure semantics

- A message that can't be parsed/classified is **never** an error: `word`
  prints `unsure`, `header` prints nothing, `full` passes the original bytes
  through untouched — all with exit 0.
- Non-zero exit (2) is reserved for genuine misuse: bad arguments, missing
  index/config, unreadable input.
- `panic = "abort"` in release, so nothing panics to stdout mid-write; all
  output paths are explicit, error-ignored writes.
- A crash mid-`learn` can't corrupt state: the index is saved via
  tmp-file + rename.

## Configuration (`config.toml`)

| Key | Default | Meaning |
|---|---|---|
| `categories` | `[]` | Auto-maintained by `learn`; not worth hand-editing |
| `dims` | `2048` | Feature-hash dimensionality (SPEC §7) |
| `k` | `15` | Neighbours consulted per inference |
| `threshold` | `0.4` | Vote share a category must reach, else `unsure` |

The category name `unsure` is reserved. Category names are `[a-z0-9_-]`.

## Tuning notes

- **False `unsure`s** → lower `threshold`, or learn more examples.
- **Wrong category confidently** → `learn <right-category> -` with the
  corrected message; the fix takes effect on the very next message.
- **Check vocabulary separation** with `analytics --all`: a category whose
  top terms are footer/legal boilerplate means its examples were noisy —
  the intended fix is better examples, or tightening the text extraction.
