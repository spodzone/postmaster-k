# Postmaster‑K

**A multi‑category, dynamically‑learning email classifier for `procmail` — one small Rust binary that never loses a message.**

## What is this?

Postmaster‑K is a fast, self‑contained email classifier that you invoke from a mail delivery agent (e.g. `procmail`) on every incoming message. Unlike a Bogofilter‑style spam filter, it isn't limited to *spam vs. ham*: it sorts mail into any set of categories **you** define — `work`, `invoices`, `newsletters`, `spam`, whatever — and keeps learning from new examples with no separate retraining step to babysit.

Under the hood, each message is parsed, its classifiable text is tokenized and feature‑hashed into a fixed‑size vector, and that vector is matched by a k‑nearest‑neighbour search against a disk‑backed (mmap'd) index. It's a single binary, no daemon, no model file, and it **fails open**: a crash or a low‑confidence guess never drops, bounces, or mangles mail — the message simply passes through unclassified.

*The name is inspired by a memorable scene set in Truro, Massachusetts. If you know, you know.*

## Why do I want it?

- **Fast enough for the hot path** — the index is mmap'd, so classifying one message is a few milliseconds. Safe to run once per email, synchronously, inside a `procmail` recipe.
- **Categorize, not just spam‑filter** — as many named buckets as you want, each deliverable to its own mailbox.
- **Learns as it goes** — there's no "training phase." Every `learn` call inserts straight into the live index, and a correction shows up on the very next `infer`.
- **One drop‑in binary** — no Python venvs, no sidecar server, no database daemon, no runtime dependencies.
- **Delivery‑safe by design** — fail‑open means Postmaster‑K can't be the thing that eats your mail.

## How?

### Building & installing

You need a working Rust toolchain (`cargo`).

```sh
git clone <url-to-repo> postmaster-k   # your host + path to postmaster-k
cd postmaster-k
cargo build --release
```

The release binary lands at `target/release/postmaster-k` — a small, LTO‑optimized, stripped, self‑contained executable. There's no installer: put it somewhere on your `$PATH`, then scaffold the state directory.

```sh
install -m 755 target/release/postmaster-k /usr/local/bin/postmaster-k
# — or:  ln -s "$PWD/target/release/postmaster-k" ~/.local/bin/postmaster-k

postmaster-k init   # creates ~/.postmaster-k/ with a default config.toml
```

> Tip: the dev profile is configured to mirror release (`opt-level=3`, LTO, strip), so a plain `cargo build` already yields a fully optimized binary if you just want to poke around quickly.

### Configuring it

All state lives under a single root — `~/.postmaster-k/` by default. Override it with the `POSTMASTER_K_HOME` environment variable, or with a per‑call `--home DIR` flag on any subcommand.

The tunable knobs are in `config.toml` at that root:

| Key | Default | Range | Meaning |
|---|---|---|---|
| `categories` | `[]` | — | Known categories. Maintained automatically by `learn`; you rarely edit it by hand. |
| `dims` | `2048` | `64`–`1048576` | Feature‑hash vector dimensionality. Higher = finer vocabulary resolution, bigger index. |
| `k` | `15` | `1`–`4096` | Nearest neighbours consulted per inference. |
| `threshold` | `0.4` | `0.0`–`1.0` | Minimum vote **share** a category must reach to win (else the result is `unsure`). |
| `threshold_factor` | `2.0` | `> 1.0` | The winner's vote weight must be at least this many × the runner‑up's (enforces a clear drop‑off). |

**`threshold`** — the winning category's share of the total vote weight, `0.0`–`1.0`. Anything below it is classed `unsure` rather than delivered. Raise it to push more ambiguous mail to manual sorting; lower it if too much ends up in `unsure`.

**`threshold_factor`** — share alone can mislead: a `0.6` share that is really a `0.6 / 0.4` split is a coin‑flip, whereas `0.6` from a `0.6 / 0.05` split is a clear call. This criterion enforces the second shape — the top category's weight must be at least `threshold_factor` times the next, otherwise the result is `unsure`. Set it close to `1.0` to trust share alone, or high (3–5) to only classify near‑unanimous matches.

Because the two interact, tune them together — or just run `autocalibrate` (below), which searches both against a sample of your real mail and picks the least conservative pair that keeps accuracy at its maximum.

### Running it

**`init`** — scaffold the state root and a default `config.toml`. Idempotent; an existing config is left untouched.

```sh
postmaster-k init
```

**`learn <category> <dir | ->`** — insert labelled examples into a category. `category` names a bucket (`a–z`, `0–9`, `_`, `-`; 1–64 chars; `unsure` is reserved). `dir` is a folder of one message per file (Maildir‑style), or `-` to read a single message from stdin. New categories are registered automatically, and a batch is only persisted if it parsed and embedded cleanly — a failed `learn` leaves the previous state intact.

```sh
postmaster-k learn invoices ~/mail/examples/invoices/

# or feed one message on stdin (e.g. straight from formail splitting an mbox):
formail -ds postmaster-k learn invoices -- < ~/Mail/archive/invoices.mbox
```

**`autocalibrate <maildir>`** — grid‑search `threshold` and `threshold_factor` against your real data and write the best pair into `config.toml`. The argument is a Maildir‑style root where **each subdirectory is a category** and its files are messages. It samples `max(20%, 200)` messages per category, scores every combination of `threshold` (0.0 → 0.6, step 0.1) and `threshold_factor` (1.0 → 5.0, step 0.25), prints a `(threshold, factor, %correct)` table, and writes back the **least conservative** pair that reaches the top accuracy. It reads the existing index — `learn` every category first; it does not modify the index itself.

```sh
postmaster-k autocalibrate ~/mail/sorted/
```

**`infer` + `.procmailrc` integration** — `infer` is the hot path: classify one message and emit a result. It always opens the index in read‑only mmap mode, and on any internal error it **fails open** (`word` → `unsure`, `header` → nothing, `full` → the original message, all exiting 0).

- `--mode=word` *(default)* — prints exactly one word: the winning category, or `unsure`.
- `--mode=header` — prints a single header line, e.g. `X-Postmaster-K-Category: invoices`.
- `--mode=full` — emits the complete, otherwise‑untouched message with the `X-Postmaster-K-Category:` header inserted.
- `--learn` — after a non‑`unsure` verdict, feeds the message back through `learn` under the category just chosen, so the classifier reinforces itself as mail flows (an `unsure` verdict is never learned; a learn failure is only a stderr warning and never affects output or the exit code). Classification itself still uses the read‑only mmap path — the reinforcement step opens the index mutably only after the verdict has been emitted.

Pipe a message through it to try it by hand:

```sh
postmaster-k infer --mode=word < mail.msg
```

**Tag then file** (header mode) — tag every message, then deliver by the added header:

```procmail
# 1) Classify: adds X-Postmaster-K-Category and passes the message on.
:0fw
| postmaster-k infer --mode=full

# 2) Deliver based on the added header.
:0:
* ^X-Postmaster-K-Category: invoices
Mail/invoices/

:0:
* ^X-Postmaster-K-Category: newsletters
Mail/newsletters/

:0:
* ^X-Postmaster-K-Category: unsure
Mail/needs-sorting/
```

**Lightweight, word mode** — drive the recipe directly off the single‑word verdict:

```procmail
CATEGORY=`formail -c | postmaster-k infer --mode=word`

:0:
* CATEGORY ?? ^invoices$
Mail/invoices/
```

**`analytics` (diagnostic)** — peek at the top‑N terms a category is actually keyed on, pulled from the un‑hashed term tables (not the vector index). Handy for sanity‑checking that a category separates on sensible vocabulary rather than absorbed boilerplate.

```sh
postmaster-k analytics invoices -n 20
postmaster-k analytics --all
```

---

*MIT‑licensed. The classifier state under `~/.postmaster-k/` (index, labels, term tables) is plain data you own — back it up like any other mailbox.*
