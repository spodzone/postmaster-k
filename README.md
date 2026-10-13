# Postmaster‑K

**A multi‑category, dynamically‑learning email classifier for `procmail` - one small Rust binary that never loses a message.**

## What is this?

Postmaster‑K is a fast, self‑contained email classifier that you invoke from a mail delivery agent (e.g. `procmail`) on every incoming message. Unlike a Bogofilter‑style spam filter, it isn't limited to *spam vs. ham*: it sorts mail into any set of categories **you** define - `work`, `invoices`, `newsletters`, `spam`, whatever - and keeps learning from new examples with no separate retraining step to babysit.

Under the hood, each message is parsed, its classifiable text is tokenized and feature‑hashed into a fixed‑size vector, and that vector is matched by a k‑nearest‑neighbour search against a disk‑backed (mmap'd) index. It's a single binary, no daemon, no model file, and it **fails open**: a crash or a low‑confidence guess never drops, bounces, or mangles mail - the message simply passes through unclassified.

*The name is inspired by a memorable scene set in Truro, Massachusetts. If you know, you know.*

## Why do I want it?

- **Fast enough for the hot path** - the index is mmap'd, so classifying one message is a few milliseconds. Safe to run once per email, synchronously, inside a `procmail` recipe.
- **Categorize, not just spam‑filter** - as many named buckets as you want, each deliverable to its own mailbox.
- **Learns as it goes** - there's no "training phase." Every `learn` call inserts straight into the live index, and a correction shows up on the very next `infer`.
- **One drop‑in binary** - no Python venvs, no sidecar server, no database daemon, no runtime dependencies.
- **Delivery‑safe by design** - fail‑open means Postmaster‑K can't be the thing that eats your mail.

## How?

### Installing:

```sh
git clone https://github.com/spodzone/postmaster-k.git
cd postmaster-k
cargo build --release
```

Drop the resulting `target/release/postmaster-k` somewhere on your `$PATH` - it's self‑contained, with no runtime dependencies.

### Using:

#### Startup

`postmaster-k init` creates the `~/.postmaster-k` directory with the config file set to defaults. The tuning knobs (`dims`, `k`, `threshold`, `threshold_factor`) live in the `config.toml` there - tweak them by hand, or let `autocalibrate` (below) pick them for you. A different state root can be selected per call with `--home DIR` or the `POSTMASTER_K_HOME` environment variable.

#### Training:

```
% postmaster-k learn spam .Junk/
% postmaster-k learn friends .friends/
```

Each source is a directory of one message per file (walked recursively), or `-` to read a single message from stdin. New categories are registered automatically, and a batch is only persisted if it parsed and embedded cleanly.

#### Inferring

To use postmaster-k with procmail, tag every message and deliver off the added header:

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

`infer` reads one message (file argument or stdin) and prints the verdict: the default `--mode=word` prints exactly one word - the winning category, or `unsure`; `--mode=header` prints only the `X-Postmaster-K-Category:` line; `--mode=full` (as above) re‑emits the message with the header inserted. It fails open - on any internal error, `word` reports `unsure`, `header` prints nothing, `full` passes the message through untouched - always exiting 0.

For dynamic reinforcement learning, use `postmaster-k infer --learn` - when the incoming document is identified as any non‑`unsure` category, it will run the `learn` cycle to add it to the vector db. (`unsure` verdicts are never learned, and a learn failure is only a stderr warning; it never affects the verdict or exit code.)

#### Other Commands

##### AutoCalibrate

Given a corpus where the subdirectory is the category name, `postmaster-k autocalibrate` will perform a grid search across its available parameters (`dims`, `threshold`, `threshold_factor`) and choose the combination that gives the best accuracy overall. It samples `max(20%, 200)` messages per category, prints the scored table, and writes the winning combination back into `config.toml` - the index is only read, so `learn` every category first.

```sh
postmaster-k autocalibrate ~/mail/sorted/
```

##### Analytics

To inspect the state of the vector database, run `postmaster-k analytics` - this will show a list of the defined categories and the strongest terms present in each.

```sh
postmaster-k analytics                  # every configured category
postmaster-k analytics invoices -n 20   # one category, top 20 terms
```

---

*MIT‑licensed. The classifier state under `~/.postmaster-k/` (index, labels, term tables) is plain data you own - back it up too!*
