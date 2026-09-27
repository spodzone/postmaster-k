# Postmaster-K

## What is this?

Postmaster‑K is a **multi‑dimensional, fast, self‑contained email classifier** designed to be called from a mail delivery agent (e.g. `procmail`).  It goes beyond the classic *spam‑vs‑ham* binary decision and lets you define arbitrary categories such as `work`, `invoices`, `newsletters`, `spam`, etc.  Each incoming message is parsed, tokenised, hashed into a fixed‑size vector and classified with a k‑nearest‑neighbour search against a disk‑backed index.  The classifier is tiny, single‑binary, and **fails open** – a crash or mis‑classification never loses the message.

The name comes from a memorable scene in **Truro, Massachusetts** (if you know, you know). 

## Why use it?

* **Speed** – mmap‑backed index makes inference a few milliseconds per message, safe for a per‑email `procmail` pipe.
* **Flexibility** – unlimited user‑defined categories, no separate training step; new examples are learned instantly.
* **Reliability** – single static binary, no external daemon, and a fail‑open design guarantees delivery.

## Modes of operation

### Learning

- **`init`** – Scaffold a fresh state directory (`~/.postmaster-k/`).
  ```sh
  postmaster-k init
  ```
- **`learn <category> <dir|->`** – Insert labelled examples from a directory of messages or from **stdin** (single message).
  ```sh
  postmaster-k learn invoices ~/mail/examples/
  # or from stdin via formail
  formail -ds postmaster-k learn invoices -- < old.mbox
  ```
- **`analytics <category> [-n N] [--all]`** – Show top‑N terms for a category (human‑readable diagnostics).
  ```sh
  postmaster-k analytics invoices -n 20
  ```

### Calibrating

- **`autocalibrate <maildir>`** – Grid‑search the confidence parameters against real data and write the best pair into `config.toml`.  The argument is a Maildir‑style root where **each subdirectory is a category** and the files within it are messages.

  A random sample of `max(20%, 200)` messages per category is classified, then every combination of `threshold` (0.0 → 0.6, step 0.1) and `threshold_factor` (1.0 → 5.0, step 0.25) is scored.  A markdown table of `(threshold, factor, %correct)` is printed, and the **lowest** threshold/factor pair that reaches maximum accuracy is written back to `config.toml`.

  ```sh
  postmaster-k autocalibrate ~/mail/sorted/
  ```

  Requires the categories to have been `learn`-ed first — autocalibrate scores against the existing index, it does not modify it.

### Inference

- **`infer [--mode=word|header|full] [file]`** – Classify a single message.
  - `--mode=word` (default) – outputs a single word, the winning category (or `unsure`).
  - `--mode=header` – outputs a header line, e.g. `X-Postmaster‑K-Category: invoices`.
  - `--mode=full` – writes the full message with the added header.

  ```sh
  postmaster-k infer --mode=word < mail.msg            # single word output
  postmaster-k infer --mode=header < mail.msg          # header line output
  postmaster-k infer --mode=full < mail.msg            # full message with header
  ```

### Example `.procmailrc` integration
```procmail
# Tag each incoming mail with a Postmaster‑K header and file by category.
:0fw
| postmaster-k infer --mode=full

# Deliver based on the added header.
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

Or the lightweight single‑word mode:
```procmail
CATEGORY=`formail -c | postmaster-k infer --mode=word`

:0:
* CATEGORY ?? ^invoices$
Mail/invoices/
```

## Configuration

Settings live in `config.toml` at the state root (`~/.postmaster-k/` by default).  `learn` appends to `categories` automatically; the other keys can be edited by hand or tuned with `autocalibrate`.

| Key | Default | Meaning |
| --- | --- | --- |
| `categories` | `[]` | Known categories; maintained automatically by `learn`. |
| `dims` | `2048` | Feature‑hash dimensionality. |
| `k` | `15` | Neighbours consulted per inference. |
| `threshold` | `0.4` | Vote share the winning category must reach to avoid `unsure`. |
| `threshold_factor` | `2.0` | Hockey‑stick criterion: the winner's vote weight must be at least this many times the runner-up. |

### `threshold`

The winner's share of the total vote weight (0.0–1.0).  A result below this is classed `unsure` rather than delivered to a category.  Raise it to send more ambiguous mail to manual sorting; lower it if too much mail ends up in `unsure`.

### `threshold_factor`

Share alone can be misleading: a 0.6 share produced by a 0.6/0.4 split between two categories is a coin‑flip, while a 0.6 share from a 0.6/0.05 split is a clear winner.  `threshold_factor` enforces the latter shape — the categories are effectively ordered by likelihood, and the greatest must be at least `threshold_factor` times the second‑greatest (a hockey‑stick / exponential drop‑off), otherwise the result is `unsure`.

* `2.0` (default) – the winner must have double the runner‑up's weight.
* Close to `1.0` – trust the share alone; `threshold_factor = 1.0` disables the criterion.
* High values (3–5) – only near‑unanimous votes classify; everything else falls to `unsure`.

Both parameters interact, so tune them together — `autocalibrate` does exactly that by scoring a sample of your real mail across the grid and picking the least conservative pair that keeps accuracy at its maximum.

---

*Documentation, issue tracking, and contribution guidelines are in the repository.*
