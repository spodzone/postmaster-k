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

---

*Documentation, issue tracking, and contribution guidelines are in the repository.*
