# Postmaster-K — Project Specification

**Binary name:** `postmaster-k`
**Language:** Rust
**License / Distribution:** single static binary, no daemon, no runtime dependencies beyond what's statically linked in

## 1. What this is

Postmaster-K is a multi-category, dynamically-learning email classifier
designed to be invoked directly from `.procmailrc` (or an equivalent MDA hook)
on every incoming message. Unlike Bogofilter-style tools, it is not
restricted to a spam/ham binary decision — it classifies mail into an
arbitrary, user-defined set of categories (`work`, `invoices`, `newsletters`,
`spam`, whatever you like), and learns new examples incrementally without a
retraining step.

Classification is done by embedding the message text into a vector and
running k-nearest-neighbour search against a disk-backed vector index
(usearch), rather than a Bayesian/Markov token-frequency model. This trades
some of the adversarial robustness of Bayesian spam filters for much better
handling of semantic/topical categories, which is the point.

## 2. Hard requirements (non-negotiable)

- **Must be fast enough to run once per incoming email, synchronously, in a
  procmail pipe, without perceptible delay.** This is the overriding design
  constraint and it shapes every other decision below — including, as of
  this revision, ruling out any local transformer/ONNX embedding model:
  on CPU-only hardware that cost is unacceptable for a per-message hot path.
  See §7.
- **Must never break mail delivery.** A crash, panic, or classification
  failure must never cause a message to be lost, mangled, or bounced. Fail
  open (pass the message through unmodified) rather than fail closed.
- **Single self-contained binary.** No Python venvs, no sidecar server
  process, no separately-running database daemon. Compile with LTO,
  `panic = "abort"`, and stripped symbols; static-link against musl where
  practical for portability across the Linux/macOS boxes this will run on.
- **Disk-backed, not fully-loaded-into-RAM.** The vector index must use
  usearch's `view()` mmap mode for inference, so cold-start cost does not
  scale linearly with corpus size as the training set grows over years of
  use.

## 3. Non-goals

- Not a spam-specific tool. Spam/ham can be *one* of the user's categories,
  but this is not trying to out-compete Bayesian filters on adversarial spam
  detection specifically.
- Not a server, not a daemon, not something with a network-facing API.
- Not using a neural embedding model. Decided (see §7): CPU-only inference
  speed rules out ONNX/transformer embeddings entirely, not just as a
  "risk to benchmark" — this is now a closed question, not an open one.

## 4. Core dependencies

| Concern | Crate | Why |
|---|---|---|
| Vector index | `usearch` | Official Rust binding, `view()` mmap mode gives near-instant cold start regardless of index size |
| Email parsing | `mail-parser` (stalwartlabs) | Zero-copy, fuzzed, RFC 5322 + full MIME compliant, battle-tested in a real production MTA (Stalwart) |
| CLI | `clap` (derive API) | Standard, low-friction subcommand + flag parsing |
| HTML-to-text | a minimal tag-stripping routine (hand-rolled or a tiny crate) | text/html parts need de-tagging before tokenization — see §7.1 |
| Embedding | hand-rolled feature hashing, no external model — see §7 | |

## 5. Data layout

All state lives under a single configurable root, default `~/.postmaster-k/`:

```
~/.postmaster-k/
  config.toml          # categories, hashing dimensionality, k, threshold, etc.
  index.usearch        # single combined vector index, every vector tagged
                        # with a category label via usearch's key→metadata
                        # side-table (see §6)
  labels.db            # flat mapping of usearch integer key -> category
                        # string + optional source path, for introspection
                        # and for pruning/relearning a bad example later
  terms/<category>.db   # PER-CATEGORY, UNHASHED term-frequency table.
                        # Not used for classification (that's the hashed
                        # vector's job) — exists purely to (a) drive the
                        # `analytics` subcommand with human-readable words,
                        # and (b) optionally weight the hashing step by
                        # inverse document frequency later. See §7.2.
```

A **single combined index** (rather than one index per category) is used so
that k-NN voting across categories is a single query, not N queries. Each
inserted vector's usearch key indexes into `labels.db` to recover its
category at classification time.

## 6. Classification algorithm

1. Parse the message with `mail-parser`.
2. Extract only the classifiable text — see §7.1 for exactly what that
   means and what gets thrown away.
3. Tokenize and embed via feature hashing (see §7).
4. Query the combined index for the `k` nearest neighbours (`k` configurable,
   default 15).
5. Each neighbour votes for its category, weighted by inverse distance.
6. The category with the highest weighted vote wins, **provided** its vote
   share clears a configurable confidence threshold (default 0.4 of total
   weight). Below threshold, the result is the reserved category `unsure`.
7. No separate "training phase" in the machine-learning sense — every
   `learn` call simply inserts new vectors into the live index. Corrections
   take effect immediately on the next `infer` call. There *is*, however, a
   required **bootstrap phase**: every category needs an initial batch of
   labelled examples before it can be voted for at all — see §8's `learn`
   spec.

## 7. Embedding: feature hashing (decided, not a placeholder)

The originally-sketched local-ONNX-model idea is dropped. On CPU-only
hardware, per-message transformer inference cannot meet the "quick af,
runs synchronously in procmail" requirement — that's not a close call, so
there's no point hedging on it.

Instead: a classic, dependency-free **hashing trick** (à la Vowpal Wabbit),
computed entirely with string hashing and arithmetic, no model file, no
runtime, no measurable startup cost:

1. Tokenize the cleaned text (§7.1) into lowercased word tokens, stripping
   punctuation, with a small built-in stopword list removed (`the`, `and`,
   `of`, etc. — the usual short high-frequency function words that add
   noise to both classification and, later, the analytics output).
2. For each token, compute a hash and fold it into a fixed-dimension vector
   (default 2048 dims — tunable in `config.toml`) using signed hashing
   (Vowpal-Wabbit style: the hash's sign bit determines whether the token's
   count adds or subtracts from its bucket, which reduces systematic bias
   from hash collisions compared to unsigned accumulation).
3. Optionally scale each bucket by that token's inverse document frequency,
   sourced from the unhashed per-category term tables in §7.2, if that
   turns out to meaningfully improve separation once there's real data to
   test against — flagged as tunable, default off for v1 simplicity.
4. L2-normalize the resulting vector before inserting/querying, so cosine
   and dot-product metrics behave sanely regardless of message length.

This is essentially free computationally — string hashing and a fixed-size
float array, no model weights, no matrix multiplies beyond the final
normalization. It directly solves the CPU-only latency problem, at the cost
of not capturing deeper semantic similarity the way a transformer embedding
would. Given the category-separation task at hand (topical buckets, not
fine-grained semantic search), this is a perfectly reasonable trade.

### 7.1 What text actually gets read

The classifier must not be fooled by, or wade through, MIME/protocol noise.
**Included:**

- The `Subject` header (decoded, RFC 2047 handled by `mail-parser`).
- The body of every `text/plain` MIME part.
- The body of every `text/html` MIME part, **after** stripping HTML tags,
  script/style contents, and collapsing entities — html is only used if no
  `text/plain` alternative exists for a given part of the message, per
  usual `multipart/alternative` semantics.

**Explicitly excluded, never tokenized, never embedded:**

- `Message-ID`, `Date`, `Received`, `Return-Path`, `MIME-Version`,
  `Content-Type`, `Content-Transfer-Encoding`, `Content-Disposition`,
  boundary strings, and any other `X-*` header noise — none of this carries
  topical signal and all of it is exactly the kind of thing that would
  otherwise pollute the token stream with garbage like MIME boundary
  markers or base64 fragments if a message were mis-parsed.
- Attachment bodies of any non-text MIME type (binaries are never
  tokenized; v2 might consider attachment *filenames* as a weak signal, see
  §11).
- Any part of the message when the whole thing fails to parse as valid
  MIME — see §2's fail-open rule instead.

### 7.2 Per-category term table

Maintained purely as a side effect of `learn`, independent of the hashed
vectors: a simple `token -> count` map per category, persisted to
`terms/<category>.db`. This is what makes the `analytics` subcommand
possible with actual readable words instead of opaque hash buckets, and is
the natural place to compute document frequencies if IDF weighting (§7
step 3) gets turned on later.

## 8. CLI specification

### `postmaster-k learn <category-name> <directory|-->`

The required bootstrap phase, and also the ongoing incremental-correction
mechanism — same command, same code path, whether you're seeding a brand
new category with a hundred examples or feeding it one corrected message.

```
postmaster-k learn invoices ~/mail-examples/invoices/
```

- Reads every file in `<directory>` as one message per file
  (Maildir-style — not mbox, to keep parsing trivial), extracts
  classifiable text (§7.1), hashes it (§7), and inserts into the index
  tagged with `<category-name>`. Also updates `terms/invoices.db`.
- Creates the category in `config.toml` if it doesn't already exist.
- **`-` instead of a directory reads a single message from stdin.** This is
  the integration point for `formail`-based splitting of an existing mbox:

  ```
  formail -ds postmaster-k learn invoices -- < old-invoices.mbox
  ```

  `formail -ds` splits a multi-message mbox and invokes the given command
  once per message, piping each individual message to it on stdin — exactly
  matching `learn`'s single-message stdin mode. This is the documented,
  intended way to bulk-bootstrap a category from an existing mail archive
  without writing one-file-per-message export scripts by hand.
- This is the *only* subcommand that opens the index in mutable (`load()`)
  mode rather than `view()` mode. It is expected to be run interactively or
  in a batch/bootstrap context, not from the procmail hot path, so its
  startup cost is not under the same latency budget as `infer`.

### `postmaster-k infer [--mode=word|header|full] [file]`

Reads a single message from `<file>` if given, else from stdin. Always the
hot path from `.procmailrc` — classification must open the index in `view()`
(mmap) mode only, never `load()`.

- `--mode=word` *(default)* — prints exactly one word: the winning category
  name (or `unsure`). Intended for `:0` recipe conditionals.
- `--mode=header` — prints a single formatted header line, e.g.
  `X-Postmaster-K-Category: invoices`, suitable for feeding to `formail -A`.
- `--mode=full` — reads the entire message, inserts the `X-Postmaster-K-*`
  header(s) at the top of the header block, and writes the **complete,
  otherwise-untouched** message back out to stdout. This is the mode used
  when Postmaster-K sits directly in a procmail pipe as the thing that
  produces the message procmail actually delivers.
- `--learn` — after a **non-`unsure`** verdict, feeds the message back
  through the `learn` path under the category just chosen, so the classifier
  reinforces itself as mail flows. The reinforcement happens strictly *after*
  the verdict is emitted (the fresh vector can never influence its own
  classification), `unsure` verdicts are never learned, and a failed learn is
  only a stderr warning — it never alters the classification output or the
  exit code. Only this step opens the index in mutable `load()` mode.

On any internal error (parse failure, corrupt index, embedding failure):
`--mode=word` prints `unsure` and exits 0; `--mode=header` prints nothing and
exits 0; `--mode=full` prints the **original input unmodified** and exits 0.
Non-zero exit is reserved for genuine misuse (bad arguments, missing index),
never for "couldn't classify this particular email" — see §2's fail-open
requirement.

### `postmaster-k analytics <category-name> [-n 20] [--all]`

Displays the top-N terms recorded for a category, ranked by frequency (or,
if IDF weighting is later enabled, by TF-IDF score) — sourced from the
unhashed per-category term table in §7.2, *not* from the hashed vector
index, since hash buckets aren't reversible into words.

```
$ postmaster-k analytics invoices -n 20
 1. invoice          (412)
 2. payment          (388)
 3. due               (301)
 4. amount           (287)
 5. account          (255)
 ...
```

- `--all` runs the report across every configured category in one pass
  (useful for a quick sanity check that categories are actually separating
  on sensible vocabulary, and for spotting a category that's accidentally
  absorbed a bunch of boilerplate — e.g. footer/legal text — as its "top
  concepts" instead of real content, which would be a sign the text
  extraction or stopword list needs tightening).
- This subcommand never touches `index.usearch` and never needs mmap-view
  speed guarantees — it's a diagnostic tool, run interactively, same
  latency tier as `learn`.

## 9. Example `.procmailrc` integration

```procmail
# Tag every incoming message with a category header, then file by category.
:0fw
| postmaster-k infer --mode=full

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

Or, lighter-weight, using the single-word mode to drive the recipe directly
without a header round-trip:

```procmail
CATEGORY=`formail -c | postmaster-k infer --mode=word`

:0:
* CATEGORY ?? ^invoices$
Mail/invoices/
```

(exact procmail quoting/escaping TBD at implementation time — sketch only)

Bootstrapping categories from an existing archive, per §8:

```sh
formail -ds postmaster-k learn invoices -- < ~/Mail/archive/invoices.mbox
formail -ds postmaster-k learn newsletters -- < ~/Mail/archive/newsletters.mbox
```

## 10. Build & packaging

- `Cargo.toml` release profile: `lto = true`, `codegen-units = 1`,
  `panic = "abort"`, `strip = true`.
- CI target matrix: `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`,
  `x86_64-apple-darwin`, `aarch64-apple-darwin`.
- No installer beyond "drop the binary somewhere on `$PATH`" plus a
  first-run `postmaster-k init` that scaffolds `~/.postmaster-k/config.toml`.

## 11. Open questions to resolve during implementation

1. Hashing dimensionality and collision behaviour — 2048 is a starting
   guess, needs tuning against a real labelled corpus once one exists.
2. Confidence threshold and default `k` — same caveat, needs real data.
3. Stopword list tuning — the built-in list is a first pass; `analytics`
   itself (§8) is the intended tool for spotting when it's letting through
   too much boilerplate noise.
4. Whether `unsure` messages should double as an implicit "please label me"
   queue that a later `learn -` pass consumes.
5. Quoted-reply-chain and signature-block stripping — real emails carry
   quoted prior messages and signature boilerplate inside the text/plain
   body itself, which §7.1's MIME-part-level filtering does *not* address
   (it operates at the MIME-part level, not within a part's text). Worth a
   v2 pass at heuristic quote/signature stripping (e.g. lines starting
   with `>`, `-- ` signature delimiters) once basic classification is
   working, since it currently isn't in scope for v1's text extraction.
6. Corpus size expectations — the whole mmap-view architecture assumes a
   "years of personal email" scale (thousands–tens of thousands of vectors),
   not web-scale; worth stating explicitly so nobody over-engineers for a
   problem this doesn't have.
