# STATUS — postmaster-k

**Date:** 2026-09-27
**State:** SPEC.md implemented end to end, v0.1.0. All four CLI subcommands
(`init`, `learn`, `infer`, `analytics`) work; 40 tests pass; release binary
built and measured. Git remote not yet configured, so the CI workflow has
never run on a real runner.

## What works (SPEC → implementation)

| SPEC area | Status | Where |
|---|---|---|
| §5 data layout | done | `config.toml`, `index.usearch`, `labels.db`, `terms/<cat>.db` under one state root |
| §6 classification | done | kNN inverse-distance vote, threshold → `unsure` (reserved name) |
| §7 feature hashing | done | signed FNV dual-hash, stoplist, L2-normalized; IDF hook default-off per spec |
| §7.1 text extraction | done | subject + text/plain; html-only fallback; nested rfc822; MIME noise excluded |
| §7.2 term tables | done | per-category `token→count→docs`, side effect of `learn` |
| §8 `learn` | done | directory batch + stdin (`-` / `formail -ds`); only command in `load()` mode |
| §8 `infer` | done | `word` / `header` / `full`; always mmap `view()` mode |
| §8 `analytics` | done | top-N per category from unhashed tables; `--all`; never touches the index |
| §10 build & packaging | done | profiles, CI matrix for 4 targets, drop-in binary, `init` scaffolder |
| §2 hard requirements | done | fail-open everywhere, single binary, mmap view for cold start |
| §11 open questions | open | see below — tuning needs a real corpus |

## Build & binary

- `cargo build` → `./postmaster-k` (symlink → `target/debug/postmaster-k`;
  the dev profile already mirrors release: `opt-level=3`, `lto`,
  `codegen-units=1`, `strip`, `debug=false`). Dev profile intentionally lacks
  `panic="abort"` so `cargo test` works.
- `cargo build --release` → 2.6 MB stripped Mach-O arm64 (adds
  `panic=abort`). Musl/darwin cross-builds are defined in
  `.github/workflows/ci.yml` but not yet exercised (no remote).
- Dependencies: `usearch 2`, `mail-parser 0.11`, `clap 4`, `toml_edit`.
  `usearch` needs cmake at build time; binary is self-contained at runtime.

## Validation performed

- **40/40 tests pass** (`cargo test`): 35 unit + 5 e2e in `tests/e2e.rs`
  covering category separation, `unsure` threshold behavior,
  misuse-vs-soft-failure exit paths, and byte-exact full-mode passthrough.
- **CLI round trip verified in debug and release**: learn two categories
  from directories, learn from stdin, infer all three modes, analytics
  single + `--all`, failure-path exit codes.
- **Hot path measured**: 10 sequential `infer --mode=word` runs ≈ 33 ms
  total (~3.3 ms each including process spawn) on macOS arm64. Meets §2's
  per-message procmail budget with wide margin.
- **Fail-open verified end to end**: corrupt/dims-mismatched index →
  `unsure` / original bytes at exit 0; missing index/config → exit 2 for
  word/header but pass-through at exit 0 for full mode.

## Semantics worth remembering

- Non-zero exit (2) means *misuse* (bad args, missing state), never
  "couldn't classify this message". A message that can't be parsed or
  classified is `unsure`, exit 0 — the message must never be lost.
- Full mode is idempotent: previously inserted `X-Postmaster-K-*` headers
  (incl. folded continuations) are stripped and re-inserted at the top of
  the header block, matching the host message's CRLF/LF style; mbox
  `From ` envelope lines stay above our headers.
- A body that merely *starts* with "From " is not treated as an mbox
  envelope unless the next line looks like a header.
- `learn` saves the index via tmp-file + rename; a crash mid-learn cannot
  corrupt `index.usearch`. Categories are registered in `config.toml` only
  after the batch yields ≥1 vector.
- `unsure` is a reserved category name; user categories are `[a-z0-9_-]`.

## Open questions (SPEC §11, still open)

1. **dims / k / threshold tuning** — defaults 2048 / 15 / 0.4 are first
   guesses; nothing validated against a real labelled corpus yet.
2. **Stopword list** — first pass only; `analytics --all` is the intended
   tool for spotting boilerplate leaking through.
3. **`unsure` as an implicit label-me queue** consumed by a later
   `learn -` pass — not implemented.
4. **Quote/signature stripping** (§11.5) — v1 extracts at MIME-part level
   only; quoted reply chains and `-- ` signatures still pollute tokens.
5. **IDF weighting** — term tables already store per-doc counts, so the
   data exists; the embed side is not wired (spec default: off).
6. **CI matrix** — defined but never run (no git remote configured).

## Suggested next steps

1. `git remote add … && git push`, confirm the 4-target CI build is green.
2. Bootstrap real categories from an archive
   (`formail -ds postmaster-k learn <cat> -- < mbox`) and sanity-check
   vocabulary separation with `analytics --all` before trusting it.
3. Revisit §11 items 1–4 once real mail has flowed through it.
