# Research-corpus read ledger

The public corpus [`1jmdev/research`](https://github.com/1jmdev/research) maps ~8,200
arXiv papers (2025-01 → 2026-09) on LLM inference and runtime work, with one Markdown
digest per paper and per-category rankings. It is the pool we mine when a speed or
quality lever is short.

The problem this folder solves: **re-reading after a corpus update.** The corpus is
rebuilt from time to time ("Rebuild with newly fetched full-text digests"), so papers
keep their arXiv id but get **new digest text**. A plain notes file cannot tell the two
apart, and reading 8,200 digests again is not an option.

## Files

| File | Purpose |
|---|---|
| `corpus-state.json` | The ledger. One entry per arXiv id: subcategory, read depth, verdict, `sha1[:12]` of the digest, date. Plus the corpus commit it was built against. |
| `check_corpus.py` | Diff a fresh clone against the ledger. Prints only **NEW** (never seen) and **CHANGED** (same id, rewritten digest). |
| `sweep-2026-10.md` | What the last sweep found: candidates, rejections with reasons, open questions. |

## Workflow

```sh
git clone --depth 1 https://github.com/1jmdev/research /tmp/research-corpus

python3 docs/research/check_corpus.py /tmp/research-corpus            # only the delta
python3 docs/research/check_corpus.py /tmp/research-corpus --update   # new ids as "unjudged"
python3 docs/research/check_corpus.py /tmp/research-corpus 2501.23456=candidate:tldr
```

`--top N` widens or narrows the ranking window that is compared (default 10 per
subcategory). A digest whose text changed is reported as CHANGED even when the id and
its rank are unchanged — that is the signal that a paper's numbers may have moved.

## Read depth

| Depth | Meaning |
|---|---|
| `ranking-only` | registered from the category ranking, not read |
| `tldr` | the paper's TL;DR digest read; no full text, no abstract |
| `abstract` | abstract read |
| `full` | full text read |

A verdict is `candidate`, `rejected`, or `unjudged`. Reasons for a rejection live in
`sweep-2026-10.md`, not in the ledger, so the ledger stays a flat, diffable file.

## Coverage of the 2026-10-25 sweep

| Metric | Value |
|---|---|
| Corpus commit | `ae20e11f` |
| Subcategories swept | 73 (all) |
| Papers registered (top 10 per subcategory) | 723 |
| Papers read (TL;DR) | 219 — 15 candidate, 204 rejected |
| Registered, not yet read | 504 |

The 504 are the honest remainder: the sweep read the top 3 of every subcategory, not
the full top 10.

## Caveats

- A digest is a **summary of claims**, not a measurement. Every number taken from it is
  labelled `paper-claim` in `sweep-2026-10.md`, never as a result.
- Rankings shift. A paper at rank 4 today can be rank 1 tomorrow; `--top 10` is the
  window that catches it, `--top 3` would not.
- The sweep is a filter for our stack (one R9700, RDNA4, MXFP4, one user), not a general
  survey. Most of the corpus assumes datacenter GPUs, multi-GPU or training; those
  categories are rejected wholesale and that is recorded rather than silent.
