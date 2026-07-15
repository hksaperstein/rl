"""Small, deliberately naive full-text search over this repo's knowledge base.

Searches `kb/wiki/**/*.md` (the LLM-compiled Obsidian wiki — see
`kb/README.md`) plus the raw sources it's compiled from
(`ROADMAP.md`, `CLAUDE.md`, `docs/superpowers/specs/*.md`,
`docs/superpowers/plans/*.md`, `docs/cloud/*.md`), and prints ranked
results: file path, a relevance score, and a couple lines of matching
context.

This is intentionally NOT a vector-DB/embedding/RAG system. At this
repo's current scale (~35 kb articles + ~110 raw-source docs, a few MB of
text total) a plain keyword search with a log-scaled TF-IDF-ish score is
the right level of engineering — see the 2026-07-14 kb-search design
discussion. No index is built ahead of time; every invocation scans the
corpus fresh (sub-second at this scale) so there's no cache/index file to
go stale.

Plain `python3`, stdlib only (`re`, `glob`, `argparse`, `json`, `math`) —
no embedding/ML libraries. Run from anywhere; paths are resolved relative
to the repo root (this file's grandparent directory), not the CWD.

Ranking: bag-of-words, case-insensitive, `[a-z0-9]+` tokenization (so
"SPOT" / "spot" / "d20" all match consistently). For each query term,
score contribution per document is
`idf(term) * log(1 + count_of_term_in_document)`
where `idf(term) = ln((N + 1) / (df(term) + 1)) + 1` (N = total docs,
df = number of docs containing the term at least once — standard smoothed
IDF, avoids zero/negative weights for terms that appear in every
document). Document score is the sum over query terms. The log(1+tf)
term is a deliberate (still naive) refinement over raw term-count: it
keeps a handful of hits in a short kb article competitive against the
same handful of hits buried in one of the much larger raw spec/plan
files, rather than letting raw-source file *length* dominate purely by
having more room to repeat a word. No artificial per-category boost is
applied between kb/wiki and raw-source hits — score is driven by content
match only; use --kb-only / --raw-only for explicit corpus control
instead of an implicit ranking thumb on the scale.

Context snippet: for each matched document, the single line with the
highest count of query-term hits is selected (ties broken by first
occurrence), and printed together with one line of context before and
after (like `grep -C1`, but the anchor line is the best-matching line in
the document, not just the first match).

Usage:

.. code-block:: bash

    cd ~/projects/rl

    # basic search, top 10 results, human-readable
    python3 scripts/kb_search.py antipodal grasp

    # restrict to the compiled wiki only (skip raw specs/plans/ROADMAP)
    python3 scripts/kb_search.py --kb-only reward hacking

    # restrict to raw sources only (skip kb/wiki) - e.g. to cross-check
    # a wiki claim against the spec/plan/ROADMAP entry it was compiled from
    python3 scripts/kb_search.py --raw-only franka pivot

    # JSON output for an LLM caller to consume programmatically
    python3 scripts/kb_search.py --json --limit 5 SPOT preemption

    # more results
    python3 scripts/kb_search.py --limit 20 curriculum
"""

import argparse
import glob
import json
import math
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_LIMIT = 10


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _collect_files(kb_only: bool, raw_only: bool) -> list[tuple[str, str]]:
    """Returns list of (absolute_path, category) where category is 'kb' or 'raw'."""
    files: list[tuple[str, str]] = []

    if not raw_only:
        kb_glob = os.path.join(REPO_ROOT, "kb", "wiki", "**", "*.md")
        for path in sorted(glob.glob(kb_glob, recursive=True)):
            files.append((path, "kb"))

    if not kb_only:
        raw_candidates = []
        raw_candidates.append(os.path.join(REPO_ROOT, "ROADMAP.md"))
        raw_candidates.append(os.path.join(REPO_ROOT, "CLAUDE.md"))
        raw_candidates += glob.glob(os.path.join(REPO_ROOT, "docs", "superpowers", "specs", "*.md"))
        raw_candidates += glob.glob(os.path.join(REPO_ROOT, "docs", "superpowers", "plans", "*.md"))
        raw_candidates += glob.glob(os.path.join(REPO_ROOT, "docs", "cloud", "*.md"))
        for path in sorted(p for p in raw_candidates if os.path.isfile(p)):
            files.append((path, "raw"))

    return files


class Document:
    def __init__(self, path: str, category: str, lines: list[str]):
        self.path = path
        self.category = category
        self.lines = lines
        self.term_counts: dict[str, int] = {}
        for line in lines:
            for tok in _tokenize(line):
                self.term_counts[tok] = self.term_counts.get(tok, 0) + 1

    @property
    def rel_path(self) -> str:
        return os.path.relpath(self.path, REPO_ROOT)


def _load_documents(kb_only: bool, raw_only: bool) -> list[Document]:
    docs = []
    for path, category in _collect_files(kb_only, raw_only):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        docs.append(Document(path, category, lines))
    return docs


def _idf(term: str, docs: list[Document]) -> float:
    n = len(docs)
    df = sum(1 for d in docs if term in d.term_counts)
    return math.log((n + 1) / (df + 1)) + 1.0


def _best_snippet(doc: Document, query_terms: list[str]) -> tuple[int, str]:
    """Returns (1-based line number, snippet text) for the best-matching line."""
    best_idx = 0
    best_count = -1
    for i, line in enumerate(doc.lines):
        line_toks = _tokenize(line)
        count = sum(line_toks.count(t) for t in query_terms)
        if count > best_count:
            best_count = count
            best_idx = i
    lo = max(0, best_idx - 1)
    hi = min(len(doc.lines), best_idx + 2)
    snippet_lines = [ln.strip() for ln in doc.lines[lo:hi] if ln.strip()]
    snippet = " / ".join(snippet_lines)
    if len(snippet) > 300:
        snippet = snippet[:297] + "..."
    return best_idx + 1, snippet


def search(query: str, kb_only: bool = False, raw_only: bool = False, limit: int = DEFAULT_LIMIT):
    query_terms = _tokenize(query)
    docs = _load_documents(kb_only, raw_only)
    if not query_terms or not docs:
        return []

    idf_cache = {t: _idf(t, docs) for t in set(query_terms)}

    scored = []
    for doc in docs:
        score = 0.0
        for term in query_terms:
            tf = doc.term_counts.get(term, 0)
            if tf == 0:
                continue
            score += idf_cache[term] * math.log(1 + tf)
        if score > 0:
            line_no, snippet = _best_snippet(doc, query_terms)
            scored.append(
                {
                    "path": doc.rel_path,
                    "category": doc.category,
                    "score": round(score, 4),
                    "line": line_no,
                    "snippet": snippet,
                }
            )

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:limit]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Naive keyword search over this repo's knowledge base "
            "(kb/wiki/ + its raw sources: ROADMAP.md, CLAUDE.md, "
            "docs/superpowers/specs, docs/superpowers/plans, docs/cloud)."
        ),
    )
    parser.add_argument("query", nargs="+", help="query terms (bag-of-words, not phrase-matched)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--kb-only", action="store_true", help="search kb/wiki/ only, skip raw sources")
    group.add_argument("--raw-only", action="store_true", help="search raw sources only, skip kb/wiki/")
    parser.add_argument("--json", action="store_true", help="emit results as a JSON array instead of formatted text")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"max results to show (default {DEFAULT_LIMIT})")
    args = parser.parse_args()

    query = " ".join(args.query)
    results = search(query, kb_only=args.kb_only, raw_only=args.raw_only, limit=args.limit)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print(f"No results for: {query!r}")
        return

    print(f"Query: {query!r} ({len(results)} result(s) shown, limit={args.limit})\n")
    for i, r in enumerate(results, 1):
        print(f"{i:2}. [{r['score']:6.2f}] ({r['category']}) {r['path']}:{r['line']}")
        print(f"     {r['snippet']}")
        print()


if __name__ == "__main__":
    sys.exit(main())
