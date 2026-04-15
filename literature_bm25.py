"""
Literature pipeline: collect → chunk → BM25 retrieve → synthesize.

4 helper functions (NOT @tools yet — kept clean so they're easy to test):

    Step 1  literature_collect        PubMed search → catalog.json
                                       (title + abstract for every hit)

    Step 2  literature_build_chunks   Read catalog → try PMC free full text
                                       → fall back to abstract → chunks.jsonl

    Step 3  literature_bm25_search    Load chunks → BM25 → bm25_evidence.txt
                                       (top-k passages ranked by keyword match)

    Step 4  literature_synthesize     LLM reads bm25_evidence.txt → final report

The notebook (Literature_BM25_Agent.ipynb) imports these functions and wraps
each one with @tool so the LangGraph planner can call them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ── simple word tokeniser for BM25 ────────────────────────────────────────────
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens for BM25 indexing."""
    return _TOKEN_RE.findall((text or "").lower())


# ── filesystem helpers ─────────────────────────────────────────────────────────
def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def reset_literature_workspace(workspace: Path) -> None:
    """Delete all files in workspace (call at the start of a fresh research run)."""
    _ensure_dir(workspace)
    for f in workspace.iterdir():
        if f.is_file():
            f.unlink()


# ── field extractors ───────────────────────────────────────────────────────────
def _extract_pmid(article: dict[str, Any]) -> str:
    for key in ("pmid", "PMID", "uid", "id"):
        v = article.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _extract_title(article: dict[str, Any]) -> str:
    return str(article.get("title") or article.get("Title") or "").strip()


def _extract_abstract(article: dict[str, Any]) -> str:
    for key in ("abstract", "Abstract", "abstract_text", "abstractText"):
        v = article.get(key)
        if v:
            return str(v).strip()
    return ""


def _extract_pmc_id(article: dict[str, Any]) -> str:
    """
    Extract the PubMed Central ID (PMC ID) from article data.
    If a PMC ID exists it means the full text is FREE to download!
    
    Example:  pmc_id = "PMC9876543"  → article is open-access ✅
              pmc_id = ""            → article is paywalled ❌
    """
    for key in ("pmc", "pmc_id", "pmcid", "PMC", "PMCID"):
        v = article.get(key)
        if v is not None and str(v).strip():
            val = str(v).strip().upper()
            # Normalise: always prefix with "PMC" if missing
            if not val.startswith("PMC"):
                val = "PMC" + val
            return val
    # Sometimes it's nested inside articleids list
    for id_obj in article.get("articleids", []):
        if isinstance(id_obj, dict) and id_obj.get("idtype", "").lower() == "pmc":
            val = str(id_obj.get("value", "")).strip().upper()
            if val:
                return "PMC" + val if not val.startswith("PMC") else val
    return ""


def _fetch_pmc_full_text(tu: Any, pmc_id: str) -> str:
    """
    Try to download free full text from PubMed Central.
    Returns the text (up to 12 000 chars) or "" if not available.
    
    Why 12 000 chars? That's roughly 2 000 words — enough context for BM25
    without making chunks enormous.
    """
    # ToolUniverse may use different names — try all plausible ones
    attempts = [
        {"name": "PubMedCentral_get_full_text_by_pmcid", "arguments": {"pmcid": pmc_id}},
        {"name": "PubMed_get_full_text_by_pmc_id",       "arguments": {"pmc_id": pmc_id}},
        {"name": "PubMedCentral_fetch_article",           "arguments": {"pmcid": pmc_id}},
    ]
    for call in attempts:
        try:
            result = tu.run_one_function(call)
            if isinstance(result, str) and len(result) > 300:
                return result[:12_000]
            if isinstance(result, dict):
                for key in ("full_text", "body", "text", "content"):
                    val = result.get(key, "")
                    if val and len(str(val)) > 300:
                        return str(val)[:12_000]
        except Exception:
            continue   # silently try the next tool name
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Search PubMed and save results to catalog.json
# ══════════════════════════════════════════════════════════════════════════════
def literature_collect(
    tu: Any,
    query: str,
    max_articles: int,
    workspace: Path,
) -> str:
    """
    Search PubMed for articles matching `query`.
    Saves a catalog.json with: pmid, title, abstract, pmc_id for each hit.
    Returns the absolute path to catalog.json.

    Why save to a file?
    ─────────────────────
    Article data can be thousands of characters. Passing giant strings between
    tools confuses the LLM. Instead every tool saves to disk and passes only
    a SHORT FILE PATH to the next tool. The next tool reads from disk itself.
    """
    _ensure_dir(workspace)

    raw = tu.run_one_function(
        {
            "name": "PubMed_search_articles",
            "arguments": {
                "query": query,
                "max_results": max(1, min(max_articles, 25)),
                "include_abstract": True,   # ask for abstracts in one call
            },
        }
    )

    if not isinstance(raw, list):
        return f"PubMed search returned unexpected type: {type(raw).__name__}. Raw: {str(raw)[:500]}"

    articles: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pmid = _extract_pmid(item)
        if not pmid:
            continue
        articles.append(
            {
                "pmid":     pmid,
                "title":    _extract_title(item),
                "abstract": _extract_abstract(item),
                "pmc_id":   _extract_pmc_id(item),   # empty = paywalled
            }
        )

    catalog_path = workspace / "catalog.json"
    catalog_path.write_text(
        json.dumps({"query": query, "count": len(articles), "articles": articles}, indent=2),
        encoding="utf-8",
    )
    return str(catalog_path.resolve())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Chunk articles (try PMC full text → fall back to abstract)
# ══════════════════════════════════════════════════════════════════════════════
def _chunk_text(text: str, chunk_size: int = 420, overlap: int = 80) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Why overlap?
    ─────────────
    If a sentence falls exactly on a chunk boundary it gets cut in half.
    Overlapping by 80 chars ensures every sentence appears fully in at
    least one chunk, so BM25 can still find it.
    
    Example (chunk_size=10, overlap=3):
      text = "ABCDEFGHIJ"
      chunks → ["ABCDEFGHIJ", "HIJKLM", "LMNOPQ", ...]
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(text), step):
        chunks.append(text[i : i + chunk_size])
    return chunks


def literature_build_chunks(tu: Any, catalog_path: str, workspace: Path) -> str:
    """
    Read catalog.json, fetch free full text from PMC where available,
    fall back to abstract for paywalled articles, chunk everything,
    save to chunks.jsonl.
    Returns absolute path to chunks.jsonl (first line of return value).

    Open-access vs paywalled logic:
    ─────────────────────────────────
    pmc_id found in catalog  →  FREE full text  →  much richer chunks!
    pmc_id missing           →  abstract only   →  still useful (150-300 words)
    """
    p = Path(catalog_path).expanduser().resolve()
    if not p.is_file():
        return f"Catalog not found: {catalog_path}"

    data = json.loads(p.read_text(encoding="utf-8"))

    rows_out: list[dict[str, str]] = []
    chunk_ix = 0
    full_text_count = 0
    abstract_only_count = 0
    skipped_count = 0

    for art in data.get("articles", []):
        pmid     = str(art.get("pmid",     "")).strip()
        title    = str(art.get("title",    "")).strip()
        abstract = str(art.get("abstract", "")).strip()
        pmc_id   = str(art.get("pmc_id",   "")).strip()

        # ── Try PMC full text first (completely free!) ─────────────────────
        full_text = ""
        if pmc_id:
            print(f"  📖 PMID {pmid}: PMC ID found ({pmc_id}) → fetching free full text…")
            full_text = _fetch_pmc_full_text(tu, pmc_id)
            if full_text:
                print(f"     ✅ Got {len(full_text)} chars of full text")
            else:
                print(f"     ⚠️  PMC fetch failed, will use abstract")

        # ── If abstract is missing, try fetching it by PMID ────────────────
        if not abstract and not full_text and pmid:
            got = tu.run_one_function(
                {"name": "PubMed_get_article", "arguments": {"pmid": pmid}}
            )
            if isinstance(got, dict):
                abstract = _extract_abstract(got)
                # Also check if this response reveals a PMC ID we missed
                if not pmc_id:
                    pmc_id = _extract_pmc_id(got)
                    if pmc_id and not full_text:
                        full_text = _fetch_pmc_full_text(tu, pmc_id)

        # ── Decide what body text to use ────────────────────────────────────
        if full_text:
            body = f"{title}\n\n{full_text}".strip()
            text_source = f"full_text_pmc_{pmc_id}"
            full_text_count += 1
        elif abstract:
            body = f"{title}\n\n{abstract}".strip()
            text_source = "abstract_only"
            abstract_only_count += 1
        else:
            # No text at all — skip this article
            print(f"  ⏭️  PMID {pmid}: no text found, skipping")
            skipped_count += 1
            continue

        for ch in _chunk_text(body):
            rows_out.append(
                {
                    "chunk_id":   f"{pmid}_{chunk_ix}",
                    "pmid":       pmid,
                    "title":      title[:200],
                    "text":       ch,
                    "text_source": text_source,   # shows full_text vs abstract
                }
            )
            chunk_ix += 1

    chunks_path = workspace / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not rows_out:
        return (
            f"No chunks written. Check catalog at {catalog_path}. "
            f"Skipped {skipped_count} articles with no text."
        )

    # Return path on line 1 (so the @tool can pass it directly to bm25_search)
    # followed by a human-readable summary
    abs_path = str(chunks_path.resolve())
    summary = (
        f"{abs_path}\n\n"
        f"✅ Chunked {chunk_ix} passages from {full_text_count + abstract_only_count} articles:\n"
        f"   📖 Full text (open-access PMC): {full_text_count} articles\n"
        f"   📄 Abstract only (paywalled):   {abstract_only_count} articles\n"
        f"   ⏭️  Skipped (no text at all):    {skipped_count} articles"
    )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BM25 keyword search over chunks
# ══════════════════════════════════════════════════════════════════════════════
EVIDENCE_FILENAME = "bm25_evidence.txt"


def literature_bm25_search(chunks_path: str, search_query: str, top_k: int = 8) -> str:
    """
    Run BM25 over all chunks and save the top-k passages to bm25_evidence.txt.
    Returns the evidence file path (first line) + a 900-char preview.

    How BM25 works (simply):
    ──────────────────────────
    1. Tokenise every chunk: "LRRK2 kinase activity" → ["lrrk2","kinase","activity"]
    2. Build a score table: how often does each word appear across ALL chunks?
       Rare words (like "LRRK2") score higher than common words (like "the").
    3. Tokenise your query the same way.
    4. For each chunk: score = sum of query-word scores in that chunk.
    5. Sort chunks by score descending → top-k are most relevant.
    
    This is exactly how traditional search engines worked before neural networks!
    """
    from rank_bm25 import BM25Okapi

    # Parse path from first line (literature_build_chunks may return multiline)
    clean_path = (chunks_path or "").strip().split("\n")[0].strip()
    p = Path(clean_path).expanduser().resolve()
    if not p.is_file():
        return f"Chunks file not found: {clean_path}\nPass the path returned by build_chunks."

    # ── Load all chunks from JSONL ──────────────────────────────────────────
    records: list[dict[str, str]] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return "No lines in chunks.jsonl — run literature_build_chunks first."

    # ── Build BM25 index ────────────────────────────────────────────────────
    # Filter out any empty chunks (BM25Okapi dislikes them)
    tokenized_corpus = [tokenize(r["text"]) for r in records]
    kept_idx, kept_docs = [], []
    for i, toks in enumerate(tokenized_corpus):
        if toks:
            kept_idx.append(i)
            kept_docs.append(toks)

    if not kept_docs:
        return "All chunks were empty after tokenization."

    bm25 = BM25Okapi(kept_docs)

    # ── Score the query ─────────────────────────────────────────────────────
    q_tokens = tokenize(search_query)
    if not q_tokens:
        return "Search query produced no tokens. Try different keywords."

    scores = bm25.get_scores(q_tokens)

    # Sort by score descending, pick top-k
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_positions = ranked[: max(1, min(top_k, len(ranked)))]

    # ── Format results ──────────────────────────────────────────────────────
    blocks: list[str] = []
    for rank, pos in enumerate(top_positions, start=1):
        rec = records[kept_idx[pos]]
        score = float(scores[pos])
        source_label = rec.get("text_source", "unknown")
        blocks.append(
            f"--- Passage {rank} | BM25 score={score:.4f} | "
            f"PMID={rec['pmid']} | Source={source_label}\n"
            f"Title: {rec['title']}\n\n"
            f"{rec['text']}\n"
        )
    full_text = "\n".join(blocks)

    # Save to disk next to chunks.jsonl
    evidence_path = p.parent / EVIDENCE_FILENAME
    evidence_path.write_text(full_text, encoding="utf-8")
    abs_ev = str(evidence_path.resolve())

    preview = full_text[:900] + ("…" if len(full_text) > 900 else "")
    return (
        f"{abs_ev}\n\n"
        f"(Saved {len(top_positions)} passages — pass ONLY that path to literature_synthesize)\n\n"
        f"--- Preview ---\n{preview}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — LLM synthesis from BM25 evidence
# ══════════════════════════════════════════════════════════════════════════════
def literature_synthesize(llm: Any, research_question: str, evidence_file_path: str) -> str:
    """
    Read bm25_evidence.txt and ask the LLM to write a structured synthesis.
    
    Key design: read from file, not from argument.
    ────────────────────────────────────────────────
    The evidence text can be thousands of characters. If we passed it as a
    tool argument the LLM context fills up fast and it may truncate or forget.
    Instead we keep it on disk and load it here — fresh, complete, every time.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Extract path from first line (bm25_search returns multiline)
    raw  = (evidence_file_path or "").strip()
    path = Path(raw.split("\n", 1)[0].strip()).expanduser().resolve()
    if not path.is_file():
        return (
            f"Evidence file not found: {path}. "
            "Pass the path returned by literature_bm25_search (first line)."
        )

    retrieved_passages = path.read_text(encoding="utf-8")
    if not retrieved_passages.strip():
        return f"Evidence file is empty: {path}"

    system = SystemMessage(
        content=(
            "You are a careful biomedical research assistant. "
            "Answer ONLY using the evidence passages provided below. "
            "If the passages are insufficient to answer fully, say so. "
            "Cite PMIDs in parentheses like (PMID: 12345678). "
            "Note which findings come from full text vs abstract-only sources. "
            "Never invent studies, gene names, or statistics not in the passages."
        )
    )
    human = HumanMessage(
        content=(
            f"Research question:\n{research_question}\n\n"
            f"Retrieved passages (PubMed, BM25-ranked):\n\n{retrieved_passages}\n\n"
            "Write a concise synthesis (3–6 short paragraphs) with PMID citations. "
            "End with a one-sentence summary answering the research question directly."
        )
    )
    out = llm.invoke([system, human])
    return str(out.content)


__all__ = [
    "reset_literature_workspace",
    "literature_collect",
    "literature_build_chunks",
    "literature_bm25_search",
    "literature_synthesize",
    "tokenize",
]
