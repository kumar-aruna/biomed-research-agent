# BioEvidence Research Agent

A LangGraph-based research agent that queries real biomedical databases — OpenTargets, PubMed, PubMed Central, UniProt — and writes structured markdown reports with gene target tables, literature citations, and PMID references. Give it a disease question and it autonomously gathers evidence across multiple sources, ranks literature passages with BM25, and synthesizes a final report.

Built for anyone exploring how LLM agents can automate biomedical literature review.

---

## Features

- **7 specialized tools** — disease lookup, gene-target evidence, article fetching, text chunking, BM25 keyword search, protein detail, and web search
- **3-node LangGraph agent** — planner (decides which tools to call), tool executor (runs them), synthesizer (writes the report)
- **BM25 literature pipeline** — fetches PubMed articles, downloads free PubMed Central full text when available, chunks the text, and ranks passages by keyword relevance
- **File-based evidence** — all intermediate data (catalog, chunks, ranked passages) is saved to disk under `lit_workspace/` so you can inspect exactly what the agent used
- **Structured reports** — gene target tables, PMID citations, disease overview, literature findings, and a summary
- **Streamlit web app** — interactive UI with example queries and live agent trace
- **Error-resilient** — each tool call is wrapped in try/except so one failed API does not crash the run

---

## Architecture

```
User question
      │
      ▼
 ┌──────────┐   tool_calls    ┌────────────────┐
 │  PLANNER │────────────────►│ TOOL EXECUTOR  │
 │ (LLM + 7 │◄────────────────│ (runs tools)   │
 │  tools)   │   ToolMessages  └────────────────┘
 └─────┬─────┘
       │ no more tool_calls
       ▼
 ┌─────────────┐
 │ SYNTHESIZER │──► Final markdown report
 │ (LLM, no    │
 │  tools)     │
 └─────────────┘
```

### How a typical run works

1. **Planner** receives the question and system instructions listing all 7 tools.
2. It calls `disease_lookup` to resolve the disease name to an OpenTargets ID.
3. `disease_evidence` returns the top associated gene targets.
4. The **BM25 literature pipeline** runs in three steps:
   - `fetch_articles` → PubMed search → saves `lit_workspace/catalog.json`
   - `build_chunks` → for each article, tries PubMed Central free full text; falls back to abstract → saves `lit_workspace/chunks.jsonl`
   - `bm25_search` → ranks chunks by keyword match → saves `lit_workspace/bm25_evidence.txt`
5. `protein_info` and `web_search` add protein function summaries and recent context.
6. Once the planner stops requesting tools, the **synthesizer** reads the full conversation and writes a structured report.

All heavy text stays on disk. Only short file paths pass through the LLM messages, keeping context usage predictable.

---

## Tools

| # | Name | Source | What it does |
|---|------|--------|--------------|
| 1 | `disease_lookup` | OpenTargets | Disease name → official EFO/MONDO ID and description |
| 2 | `disease_evidence` | OpenTargets | Disease ID → top 10 gene targets with association scores |
| 3 | `fetch_articles` | PubMed | Query → article metadata (PMID, title, abstract, PMC ID) saved to `catalog.json` |
| 4 | `build_chunks` | PubMed Central + local | Reads catalog, downloads free full text where available, chunks text → `chunks.jsonl` |
| 5 | `bm25_search` | rank-bm25 | BM25 keyword ranking over chunks → top passages saved to `bm25_evidence.txt` |
| 6 | `protein_info` | UniProt | Accession ID → protein name, gene, organism, function |
| 7 | `web_search` | DuckDuckGo | General web search for clinical trials, news, supplementary context |

---

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/ArunaK-netizen/Bio-Evidence-Research-Agent.git
cd Bio-Evidence-Research-Agent
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` with your Azure OpenAI credentials:

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY` | API key from the Azure portal |
| `AZURE_OPENAI_API_VERSION` | API version |
| `AZURE_OPENAI_LLM_MODEL_DEPLOYMENT_ID` | Deployment name for your chat model (GPT-4o recommended) |

> **Note:** The notebook also imports `AzureOpenAIEmbeddings` for optional use. If you configure an embedding deployment in `.env`, set `AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_ID` as well. The core BM25 pipeline does not require embeddings.

ToolUniverse and DuckDuckGo require no additional API keys.

### 3. Run the notebook

```bash
jupyter notebook Research_Agent.ipynb
```

Walk through the cells top to bottom: environment setup → tool definitions with smoke tests → agent graph → `research()` helper → example queries.

### 4. Or run the Streamlit app

```bash
streamlit run app.py
```

The web app provides an interactive interface with example queries, a live agent trace, and rendered reports.

---

## Project structure

```
├── Research_Agent.ipynb      # Main notebook — tools, agent graph, research() helper
├── literature_bm25.py        # BM25 pipeline functions (collect, chunk, search, synthesize)
├── app.py                    # Streamlit web app
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment variables
├── .gitignore
├── lit_workspace/             # Created at runtime — BM25 intermediate files
│   ├── catalog.json           #   Article metadata from PubMed
│   ├── chunks.jsonl           #   Chunked article text
│   └── bm25_evidence.txt     #   Top-k ranked passages (what the synthesizer reads)
├── reports/                   # Saved markdown reports (one per research() call)
├── learn.ipynb                # Earlier notebook — LangChain basics and simple agent loops
├── tooluniverse_learning.ipynb # Earlier notebook — ToolUniverse exploration
└── server/                    # FastAPI deployment (Docker + Compose)
```

### Key files

- **`literature_bm25.py`** — Plain Python functions with no LangGraph dependency. The notebook imports them and wraps each with `@tool`. This separation makes it easy to test the pipeline functions individually.
- **`lit_workspace/`** — Regenerated on every `research()` call. Open `bm25_evidence.txt` after a run to see exactly which passages the agent used for the literature section.
- **`reports/`** — Each `research()` call saves a timestamped markdown report with the query, runtime, agent trace, and full report body.

---

## Example queries

| Query | What happens |
|-------|-------------|
| *What gene targets are associated with Parkinson's disease and what does recent research say about them?* | Disease lookup → gene evidence → PubMed fetch → PMC full text → BM25 ranking → protein detail → web context → report |
| *What is the role of BRCA1 and BRCA2 in breast cancer?* | Same pipeline, different disease and gene focus |
| *Find evidence linking Type 2 Diabetes to inflammatory pathways* | Multi-source research across OpenTargets, PubMed, and web |

---

## Sample output

Want to see what the agent produces before running it? Real reports are checked in under [`reports/`](reports/):

- [Gene targets for Parkinson's disease (2026-04-13)](reports/Gene_Targets_Associated_With_Parkinsons_2026-04-13.md)
- [Gene targets for Parkinson's disease (2026-04-14)](reports/Gene_Targets_Associated_With_Parkinsons_2026-04-14.md)

Each file includes the query, the agent's tool-call trace, a ranked gene target table, literature findings with PMIDs, and a synthesized summary. See [`reports/README.md`](reports/README.md) for a breakdown of the report structure.

---

## Limitations and future work

- **BM25 is keyword-based** — it works well for gene and drug names but may miss semantically related terms. A hybrid approach (BM25 + vector embeddings) could improve recall.
- **PubMed Central coverage** — only open-access articles have downloadable full text. Paywalled articles fall back to abstracts, which are shorter.
- **No persistent storage** — `lit_workspace/` is cleared on each run. For multi-session workflows, the pipeline could write to a persistent database.
- **Single LLM provider** — currently tied to Azure OpenAI. Swapping to another provider means changing the LLM initialization in one place.
- **No automated evaluation** — adding a test harness that checks whether the agent calls the expected tools for known queries would improve reliability.

---

## License

MIT

---

## Author

**Aruna Kumar** — Bioinformatics student building AI agents for biomedical research.
