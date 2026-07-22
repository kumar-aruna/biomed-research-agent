"""
BioEvidence Research Agent — Streamlit Web App

A multi-tool AI agent that queries real biomedical databases (OpenTargets,
PubMed, UniProt) and writes structured research reports with citations.

Run with:  streamlit run app.py
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import literature_bm25 as lit_pipe
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from tooluniverse import ToolUniverse
import chromadb
from ddgs import DDGS

load_dotenv(override=True)

# ── LLM provider selection ───────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "AZURE_OPENAI").upper()

# ═══════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BioEvidence Research Agent",
    page_icon="🧬",
    layout="wide",
)


# ═══════════════════════════════════════════════════════════════════════
# Cached setup — runs once, shared across all users / reruns
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading LLM, ToolUniverse & ChromaDB...")
def setup():
    if LLM_PROVIDER == "GEMINI":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_retries=2,
            max_output_tokens=8192,
        )
        # Separate, more capable model for synthesis (1 call/query vs many for planning).
        # Generous output budget so a "thinking" model has room to emit the full
        # report instead of returning empty content after its reasoning pass.
        llm_synth = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_SYNTH_MODEL", "gemini-3-flash-preview"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_retries=2,
            max_output_tokens=8192,
        )
        embed = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    elif LLM_PROVIDER == "OPENROUTER":
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import FakeEmbeddings

        _or_base = "https://openrouter.ai/api/v1"
        _or_key = os.getenv("OPENROUTER_API_KEY")
        _or_headers = {
            "HTTP-Referer": "https://github.com/bioevidence-agent",
            "X-Title": "BioEvidence Research Agent",
        }
        llm = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free"),
            openai_api_key=_or_key,
            openai_api_base=_or_base,
            default_headers=_or_headers,
            temperature=0,
            max_retries=2,
        )
        llm_synth = ChatOpenAI(
            model=os.getenv("OPENROUTER_SYNTH_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")),
            openai_api_key=_or_key,
            openai_api_base=_or_base,
            default_headers=_or_headers,
            temperature=0,
            max_retries=2,
        )
        embed = FakeEmbeddings(size=1536)  # OpenRouter has no embedding endpoint; BM25 pipeline doesn't need real embeddings
    else:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL_DEPLOYMENT_ID"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        llm_synth = llm  # Azure uses the same model for both roles
        embed = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_ID"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

    tu = ToolUniverse()
    tu.load_tools()

    chroma = chromadb.Client()

    return llm, llm_synth, embed, tu, chroma


llm, llm_synth, embeddings, tu, chroma_client = setup()

LITERATURE_WS = Path("reports/literature_workspace/streamlit_current")
LITERATURE_WS.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Per-session ChromaDB collection (reset each query)
# ═══════════════════════════════════════════════════════════════════════

if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.get_or_create_collection(
        name="research_docs"
    )

collection = st.session_state.collection


# ═══════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════

@tool
def disease_lookup(disease_name: str) -> str:
    """Look up a disease by name to get its EFO/MONDO ID and description
    from the OpenTargets database.

    Args:
        disease_name: Name of the disease (e.g. "Alzheimer's disease")
    """
    result = tu.run_one_function({
        "name": "OpenTargets_get_disease_id_description_by_name",
        "arguments": {"diseaseName": disease_name},
    })
    return str(result)


@tool
def disease_evidence(efo_id: str) -> str:
    """Fetch gene targets associated with a disease from OpenTargets.
    Returns the top 10 targets with association scores.

    Args:
        efo_id: Disease EFO/MONDO ID (e.g. "MONDO_0004975")
    """
    result = tu.run_one_function({
        "name": "OpenTargets_get_associated_targets_by_disease_efoId",
        "arguments": {"efoId": efo_id},
    })
    if isinstance(result, dict) and "data" in result:
        targets = result["data"].get("disease", {}).get("associatedTargets", {})
        rows = targets.get("rows", [])[:10]
        summary = {
            "total_targets": targets.get("count", 0),
            "top_10": [
                {
                    "gene": r.get("target", {}).get("approvedSymbol", "?"),
                    "ensembl_id": r.get("target", {}).get("id", "?"),
                    "score": round(r.get("score", 0), 4),
                }
                for r in rows
            ],
        }
        return str(summary)
    return str(result)


@tool
def pubmed_search(query: str, max_results: int = 5) -> str:
    """Search PubMed for scientific articles about a biomedical topic.

    Args:
        query: Search terms (e.g. "BRCA1 breast cancer therapy")
        max_results: Number of articles to return (default 5)
    """
    result = tu.run_one_function({
        "name": "PubMed_search_articles",
        "arguments": {"query": query, "max_results": max_results},
    })
    if isinstance(result, list):
        articles = [
            {
                "title": a.get("title", ""),
                "authors": a.get("authors", [])[:3],
                "journal": a.get("journal", ""),
                "year": a.get("pub_year", ""),
                "doi": a.get("doi", ""),
                "url": a.get("url", ""),
            }
            for a in result[:max_results]
        ]
        return str(articles)
    return str(result)


# UniProt accession pattern (e.g. P38398, Q9Y6K9). Note position 2 is always a
# digit, which is what distinguishes an accession from a gene symbol like "BRCA1".
_ACCESSION_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


@tool
def protein_info(protein: str) -> str:
    """Get protein details from UniProt — name, gene, organism, and function.

    Accepts either a gene symbol (e.g. "BRCA1") or a UniProt accession
    (e.g. "P38398"); gene symbols are resolved to the reviewed human entry.

    Args:
        protein: Gene symbol or UniProt accession
    """
    try:
        term = protein.strip()
        if _ACCESSION_RE.match(term.upper()):
            query = f"accession:{term}"
        else:
            query = f"gene_exact:{term} AND organism_id:9606 AND reviewed:true"
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": query, "format": "json", "size": 1}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "BioEvidenceAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return f"UniProt lookup failed for '{protein}': {e}"

    results = data.get("results", [])
    if not results:
        return f"No UniProt entry found for '{protein}'."

    r = results[0]
    function = ""
    for comment in r.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                function = texts[0].get("value", "")[:500]
            break
    summary = {
        "protein": r.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", ""),
        "accession": r.get("primaryAccession", ""),
        "gene": r.get("genes", [{}])[0].get("geneName", {}).get("value", ""),
        "organism": r.get("organism", {}).get("scientificName", ""),
        "function": function,
    }
    return str(summary)


@tool
def web_search(query: str) -> str:
    """Search the web for additional information when biomedical databases
    aren't enough.

    Args:
        query: Search query
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return str([
            {"title": r.get("title", ""), "body": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ])
    except Exception as e:
        return f"Web search failed: {e}. Try pubmed_search instead."


@tool
def literature_collect(query: str, max_articles: int = 12) -> str:
    """STEP 1 — PubMed search; save catalog.json; return its path."""
    return lit_pipe.literature_collect(tu, query, max_articles, LITERATURE_WS)


@tool
def literature_build_chunks(catalog_path: str) -> str:
    """STEP 2 — Read catalog; chunk abstracts; save chunks.jsonl; return path."""
    return lit_pipe.literature_build_chunks(tu, catalog_path, LITERATURE_WS)


@tool
def literature_bm25_search(chunks_path: str, search_query: str, top_k: int = 8) -> str:
    """STEP 3 — BM25; saves passages to bm25_evidence.txt; first line of output is the path."""
    return lit_pipe.literature_bm25_search(chunks_path, search_query, top_k)


@tool
def literature_synthesize(research_question: str, evidence_file_path: str) -> str:
    """STEP 4 — Read evidence file from disk (path from step 3); grounded synthesis."""
    return lit_pipe.literature_synthesize(llm_synth, research_question, evidence_file_path)


# ═══════════════════════════════════════════════════════════════════════
# Build the agent graph
# ═══════════════════════════════════════════════════════════════════════

tools_list = [
    disease_lookup,
    disease_evidence,
    pubmed_search,
    protein_info,
    web_search,
    literature_collect,
    literature_build_chunks,
    literature_bm25_search,
    literature_synthesize,
]
tools_by_name = {t.name: t for t in tools_list}
llm_with_tools = llm.bind_tools(tools_list)

PLANNER_PROMPT = """\
You are a biomedical research assistant with access to real scientific databases.
Your job is to gather a thorough, citable evidence trail before a report is
written. Use EVERY relevant tool — comprehensive evidence is required.

Run this full workflow:

TRACK A — Structured biomedical data
1. disease_lookup(disease_name) → resolve the topic to an EFO/MONDO disease ID
2. disease_evidence(efo_id) → top associated gene targets with scores
3. protein_info(protein) → function of the 1–2 most relevant genes
   (pass the gene SYMBOL, e.g. "BRCA1" — it is resolved to UniProt automatically)
4. pubmed_search(query) → a quick list of recent papers

TRACK B — Literature evidence (BM25 pipeline; ALWAYS run, strictly in order)
5. literature_collect(query) → returns catalog_path
6. literature_build_chunks(catalog_path=<path from 5>)
7. literature_bm25_search(chunks_path=<path from 6>, search_query=<keywords>)
   → the FIRST line of the tool output is the absolute path to bm25_evidence.txt
8. literature_synthesize(research_question=..., evidence_file_path=<first line from 7>)

9. web_search(query) → recent clinical-trial / news / therapy context

Rules:
- Do Track A steps 1–4, then the FULL Track B pipeline 5→8 in order, then step 9.
- For step 8 pass the file PATH only (never paste the BM25 text into the tool).
- Do NOT write the final report yourself. Once the evidence is gathered, stop
  calling tools — a dedicated writer will synthesize the report.
"""

SYNTHESIZER_PROMPT = """\
You are a biomedical research report writer for a precision-oncology audience.
Using ONLY the evidence gathered in this conversation (OpenTargets targets,
UniProt protein data, PubMed/PMC literature, the literature_synthesize output,
and web results), write a rigorous, well-structured markdown report.

Begin your response with "## Disease Overview" and use EXACTLY these sections:

## Disease Overview
2–3 sentences on the disease/topic, grounded in the retrieved disease description.

## Top Gene Targets
A markdown table with columns: | Gene | Ensembl ID | Association Score |
Fill it from the disease_evidence data — up to 10 genes, highest score first.

## Protein Function
For the key gene(s), summarize the UniProt entry (protein name, accession,
function). If no protein data was retrieved, say so.

## Key Literature Findings
Summarize the literature and the literature_synthesize output. Cite specific
papers as [PMID:xxxxxxxx] wherever a PMID is available.

## Clinical & Translational Relevance
Therapeutic implications, biomarkers, or trials drawn from web_search / literature
(e.g. targeted therapies, companion diagnostics).

## Summary
A concise 2–3 sentence answer to the user's original question.

Rules:
- Ground EVERY claim in retrieved data — never invent genes, scores, PMIDs, or facts.
- Always render the gene-target table when disease_evidence returned data.
- If a section genuinely has no data, write "No data retrieved for this section."
- Be precise with gene symbols, association scores, and citations.
"""


def planner(state: MessagesState):
    system = SystemMessage(content=PLANNER_PROMPT)
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


def tool_executor(state: MessagesState):
    results = []
    for call in state["messages"][-1].tool_calls:
        try:
            tool_fn = tools_by_name[call["name"]]
            output = tool_fn.invoke(call["args"])
        except Exception as e:
            output = f"Error calling {call['name']}: {e}"
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


def synthesizer(state: MessagesState):
    system = SystemMessage(content=SYNTHESIZER_PROMPT)
    response = llm_synth.invoke([system] + state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "tool_executor"
    return "synthesizer"


@st.cache_resource
def build_agent():
    graph = StateGraph(MessagesState)
    graph.add_node("planner", planner)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("synthesizer", synthesizer)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", should_continue, {
        "tool_executor": "tool_executor",
        "synthesizer": "synthesizer",
    })
    graph.add_edge("tool_executor", "planner")
    graph.add_edge("synthesizer", END)
    return graph.compile()


agent = build_agent()


# ═══════════════════════════════════════════════════════════════════════
# Error reporting
# ═══════════════════════════════════════════════════════════════════════

def _explain_failure(exc: Exception) -> str:
    """Turn a provider or graph exception into a message the user can act on.

    A research question costs one LLM call per planner turn, so a rate limit is
    the failure most likely to be hit on a free-tier key.
    """
    name = type(exc).__name__
    detail = str(exc)

    if "RateLimit" in name or "429" in detail:
        return (
            f"**{LLM_PROVIDER} is rate-limiting this key.** Each research question "
            "costs one model call per planning step, so a low quota is exhausted "
            "quickly. Wait a minute and retry, or switch `LLM_PROVIDER` to a key "
            "with more headroom."
        )
    if "Authentication" in name or "401" in detail:
        return (
            f"**{LLM_PROVIDER} rejected the credentials.** Check the API key and "
            "deployment name in the app's secrets."
        )
    if "GraphRecursionError" in name:
        return (
            "**The agent exceeded its step limit.** It kept calling tools without "
            "reaching a conclusion — try a narrower question."
        )
    return f"**Research failed:** {name}: {detail[:300]}"


# ═══════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════

st.title("🧬 BioEvidence Research Agent")
st.markdown(
    "Ask any biomedical research question and the agent will **autonomously** "
    "query OpenTargets, PubMed, UniProt, and the web to build a structured "
    "research report with citations."
)

# ── Sidebar ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown(
        "This agent uses **LangGraph** to orchestrate **9 tools** across "
        "real biomedical databases, plus a **BM25 literature pipeline** "
        "(on-disk catalog + chunks + keyword retrieval + grounded synthesis)."
    )

    st.divider()
    st.subheader("Example Queries")
    examples = [
        "What gene targets are associated with Parkinson's disease?",
        "What is the role of BRCA1 and BRCA2 in breast cancer?",
        "Find evidence linking Type 2 Diabetes to inflammatory pathways",
        "What are the latest gene therapy approaches for Alzheimer's disease?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

    st.divider()
    st.subheader("Tech Stack")
    if LLM_PROVIDER == "GEMINI":
        llm_label = f"**Google Gemini** (`{os.getenv('GEMINI_MODEL', 'gemini-3.1-flash-lite-preview')}`)"
    elif LLM_PROVIDER == "OPENROUTER":
        llm_label = f"**OpenRouter** (`{os.getenv('OPENROUTER_MODEL', 'openai/gpt-oss-120b:free')}`)"
    else:
        llm_label = f"**Azure OpenAI** (`{os.getenv('AZURE_OPENAI_LLM_MODEL_DEPLOYMENT_ID', 'GPT-4o')}`)"
    st.markdown(f"""
    - **LangGraph** — Agent orchestration
    - {llm_label} — Reasoning
    - **ToolUniverse** — 2,000+ biomedical APIs
    - **BM25 + files** — Literature workspace under `reports/`
    - **DuckDuckGo** — Web search
    """)

    st.divider()
    st.caption("Built by **Aruna Kumar** · Bioinformatics")

# ── Main input ──────────────────────────────────────────────────────────

query = st.text_area(
    "Enter your research question:",
    value=st.session_state.get("query_input", ""),
    height=80,
    placeholder="e.g. What gene targets are associated with Parkinson's disease?",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("🔬 Research", type="primary", use_container_width=True)
with col2:
    st.empty()

# ── Run the agent ───────────────────────────────────────────────────────

if run_btn and query.strip():
    lit_pipe.reset_literature_workspace(LITERATURE_WS)
    # Reset Chroma collection for a clean session (optional legacy use)
    try:
        chroma_client.delete_collection("research_docs")
    except Exception:
        pass
    st.session_state.collection = chroma_client.get_or_create_collection(
        name="research_docs"
    )

    # Run agent with live status
    trace_container = st.expander("📋 Agent Trace (click to expand)", expanded=True)
    report_container = st.container()

    response = None
    with trace_container:
        status = st.status("Researching...", expanded=True)
        start = time.time()

        with status:
            try:
                response = agent.invoke(
                    {"messages": [HumanMessage(query)]},
                    config={"recursion_limit": 25},
                )
            except Exception as exc:
                status.update(label="Research failed", state="error", expanded=True)
                st.error(_explain_failure(exc))
            else:
                elapsed = time.time() - start

                step = 1
                for msg in response["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            arg_preview = ", ".join(
                                f"{k}={v}" for k, v in (tc.get("args") or {}).items()
                            )
                            if len(arg_preview) > 70:
                                arg_preview = arg_preview[:70] + "…"
                            st.write(
                                f"**Step {step}:** 🔧 `{tc['name']}({arg_preview})`"
                            )
                            step += 1
                    elif isinstance(msg, ToolMessage):
                        preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                        st.caption(f"↳ {preview}")

                status.update(
                    label=f"Research complete in {elapsed:.1f}s ({step - 1} tool calls)",
                    state="complete",
                    expanded=False,
                )

    if response is not None:
        with report_container:
            st.divider()
            st.subheader("📄 Research Report")

            def _extract_text(msg) -> str:
                """Pull display text out of a message, whatever shape it takes.

                Different providers return content as a plain string, a list of
                content blocks, or an empty string with the text stashed in a
                reasoning channel. Handle all three.
                """
                raw = getattr(msg, "content", "")
                if isinstance(raw, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in raw
                    )
                else:
                    text = raw or ""
                if not text.strip():
                    extra = getattr(msg, "additional_kwargs", {}) or {}
                    text = extra.get("reasoning_content") or extra.get("reasoning") or ""
                return text

            # The final message is usually the report, but some providers end the
            # run with an empty message — so fall back to the most recent message
            # that actually carries text.
            final_report = ""
            for msg in reversed(response["messages"]):
                # Skip tool outputs; we only want the model's written answer.
                if isinstance(msg, ToolMessage):
                    continue
                if getattr(msg, "tool_calls", None):
                    continue
                candidate = _extract_text(msg)
                if candidate.strip():
                    final_report = candidate
                    break

            if final_report.strip():
                st.markdown(final_report)
            else:
                st.warning(
                    "The agent finished but no report text was returned. Try running "
                    "the query again, pick a different synthesis model, or switch "
                    "`LLM_PROVIDER`."
                )

elif run_btn:
    st.warning("Please enter a research question first.")
