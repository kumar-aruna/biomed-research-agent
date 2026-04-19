"""
BioEvidence Research Agent — Streamlit Web App

A multi-tool AI agent that queries real biomedical databases (OpenTargets,
PubMed, UniProt) and writes structured research reports with citations.

Run with:  streamlit run app.py
"""

import os
import time
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
        )
        # Separate, more capable model for synthesis (1 call/query vs many for planning)
        llm_synth = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_SYNTH_MODEL", "gemini-3-flash-preview"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_retries=2,
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


@tool
def protein_info(uniprot_id: str) -> str:
    """Get protein details from UniProt — name, gene, organism, and function.

    Args:
        uniprot_id: UniProt accession ID (e.g. "P05067")
    """
    result = tu.run_one_function({
        "name": "UniProt_get_entry_by_accession",
        "arguments": {"accession": uniprot_id},
    })
    if isinstance(result, dict):
        summary = {
            "protein": result.get("uniProtkbId", ""),
            "accession": result.get("primaryAccession", ""),
            "gene": result.get("genes", [{}])[0].get("geneName", {}).get("value", ""),
            "organism": result.get("organism", {}).get("scientificName", ""),
            "function": "",
        }
        for comment in result.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    summary["function"] = texts[0].get("value", "")[:500]
                break
        return str(summary)
    return str(result)


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

Gather evidence in two tracks:

TRACK A — Structured data
1. disease_lookup → disease ID
2. disease_evidence → gene targets
3. pubmed_search → quick paper list
4. protein_info → key proteins
5. web_search → trials / news

TRACK B — Literature (BM25 file pipeline; run in order when used)
6. literature_collect → returns catalog_path
7. literature_build_chunks(catalog_path=<path from 6>)
8. literature_bm25_search(chunks_path=<path from 7>, search_query=<keywords>)
   → saves bm25_evidence.txt; first line of the tool output is its absolute path.
9. literature_synthesize(research_question=..., evidence_file_path=<first line from step 8>)

Rules:
- Finish 6→7→8→9 in order when you start TRACK B.
- For step 9 pass the file path only (do not paste full BM25 text into the tool).
- Stop calling tools when you have enough for a solid report.
"""

SYNTHESIZER_PROMPT = """\
You are a biomedical research report writer. Based on ALL the research data \
gathered in this conversation, write a clear, well-structured report.

Format:

## Disease Overview
Brief description of the disease or topic.

## Top Gene Targets
| Gene Symbol | Ensembl ID | Association Score |
|-------------|-----------|-------------------|
(Fill from the evidence data. Include up to 10 genes.)

## Key Literature Findings
Summarize publications and any **literature_synthesize** output. Use [PMID:…] where available.

## Additional Insights
Any notable findings from web search or cross-referencing.

## Summary
A concise 2-3 sentence conclusion answering the user's original question.

Rules:
- ONLY include information that was actually retrieved — never make up data
- If a section has no data, say "No data retrieved for this section"
- Be precise with gene names, scores, and citations
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

    with trace_container:
        status = st.status("Researching...", expanded=True)
        start = time.time()

        with status:
            response = agent.invoke(
                {"messages": [HumanMessage(query)]},
                config={"recursion_limit": 25},
            )
            elapsed = time.time() - start

            step = 1
            for msg in response["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    called = [tc["name"] for tc in msg.tool_calls]
                    st.write(f"**Step {step}:** Planner called → `{', '.join(called)}`")
                    step += 1
                elif isinstance(msg, ToolMessage):
                    preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                    st.caption(f"↳ {preview}")

            status.update(
                label=f"Research complete in {elapsed:.1f}s ({step - 1} tool calls)",
                state="complete",
                expanded=False,
            )

    with report_container:
        st.divider()
        st.subheader("📄 Research Report")
        raw_content = response["messages"][-1].content
        if isinstance(raw_content, list):
            final_report = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        else:
            final_report = raw_content or ""
        st.markdown(final_report)

elif run_btn:
    st.warning("Please enter a research question first.")
