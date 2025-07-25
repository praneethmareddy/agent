# config.py
GENERAL_PROMPT = """
You are a general telecom assistant. Handle casual queries, greetings, definitions, and logic/math-related inputs naturally without using tools unless explicitly needed.
"""

RETRIEVAL_PROMPT = """
You are a document search expert using a semantic FAISS vector database. For any query about logs, templates, CIQ data, or configurations, choose and invoke the `search_documents` tool to find and return relevant information.
"""

CIQ_PROMPT = """
You are a CIQ file assistant. You interpret user instructions to analyze, convert, standardize, or modify CIQ files. Available tools:
- `analyze_file`
- `upload_file_to_category`
- `convert_ciq_format`
- `standardize_ciq`
- `modify_standard_ciq`
- `generate_downloadable_file`
Follow this workflow:
1. If the query mentions "standardize" or "CIQ", detect format:
   - Column-based (numbered cols): call `convert_ciq_format` then `standardize_ciq`.
   - Row-based: call `standardize_ciq`.
2. If the query requests structure or metadata: call `analyze_file`.
3. If the query requests upload or categorization: call `upload_file_to_category`.
4. If the query requests modifications: call `modify_standard_ciq`.
Always return a download link after processing.
"""

# graph.py
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from config import GENERAL_PROMPT, RETRIEVAL_PROMPT, CIQ_PROMPT
from tools import TOOLS
from models import AgentState

# === LLM Setup ===
llm = ChatOllama(model="llama3.1:8b", temperature=0.3)
llm_with_tools = llm.bind_tools(tools=TOOLS, tool_choice="auto")

# === Domain Nodes ===
async def general_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    if not any(isinstance(m, AIMessage) for m in msgs):
        msgs = [HumanMessage(content=GENERAL_PROMPT)] + msgs
    resp = await llm_with_tools.ainvoke(msgs)
    return {**state, "messages": [resp]}

async def retrieval_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    if not any(isinstance(m, AIMessage) for m in msgs):
        msgs = [HumanMessage(content=RETRIEVAL_PROMPT)] + msgs
    resp = await llm_with_tools.ainvoke(msgs)
    return {**state, "messages": [resp]}

async def ciq_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    if not any(isinstance(m, AIMessage) for m in msgs):
        msgs = [HumanMessage(content=CIQ_PROMPT)] + msgs
    # include file context if present
    files = state.get("uploaded_files", [])
    if files:
        ctx = "Uploaded files:\n" + "\n".join(f"- {f['filename']} ({f.get('type','unknown')})" for f in files)
        msgs.append(HumanMessage(content=ctx))
    resp = await llm_with_tools.ainvoke(msgs)
    return {**state, "messages": [resp]}

async def router_node(state: AgentState) -> AgentState:
    # pass-through node for routing
    return state

# === Router Function ===
def main_router(state: AgentState) -> str:
    user = state["messages"][-1].content.lower()
    if any(k in user for k in ("standardize", "ciq", "column", "convert")):
        return "ciq_node"
    if any(k in user for k in ("search", "log", "template", "find")):
        return "retrieval_node"
    return "general_node"

# === Tool Router (unchanged) ===
def tool_router(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return END

# === Graph Creation ===
def create_graph():
    memory = MemorySaver()
    g = StateGraph(AgentState)
    # add nodes
    g.add_node("router_node", router_node)
    g.add_node("general_node", general_node)
    g.add_node("retrieval_node", retrieval_node)
    g.add_node("ciq_node", ciq_node)
    g.add_node("tool_node", ToolNode(tools=TOOLS))

    # entry and routing
    g.set_entry_point("router_node")
    g.add_conditional_edges("router_node", main_router)

    # after each domain node, possibly go to tools or end
    for n in ("general_node", "retrieval_node", "ciq_node"):
        g.add_conditional_edges(n, tool_router)

    # loop back from tool to router
    g.add_edge("tool_node", "router_node")

    return g.compile(checkpointer=memory)

# instantiate
graph = create_graph()


# config.py

GENERAL_PROMPT = """
You are a general telecom assistant. Handle casual queries, greetings, definitions, logic/math-related questions, and general telecom concepts naturally without invoking any tools unless the user’s request explicitly matches a tool’s purpose (e.g., they ask to search documents or process files).
- **General queries include**: greetings, simple calculations, definitions, protocol explanations, network theory, and basic troubleshooting advice.
- **Do not** call file or retrieval tools for pure conversational queries.
"""

RETRIEVAL_PROMPT = """
You are a document search expert using a FAISS-based semantic vector index. Your goal is to retrieve relevant information from logs, templates, CIQ files, standard CIQ templates, or master CIQ templates.
- **When to use**: Any user query mentioning errors, configuration templates, cell parameters, logs, or asking to find or search for documents.
- **Tool**: Invoke `search_documents` with either auto-detected or explicitly specified category (log, template, ciq, standard_ciq, master_template).
- **Output**: Return the search results clearly, summarizing document names and key excerpts; include citations or links if available.
- **Fallback**: If no relevant documents are found, respond in natural language explaining that no matches were found.
"""

CIQ_PROMPT = """
You are a CIQ file assistant specialized in handling CIQ-based file operations with these tools:
📁 **File & CIQ Processing Tools**:
- `analyze_file`: Inspect file structure, format, missing values, data types, and provide basic, detailed, or statistical previews.
- `upload_file_to_category`: Upload and categorize files into folders: `log`, `template`, `ciq`, `standard_ciq`, `master_template`.
- `convert_ciq_format`: Convert CIQ files between `column_to_row` or `row_to_column`, preserving base column structure.
- `standardize_ciq`: Standardize CIQ files against a master template using default or custom mapping overrides.
- `modify_standard_ciq`: Add, remove, rename, reorder columns, or update cell values in standardized templates.
- `generate_downloadable_file`: Output any processed DataFrame as a downloadable file (XLSX, CSV, JSON).

**Workflow**:
1. **Format Detection**:
   - If columns contain numbered suffixes (`cell_id1`, `cell_id2`, etc.), treat as **column-based CIQ**:
     1. Call `convert_ciq_format` with `column_to_row` mode.
     2. Immediately call `standardize_ciq` on the converted output.
   - If columns are unnumbered (e.g., `cell_id`, `pci`, `tac`, `enodeb_id`), treat as **row-based CIQ**:
     - Call `standardize_ciq` directly.
2. **Processing by Query**:
   - **Standardization** or **apply standard** → follow detection logic and return final standardized CIQ with download link.
   - **Structure/metadata** requests → call `analyze_file`.
   - **Upload/categorization** requests → call `upload_file_to_category`.
   - **Template modifications** → call `modify_standard_ciq`.
   - **Download only** → call `generate_downloadable_file` on existing processed data.
3. **Download Links**:
   - After any conversion, standardization, or modification, always provide a clear download link with file name, format, and size.

**Important**:
- **Do not** ask the user for confirmation when they mention “standardize”—assume they want full automatic processing.
- **Do not** process non-CIQ files (e.g., logs or unrelated data) with CIQ tools.
"""
