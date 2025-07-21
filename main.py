from typing import TypedDict, Annotated, Optional, List, Dict, Any, Union
from langgraph.graph import StateGraph, add_messages, END
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import json
import uuid
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import asyncio
import uvicorn
from langchain_core.documents import Document
import glob
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from pathlib import Path
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI



# === Setup ===
memory = MemorySaver()
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
FAISS_INDEX_DIR = "faiss_indexes"
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed_files"
FOLDERS = {
    "ciq": "ciq_files",
    "standard_ciq": "standard_ciq",
    "template": "templates",
    "log": "logs",
    "master_template": "master_templates"
}

# Ensure directories exist
for dir_path in [FAISS_INDEX_DIR, UPLOAD_DIR, PROCESSED_DIR] + list(FOLDERS.values()):
    os.makedirs(dir_path, exist_ok=True)

# === State Schema ===
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    uploaded_files: Optional[List[Dict[str, Any]]]
    processing_context: Optional[Dict[str, Any]]
    user_intent: Optional[str]

# === Request Models ===
class ChatRequest(BaseModel):
    message: str
    checkpoint_id: Optional[str] = None
    files: Optional[List[str]] = None



# === FAISS Functions ===
def load_ciq_documents(folder: str, doc_type: str = "ciq") -> list[Document]:
    documents = []
    for filepath in glob.glob(os.path.join(folder, "*.xlsx")):
        try:
            filename = os.path.basename(filepath)
            df_dict = pd.read_excel(filepath, sheet_name=None)
            sheet_contents = []
            for sheet_name, df in df_dict.items():
                if not df.empty:
                    sheet_content = f"Sheet: {sheet_name}\n"
                    sheet_content += f"Columns: {', '.join(df.columns.astype(str))}\n"
                    sheet_content += f"Sample data (first 5 rows):\n{df.head().to_string()}\n"
                    sheet_contents.append(sheet_content)
            full_content = f"[{filename}]\n" + "\n".join(sheet_contents)
            documents.append(Document(page_content=full_content, metadata={"type": doc_type, "filename": filename}))
        except Exception as e:
            continue
    return documents

def load_generic_documents(folder: str, doc_type: str) -> list[Document]:
    documents = []
    for pattern in ["*.txt", "*.log", "*.cfg", "*.conf", "*.json"]:
        for filepath in glob.glob(os.path.join(folder, "**", pattern), recursive=True):
            try:
                filename = os.path.basename(filepath)
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                documents.append(Document(page_content=f"[{filename}]\n{content}", metadata={"type": doc_type, "filename": filename}))
            except Exception as e:
                continue
    return documents

def build_faiss_index(docs: list[Document], doc_type: str) -> bool:
    try:
        if not docs:
            return False
        texts = [d.page_content for d in docs]
        embeddings = EMBED_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        with open(os.path.join(FAISS_INDEX_DIR, f"{doc_type}_docs.pkl"), "wb") as f:
            pickle.dump(docs, f)
        faiss.write_index(index, os.path.join(FAISS_INDEX_DIR, f"{doc_type}.index"))
        return True
    except Exception as e:
        return False

def load_faiss_index(doc_type: str):
    try:
        index_path = os.path.join(FAISS_INDEX_DIR, f"{doc_type}.index")
        docs_path = os.path.join(FAISS_INDEX_DIR, f"{doc_type}_docs.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            return None, []
            
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        return index, docs
    except Exception as e:
        return None, []

def _faiss_retrieve(query: str, doc_type: str, top_k: int = 1) -> str:
    try:
        index, docs = load_faiss_index(doc_type)
        if index is None or not docs:
            return f"No {doc_type} documents found or index not built."
        
        vec = EMBED_MODEL.encode([query], convert_to_numpy=True)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        
        dists, inds = index.search(vec, min(top_k, len(docs)))
        
        if inds[0].size > 0 and inds[0][0] < len(docs):
            return docs[inds[0][0]].page_content
        return f"No relevant {doc_type} context found."
    except Exception as e:
        return f"Error retrieving {doc_type} documents: {str(e)}"

# === Retrieval Tools ===
# === Optimized System Prompt for Llama 3.1:8b ===

SYSTEM_PROMPT = """
You are an advanced telecom assistant that intelligently handles 3 task types: general queries, retrieval, and CIQ-based file processing.

You have access to the following specialized tools:

📁 File & CIQ Processing Tools:
- `upload_and_analyze_file`: Analyze structure, format, missing values, and type of uploaded files.
- `convert_column_ciq_to_row_ciq`: Convert column-based CIQ files to row-based format.
- `standardize_ciq_df`: Standardize a CIQ DataFrame’s column names using reference templates.
- `map_with_semantic_and_exceptions`: Perform semantic mapping from unstandardized to standard columns with fallbacks.

📂 Retrieval Tools (via FAISS index):
- `retrieve_logs`: Retrieve the most relevant log content for a query.
- `retrieve_templates`: Fetch NE templates based on a semantic query.
- `retrieve_ciq`: Fetch relevant CIQ content.
- `retrieve_standard_ciq`: Retrieve standardized CIQ files or references.
- `retrieve_master_template`: Fetch master NE templates or mappings.

🔍 INTELLIGENT WORKFLOW:

1. **GENERAL QUERY HANDLING**  
   - If the query is generic (e.g., greetings, "2+2", definitions), answer naturally without using tools.
   - If it matches a tool’s purpose (via docstring or name), use the relevant tool.

2. **RETRIEVAL TASKS (NO FILES)**  
   - Use `retrieve_*` tools when query semantically matches any tool's docstring.
   - If not, respond using your own knowledge.

3. **CIQ FILE-BASED PROCESSING (With Uploaded File)**  
   Based on the query and the uploaded DataFrame:
   - If it's column-based & query mentions standardization →  
     → Call `convert_column_ciq_to_row_ciq` → then `standardize_ciq_df`
   - If it's row-based & query mentions standardization →  
     → Call `standardize_ciq_df` directly
   - If query requests structure insight or metadata →  
     → Call `upload_and_analyze_file`
   - Do **not** standardize or convert if it’s a generic or log/config file.

🚦 Always decide what to do based on:
- The user’s query (primary indicator)
- The file’s structure and contents (if uploaded)

Never ask the user what to do — infer intent and act intelligently. Provide clean, structured output (DataFrame, summary, or tool results).
"""

# === Optimized Tool Descriptions ===
COLUMN_MAP = {
    "cell_id": r"cell_id\d+",
    "cell_name": r"cell_name\d+",
    "pci": r"pci\d+",
    "tac": r"tac\d+"
}
@tool
def retrieve_logs(query: str) -> str:
    """
    Use ONLY when user says:
    - "search the log files"
    - "check logs for errors"
    - "find in uploaded logs"
    - "look in log documents"
    
    DO NOT use for: "what are logs?", "explain logging", "how to read logs"
    """
    return _faiss_retrieve(query, "log")

@tool
def retrieve_templates(query: str) -> str:
    """
    Use ONLY when user says:
    - "search template files"
    - "find NE template"
    - "look in template documents"
    - "check template x1/x2/x3"
    
    DO NOT use for: "what is a template?", "explain templates", "template format"
    """
    return _faiss_retrieve(query, "template")

@tool
def retrieve_ciq(query: str) -> str:
    """
    Use ONLY when user says:
    - "search CIQ files"
    - "find in CIQ data"
    - "look up cell info in CIQ"
    - "check CIQ documents"
    
    DO NOT use for: "what is CIQ?", "explain CIQ format", "CIQ structure"
    """
    return _faiss_retrieve(query, "ciq")

@tool
def retrieve_standard_ciq(query: str) -> str:
    """
    Use ONLY when user says:
    - "search standard CIQ"
    - "find standard format"
    - "look in standardized CIQ"
    - "check reference schema"
    
    DO NOT use for: "what is standard CIQ?", "explain standardization"
    """
    return _faiss_retrieve(query, "standard_ciq")

@tool
def retrieve_master_template(query: str) -> str:
    """
    Use ONLY when user says:
    - "search master template"
    - "find master template"
    - "look in master template files"
    
    DO NOT use for: "what is master template?", "explain master template"
    """
    return _faiss_retrieve(query, "master_template")

@tool
def upload_and_analyze_file(file_content: str, filename: str, user_query: str = "") -> str:
    """
    Use ONLY when user says:
    - "analyze this file"
    - "analyze uploaded file"
    - "show file structure"
    - "inspect this file"
    - "tell me about this file"
    
    DO NOT use for: "how to analyze files?", "file analysis process"
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)  # Use full path
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
        
        # Use the full file_path instead of just filename
        if filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)  # Changed from filename to file_path
        elif filename.endswith('.csv'):
            df = pd.read_csv(file_path)    # Changed from filename to file_path
        else:
            return f"Unsupported file format: {filename}"

        analysis = {
            'filename': filename,
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }

        return json.dumps(analysis, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing file: {str(e)}"
@tool
def convert_column_ciq_to_row_ciq(
    df_data: List[Dict[str, Any]],
    base_cols: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Use ONLY when user says:
    - "convert column CIQ to row"
    - "convert this column-based CIQ"
    - "change column format to row"
    - "transform column CIQ"
    
    DO NOT use for: "how to convert?", "explain conversion", "conversion process"
    """
    column_map = COLUMN_MAP
    df = pd.DataFrame(df_data)
    if base_cols is None:
        base_cols = [col for col in df.columns if not any(re.match(pat, col, re.IGNORECASE) for pat in column_map.values())]

    column_matches = {}
    for logical_name, pattern in column_map.items():
        for col in df.columns:
            if re.fullmatch(pattern, col, flags=re.IGNORECASE):
                idx_match = re.search(r'\d+', col)
                idx = idx_match.group() if idx_match else "0"
                operator_match = re.match(r'(vf|tef)', col, re.IGNORECASE)
                op = operator_match.group(1).lower() if operator_match else ""
                column_matches.setdefault((op, idx), {})[logical_name] = col

    all_rows = []
    for _, row in df.iterrows():
        base_data = {col: row[col] for col in base_cols}
        for (op, idx), logical_cols in column_matches.items():
            new_row = base_data.copy()
            for logical_name, actual_col in logical_cols.items():
                new_row[logical_name] = row.get(actual_col, "")
            new_row["operator"] = op
            new_row["cell_index"] = idx
            all_rows.append(new_row)

    return pd.DataFrame(all_rows).to_dict(orient="records")

@tool
def standardize_ciq_df(
    unstandard_df_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Use ONLY when user says:
    - "standardize this CIQ"
    - "standardize CIQ file"
    - "apply standard format"
    - "convert to standard schema"
    
    DO NOT use for: "what is standardization?", "explain standardization", "how to standardize"
    """
    unstandard_df = pd.DataFrame(unstandard_df_data)

    # Load default standard CIQ from x.xlsx
    default_path = os.path.join(FOLDERS["standard_ciq"], "x.xlsx")
    if not os.path.exists(default_path):
        return {"error": "Default standard CIQ file 'x.xlsx' not found in standard_ciq folder."}

    try:
        standard_df = pd.read_excel(default_path)
        
    except Exception as e:
        return {"error": f"Failed to read default standard CIQ: {e}"}

    exception_map = {
        'enodeb_id': 'eNodeB_ID',
        'enodeb_name': 'eNodeB_Name',
        'cell_id': 'Cell_ID',
        'cell_name': 'Cell_Name',
        'pci': 'PCI',
        'tac': 'TAC',
    }

    unstandard_cols = list(unstandard_df.columns)
    standard_cols = list(standard_df.columns)
    print(unstandard_cols)
    print(standard_cols)
    mapping = map_with_semantic_and_exceptions(
        unstandard_cols, standard_cols, threshold=0.75, exception_map=exception_map
    )
    print("Mapping:", mapping)
    standardized_data = {}
    for std_col in standard_cols:
        matched_cols = [col for col, mapped_to in mapping.items() if mapped_to == std_col]
        if matched_cols:
            standardized_data[std_col] = unstandard_df[matched_cols[0]]
        else:
            standardized_data[std_col] = [pd.NA] * len(unstandard_df)

    unmatched_cols = [col for col, mapped_to in mapping.items() if mapped_to is None]
     
    return {
        "standardized_data": pd.DataFrame(standardized_data).to_dict(orient="records"),
        "unmatched_columns": unmatched_cols
    }


def map_with_semantic_and_exceptions(unstandard_cols, standard_cols, threshold=0.75, exception_map=None):
    """
    INTERNAL TOOL - Do not call directly from user input.
    Used internally by standardize_ciq_df for column mapping.
    """
    exception_map  = {
    # ENODEB identifiers
   "enodeb_id":"neid",
   "cell_id":"cellid",
   "cell_name":"cellname",
    
}

    un_embeds = EMBED_MODEL.encode(unstandard_cols, convert_to_numpy=True)
    std_embeds = EMBED_MODEL.encode(standard_cols, convert_to_numpy=True)

    def normalize(v):
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-10, None)

    un_embeds = normalize(un_embeds)
    std_embeds = normalize(std_embeds)

    sim_matrix = np.dot(un_embeds, std_embeds.T)
    mapping = {}

    for i, un_col in enumerate(unstandard_cols):
        best_idx = np.argmax(sim_matrix[i])
        best_score = sim_matrix[i][best_idx]

        if best_score >= threshold:
            mapping[un_col] = standard_cols[best_idx]
        elif un_col.lower() in exception_map:
            mapping[un_col] = exception_map[un_col.lower()]
        else:
            mapping[un_col] = None

    return mapping


# === Tools List ===
TOOLS = [
    retrieve_logs, retrieve_templates, retrieve_ciq, retrieve_standard_ciq, retrieve_master_template,
    upload_and_analyze_file, convert_column_ciq_to_row_ciq, standardize_ciq_df,map_with_semantic_and_exceptions
]

# === LLM Setup ===
# llm = ChatOllama(model="mistral-nemo:latest", temperature=0.3)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key="AIzaSyDvY2JqtHJLibsONSMBGR-ENTnBytxZ3PQ"
)
llm_with_tools = llm.bind_tools(tools=TOOLS,tool_choice="auto")


# === Graph Nodes ===
async def llm_node(state: AgentState) -> AgentState:
    messages = state["messages"]

    if not any(isinstance(msg, AIMessage) for msg in messages):
        messages = [HumanMessage(content=SYSTEM_PROMPT)] + messages

    if state.get("uploaded_files"):
        file_context = "Currently uploaded files:\n"
        for file_info in state["uploaded_files"]:
            file_context += f"- {file_info['filename']} ({file_info.get('type', 'unknown')})\n"
        messages.append(HumanMessage(content=file_context))

    try:
        
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"⚠️ LLM error: {str(e)}")]}

def tool_router(state: AgentState) -> str:
    try:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return END
    except Exception as e:
        return END

# === Graph Creation ===
def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("llm", llm_node)
    workflow.add_node("tool_node", ToolNode(tools=TOOLS))
    workflow.set_entry_point("llm")
    workflow.add_conditional_edges("llm", tool_router)
    workflow.add_edge("tool_node", "llm")
    return workflow.compile(checkpointer=memory)

graph = create_graph()

# === FastAPI App ===
app = FastAPI(title="Telecom Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# === File Management ===
def save_uploaded_file(file: UploadFile) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        content = file.file.read()
        buffer.write(content)
        buffer.flush()  # Ensure it's fully written
        os.fsync(buffer.fileno())
    return file_path

def get_file_info(file_path: str) -> Dict[str, Any]:
    try:
        stat = os.stat(file_path)
        return {
            'filename': os.path.basename(file_path),
            'path': file_path,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': Path(file_path).suffix
        }
    except Exception as e:
        return {}

# === Streaming ===
def serialize_message_content(message):
    if hasattr(message, 'content'):
        return message.content
    elif isinstance(message, dict) and 'content' in message:
        return message['content']
    else:
        return str(message)


# === Update generate_chat_stream to remove detect_file_intent ===
async def generate_chat_stream(message: str, checkpoint_id: Optional[str] = None, uploaded_files: Optional[List[str]] = None):
    try:
        is_new = checkpoint_id is None
        thread_id = str(uuid.uuid4()) if is_new else checkpoint_id
        config = {"configurable": {"thread_id": thread_id}}

        if is_new:
            yield f"data: {json.dumps({'type': 'checkpoint', 'checkpoint_id': thread_id})}\n\n"

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "uploaded_files": [],
            "processing_context": {},
            "user_intent": None
        }

        if uploaded_files:
            for file_path in uploaded_files:
                file_info = get_file_info(file_path)
                if file_info:
                    initial_state["uploaded_files"].append(file_info)

                    try:
                        if file_path.endswith(('.xlsx', '.csv')):
                            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

                            # File analysis
                            analysis = {
                                'filename': file_info['filename'],
                                'shape': df.shape,
                                'columns': list(df.columns),
                                'data_types': df.dtypes.astype(str).to_dict(),
                                'missing_values': df.isnull().sum().to_dict()
                            }
                            yield f"data: {json.dumps({'type': 'file_analysis', 'file_info': analysis})}\n\n"

                            # 👇 Inject preview of file into LLM message context
                            df_preview = df.head(5).to_dict(orient="records")
                            preview_text = f"Preview of uploaded CIQ file `{file_info['filename']}`:\n{json.dumps(df_preview, indent=2)}"
                            initial_state["messages"].append(HumanMessage(content=preview_text))

                            # 👇 Store full data in context (optional: in case tool needs access)
                            initial_state["processing_context"]["ciq_data"] = df.to_dict(orient="records")

                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'⚠️ Failed to process file: {str(e)}'})}\n\n"

        async for event in graph.astream_events(initial_state, version="v2", config=config):
            event_type = event.get("event", "")

            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", {})
                content = serialize_message_content(chunk)
                if content:
                    safe_content = json.dumps(content)
                    yield f"data: {json.dumps({'type': 'content', 'content': safe_content})}\n\n"

            elif event_type == "on_tool_end":
                tool_name = event.get("name", "")
                output = event.get("data", {}).get("output", "")
                yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'output': str(output)})}\n\n"

        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'⚠️ Stream crashed: {str(e)}'})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

# === Main Chat Endpoint ===
@app.post("/chat")
async def chat(
    message: str = Form(...),
    checkpoint_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """Main chat endpoint that handles both messages and optional file uploads."""
    try:
        uploaded_files = []
        
        if files:
            for file in files:
                # Validate file type
                allowed_extensions = ['.xlsx', '.csv', '.txt', '.log', '.json', '.cfg', '.conf']
                if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
                
                file_path = save_uploaded_file(file)
                uploaded_files.append(file_path)
        
        return StreamingResponse(
            generate_chat_stream(message, checkpoint_id, uploaded_files),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Alternative JSON endpoint ===
@app.post("/chat_json")
async def chat_json(request: ChatRequest):
    """Alternative JSON endpoint for programmatic access."""
    try:
        uploaded_files = []
        if request.files:
            for filename in request.files:
                file_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.exists(file_path):
                    uploaded_files.append(file_path)
        
        return StreamingResponse(
            generate_chat_stream(request.message, request.checkpoint_id, uploaded_files),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Root endpoint ===
@app.get("/")
async def root():
    return {"message": "Telecom Assistant API", "version": "1.0.0"}

# === Preprocessing ===
def preprocess_all():
    for doc_type in FOLDERS:
        folder = FOLDERS[doc_type]
        if doc_type in ["ciq", "standard_ciq"]:
            docs = load_ciq_documents(folder, doc_type)
        else:
            docs = load_generic_documents(folder, doc_type)
        build_faiss_index(docs, doc_type)


# === Main Entry Point ===
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )