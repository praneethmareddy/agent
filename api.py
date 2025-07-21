import json
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from models import ChatRequest
from config import UPLOAD_DIR
from graph import graph
import os
from langchain_core.messages import HumanMessage
from config import PROCESSED_DIR

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
from fastapi.responses import FileResponse

@app.get("/download/{filename}")
def download_file(filename: str):
    filepath = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(path=filepath, filename=filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    raise HTTPException(status_code=404, detail="File not found")

# === Root endpoint ===
@app.get("/")
async def root():
    return {"message": "Telecom Assistant API", "version": "1.0.0"} 

