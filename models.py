from typing import TypedDict, Annotated, Optional, List, Dict, Any
from pydantic import BaseModel
from langgraph.graph import add_messages

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