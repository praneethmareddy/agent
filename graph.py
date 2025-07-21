import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from models import AgentState
from config import SYSTEM_PROMPT
from tools import TOOLS
from langchain_ollama import ChatOllama
# === LLM Setup ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key="AIzaSyDvY2JqtHJLibsONSMBGR-ENTnBytxZ3PQ"
)
# llm = ChatOllama(model="mistral-nemo:latest", temperature=0.3) 
# llm = ChatOllama(model="llama3.1:8b", temperature=0.3) 
llm_with_tools = llm.bind_tools(tools=TOOLS, tool_choice="auto")

# === Graph Nodes ===
async def llm_node(state: AgentState) -> AgentState:
    messages = state["messages"]

    if not any(isinstance(msg, AIMessage) for msg in messages):
        messages = [HumanMessage(content=SYSTEM_PROMPT)] + messages

    uploaded_files = state.get("uploaded_files") or []
    if uploaded_files:
        file_context = "Currently uploaded files:\n"
        for file_info in uploaded_files:
            file_context += f"- {file_info['filename']} ({file_info.get('type', 'unknown')})\n"
        messages.append(HumanMessage(content=file_context))

    try:
        response = await llm_with_tools.ainvoke(messages)
        return {
            "messages": [response],
            "uploaded_files": state.get("uploaded_files", []),
            "processing_context": state.get("processing_context", {}),
            "user_intent": state.get("user_intent")
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"⚠️ LLM error: {str(e)}")],
            "uploaded_files": state.get("uploaded_files", []),
            "processing_context": state.get("processing_context", {}),
            "user_intent": state.get("user_intent")
        }

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
    memory = MemorySaver()
    workflow = StateGraph(AgentState)
    workflow.add_node("llm", llm_node)
    workflow.add_node("tool_node", ToolNode(tools=TOOLS))
    workflow.set_entry_point("llm")
    workflow.add_conditional_edges("llm", tool_router)
    workflow.add_edge("tool_node", "llm")
    return workflow.compile(checkpointer=memory)

# Create the graph instance
graph = create_graph()

# Optional: Visualize the graph (uncomment if needed)
