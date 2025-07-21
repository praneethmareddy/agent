# Telecom Assistant API - Modularized Version

This is a modularized version of the Telecom Assistant API, broken down into logical components for better maintainability and organization.

## Project Structure

```
agent/
├── config.py           # Configuration and constants
├── models.py           # Data models and schemas
├── faiss_utils.py      # FAISS indexing and retrieval functions
├── tools.py            # All tool functions
├── graph.py            # LangGraph workflow setup
├── api.py              # FastAPI endpoints
├── main_new.py         # Entry point (new modularized version)
├── main.py             # Original monolithic file
├── requirements.txt    # Python dependencies
└── venv/               # Virtual environment
```

## Module Descriptions

### `config.py`
- Contains all configuration constants
- System prompt for the LLM
- Directory paths and folder mappings
- Column mapping for CIQ conversion

### `models.py`
- Data models and schemas
- `AgentState` TypedDict for LangGraph state
- `ChatRequest` Pydantic model for API requests

### `faiss_utils.py`
- FAISS document loading functions
- Index building and retrieval utilities
- Document preprocessing functions

### `tools.py`
- All LangChain tool functions
- Retrieval tools for different document types
- CIQ processing and standardization tools
- File analysis tools

### `graph.py`
- LangGraph workflow setup
- LLM configuration and tool binding
- Graph nodes and routing logic

### `api.py`
- FastAPI application setup
- Chat endpoints with streaming support
- File upload and management functions

### `main_new.py`
- Entry point for the modularized application
- Initializes FAISS indexes
- Starts the FastAPI server

## Setup and Installation

1. **Create Virtual Environment** (already done):
   ```bash
   python -m venv venv
   ```

2. **Activate Virtual Environment**:
   ```bash
   # Windows
   .\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies** (already done):
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Using the Modularized Version (Recommended)
```bash
python main_new.py
```

### Using the Original Monolithic Version
```bash
python main.py
```

Both versions will:
1. Build FAISS indexes from documents in the configured folders
2. Start the FastAPI server on `http://0.0.0.0:8000`

## API Endpoints

- `GET /` - Root endpoint with API info
- `POST /chat` - Main chat endpoint with file upload support
- `POST /chat_json` - Alternative JSON endpoint for programmatic access

## Key Features

- **Modular Architecture**: Clean separation of concerns
- **FAISS Integration**: Semantic search across documents
- **File Processing**: Support for CIQ files, logs, templates
- **Streaming Responses**: Real-time chat responses
- **Tool Integration**: LangChain tools for specialized tasks

## Dependencies

The application uses several key libraries:
- `fastapi` - Web framework
- `langchain` - LLM framework
- `langgraph` - Workflow orchestration
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embeddings
- `pandas` - Data manipulation
- `uvicorn` - ASGI server

## Notes

- The linter errors you see are due to the virtual environment not being activated in the IDE
- All imports will resolve correctly when running the application in the activated virtual environment
- The modularized version maintains the exact same functionality as the original monolithic version 