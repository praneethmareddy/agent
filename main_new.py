import uvicorn
from api import app
from faiss_utils import preprocess_all

# === Main Entry Point ===
if __name__ == "__main__":
    # Preprocess all documents to build FAISS indexes
    print("Building FAISS indexes...")
    preprocess_all()
    print("FAISS indexes built successfully!")
    
    # Start the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 