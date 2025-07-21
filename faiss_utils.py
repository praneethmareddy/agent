import os
import glob
import pickle
import pandas as pd
import numpy as np
from langchain_core.documents import Document
from config import EMBED_MODEL, FAISS_INDEX_DIR, FOLDERS
import faiss

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

def preprocess_all():
    for doc_type in FOLDERS:
        folder = FOLDERS[doc_type]
        if doc_type in ["ciq", "standard_ciq"]:
            docs = load_ciq_documents(folder, doc_type)
        else:
            docs = load_generic_documents(folder, doc_type)
        build_faiss_index(docs, doc_type) 