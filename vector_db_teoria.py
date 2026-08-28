import os
import chromadb
from langchain_chroma import Chroma
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from setup import chunks

class VectorMemoryManager:
    def __init__(self, persist_path: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    def upsert_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str,Any]]):
        try:
            self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
            )
            print(f"Éxito: {len(ids)} documentos procesados.")
        except Exception as e:
            print(f"Error en upsert: {e}")


PERSIST_DIR = "./vectors_db"
COLLECTION_NAME = "Statistics_and_methodolgy_texts"

# ⚠️ Mismo modelo de embeddings para indexar Y para consultar (evita el error #1 de la consigna)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

ya_existe_indice = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

if ya_existe_indice:
    print("♻️ Índice existente detectado — cargando sin reindexar")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
else:
    print("🆕 No hay índice previo — indexando documentos por primera vez")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

print(f"📦 Documentos en la colección: {vectorstore._collection.count()}")