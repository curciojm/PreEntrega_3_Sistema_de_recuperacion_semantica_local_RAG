# 2 # ejecutar segundo si borre la base

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from setup import documentos_limpios

PERSIST_DIR = "./vectors_db"
COLLECTION_NAME = "Statistics_and_methodolgy_texts"

# el modelo embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# si no existe el directorio y no tiene nada
ya_existe_indice = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

# si se cumple lo de arriba lo crea
if ya_existe_indice:
    print("♻️ Índice existente detectado — cargando sin reindexar")
    vectors_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        # distancia coseno NO SIMILITUD
        collection_metadata={"hnsw:space": "cosine"}
    )
else:
    print("🆕 No hay índice previo — indexando documentos por primera vez")
    vectors_db = Chroma.from_documents(
        documents=documentos_limpios,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        # distancia coseno NO SIMILITUD
        collection_metadata={"hnsw:space": "cosine"}
    )

print(f"📦 Documentos en la colección: {vectors_db._collection.count()}")