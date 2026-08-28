import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from setup import documentos_procesados


# Creación de la base vectorial
PERSIST_DIR = "./vectors_db"
COLLECTION_NAME = "Statistics_and_methodolgy_texts"


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


ya_existe_indice = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0


if ya_existe_indice:
    print("\n♻️ Índice existente detectado — cargando sin reindexar")
    vectors_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
else:
    print("\n🆕 No hay índice previo — indexando documentos por primera vez")
    vectors_db = Chroma.from_documents(
        documents=documentos_procesados,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )

print(f"\n📦 Documentos en la colección: {vectors_db._collection.count()}\n")