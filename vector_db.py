import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from logging_config import logger
from setup import documentos_procesados


# Creación de la base vectorial
PERSIST_DIR = "./vectors_db"
COLLECTION_NAME = "Statistics_and_methodolgy_texts"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ya_existe_indice = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

if ya_existe_indice:
    logger.info("Índice existente detectado — cargando sin reindexar")

    vectors_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        # Se utiliza distancia coseno para medir la similitud entre embeddings.
        collection_metadata={"hnsw:space": "cosine"},
    )

else:
    logger.info("No hay índice previo — indexando documentos por primera vez")

    vectors_db = Chroma.from_documents(
        documents=documentos_procesados,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )

cantidad_documentos = vectors_db._collection.count()

logger.info(f"Creada la base con: {cantidad_documentos} documentos")