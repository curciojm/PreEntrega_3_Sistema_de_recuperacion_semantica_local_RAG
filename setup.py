# 1 # ejecutar primero si borre la base

from chunking import DocumentProcessor
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

# 1. Cargar todos los .txt de /data
loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

# loader agrega meta data
documentos_crudos = loader.load()
# donde:
# Document(
#     page_content="texto del documento...",
#     metadata={"source": "data/archivo.txt"}
# )
# y cada uno:
# documento.page_content es el texto
# documento.metadata es la meta data

# 2. Crear el procesador
processor = DocumentProcessor()

# 3. Procesar cada documento
chunks = []

def Clean(documentos_crudos):
    documentos_limpios = []
    for documento in documentos_crudos:
        chunks = processor.process_document(documento.page_content)
        for chunk in chunks:
            documentos_limpios.append(
                Document(
                    page_content=chunk,
                    metadata=documento.metadata
                )
            )

    return documentos_limpios

documentos_limpios = Clean(documentos_crudos)

print(type(documentos_limpios[0]))
print(documentos_limpios[0].page_content)
print(documentos_limpios[0].metadata)