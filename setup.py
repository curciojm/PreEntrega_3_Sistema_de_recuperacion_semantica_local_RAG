from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from chunking import DocumentProcessor

# Ingesta de los datos y procesamiento
loader = DirectoryLoader(
    "data", glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
)

documentos_crudos = loader.load()
processor = DocumentProcessor()


def procesar_documentos(documentos_crudos):
    documentos_procesados = []
    for documento in documentos_crudos:
        chunks = processor.process_document(documento.page_content)
        for chunk in chunks:
            documentos_procesados.append(
                Document(page_content=chunk, metadata=documento.metadata)
            )

    return documentos_procesados


documentos_procesados = procesar_documentos(documentos_crudos)
