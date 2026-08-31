from langchain_core.documents import Document

from chain import formatear_documentos


def test_formatear_documentos():
    documentos = [
        Document(
            page_content="La regresión permite realizar predicciones.",
            metadata={"source": "regresion.txt"},
        ),
        Document(
            page_content="La correlación estudia la relación entre variables.",
            metadata={"source": "correlacion.txt"},
        ),
    ]

    resultado = formatear_documentos(documentos)

    assert "La regresión permite realizar predicciones." in resultado
    assert "La correlación estudia la relación entre variables." in resultado
    assert "regresion.txt" in resultado
    assert "correlacion.txt" in resultado
