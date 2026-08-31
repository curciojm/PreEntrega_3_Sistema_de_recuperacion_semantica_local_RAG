import pytest
from langchain_core.documents import Document

from main import get_rag_response
from schemas import RAGResponse, RespuestaLLM


@pytest.mark.asyncio
async def test_get_rag_response(monkeypatch):

    # simula que encuentra dos documentos
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

    class MockRetriever:
        async def ainvoke(self, query):
            return documentos

    class MockChain:
        async def ainvoke(self, datos):
            return RespuestaLLM(respuesta="La regresión permite realizar predicciones.")

    # incluido en pytest
    # set attr reemplaza temporalmente retriever por MockRetriever
    # main.retriever ruta del atributo original que se simula
    monkeypatch.setattr("main.retriever", MockRetriever())
    monkeypatch.setattr("main.chain", MockChain())

    resultado = await get_rag_response("¿Qué es la regresión?")

    assert isinstance(resultado, RAGResponse)
    assert resultado.respuesta
    assert resultado.respuesta == "La regresión permite realizar predicciones."
    assert resultado.fragmentos_recuperados > 0
    assert resultado.fragmentos_recuperados == 2
    assert len(resultado.fuentes) > 0
    assert "regresion.txt" in resultado.fuentes
    assert "correlacion.txt" in resultado.fuentes
