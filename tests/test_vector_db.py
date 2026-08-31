from vector_db import vectors_db


def test_vector_db_tiene_documentos():
    cantidad = vectors_db._collection.count()

    assert cantidad > 0


def test_retriever_devuelve_documentos():
    retriever = vectors_db.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    resultados = retriever.invoke("¿Qué es la regresión?")

    # Debe devolver resultados
    assert isinstance(resultados, list)

    # Debe recuperar al menos un fragmento
    assert len(resultados) > 0

    # Cada resultado debe ser un Document
    assert all(hasattr(doc, "page_content") for doc in resultados)
