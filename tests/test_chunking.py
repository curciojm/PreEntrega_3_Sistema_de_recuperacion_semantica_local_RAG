from chunking import DocumentProcessor


def test_process_document_crea_chunks():
    processor = DocumentProcessor()

    texto = """
    La regresión es una técnica estadística utilizada para realizar predicciones.

    La correlación estudia la relación entre dos variables.

    Ambas técnicas utilizan datos obtenidos de los mismos sujetos.
    """

    chunks = processor.process_document(texto)

    # Debe devolver una lista
    assert isinstance(chunks, list)

    # Debe haber generado al menos un chunk
    assert len(chunks) > 0

    # Cada chunk debe ser un string
    assert all(isinstance(chunk, str) for chunk in chunks)

    # Ningún chunk debe estar vacío
    assert all(chunk.strip() for chunk in chunks)
