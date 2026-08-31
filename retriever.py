from vector_db import vectors_db

# Se recuperan los 4 fragmentos más similares a la consulta.
retriever = vectors_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
