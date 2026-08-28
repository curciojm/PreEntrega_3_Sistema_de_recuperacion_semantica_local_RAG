# 3 # ejecutar tercero si borre la base

from vector_db import vectors_db

# vectors_db es la base hecha con hugging, por lo que su modelo es el que tambien es usado para hacer el retriever
retriever = vectors_db.as_retriever(
    #search type dice la accion a realizar. lo mas similar, tambien puede ser: "mmr" que es un equilibrio entre query y diversidad
    search_type="similarity",
    # cantidad de chunks, recorda que los documentos se convierten en chunks
    search_kwargs={"k": 4}
)

# query = "¿Qué es la regresión?"

# # si quiero score es:
# resultados = vectors_db.similarity_search_with_score(
#     query,
#     k=4
# )

# for i, (doc, score) in enumerate(retriever):
#     print(f"\n--- Resultado {i+1} ---")
#     print(f"Score: {score}")
#     print(f"Metadata: {doc.metadata}")
#     print(doc.page_content[:300])

# Prueba rápida del retriever solo
resultados_prueba = retriever.invoke("¿Que es la regresion?")
for i, doc in enumerate(resultados_prueba, 1):
    print(f"--- Fragmento {i} (fuente: {doc.metadata['source']}) ---")
    print(doc.page_content[:150], "...\n")