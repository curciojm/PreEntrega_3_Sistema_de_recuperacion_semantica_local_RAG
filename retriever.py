from vector_db import vectors_db

# Retriever
retriever = vectors_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)