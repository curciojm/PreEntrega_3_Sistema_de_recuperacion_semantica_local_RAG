

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

resultado = embeddings.embed_documents([
    "Este es un texto de prueba.",
    "Este es otro texto."
])

print(type(resultado))
print(len(resultado))
print(len(resultado[0]))