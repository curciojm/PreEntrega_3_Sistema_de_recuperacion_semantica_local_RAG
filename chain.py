from retriever import retriever
from schemas import RespuestaLLM, RAGResponse
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt_config import prompt

parser_llm = PydanticOutputParser(pydantic_object=RespuestaLLM)

# Uso el modelo mas viejo disponible para mayor disponibilidad
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# asi queda la fuente arriba
def formatear_documentos(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Fuente: {d.metadata.get('source', 'desconocida')}]\n{d.page_content}"
        for d in docs
    )

# Cadena LCEL: prompt -> llm -> parser (recibe contexto y pregunta ya armados)
chain = prompt | llm | parser_llm

# RETRIEVER enfoque del profe
async def get_rag_response(query: str) -> RAGResponse:
    # a. Búsqueda de similitud en ChromaDB
    docs = await retriever.ainvoke(query)

    # b. Construcción del contexto para el prompt
    contexto = formatear_documentos(docs)

    # c. Llamada asíncrona al LLM
    salida_llm: RespuestaLLM = await chain.ainvoke({
        "contexto": contexto,
        "pregunta": query,
        "formato": parser_llm.get_format_instructions(),
    })

    # d. Ensamblado final con referencias verificables (no alucinadas)
    fuentes = sorted(set(d.metadata.get("source", "desconocida") for d in docs))

    return RAGResponse(
        respuesta=salida_llm.respuesta,
        fuentes=fuentes,
        fragmentos_recuperados=len(docs),
    )