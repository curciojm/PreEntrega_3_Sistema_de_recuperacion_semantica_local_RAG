import asyncio
from schemas import RespuestaLLM, RAGResponse
from chain import formatear_documentos, parser_llm, chain
from retriever import retriever
from errors import classify_error

# Orquestación
async def get_rag_response(query: str) -> RAGResponse:
        try:
            docs = await retriever.ainvoke(query)

            contexto = formatear_documentos(docs)

            salida_llm: RespuestaLLM = await chain.ainvoke({
                "contexto": contexto,
                "pregunta": query,
                "formato": parser_llm.get_format_instructions(),
            })

            fuentes = sorted(
                set(d.metadata.get("source", "desconocida") for d in docs)
            )

            return RAGResponse(
                respuesta=salida_llm.respuesta,
                fuentes=fuentes,
                fragmentos_recuperados=len(docs),
            )
        except Exception as e:
             raise classify_error(e)

async def main():
    respuesta_ok = await get_rag_response("¿Que es la regresion?")
    print("\nRESPUESTA:", respuesta_ok.respuesta)
    print("\nFUENTES:", respuesta_ok.fuentes)
    print("\nFragmentos usados:", respuesta_ok.fragmentos_recuperados)

if __name__ == "__main__":
    asyncio.run(main())