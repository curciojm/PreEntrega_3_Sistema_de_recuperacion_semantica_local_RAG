import asyncio

from chain import chain, formatear_documentos, parser_llm
from errors import classify_error
from logging_config import logger
from retriever import retriever
from schemas import RAGResponse, RespuestaLLM


# Orquestación
async def get_rag_response(query: str) -> RAGResponse:
    try:
        logger.info(f"Procesando consulta: {query}")

        docs = await retriever.ainvoke(query)

        contexto = formatear_documentos(docs)

        salida_llm: RespuestaLLM = await chain.ainvoke(
            {
                "contexto": contexto,
                "pregunta": query,
                "formato": parser_llm.get_format_instructions(),
            }
        )

        fuentes = sorted({d.metadata.get("source", "desconocida") for d in docs})
        logger.info(f"Fragmentos recuperados: {len(docs)}")

        return RAGResponse(
            respuesta=salida_llm.respuesta,
            fuentes=fuentes,
            fragmentos_recuperados=len(docs),
        )
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        # Excepciones centralizadas para convertirlas en errores legibles para la aplicación.
        raise classify_error(e)


async def main():
    respuesta_ok = await get_rag_response("¿Cual es el sentido de la vida?")
    print("\nRESPUESTA:", respuesta_ok.respuesta)
    print("\nFUENTES:", respuesta_ok.fuentes)
    print("\nFragmentos usados:", respuesta_ok.fragmentos_recuperados)


if __name__ == "__main__":
    asyncio.run(main())
