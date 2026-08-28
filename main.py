import asyncio
from chain import get_rag_response

async def main():
    respuesta_ok = await get_rag_response("¿Que es la regresion?")

    print("RESPUESTA:", respuesta_ok.respuesta)
    print("FUENTES:", respuesta_ok.fuentes)
    print("Fragmentos usados:", respuesta_ok.fragmentos_recuperados)

if __name__ == "__main__":
    asyncio.run(main())