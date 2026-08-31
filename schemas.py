from enum import Enum

from pydantic import BaseModel, Field


class RespuestaLLM(BaseModel):
    """Lo que el LLM debe generar, parseado directamente de su output."""

    respuesta: str = Field(
        description="Respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en el CONTEXTO. "
        "Si la información no está en el contexto, decir explícitamente que no se cuenta con esa información."
    )


class RAGResponse(BaseModel):
    """Objeto final que devuelve get_rag_response — combina el output del LLM con metadata verificable."""

    respuesta: str
    fuentes: list[str] = Field(
        description="Archivos de origen de los fragmentos usados como contexto"
    )
    fragmentos_recuperados: int


# Errores posibles
class LLMErrorType(str, Enum):
    """Tipos de errores utilizados para clasificar las excepciones."""

    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


## Excepción personalizada para unificar el manejo de errores del cliente.
class LLMError(Exception):
    """Permite clasificar el error y proporcionar un mensaje legible al usuario."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)
