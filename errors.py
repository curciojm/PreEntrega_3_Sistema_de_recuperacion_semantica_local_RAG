from schemas import LLMErrorType, LLMError
from google.genai.errors import ClientError

# No solicitado en la tarea pero RATE_LIMIT es el error mas comun con genai porque la versión gratuita es muy limitada
def classify_error(error: Exception) -> LLMError:

    if isinstance(error, ClientError):
        if getattr(error, "status_code", None) == 429:
            return LLMError(
                LLMErrorType.RATE_LIMIT,
                "El proveedor alcanzó el límite de solicitudes."
            )

    return LLMError(
        LLMErrorType.UNKNOWN,
        "Ocurrió un error inesperado al comunicarse con el modelo."
    )