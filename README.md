# Sistema de Recuperación Semántica Local (Pre-Entrega 3)

Proyecto correspondiente a la Pre-Entrega 3 del curso de AI Engineering.

## Descripción

Sistema local de recuperación semántica y generación aumentada por recuperación (RAG) utilizando LangChain, Chroma, embeddings y Google Gemini.

El proyecto implementa un pipeline RAG orientado a la consulta de documentos académicos relacionados con metodología de la investigación y estadística.

El sistema permite:

* Ingestar documentos de texto en formato `.md`.
* Limpiar y dividir los documentos en fragmentos (*chunks*).
* Medir el tamaño de los fragmentos utilizando tokens mediante `tiktoken`.
* Generar embeddings de los fragmentos mediante `sentence-transformers/all-MiniLM-L6-v2`.
* Almacenar los embeddings en una base vectorial Chroma persistente.
* Utilizar similitud coseno para la recuperación semántica.
* Recuperar los fragmentos más relevantes mediante un retriever.
* Construir el contexto que será enviado al modelo.
* Generar respuestas mediante Google Gemini.
* Validar la salida del modelo mediante Pydantic.
* Informar las fuentes utilizadas y la cantidad de fragmentos recuperados.
* Ejecutar las operaciones de recuperación y generación de manera asíncrona mediante `.ainvoke()`.
* Clasificar errores del proveedor mediante excepciones personalizadas.
* Realizar pruebas unitarias y de integración mediante `pytest`.
* Utilizar *mocking* para probar componentes sin depender de llamadas reales al modelo.

El modelo LLM se configura con un rol de **docente universitario especializado en metodología de la investigación y estadística**. El prompt establece que la respuesta debe basarse exclusivamente en la información recuperada de los documentos y que, si la información solicitada no está disponible en el contexto, debe indicarlo explícitamente.

## Requisitos

* Python 3.12+
* API key de Google Gemini.

## Tecnologías utilizadas

* Python
* asyncio
* Pydantic
* LangChain
* LangChain Core
* LangChain Text Splitters
* LangChain Chroma
* LangChain Google GenAI
* Chroma
* Hugging Face
* Sentence Transformers
* tiktoken
* pytest
* pytest-asyncio
* Ruff

## Variables de entorno

El proyecto utiliza la siguiente variable de entorno:

* `GOOGLE_API_KEY`

Crear un archivo `.env` a partir de `.env.example` y completar la API key correspondiente; en caso de no encontrarse configurada como variable de entorno del sistema.

La API key real no se incluye en el repositorio.

## Pipeline RAG

El procesamiento se organiza en las siguientes etapas:

```text
Documentos .md
      ↓
Ingesta
      ↓
Limpieza y chunking
      ↓
Embeddings
      ↓
Chroma
      ↓
Retriever
      ↓
Fragmentos relevantes
      ↓
Construcción del contexto
      ↓
Prompt
      ↓
Google Gemini
      ↓
PydanticOutputParser
      ↓
RAGResponse
```

Los documentos son divididos en fragmentos de hasta 500 tokens, con un solapamiento (*overlap*) de 50 tokens.

El retriever recupera los 4 fragmentos más similares a la consulta.

## Ejecución

El script principal realiza una consulta de prueba sobre los documentos disponibles.

Para ejecutar el sistema:

```bash
python main.py
```

La respuesta incluye:

* La respuesta generada por el modelo.
* Los archivos utilizados como fuente.
* La cantidad de fragmentos recuperados.

## Manejo de errores

Se implementó una clasificación de errores mediante `LLMErrorType` y la excepción personalizada `LLMError`.

Actualmente se contemplan:

* `RATE_LIMIT`: límite de solicitudes alcanzado por el proveedor.
* `UNKNOWN`: error no contemplado específicamente.

Los errores `429` provenientes de `ClientError` de Google GenAI son identificados como errores de límite de solicitudes.

## Logging

Se incorporó el módulo estándar `logging` de Python para registrar eventos relevantes durante la ejecución del pipeline.

Los mensajes se clasifican según su nivel de importancia:

* `INFO`: información sobre el procesamiento, como creación de chunks, carga de la base vectorial y cantidad de fragmentos recuperados.
* `ERROR`: errores ocurridos durante la ejecución del pipeline.

La configuración utiliza el nivel `INFO`, por lo que se muestran en consola los mensajes `INFO` y los niveles superiores (`WARNING`, `ERROR` y `CRITICAL`).

Los logs permiten observar el funcionamiento del sistema durante la ejecución y facilitan la identificación de errores.


## Testing

Se incorporaron pruebas automatizadas utilizando `pytest` y `pytest-asyncio`.

Las pruebas permiten verificar distintos componentes del sistema sin depender de llamadas reales al modelo.

Se incluyen pruebas para:

* Procesamiento y generación de chunks.
* Creación y funcionamiento de la base vectorial.
* Formateo y ejecución de la cadena LCEL.
* Orquestación de `get_rag_response()`.
* Clasificación de errores.
* Manejo de errores mediante objetos simulados (*mocking*).

### Ejecución de los tests

Los tests se encuentran dentro de la carpeta `tests/`.

Para ejecutar una prueba específica:

```bash
python -m pytest tests/test_chunking.py
```

Por ejemplo:

```bash
python -m pytest tests/test_vector_db.py
```

```bash
python -m pytest tests/test_chain.py
```

```bash
python -m pytest tests/test_main.py
```

```bash
python -m pytest tests/test_errors.py
```

También pueden ejecutarse todas las pruebas del proyecto mediante:

```bash
python -m pytest
```

Las pruebas asíncronas utilizan `pytest-asyncio` para permitir la ejecución de funciones definidas con `async def`.

En las pruebas de integración de `main.py` se utiliza *mocking* para reemplazar temporalmente el retriever y la cadena LLM por implementaciones simuladas. De esta manera se puede verificar la lógica de orquestación sin realizar una llamada real al modelo.

## Calidad y buenas prácticas

Como mejora respecto de entregas anteriores, se incorporó Ruff como herramienta de análisis y formateo del código.

Para formatear automáticamente el proyecto:

```bash
ruff format .
```

Para analizar el código sin modificarlo:

```bash
ruff check .
```

El formateo automático se utiliza para mantener una estructura consistente del código, mientras que las sugerencias de `ruff check` se revisan manualmente.

## Estructura del proyecto

```text
├── data/                       # Documentos utilizados como fuente del RAG
│   ├── Pagano 2006, correlacion y regresion.md
│   ├── Sampieri 2018, cap 3 planteamiento del problema en la ruta cuantitativa.md
│   └── Sampieri 2018, cap 7 diseño experimental.md
│
├── tests/                      # Pruebas automatizadas
│   ├── test_chunking.py        # Tests del procesamiento y chunking
│   ├── test_vector_db.py       # Tests de la base vectorial
│   ├── test_chain.py           # Tests de la cadena LCEL
│   ├── test_main.py            # Tests de la orquestación RAG
│   └── test_errors.py          # Tests de clasificación de errores
│
├── .env.example                # Ejemplo de variables de entorno
├── .gitignore
├── chain.py                    # Modelo Gemini, parser y cadena LCEL
├── chunking.py                 # Limpieza y división de documentos
├── errors.py                   # Clasificación y manejo de errores
├── main.py                     # Orquestación y punto de entrada principal
├── logging_config.py           # Configuración del sistema de logs
├── prompt_config.py            # Configuración del prompt
├── retriever.py                # Configuración del retriever
├── schemas.py                  # Modelos Pydantic y tipos de error
├── setup.py                    # Ingesta y procesamiento de documentos
├── vector_db.py                # Creación y carga de la base vectorial
├── README.md
└── requirements.txt            # Dependencias del proyecto
```

La carpeta `vectors_db/` se utiliza para almacenar la base vectorial persistente y se encuentra excluida del repositorio mediante `.gitignore`.

## Sobre el código

El proyecto fue desarrollado tomando como referencia:

* Código base proporcionado por el profesor como guía para la Pre-Entrega 3.
* Ejemplos y contenidos incluidos en el temario de la plataforma sobre RAG, embeddings, recuperación semántica y LangChain.
* Documentación oficial y recursos disponibles en Internet.
* ChatGPT como herramienta de asistencia durante el desarrollo.

Las mejoras de testing y calidad de código fueron incorporadas a partir de recomendaciones recibidas en entregas anteriores, incluyendo el uso de `pytest`, `pytest-asyncio`, *mocking* y Ruff.
