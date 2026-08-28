# este no hace falta ejecutarlo en un orden las clases creadas aca se ejecutan en vector_db

import re
import tiktoken
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, model_encoding: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model_encoding)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=self.calculate_tokens,
            separators=["\n\n", "\n", ".", " ", ""]
            )

# sacamos espacios excesivos antes y despues, importante para ecuaciones
    def clean_text(self, text: str) -> str:
        """Limpia espacios excesivos manteniendo la estructura del documento."""
        # Elimina espacios al principio y final de cada línea
        text = re.sub(r'[ \t]+', ' ', text)
        # Elimina espacios en líneas vacías, pero conserva los saltos de línea
        text = re.sub(r'\n[ \t]+', '\n', text)
        # Reduce más de dos saltos de línea consecutivos a dos
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        """Calcula la cantidad de tokens usando el tokenizer de tiktoken."""
        return len(self.tokenizer.encode(text))

    def process_document(self, raw_text: str) -> List[str]:
        """Pipeline principal: Limpieza -> Chunking -> Validación."""
        cleaned_text = self.clean_text(raw_text)
        chunks = self.splitter.split_text(cleaned_text)

        for i, chunk in enumerate(chunks):
            token_count = self.calculate_tokens(chunk)
            print(f"Chunk {i} creado: {token_count} tokens.")
        return chunks
