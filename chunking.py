import re

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Procesador del documento: limpieza y chunking
class DocumentProcessor:
    def __init__(self, model_encoding: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model_encoding)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=self.calculate_tokens,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def clean_text(self, text: str) -> str:
        """Limpia espacios excesivos manteniendo la estructura del documento."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        """Calcula la cantidad de tokens usando tiktoken."""
        return len(self.tokenizer.encode(text))

    def process_document(self, raw_text: str) -> list[str]:
        """Pipeline: limpieza -> chunking."""
        cleaned_text = self.clean_text(raw_text)
        chunks = self.splitter.split_text(cleaned_text)

        for i, chunk in enumerate(chunks):
            token_count = self.calculate_tokens(chunk)
            print(f"Chunk {i} creado: {token_count} tokens.")
        return chunks
