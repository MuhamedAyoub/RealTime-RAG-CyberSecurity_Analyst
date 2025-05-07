from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document  # or just from langchain import Document

from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import logging
from functools import lru_cache
from langchain_core.prompts import PromptTemplate
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING", "postgresql+psycopg2://postgres:password@localhost:5432/vectordb")

logger = logging.getLogger(__name__)
class LogEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        logger.info(f"Db connection string {PGVECTOR_CONNECTION_STRING}")

    async def embed_logs_async(self, logs: List[Dict[str, Any]]) -> None:
        """Asynchronously embed logs and store in pgVector."""
        try:
            documents = [
                Document(
                    page_content=log["normalized_message"],
                    metadata={k: v for k, v in log.items() if k != "normalized_message"}
                ) for log in logs
            ]
            self.vector_store = PGVector.from_documents(
                documents=documents,
                embedding=self.model,
                connection_string=PGVECTOR_CONNECTION_STRING,
                collection_name="soc_logs"
            )
            logger.info(f"Embedded and stored {len(logs)} logs in pgVector")
        except Exception as e:
            logger.error(f"Error embedding logs: {e}")

    def get_retriever(self, k: int = 5):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

