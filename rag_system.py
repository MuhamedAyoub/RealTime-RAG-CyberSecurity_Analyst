import pandas as pd
import numpy as np
import spacy
import psycopg2
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import requests
import json
import logging
import asyncio
import configparser
from typing import List, Dict, Any, Tuple
from datetime import datetime
from functools import lru_cache
from langchain_community.vectorstores import PGVector
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import ipywidgets as widgets
from IPython.display import display, HTML
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy
import psycopg2
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from openai import OpenAI
import logging
import asyncio
import configparser
from typing import List, Dict, Any
from datetime import datetime
from functools import lru_cache
from langchain_community.vectorstores import PGVector
from langchain.docstore.document import Document
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from dotenv import load_dotenv
load_dotenv()
# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')
ELASTICSEARCH_URL = config.get('DEFAULT', 'ElasticsearchURL', fallback='http://localhost:9200')
DEEPSEEK_API_KEY = config.get('DEFAULT', 'DeepSeekAPIKey', fallback='your-api-key')
PGVECTOR_CONNECTION_STRING = config.get('DEFAULT', 'PGVectorConnectionString', fallback='postgresql+psycopg2://postgres:password@localhost:5432/vectordb')
BATCH_SIZE = config.getint('DEFAULT', 'BatchSize', fallback=32)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy for NER and tokenization
nlp = spacy.load("en_core_web_sm")


# Logs Ingestion
import re
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logger (for demonstration; adjust as needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_log(log: Dict[str, Any]) -> bool:
    # Ensure the log dict contains all required keys.
    required_keys = {"timestamp", "source_ip", "message", "event_type", "severity"}
    return all(key in log for key in required_keys)

def parse_logs(file_path: str) -> List[Dict[str, Any]]:
    sample_logs = []
    # Use ^ to anchor, and \s+ to allow variable whitespace.
    log_pattern = re.compile(
        r"^-?\s*(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(?P<source_ip>\d{1,3}(?:\.\d{1,3}){3})\s+-\s+"
        r"(?P<user>\w+)\s+\["
        r"(?P<method>[A-Z]+)\s+"
        r"(?P<endpoint>[^\]]+)\]\s+"
        r"\"(?P<status_msg>\d{3}\s+[^\"]+)\"\s+"
        r"\"(?P<user_agent>[^\"]+)\""
    )

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            match = log_pattern.match(line)
            if not match:
                logger.warning(f"Line doesn't match: {line}")
                continue

            groups = match.groupdict()
            timestamp_str = groups["timestamp"]
            # Convert to ISO format
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").isoformat()
            except ValueError as e:
                logger.error(f"Timestamp parsing error: {e}")
                continue

            source_ip = groups["source_ip"]
            user = groups["user"]
            method = groups["method"]
            endpoint = groups["endpoint"]
            status_msg = groups["status_msg"]
            user_agent = groups["user_agent"]

            # Determine severity based on status code
            try:
                status_code = int(status_msg.split()[0])
            except ValueError:
                logger.error(f"Invalid status message: {status_msg}")
                continue

            severity = "high" if status_code >= 500 else "medium" if status_code >= 400 else "low"

            message = f"{status_msg} on {method} {endpoint} by {user} using {user_agent}"

            log_entry = {
                "timestamp": timestamp,
                "source_ip": source_ip,
                "message": message,
                "event_type": "http_request",
                "severity": severity
            }

            if validate_log(log_entry):
                sample_logs.append(log_entry)
            else:
                logger.warning(f"Invalid log: {log_entry}")

    return sample_logs

logs = parse_logs("./logs/logs1.md")
logger.info(f"Generated {len(logs)} valid sample logs")




# Tokenization and NER
def tokenize_message(message: str) -> List[str]:
    """Tokenize a log message using spaCy."""
    try:
        doc = nlp(message)
        tokens = [token.text for token in doc]
        return tokens
    except Exception as e:
        logger.error(f"Error tokenizing message: {e}")
        return message.split()

def preprocess_log(log: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a log entry with tokenization and NER."""
    try:
        message = log.get("message", "")
        tokens = tokenize_message(message)
        tokenized_message = " ".join(tokens)
        doc = nlp(message)
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["GPE", "ORG"]]
        normalized_message = tokenized_message.lower().replace("err", "error")
        
        processed_log = log.copy()
        processed_log["normalized_message"] = normalized_message
        processed_log["entities"] = entities
        processed_log["tokens"] = tokens
        return processed_log
    except Exception as e:
        logger.error(f"Error preprocessing log: {e}")
        return log

# Preprocess logs
processed_logs = [preprocess_log(log) for log in logs]
logger.info("Completed log preprocessing with tokenization")


# Embedding 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document  # or just from langchain import Document

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

    def get_retriever(self, k: int = 10):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# Initialize embedder
embedder = LogEmbedder()
import asyncio
# Embed logs
async def embed_logs():
    await embedder.embed_logs_async(processed_logs)

asyncio.run(embed_logs()) 
 
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


## LLM
class DeepSeekLLM(BaseLLM):
    """LangChain-compatible LLM for DeepSeek using OpenAI SDK."""
    client: OpenAI
    model: str = "deepseek-chat"  # Adjust based on DeepSeek's available models
    api_key: str
    base_url: str

    def __init__(self):
        """Initialize DeepSeek client with OpenAI SDK."""
        # Initialize client first
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        
        # Initialize parent with required fields
        super().__init__(
            client=client,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model="deepseek-chat"
        )

    @lru_cache(maxsize=100)
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        """Synchronous call to DeepSeek API for text generation."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for SOC log analysis, providing concise and accurate responses in JSON format when requested."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return f"Error generating response: {str(e)}"

    def _generate(self, prompts: List[str], stop: List[str] = None) -> LLMResult:
        """Generate responses for a list of prompts, returning an LLMResult."""
        try:
            generations = []
            for prompt in prompts:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for SOC log analysis, providing concise and accurate responses in JSON format when requested."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7,
                    stop=stop
                )
                text = response.choices[0].message.content
                generations.append([Generation(text=text)])
            return LLMResult(generations=generations)
        except Exception as e:
            logger.error(f"Error in _generate: {e}")
            return LLMResult(generations=[[Generation(text=f"Error generating response: {str(e)}")]])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for LangChain."""
        return {"model": self.model, "base_url": self.base_url}

    @property
    def _llm_type(self) -> str:
        """Return LLM type for LangChain."""
        return "deepseek"

# Initialize LangChain components
deepseek_llm = DeepSeekLLM()
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Based on these logs:
    {context}
    Answer the following question: {question}
    Provide a concise explanation and structured data in JSON format.
    """
)

# Create RetrievalQA chain
retriever = embedder.get_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=deepseek_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

## Query 

class QueryHandler:
    def __init__(self, qa_chain, logs: List[Dict[str, Any]]):
        self.qa_chain = qa_chain
        self.logs = logs
        self.es = Elasticsearch(ELASTICSEARCH_URL)

    @lru_cache(maxsize=100)
    def process_query(self, query: str, is_visualization: bool = False) -> Dict[str, Any]:
        """Process a query using LangChain."""
        try:
            if is_visualization:
                docs = self.qa_chain.retriever.get_relevant_documents(query)
                es_data = [
                    {
                        "timestamp": doc.metadata["timestamp"],
                        "source_ip": doc.metadata["source_ip"],
                        "event_type": doc.metadata["event_type"],
                        "severity": doc.metadata["severity"],
                        "query": query
                    } for doc in docs
                ]
                for doc in es_data:
                    self.es.index(index="soc_logs", body=doc)
                logger.info("Indexed logs for Kibana")
                return {"status": "Indexed for visualization", "data": es_data}
            else:
                result = self.qa_chain({"query": query})
                return {
                    "text": result["result"],
                    "source_logs": [doc.page_content for doc in result["source_documents"]]
                }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"text": "Error processing query", "source_logs": []}

# Initialize handler
query_handler = QueryHandler(qa_chain, processed_logs)