import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import logging
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
import logging
import configparser
from typing import List, Dict, Any
from datetime import datetime
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
