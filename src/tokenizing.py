import logging
import os
import re
from datetime import datetime
from typing import List, Dict, Any
import spacy
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_community.vectorstores import PGVector
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate

from src.preprocessing import logs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy for NER and tokenization
nlp = spacy.load("en_core_web_sm")

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