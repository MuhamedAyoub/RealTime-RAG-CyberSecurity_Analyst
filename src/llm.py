from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_community.vectorstores import PGVector
from openai import OpenAI
from typing import List, Dict, Any
from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import asyncio
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
import os 
import logging
load_dotenv()
logger = logging.getLogger(__name__)
from src.embedding import LogEmbedder
from src.tokenizing import processed_logs
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
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



# Initialize embedder
embedder = LogEmbedder()

# Embed logs
async def embed_logs():
    await embedder.embed_logs_async(processed_logs)

asyncio.run(embed_logs())


# Create RetrievalQA chain
retriever = embedder.get_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=deepseek_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)