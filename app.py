# Save this as app.py
import streamlit as st
import pandas as pd
import logging
import configparser
from rag_system import LogEmbedder, DeepSeekLLM, QueryHandler, logs as geenrated_logs, preprocess_log
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')
ELASTICSEARCH_URL = config.get('DEFAULT', 'ElasticsearchURL', fallback='http://localhost:9200')
PGVECTOR_CONNECTION_STRING = config.get('DEFAULT', 'PGVectorConnectionString', fallback='postgresql+psycopg2://user:password@localhost:5432/soc_logs')
BATCH_SIZE = config.getint('DEFAULT', 'BatchSize', fallback=16)

# Streamlit app
st.title("SOC Log Analysis RAG System")
st.markdown("""
Welcome to the SOC Log Analysis RAG System. Input a query to retrieve and analyze server logs.
Choose between natural language responses or visualization outputs for Kibana integration.

**Note**: Run this app with `streamlit run app.py --server.fileWatcherType none` to avoid PyTorch compatibility issues.
Alternatively, set the environment variable `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none`.
""")

# Initialize session state
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
    st.session_state.query_handler = None
    st.session_state.initialized = False

# Initialize RAG system
@st.cache_resource
def initialize_rag_system():
    try:
        logs = geenrated_logs
        processed_logs = [preprocess_log(log) for log in logs]
        logger.info("Generated and preprocessed sample logs")

        embedder = LogEmbedder()
        asyncio.run(embedder.embed_logs_async(processed_logs))
        logger.info("Embedded logs in pgVector")

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
        retriever = embedder.get_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=deepseek_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        query_handler = QueryHandler(qa_chain, processed_logs)
        return embedder, query_handler
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        st.error("Failed to initialize RAG system.")
        return None, None

if not st.session_state.initialized:
    st.session_state.embedder, st.session_state.query_handler = initialize_rag_system()
    st.session_state.initialized = True

# Query input and options
query = st.text_input("Enter your query:", placeholder="e.g., What caused the failed login attempts?")
response_type = st.radio("Response Type:", ("Natural Language", "Visualization"))

# Process query
if st.button("Submit Query"):
    if not query:
        st.warning("Please enter a query.")
    elif st.session_state.query_handler is None:
        st.error("RAG system not initialized.")
    else:
        try:
            with st.spinner("Processing query..."):
                is_visualization = response_type == "Visualization"
                response = st.session_state.query_handler.process_query(query, is_visualization=is_visualization)

            if is_visualization:
                st.success(response["status"])
                st.subheader("Visualization Data")
                df = pd.DataFrame(response["data"])
                st.dataframe(df)
                st.markdown("This data is indexed in Elasticsearch for Kibana visualization.")
            else:
                st.subheader("Natural Language Response")
                st.write(response["text"])
                st.subheader("Source Logs")
                for log in response["source_logs"]:
                    st.write(f"- {log}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            st.error("An error occurred while processing the query.")

st.markdown("""
### Learning Note
This Streamlit app integrates the RAG system, showing how NLP backend logic connects to a frontend. Experiment with queries and response types to understand SOC analyst workflows.
If you encounter PyTorch compatibility issues, use the `--server.fileWatcherType none` flag or downgrade Streamlit to 1.29.0.
""")