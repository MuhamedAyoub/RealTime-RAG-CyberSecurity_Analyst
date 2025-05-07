# SOC Log Analysis RAG System

A Retrieval-Augmented Generation (RAG) system for automating Security Operations Center (SOC) log analysis.

## Overview

This project implements a RAG system that combines semantic log retrieval with generative response capabilities to automate SOC log analysis. The system leverages advanced NLP techniques to process, analyze, and generate insights from security logs.

## Features

- **Log Preprocessing**: Tokenization and Named Entity Recognition (NER) via spaCy
- **Semantic Search**: Vectorization with Sentence Transformers and scalable storage in pgVector
- **Response Generation**: Query processing with LangChain and DeepSeek
- **Interactive Interface**: Streamlit application for SOC analysts
- **Visualization**: Elasticsearch integration for Kibana dashboards

## Architecture

<!-- add arch image -->

![Architecture Diagram](assets/architecture_diagram.png)
```
Log Ingestion → Preprocessing → Vectorization → Query Processing → Interface/Visualization
```

## Technologies Used

- **NLP**: spaCy, Sentence Transformers
- **Vector Storage**: pgVector. 
- **Query Processing**: LangChain, DeepSeek with OpenAI SDK
- **Interface**: Streamlit
- **Visualization**: Elasticsearch,Fluentd , Kibana

## Installation

```bash
# Clone the repository
git clone https://github.com/MuhamedAyoub/RealTime-RAG-CyberSecurity_Analyst.git
cd RealTime-RAG-CyberSecurity_Analyst

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


```
#### Setup your Docker Compose for EFK stack and pgVector

```bash
docker-compose up -d
```

## Configuration

Create a `config.init` file in the project root with the following variables:

```
PGVECTOR_CONNECTION_STRING=postgresql://user:password@localhost:5432/soc_logs
ELASTICSEARCH_HOST=http://localhost:9200
DEEPSEEK_API_KEY=your_api_key_here
```

## Usage

### Start the Streamlit Application

```bash
streamlit run app.py --server.fileWatcherType none

```

### Example Query

Enter queries like:

- "What caused the recent login failures?"
- "Show me all failed SSH attempts from external IPs"
- "Analyze access patterns for the database server"

### Visualization Data

For visualization queries, the system will:
1. Process the query
2. Retrieve relevant logs
3. Index them in Elasticsearch
4. Return structured data for Kibana visualization

## Project Structure

```
soc-log-rag/
├── app.py                # Streamlit application
├── src/
│   ├── preprocessing.py  # Log preprocessing 
│   ├── tokenizing.py     # tokenizing the  new logs 
│   ├── embedding.py      # embedding and storing logs in pgVector
│   ├── llm.py            # DeepSeek LLM and LangChain RetrievalQA
├── config.ini
├── logs/
│   └── logs.md/      # Sample SOC logs for testing
├── reports/
│   └── report.pdf        # Project report
├── compose.yml           # Docker Compose file
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Future Enhancements

- Real-time log streaming with Fluentd
- Multi-modal analysis (combining logs with network data)

## Ethical Considerations

- Data anonymization for sensitive information
- Secure storage with encrypted connections
- Bias-aware response generation

## License

[MIT License](LICENSE)

## Author

Ameri Mohamed Ayoub
[LinkedIn](https://www.linkedin.com/in/mohamed-ayoub-ameri-68935a221/)
[Email](mailto:ma.ameri@esi-sba.dz)

---

*The Higher School of Computer Science Engineering ESI-SBA, May 2025*
