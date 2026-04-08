# RAG Application Using Type Sense

This code shows how to use Type Sense as a vector store for a Retrieval-Augmented Generation (RAG) application. It demonstrates how to set up a Type Sense client, create a collection, import data, and perform search queries. Finally, it integrates Type Sense with Langchain for document retrieval based on semantic similarity.

## Key Focus Areas
- Using Typesense Vector Search
- Using Langchain for RAG
- HuggingFace Embeddings
  


## Run My Project Locally

Here's how I run it on my machine:
```bash
git clone https://github.com/nehasinghaniya21/rag-typesense.git
cd rag-typesense
uv init
uv venv
source .venv/bin/activate
uv add -r requirements.txt
python langgraph_typesense.py
```
