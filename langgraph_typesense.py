# RAG Application Using Type Sense
# This code shows how to use Type Sense as a vector store for a Retrieval-Augmented Generation (RAG) application. It demonstrates how to set up a Type Sense client, create a collection, import data, and perform search queries. Finally, it integrates Type Sense with Langchain for document retrieval based on semantic similarity.

import os
from dotenv import load_dotenv
import typesense
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Typesense
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

load_dotenv()
os.environ["TYPESENSE_API_KEY"] = os.getenv("TYPESENSE_API_KEY")
os.environ["TYPESENSE_HOST"] = os.getenv("TYPESENSE_HOST")
os.environ["TYPESENSE_PORT"] = os.getenv("TYPESENSE_PORT")
os.environ["TYPESENSE_PROTOCOL"] = os.getenv("TYPESENSE_PROTOCOL")

client=typesense.Client({
  'nodes': [{
    'host': os.environ["TYPESENSE_HOST"],  # For Typesense Cloud use xxx.a1.typesense.net
    'port': os.environ["TYPESENSE_PORT"],       # For Typesense Cloud use 443
    'protocol': os.environ["TYPESENSE_PROTOCOL"]    # For Typesense Cloud use https
  }],
  'api_key': os.environ["TYPESENSE_API_KEY"],
  'connection_timeout_seconds': 2
})
print(client)

 # Define a collection schema for books
books_schema = {
  'name': 'books',
  'fields': [
    {'name': 'title', 'type': 'string'},
    {'name': 'authors', 'type': 'string[]', 'facet': True},
    {'name': 'publication_year', 'type': 'int32', 'facet': True},
    {'name': 'ratings_count', 'type': 'int32'},
    {'name': 'average_rating', 'type': 'float'}
  ],
  'default_sorting_field': 'ratings_count'
}
# print(client.collections.create(books_schema))

# Get the directory where the script is located
script_dir = Path(__file__).parent.resolve()
# Join that directory with the filename
file_path = script_dir / 'books.jsonl'

# Importing data into Typesense
with open(file_path, 'r', encoding='utf-8') as jsonl_file:
    data = jsonl_file.read()
    client.collections['books'].documents.import_(data)

# Search examples
print("Basic Search:--------")
search_parameters={
    'q':"harry potter",
    'query_by':"title,authors",
    'sort_by':"ratings_count:desc"
}
print(client.collections['books'].documents.search(search_parameters))

# Perform a search query with filtering and sorting
print("Filtered and Sorted Search:--------")
search_parameters = {
  'q'         : 'harry potter',
  'query_by'  : 'title',
  'filter_by' : 'publication_year:<1998',
  'sort_by'   : 'publication_year:desc'
}
print(client.collections['books'].documents.search(search_parameters))

# Perform a search query with faceting
print("Faceted Search:--------")
search_parameters = {
  'q'         : 'experyment',
  'query_by'  : 'title',
  'facet_by'  : 'authors',
  'sort_by'   : 'average_rating:desc'
}
print(client.collections['books'].documents.search(search_parameters))

print("Search with Langchain and Typesense:--------")
# Using Langchain with Typesense
loader = TextLoader(script_dir /"data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

docsearch=Typesense.from_documents(
    docs,
    embeddings,
    typesense_client_params={
        "host": os.environ["TYPESENSE_HOST"],  # Use xxx.a1.typesense.net for Typesense Cloud
        "port": os.environ["TYPESENSE_PORT"],  # Use 443 for Typesense Cloud
        "protocol": os.environ["TYPESENSE_PROTOCOL"],  # Use https for Typesense Cloud
        "typesense_api_key": os.environ["TYPESENSE_API_KEY"],
        "typesense_collection_name": "lang-chain"
    },
)

query = "What is machine learning?"
found_docs = docsearch.similarity_search(query)
print("found docs: {}".format(found_docs[0].page_content))

### Retriever
retriever = docsearch.as_retriever()
print(retriever)