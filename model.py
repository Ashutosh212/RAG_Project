# import logging
# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
# from llama_index.vector_stores.chroma import ChromaVectorStore
# import chromadb
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Global query engine
# query_engine = None

# def init_llm():
#     llm = Ollama(model="llama2", request_timeout=300.0)
#     embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#     Settings.llm = llm
#     Settings.embed_model = embed_model


# def init_index():
#     reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
#     documents = reader.load_data()

#     logging.info("Index created with %d documents", len(documents))

#     # Create Chroma vector store in memory
#     chroma_client = chromadb.EphemeralClient()
#     chroma_collection = chroma_client.create_collection("iollama")

#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

#     return index


# def init_query_engine(index):
#     global query_engine

#     # Build the query engine using the created index
#     query_engine = index.as_query_engine(similarity_top_k=3)
#     logging.info("Query engine initialized.")
#     return query_engine


# def chat(input_question):
#     global query_engine

#     response = query_engine.query(input_question)
#     logging.info("Response from LLM: %s", response)

#     return response.response


import logging
import sys
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

query_engine = None

def init_llm():
    llm = Ollama(model="llama2", request_timeout=300.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model

def init_index():
    reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
    documents = reader.load_data()

    logging.info("Index created with %d documents", len(documents))

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("iollama")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

    return index

def init_query_engine(index):
    global query_engine

    query_engine = index.as_query_engine(similarity_top_k=3)
    logging.info("Query engine initialized.")
    return query_engine

def chat(input_question):
    global query_engine

    response = query_engine.query(input_question)
    logging.info("Response from LLM: %s", response)

    return response.response

def main():
    # Initialize LLM, Index, and Query Engine
    init_llm()
    index = init_index()
    init_query_engine(index)

    print("Welcome to the LLM-based chat! Type 'exit' to quit.")

    while True:
        # Take user input (question)
        question = input("You: ")

        # Exit condition
        if question.lower() == 'exit':
            print("Goodbye!")
            break

        # Get and print the model's response
        answer = chat(question)
        print("Model:", answer)

if __name__ == '__main__':
    main()
