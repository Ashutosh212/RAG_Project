import os

# vector index persist directory
INDEX_PERSIST_DIRECTORY = os.getenv('INDEX_PERSIST_DIRECTORY', './data/chromadb')

# HTTP API port
HTTP_PORT = os.getenv('HTTP_PORT', 7654)
