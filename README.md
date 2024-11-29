# Chat with your PDFs

This project demonstrates the use of LLM (Large Language Models) to build a simple chatbot that can respond to user queries based on the documents stored in a specific folder. The bot is powered by the Llama2 model and utilizes HuggingFace embeddings for semantic search. The documents are indexed using Chroma vector store, and you can easily interact with the bot through a command-line interface.

## Features:
- **Document-based Querying:** The chatbot can answer questions based on the documents you place in the `docs` folder.
- **LLM Integration:** The chatbot uses the Llama2 model for natural language understanding and HuggingFace embeddings for document retrieval.
- **Chroma Vector Store:** The documents are indexed and stored using Chroma, which allows for efficient similarity search and retrieval.

## Requirements

You will need Python 3.7 or higher to run the project. The required dependencies are listed in the `requirements.txt` file.

Project Structure


- **`app.py`**: The main script that runs the chatbot and handles the logic for document indexing, querying, and generating responses.
- **`docs/`**: A folder where you should place your documents (in `.txt` format) for the chatbot to index and use for answering queries.
- **`requirements.txt`**: A file that lists all the necessary Python packages and dependencies for running the project.
- **`README.md`**: This file, providing an overview and instructions for using the project.



