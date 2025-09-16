# PubMed Abstract Chatbot

## Introduction

The rapid proliferation of biomedical literature presents a significant challenge for clinicians and researchers who need to stay abreast of the latest findings. Traditional search engines, while effective at keyword-based document retrieval, often require users to manually sift through numerous articles to find and synthesize information. This project proposes and implements a conversational question-answering system designed to alleviate this burden.

The application leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide users with synthesized, context-aware answers to complex biomedical questions. The system uses the vast repository of PubMed as its primary knowledge source, transforming it from a passive database into an interactive knowledge base. By integrating advanced language models with a robust retrieval pipeline, the system can understand user intent, fetch relevant scientific abstracts, and generate coherent, explanatory answers, complete with a conversational interface for follow-up inquiries.

## Usage

Once you have installed the dependencies and configured the environment variables, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will start the web server and open the application in your browser.

## System Architecture

The system is designed based on a modular, multi-layer RAG architecture that separates concerns into a presentation layer, a backend logic layer, and a data storage layer.

-   **Presentation Layer**: A web-based user interface built with Streamlit (`app.py`). It is responsible for capturing user input (both direct questions and uploaded medical notes), rendering the generated answers, and managing the interactive chat component.

-   **Backend Logic & Orchestration Layer**: The core of the system, which executes the RAG pipeline. This layer is responsible for orchestrating the flow of data between the UI, the external PubMed API, the vector database, and the Large Language Model (LLM).

-   **Data & Storage Layer**: This layer consists of two primary components:
    1.  A **Vector Database** (ChromaDB) that stores the embeddings of scientific abstracts for efficient semantic retrieval.
    2.  A **Local JSON-based Metadata Store** that maintains records of user queries, retrieved abstracts, and generated answers, enabling session persistence and chat history.


## Project Structure

```
├───__init__.py
├───.env.example
├───.gitignore
├───app.py
├───README.md
├───requirements.txt
├───backend
│   ├───abstract_retrieval
│   │   ├───interface.py
│   │   ├───pubmed_query_simplification.py
│   │   └───pubmed_retriever.py
│   ├───chromadb_storage
│   ├───data_repository
│   │   ├───interface.py
│   │   ├───local_storage.py
│   │   └───models.py
│   └───rag_pipeline
│       ├───chromadb_rag.py
│       ├───embeddings.py
│       ├───interface.py
│       ├───pubmed_bert_embedder.py
│       └───sap_embedder.py
├───components
│   ├───chat_prompts.py
│   ├───chat_utils.py
│   ├───layout_extensions.py
│   └───llm.py
├───config
│   ├───logging_config.py
│   └───setting.py
└───evaluation
    ├───evaluation.py
    ├───README.md
    └───training12b_new.json
```

-   **`app.py`**: The main entry point for the Streamlit application.
-   **`requirements.txt`**: A list of all the Python packages required to run the project.
-   **`.env.example`**: An example file for setting up environment variables.
-   **`backend/`**: Contains the core logic of the RAG pipeline.
    -   **`abstract_retrieval/`**: Handles the retrieval of abstracts from PubMed.
    -   **`chromadb_storage/`**: Manages the ChromaDB vector database.
    -   **`data_repository/`**: Handles the storage and retrieval of data.
    -   **`rag_pipeline/`**: Contains the implementation of the RAG pipeline.
-   **`components/`**: Contains utility functions and classes used by the Streamlit app.
-   **`config/`**: Contains configuration files for the project.
-   **`evaluation/`**: Contains scripts and data for evaluating the performance of the system.


#### Acknowledgements
Parts of this project are based on code from [pubmed-rag-screener](https://github.com/milieere/pubmed-rag-screener), licensed under the MIT License.

