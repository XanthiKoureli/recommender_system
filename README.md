# A Retrieval-Augmented Generation System for Biomedical Question Answering

## 1. Introduction

The rapid proliferation of biomedical literature presents a significant challenge for clinicians and researchers who need to stay abreast of the latest findings. Traditional search engines, while effective at keyword-based document retrieval, often require users to manually sift through numerous articles to find and synthesize information. This project proposes and implements a conversational question-answering system designed to alleviate this burden. 

Our solution leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide users with synthesized, context-aware answers to complex biomedical questions. The system uses the vast repository of PubMed as its primary knowledge source, transforming it from a passive database into an interactive knowledge base. By integrating advanced language models with a robust retrieval pipeline, the system can understand user intent, fetch relevant scientific abstracts, and generate coherent, explanatory answers, complete with a conversational interface for follow-up inquiries.

## 2. System Architecture

The system is designed based on a modular, multi-layer RAG architecture that separates concerns into a presentation layer, a backend logic layer, and a data storage layer.

- **Presentation Layer**: A web-based user interface built with Streamlit (`app.py`). It is responsible for capturing user input (both direct questions and uploaded medical notes), rendering the generated answers, and managing the interactive chat component.

- **Backend Logic & Orchestration Layer**: The core of the system, which executes the RAG pipeline. This layer is responsible for orchestrating the flow of data between the UI, the external PubMed API, the vector database, and the Large Language Model (LLM).

- **Data & Storage Layer**: This layer consists of two primary components:
    1.  A **Vector Database** (ChromaDB) that stores the embeddings of scientific abstracts for efficient semantic retrieval.
    2.  A **Local JSON-based Metadata Store** that maintains records of user queries, retrieved abstracts, and generated answers, enabling session persistence and chat history.

## 3. Methodology and Implementation

The system's workflow is divided into four key stages: Data Acquisition, Knowledge Base Construction, Retrieval-Augmented Generation, and the Interactive Conversational Component.

### 3.1. Data Acquisition and Pre-processing

The process begins with user input, which is then standardized into a machine-readable query.

1.  **Dual Input Modalities**: The system accepts two forms of input: a direct, structured question or unstructured medical notes provided as a text file.
2.  **Query Generation from Notes**: In the case of uploaded medical notes, an LLM is employed to analyze the text and synthesize a concise, relevant research question. This abstracts the complexity of the raw notes into a focused query.
3.  **Query Simplification**: The resulting query is then passed to a simplification module (`pubmed_query_simplification.py`), which refines the language to be more effective for the PubMed API, increasing the relevance of search results.
4.  **Abstract Retrieval**: The processed query is used to fetch corresponding abstracts from PubMed via the `metapub` library.

### 3.2. Knowledge Base Construction

Once abstracts are retrieved, they are transformed into a structured knowledge base for semantic search.

1.  **Text Embedding**: Each abstract is processed by a sophisticated embedding model to generate a high-dimensional vector representation. This vector captures the semantic meaning of the text.
2.  **Embedding Model Selection**: The system is designed to be modular, supporting several state-of-the-art embedding models, including `OpenAI`, `Mistral`, `PubMedBERT`, and `SapBERT`. `SapBERT`, a model pre-trained on biomedical literature and fine-tuned for entity representation, was selected for its superior performance in capturing nuanced semantic relationships within this specific domain.
3.  **Vector Storage**: The generated embeddings are stored in a ChromaDB vector database. A key architectural choice is the creation of a **separate, isolated collection for each user query**. This ensures that the context for each question remains distinct and prevents information leakage between different user sessions.

### 3.3. Retrieval-Augmented Generation

This stage involves retrieving relevant context and generating a final answer.

1.  **Retrieval**: When a question is posed, the system performs a vector similarity search on the corresponding ChromaDB collection. The user's question is embedded, and the system retrieves the top-N most semantically similar abstracts.
2.  **Prompt Engineering**: The retrieved abstracts (the context) are formatted and injected into a structured prompt template (`qa_template`). This prompt explicitly instructs the LLM to synthesize an answer based *only* on the provided scientific texts.
3.  **Generation**: The final prompt is sent to a Large Language Model (e.g., from OpenAI), which generates a comprehensive, human-readable answer based on the retrieved context.

### 3.4. Interactive Conversational Component

The system supports stateful, multi-turn conversations.

1.  **Session Management**: The `ChatAgent` class manages the conversational state, including the chat history for each query.
2.  **Contextual Follow-up**: When a user asks a follow-up question, the system constructs a new prompt that includes not only the new question and retrieved documents but also the preceding conversation history. This enables the LLM to generate responses that are contextually aware and relevant to the ongoing dialogue.

## 4. Evaluation Methodology

To quantitatively assess the system's performance, a dedicated evaluation pipeline was developed (`evaluation/evaluation.py`). The evaluation focuses on both the retrieval and generation components.

- **Dataset**: The evaluation is performed on the `training12b_new.json` dataset, which contains a series of biomedical questions, a list of "gold standard" relevant PubMed articles (PMIDs), and an ideal, human-written answer.

- **Retrieval Evaluation**: The performance of the abstract retrieval component is measured using **Recall@5**. This metric calculates the proportion of gold-standard documents that are successfully retrieved within the top 5 results returned by the system.

- **Generation Evaluation**: The quality of the LLM-generated answers is assessed using two standard metrics:
    1.  **BERTScore**: This metric evaluates semantic similarity by comparing the contextual embeddings of the generated answer against the ideal answer. It is robust to differences in phrasing as long as the underlying meaning is preserved.
    2.  **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: This metric measures lexical overlap by counting the number of n-grams (word sequences) that are common between the generated and ideal answers. It provides a measure of surface-level similarity.

## 5. Conclusion and Future Work

This project successfully demonstrates the viability of a Retrieval-Augmented Generation system for complex biomedical question answering. By combining a powerful retrieval pipeline with the generative capabilities of large language models, the system provides a valuable tool for researchers and clinicians. 

Future work could focus on several areas for improvement. This includes expanding the knowledge base beyond PubMed to other sources like clinical trial databases, experimenting with more advanced multi-modal models capable of interpreting figures and tables within articles, and conducting a formal user study to evaluate the system's real-world utility and impact on clinical decision-making or research workflows.