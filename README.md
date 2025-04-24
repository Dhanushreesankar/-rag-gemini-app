# RAG Gemini App

A Streamlit-based Retrieval-Augmented Generation (RAG) application built on Google’s Gemini 1.5 Pro model.  
This app demonstrates how to combine document retrieval with a powerful generative AI to answer user questions based on a PDF knowledge source.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Architecture](#architecture)  
- [Getting Started](#getting-started)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Configuration](#configuration)  
- [Dependencies](#dependencies)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

Retrieval‑Augmented Generation (RAG) combines the strengths of information retrieval and generative AI. Instead of relying solely on what the model “remembers,” a RAG system first retrieves relevant context from an external source—in this case, a PDF file—and then uses that context to generate precise, up‑to‑date answers.

This repository provides a minimal, end‑to‑end example:

1. **Document Ingestion**: Load a PDF and split it into manageable text chunks.  
2. **Embedding & Indexing**: Convert each chunk into vector embeddings via Google Generative AI Embeddings and store them in Chroma.  
3. **Retrieval**: Given a user query, fetch the top‐k most relevant chunks from Chroma.  
4. **Generation**: Pass the retrieved context and user query to the Gemini 1.5 Pro model to produce a concise answer.  
5. **User Interface**: Expose a simple chat interface using Streamlit for interactive Q&A.

---

## Features

- **PDF Loader**: Automatically ingest any PDF document.  
- **Text Splitting**: Break long documents into 1,000‑character chunks for efficient embedding.  
- **Vector Embeddings**: Generate and index embeddings using Google Generative AI Embeddings.  
- **Chroma Vector Store**: Fast similarity search over document embeddings.  
- **Gemini 1.5 Pro LLM**: High‑quality, low‑latency answers via Google’s latest generative model.  
- **Streamlit Frontend**: One‑click deployment and interactive chat interface.  

---

## Architecture

flowchart TD
    A[PDF Document] --> B[PyPDFLoader]
    B --> C[TextSplitter]
    C --> D[GoogleGenerativeAIEmbeddings]
    D --> E[Chroma Vector Store]
    F[User Query] --> G[Retriever]
    E --> G
    G --> H[Gemini-1.5-Pro LLM]
    H --> I[Streamlit UI: Answer]
