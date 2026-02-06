# Codebase Agent - Technical Documentation

## 1. System Overview
**Codebase Agent** is an advanced AI-powered coding assistant designed to help developers navigate, understand, and modify large codebases. It uses a **Retrieval-Augmented Generation (RAG)** pipeline combined with **Agentic Workflows** to provide accurate, context-aware answers.

### Key Features
- **Deep Indexing**: Uses AST-based chunking (Tree-sitter) to understand code structure (Classes, Functions) rather than just text.
- **Hybrid Search**: Combines Dense Vector Search (semantic) with BM25 (keyword) and LLM-based reranking for high precision.
- **Agentic Capabilities**: Can autonomously plan multi-step refactoring tasks, analyze dependencies, and generate complex code.
- **Full-Width UI**: A modern Streamlit interface with full-screen Chat and Code Editor tabs.

---

## 2. Architecture

The system is built on a modular **Python** backend using **LangChain** and **Streamlit**.

### High-Level Components

```mermaid
graph TD
    User[User Interface (Streamlit)] <--> Backend[App Backend]
    Backend <--> Ingestion[Ingestion Engine]
    Backend <--> Retrieval[Retrieval Engine]
    Backend <--> Agents[Agent Workflow]
    
    Ingestion --> DB[(Vector DB / Chroma)]
    Retrieval <--> DB
    Agents <--> Retrieval
    Agents <--> LLM[LLM (Gemini/Groq)]
```

### Directory Structure

The codebase is organized into modular packages under `code_chatbot/`:

| Package | Description | Key Files |
| :--- | :--- | :--- |
| **`code_chatbot/ingestion`** | Handles file parsing, chunking, and embedding. | `chunker.py`, `indexer.py`, `universal_ingestor.py` |
| **`code_chatbot/retrieval`** | Implements RAG, Search, and Context assembly. | `rag.py`, `reranker.py`, `vector_store.py` |
| **`code_chatbot/agents`** | Autonomous agent logic using LangGraph. | `agent_workflow.py`, `tools.py` |
| **`code_chatbot/analysis`** | Static analysis tools (AST) for code understanding. | `ast_analysis.py`, `code_symbols.py` |
| **`code_chatbot/core`** | specialized configuration and utilities. | `config.py`, `prompts.py` |
| **`pages/`** | Streamlit UI pages. | `1_⚡_Code_Studio.py` |

---

## 3. Key Subsystems

### 3.1 Ingestion Engine (`code_chatbot/ingestion`)
This module is responsible for turning raw code files into searchable embeddings.

*   **`universal_ingestor.py`**: The entry point. It accepts GitHub URLs, Zip files, or Local paths. It filters out junk files (`.git`, `node_modules`, etc.) before processing.
*   **`chunker.py`**: The core intelligence. It uses **Tree-sitter** to parse code into an Abstract Syntax Tree (AST).
    *   *Strategy*: It attempts to keep Functions and Classes intact. If a function is too large, it recursively splits it.
    *   *Metadata*: Extracts function names, imported libraries, and cyclomatic complexity scores.
*   **`indexer.py`**: Manages the Vector Database (ChromaDB). It handles incremental indexing (hashing files to avoid re-indexing unchanged code).

### 3.2 Retrieval Engine (`code_chatbot/retrieval`)
This module powers the "Chat" interface.

*   **`rag.py`**: Contains the `ChatEngine` class.
    *   *Linear Mode*: Specific Question -> Embed -> Search -> Rerank -> LLM Answer.
    *   *Agent Mode*: Delegates complex queries to the Agent Graph.
    *   *Cleaning*: Includes robust logic to strip raw HTML artifacts (Source Chips) from LLM responses.
*   **`reranker.py`**: Uses a Cross-Encoder or LLM to re-score search results, ensuring the most relevant code snippets are at the top.

### 3.3 Agentic Workflow (`code_chatbot/agents`)
Used for complex requests like "Refactor the authentication system" or "Draw a diagram of the architecture".

*   **`agent_workflow.py`**: Defines a State Graph (using LangGraph).
    *   *Planner Node*: Breaks the user request into steps.
    *   *Executor Node*: Executes steps (Search, Read File, Write File).
    *   *Reflector Node*: Verifies the output.

---

## 4. UI Layout (`pages/1_⚡_Code_Studio.py`)

The User Interface has been redesigned for maximum readability:

*   **Sidebar**: Contains the File Explorer (navigation) and View Settings.
*   **Main Area (Tabs)**:
    1.  **💬 Chat**: Full-width chat interface for disturbance-free reading.
    2.  **📝 Code Editor**: Monaco-style code viewer.
    3.  **✨ Refactor**: Specialized UI for generating new code or refactoring existing files.
    4.  **🔍 Search**: Regex-enabled semantic search tool.

---

## 5. Deployment

The application is deployed on **Hugging Face Spaces**.

### Deployment Steps
1.  **Stage Changes**: `git add .`
2.  **Commit**: `git commit -m "Update message"`
3.  **Push**: `git push huggingface main`

### Environment Variables
Required secrets in `.env` or Hugging Face Settings:
*   `GOOGLE_API_KEY`: For Gemini 1.5/2.0 models.
*   `GROQ_API_KEY`: For Llama-3 models (optional).
