# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) system that integrates with Alibaba Cloud's Qwen (千问) models through OpenAI-compatible APIs. The system processes PDF documents, creates embeddings using Qwen's text-embedding models, and performs question-answering using the qwen-plus language model.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies with uv
uv pip install -r requirements.txt
```

### Running the Application

#### Web UI Applications (推荐)
```bash
# 一键启动完整的RAG系统（包含GitHub OAuth登录）
python start_with_oauth.py

# 或者分别启动两个服务：
# 1. OAuth服务（端口8001）
python gradio_oauth_app.py

# 2. 主应用（端口7860）
python gradio_app.py
```

#### 命令行应用
```bash
# 传统RAG管道（无UI）
python qianwen_paper_qa.py

# 基础嵌入测试
python embedding.py

# 简单入口点
python main.py
```

### Testing
```bash
# Test custom LLM integration
python test/test_custom_llm.py

# Test retrieval functionality only
python test/test_retrieval_only.py

# Test ChromaDB integration
python test/test_chroma_integration.py

# Debug DashScope API
python test/debug_dashscope.py
```

## Architecture

### Core Components

1. **CustomQwenEmbeddings** (`custom_qwen_embeddings.py`): Custom embedding class that wraps Qwen's text-embedding-v4 model using OpenAI-compatible interface
2. **Main RAG Pipeline** (`qianwen_paper_qa.py`): Complete RAG implementation with PDF processing, embedding caching, and Q&A functionality
3. **Caching System**: Intelligent PDF embedding cache using MD5 hashing to avoid reprocessing documents

### Key Features

- **PDF Processing**: Uses PyPDFLoader to extract text from PDF documents
- **Embedding Caching**: Implements MD5-based caching system in `pdf_embeddings_cache/` directory to avoid regenerating embeddings
- **Batch Processing**: Handles embeddings in batches of 4 with retry mechanism and exponential backoff
- **Vector Storage**: Uses FAISS for efficient similarity search with persistent storage in `qianwen_faiss_index/`
- **Chat History**: SQLite database for persistent chat history storage in `sqlite/chat_history.db`
- **GitHub OAuth Login**: Multi-user support with secure authentication
- **User Data Isolation**: Each user has independent chat history and API key storage
- **Encrypted API Key Storage**: User API keys are encrypted and stored securely
- **Auto-recovery**: Saved API keys are automatically loaded on user login

### API Configuration

The system uses Alibaba Cloud's DashScope API through OpenAI-compatible endpoints:
- **Base URL**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Embedding Model**: `text-embedding-v4` (1024 dimensions)
- **LLM Model**: `qwen-plus`
- **API Key**: Set `DASHSCOPE_API_KEY` environment variable

### Data Flow

1. PDF document loaded and hashed for cache lookup
2. If cached embeddings exist, load from `pdf_embeddings_cache/{hash}/`
3. Otherwise, split document into chunks (1000 chars, 200 overlap)
4. Generate embeddings using CustomQwenEmbeddings in batches
5. Create FAISS vector store and save cache
6. Use retriever with k=5 similar chunks for context
7. Generate answers using qwen-plus model via OpenAI-compatible interface

### Dependencies

- **Core**: langchain, langchain-community, langchain-openai, openai
- **Vector DB**: faiss-cpu, langchain-chroma
- **PDF Processing**: pypdf
- **Environment**: python-dotenv
- **Database**: sqlite3 (built-in)
- **UI**: gradio
- **Special**: langchain-dashscope for native integration (used in some files)

## Important Notes

- Always use the caching system to avoid unnecessary API calls and improve performance
- The system expects PDF files to be in the root directory
- Cache files are automatically managed but can be manually cleared if needed
- Embedding dimensions are fixed at 1024 for consistency with Qwen's text-embedding-v4 model