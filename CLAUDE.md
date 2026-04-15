# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Modular RAG MCP Server** — a pluggable Retrieval-Augmented Generation framework that exposes tools via the Model Context Protocol (MCP). It implements hybrid search (BM25 + Dense Embedding), reranking, multimodal image processing, and provides a Streamlit dashboard for observability.

## Common Commands

```bash
# Run the project (loads settings, initializes)
python main.py

# Run tests
pytest                          # Run all tests
pytest tests/unit/              # Run only unit tests
pytest tests/unit/test_llm_factory.py -v  # Run single test file
pytest -m unit                  # Run tests with specific marker

# Run the Streamlit dashboard
streamlit run src/observability/dashboard/app.py

# Run MCP server
# (depends on implementation in src/mcp_server/server.py)
```

## Architecture

### Module Organization

```
src/
├── core/              # Core RAG logic (query engine, response, tracing, settings)
├── ingestion/         # Data ingestion pipeline (chunking, embedding, storage)
├── libs/              # Pluggable abstractions with factory pattern
│   ├── llm/           # LLM providers (Azure, OpenAI, DeepSeek, Ollama)
│   ├── embedding/     # Embedding models
│   ├── reranker/      # Reranking (Cross-Encoder, LLM-based)
│   ├── vector_store/  # Vector storage backends (Chroma)
│   ├── splitter/      # Text splitting strategies
│   └── loader/        # Document loaders (PDF)
├── mcp_server/        # MCP protocol server exposing tools
└── observability/     # Dashboard and evaluation
```

### Pluggable Architecture Pattern

Every core component follows the **Base Class + Factory** pattern:

1. **Base abstraction** (e.g., `src/libs/llm/base_llm.py`) defines the interface
2. **Provider implementations** (e.g., `azure_llm.py`, `openai_llm.py`) implement it
3. **Factory** (e.g., `src/libs/llm/llm_factory.py`) instantiates based on config

Configuration lives in `config/settings.yaml` and drives factory instantiation. Key config sections: `llm`, `embedding`, `vector_store`, `retrieval`, `rerank`, `evaluation`, `observability`.

### Key Abstractions

- **LLM**: `BaseLLM` → `AzureLLM`, `OpenAILLM`, `DeepSeekLLM`, `OllamaLLM`
- **Embedding**: `BaseEmbedding` → `OpenAIEmbedding`, `AzureEmbedding`, `OllamaEmbedding`
- **Reranker**: `BaseReranker` → `CrossEncoderReranker`, `LLMReranker`
- **VectorStore**: `BaseVectorStore` → `ChromaStore`
- **Splitter**: `BaseSplitter` → `RecursiveSplitter`, `SemanticSplitter`, `FixedLengthSplitter`
- **Loader**: `BaseLoader` → `PDFLoader`

### MCP Tools

The MCP server exposes these tools (defined in `src/mcp_server/tools/`):
- `query_knowledge_hub` — Search the knowledge base
- `list_collections` — List available collections
- `get_document_summary` — Get document summary

## Configuration

All settings are in `config/settings.yaml`. The `Settings` class in `src/core/settings.py` validates required fields:
- `llm.provider`, `llm.model`
- `embedding.provider`, `embedding.model`
- `vector_store.backend`
- `retrieval.sparse_backend`, `retrieval.fusion_algorithm`
- `rerank.backend`
- `evaluation.backends`
- `observability.enabled`

## Testing

Tests use pytest markers defined in `pyproject.toml`:
- `@pytest.mark.unit` — Unit tests
- `@pytest.mark.integration` — Integration tests
- `@pytest.mark.e2e` — End-to-end tests

## Skills System

The project includes Agent Skills in `.github/skills/` that drive development:
- **auto-coder**: Generates code from DEV_SPEC
- **qa-tester**: Automated testing
- **setup**: Environment configuration
- **resume-writer**: Generate resume content
- **skill-creator**: Create new skills

These are invoked through the Copilot/Claude dialog (e.g., "setup", "写简历").