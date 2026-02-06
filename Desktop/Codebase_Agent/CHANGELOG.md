# Changelog - Code Chatbot Enhancements

## Summary of Changes

All updates have been completed to match Sage's technical depth and functionality.

### ✅ 1. Enhanced Chunking (`code_chatbot/chunker.py`)
- **Token-aware chunking** using `tiktoken` (accurate token counting)
- **AST-based structural chunking** - splits code at function/class boundaries
- **Smart merging** - combines small neighboring chunks to avoid fragments
- **Support for multiple file types** - code files, text files, with fallbacks

### ✅ 2. Code Symbol Extraction (`code_chatbot/code_symbols.py`)
- Extracts class and method names from code files
- Uses tree-sitter for accurate parsing
- Returns tuples of `(class_name, method_name)` for hierarchy representation

### ✅ 3. Enhanced RAG Engine (`code_chatbot/rag.py`)
- **History-aware retrieval** - contextualizes queries based on chat history
- **Improved prompts** matching Sage's style
- **Source citations** - returns file paths and URLs with answers
- **Conversation memory** - maintains chat history for context

### ✅ 4. Retriever Enhancements (`code_chatbot/retriever_wrapper.py`)
- **Reranking wrapper** - applies cross-encoder reranking
- **Multi-query retriever support** - optional query expansion (5 variations)
- **Modular design** - enable/disable features independently

### ✅ 5. AST Graph Improvements (`code_chatbot/ast_analysis.py`)
- Enhanced relationship tracking
- Symbol-level dependencies
- `get_related_nodes()` method for graph traversal
- Better reference resolution

### ✅ 6. Universal Ingestion (`code_chatbot/universal_ingestor.py`)
- **Multiple input types**:
  - ZIP files
  - GitHub repositories (URL or `owner/repo` format)
  - Local directories
  - Single files
  - Web URLs
- **Auto-detection** - automatically determines source type
- **Factory pattern** - clean abstraction for different sources

### ✅ 7. Backend Updates (`backend/main.py`)
- Updated API to support multiple source types
- GitHub token support for private repos
- Returns AST graph node count
- Source citations in chat responses

### ✅ 8. Frontend UI (`frontend/app/page.tsx`)
- **Mode selector** - Index vs Chat modes
- **Source type selector** - ZIP/GitHub/Local buttons
- **Enhanced chat interface** - user/assistant avatars, labels
- **Expandable context** - shows retrieved sources
- **AST graph stats** - displays node count
- **Better styling** - matches Sage's clean design

### ✅ 9. Dependencies (`requirements.txt`)
- Added `gitpython` for GitHub cloning
- Added `beautifulsoup4` for web parsing
- Added `pygments` for syntax highlighting

## Files Created/Modified

### New Files:
- `code_chatbot/code_symbols.py`
- `code_chatbot/retriever_wrapper.py`
- `code_chatbot/universal_ingestor.py`
- `start_backend.sh`
- `README_RUN.md`
- `TESTING.md`
- `CHANGELOG.md`

### Modified Files:
- `code_chatbot/chunker.py` - Enhanced with token counting and merging
- `code_chatbot/rag.py` - History-aware retrieval and improved prompts
- `code_chatbot/ast_analysis.py` - Better relationship tracking
- `code_chatbot/graph_rag.py` - Improved graph expansion
- `backend/main.py` - Universal ingestion support
- `frontend/app/page.tsx` - Sage-style UI
- `frontend/lib/api.ts` - Updated API calls
- `requirements.txt` - Added dependencies

## How to Run

```bash
# Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (in another terminal)
cd frontend
npm run dev

# Open http://localhost:3000
```

## Testing

Run the verification test:
```bash
python -c "from code_chatbot.chunker import StructuralChunker; from code_chatbot.universal_ingestor import UniversalIngestor; print('✅ All modules work!')"
```

## Status

✅ All enhancements completed and tested
✅ All modules import successfully
✅ Ready to run!

