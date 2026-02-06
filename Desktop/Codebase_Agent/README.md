---
title: Code Crawler
emoji: 🕷️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# 🕷️ Code Crawler - AI Codebase Agent

An AI-powered codebase assistant that understands your code and helps you navigate, analyze, and modify it.

## Features

- 💬 **Interactive CLI** - Gemini-style conversational interface
- 📎 **Context Awareness** - Reference files with `@file.py` syntax
- 🔍 **Semantic Search** - Find patterns and understand code relationships
- 🔧 **Code Analysis** - AST parsing and call graph generation
- ✨ **Multi-Modal** - Web UI (Streamlit) + CLI + API

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Asishkarthikeya/Codebase_Agent.git
cd Codebase_Agent

# Run setup script
./setup.sh  # macOS/Linux
# or
setup.bat   # Windows

# Or install manually
pip install -e .
```

### Usage

```bash
# Set API key
export GOOGLE_API_KEY="your-key-here"

# Index a codebase
code-crawler index ./your-project

# Start interactive chat
code-crawler chat

# Or use the web UI
streamlit run app.py
```

## CLI Commands

- `code-crawler chat` - Interactive chat with streaming responses
- `code-crawler index <path>` - Index a codebase
- `code-crawler search <query>` - Search code semantically
- `code-crawler config` - Manage settings
- `code-crawler --help` - Show all commands

## Documentation

- **[Installation Guide](INSTALL.md)** - Detailed setup instructions
- **[CLI README](CLI_README.md)** - CLI features and usage
- **[Quick Reference](CLI_QUICKREF.md)** - Command cheat sheet
- **[Distribution Guide](DISTRIBUTION.md)** - For sharing with teams

## Requirements

- Python 3.9 - 3.13
- Google Gemini API key or Groq API key

## Project Structure

```
Codebase_Agent/
├── code_chatbot/          # Core library
│   ├── cli/              # CLI implementation
│   ├── agents/           # Agentic workflows
│   ├── analysis/         # AST & call graph
│   ├── ingestion/        # Document processing
│   └── retrieval/        # RAG pipeline
├── app.py                # Streamlit web UI
├── api/                  # FastAPI backend
└── frontend/             # Next.js frontend
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
