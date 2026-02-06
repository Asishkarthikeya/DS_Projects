# Code Crawler CLI - Quick Reference

## Installation
```bash
pip install -e .
```

## Setup
```bash
export GOOGLE_API_KEY="your-key-here"
```

## Commands

### Index a Codebase
```bash
code-crawler index <path>
code-crawler index ./my-project
code-crawler index https://github.com/user/repo
code-crawler index --provider groq --vector-db faiss
```

### Interactive Chat
```bash
code-crawler chat
code-crawler chat --file rag.py
code-crawler chat --model gemini-2.0-flash
```

### Search
```bash
code-crawler search "query"
code-crawler search "authentication" --limit 10
```

### Configuration
```bash
code-crawler config list
code-crawler config set model gemini-2.5-flash
code-crawler config get provider
```

## Chat Commands
- `/help` - Show commands
- `/exit` - Quit
- `/clear` - Clear history
- `/context` - Show contexts
- `/files` - List indexed files
- `/save` - Save session

## Context Syntax
- `@file.py` - Include file
- `@file.py:function` - Include function
- `@file.py:10-20` - Include lines 10-20

## File Locations
- Config: `~/.code-crawler/config.json`
- Sessions: `~/.code-crawler/sessions/`
- History: `~/.code-crawler/history/`
