# Troubleshooting Guide

## Common Issues and Solutions

### 1. Chroma "Different Settings" Error

**Error:**
```
ValueError: An instance of Chroma already exists for /path/to/vector_db with different settings
```

**Cause:** Multiple Chroma instances with conflicting configurations.

**Solution:**
```bash
# Option 1: Clear the temp directory
rm -rf /tmp/vector_db

# Option 2: Use a different vector DB
code-crawler index . --vector-db faiss

# Option 3: Force re-indexing
code-crawler index . --force
```

**Fixed in:** Latest version uses shared Chroma client to prevent this issue.

---

### 2. API Key Not Found

**Error:**
```
❌ API key not found for gemini
Set GOOGLE_API_KEY environment variable
```

**Solution:**
```bash
# Set environment variable
export GOOGLE_API_KEY="your-key-here"

# Or store in config
code-crawler config set GOOGLE_API_KEY your-key-here

# Verify it's set
echo $GOOGLE_API_KEY
```

---

### 3. Command Not Found

**Error:**
```
bash: code-crawler: command not found
```

**Solution:**
```bash
# Check if installed
pip show storia-sage

# Reinstall
pip install -e . --force-reinstall

# Add to PATH
export PATH="$PATH:$HOME/.local/bin"

# Make permanent (add to ~/.zshrc or ~/.bashrc)
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc
```

---

### 4. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'typer'
```

**Solution:**
```bash
# Reinstall dependencies
pip install -e . --force-reinstall

# Or install specific package
pip install typer rich prompt-toolkit
```

---

### 5. Rate Limit Errors

**Error:**
```
429 RESOURCE_EXHAUSTED
```

**Solution:**
```bash
# Use Groq instead of Gemini
export GROQ_API_KEY="your-groq-key"
code-crawler index . --provider groq

# Or wait and retry
# Gemini free tier: 15 requests/minute
# Wait 1-2 minutes and try again
```

---

### 6. Empty Index / No Results

**Error:**
```
Collection 'codebase' is empty!
```

**Solution:**
```bash
# Re-index with force flag
code-crawler index . --force

# Check if files were found
code-crawler index . --verbose

# Verify indexed files
code-crawler chat
You: /files
```

---

### 7. Session Not Found

**Error:**
```
Session '2026-02-05_23-10' not found
```

**Solution:**
```bash
# List available sessions
ls ~/.code-crawler/sessions/

# Start new session
code-crawler chat

# Sessions are auto-saved with timestamp format:
# YYYY-MM-DD_HH-MM.json
```

---

### 8. Python Version Mismatch

**Error:**
```
Package requires a different Python: 3.12.7 not in '>=3.9,<=3.13'
```

**Solution:**
```bash
# Check Python version
python --version

# Use compatible version (3.9-3.13)
# If needed, create virtual environment with correct version
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

---

### 9. Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied: '/tmp/vector_db'
```

**Solution:**
```bash
# Clear the directory
sudo rm -rf /tmp/vector_db

# Or use custom directory
export VECTOR_DB_PATH="$HOME/.code-crawler/vector_db"
code-crawler index .
```

---

### 10. Slow Indexing

**Issue:** Indexing takes a very long time

**Solution:**
```bash
# Use FAISS (faster than Chroma)
code-crawler index . --vector-db faiss

# Use Groq (faster than Gemini)
code-crawler index . --provider groq

# Index smaller directory
code-crawler index ./src  # Instead of entire project
```

---

## Debugging Tips

### Enable Verbose Logging

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run command
code-crawler index .
```

### Check Configuration

```bash
# View all settings
code-crawler config list

# Check specific value
code-crawler config get model
code-crawler config get provider
```

### Clear All Data

```bash
# Remove all CLI data
rm -rf ~/.code-crawler

# Remove vector database
rm -rf /tmp/vector_db

# Reinstall
pip uninstall storia-sage
pip install -e .
```

### Test Connection

```bash
# Test API key works
python -c "
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
api_key = os.getenv('GOOGLE_API_KEY')
embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001', google_api_key=api_key)
print('API key works!')
"
```

---

## Platform-Specific Issues

### macOS

**Issue:** SSL certificate errors

**Solution:**
```bash
# Install certificates
/Applications/Python\ 3.*/Install\ Certificates.command
```

### Windows

**Issue:** Path too long errors

**Solution:**
```powershell
# Enable long paths
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Linux

**Issue:** Missing system dependencies

**Solution:**
```bash
# Install build tools
sudo apt-get install python3-dev build-essential

# Or on Fedora/RHEL
sudo dnf install python3-devel gcc
```

---

## Getting Help

1. **Check Documentation**
   - [INSTALL.md](INSTALL.md)
   - [CLI_QUICKREF.md](CLI_QUICKREF.md)
   - [EXAMPLES.md](EXAMPLES.md)

2. **GitHub Issues**
   - Search existing issues
   - Create new issue with error details

3. **Include in Bug Reports**
   - Python version: `python --version`
   - OS: `uname -a` (Linux/Mac) or `ver` (Windows)
   - CLI version: `code-crawler version`
   - Full error traceback
   - Steps to reproduce

---

## Quick Fixes Checklist

- [ ] API key is set correctly
- [ ] Python version is 3.9-3.13
- [ ] CLI is installed: `pip show storia-sage`
- [ ] Command is in PATH
- [ ] Vector DB directory is writable
- [ ] No conflicting Chroma instances
- [ ] Dependencies are installed
- [ ] Internet connection is working (for API calls)
