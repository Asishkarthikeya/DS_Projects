# Installation Guide

## Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/Asishkarthikeya/Codebase_Agent.git
cd Codebase_Agent

# 2. Install
pip install -e .

# 3. Set API key
export GOOGLE_API_KEY="your-key-here"

# 4. Start using
code-crawler --help
```

## System Requirements

- Python 3.9 - 3.13
- pip (Python package manager)
- Git (for cloning)

## Installation Methods

### Method 1: From GitHub (Recommended)

```bash
git clone https://github.com/Asishkarthikeya/Codebase_Agent.git
cd Codebase_Agent
pip install -e .
```

### Method 2: From PyPI (Coming Soon)

```bash
pip install code-crawler-cli
```

### Method 3: From ZIP File

```bash
# Download ZIP from GitHub
unzip Codebase_Agent.zip
cd Codebase_Agent
pip install -e .
```

## Setup

### 1. Get API Key

**For Gemini (Google):**
1. Go to https://makersuite.google.com/app/apikey
2. Create an API key
3. Copy the key

**For Groq:**
1. Go to https://console.groq.com
2. Create an API key
3. Copy the key

### 2. Configure API Key

**Option A: Environment Variable**
```bash
# macOS/Linux
export GOOGLE_API_KEY="your-key-here"

# Windows PowerShell
$env:GOOGLE_API_KEY="your-key-here"

# Windows CMD
set GOOGLE_API_KEY=your-key-here
```

**Option B: CLI Config (Persistent)**
```bash
code-crawler config set GOOGLE_API_KEY your-key-here
```

### 3. Verify Installation

```bash
code-crawler --help
code-crawler version
```

## First Steps

### Index a Codebase

```bash
# Index current directory
code-crawler index .

# Index a specific project
code-crawler index /path/to/project

# Index from GitHub
code-crawler index https://github.com/user/repo
```

### Start Chatting

```bash
code-crawler chat
```

## Platform-Specific Notes

### macOS

```bash
# May need to add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Linux

```bash
# May need to add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Windows

```powershell
# Run as Administrator if needed
pip install -e .

# Add to PATH permanently
[Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "your-key", "User")
```

## Troubleshooting

### "command not found: code-crawler"

```bash
# Find where pip installed it
pip show storia-sage

# Add to PATH
export PATH="$PATH:$HOME/.local/bin"
```

### "No module named 'typer'"

```bash
# Reinstall dependencies
pip install -e . --force-reinstall
```

### "API key not found"

```bash
# Check if set
echo $GOOGLE_API_KEY

# Set it
export GOOGLE_API_KEY="your-key"
```

## Uninstall

```bash
# Remove package
pip uninstall storia-sage

# Remove config (optional)
rm -rf ~/.code-crawler
```

## Next Steps

1. Read the [Quick Reference](CLI_QUICKREF.md)
2. Check the [CLI README](CLI_README.md)
3. Try `code-crawler chat --help`

## Support

- Issues: https://github.com/Asishkarthikeya/Codebase_Agent/issues
- Docs: See README.md files in the repository
