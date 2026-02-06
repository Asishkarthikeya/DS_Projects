# Code Crawler CLI - Distribution Guide

## For End Users

### Option 1: Install from GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/Asishkarthikeya/Codebase_Agent.git
cd Codebase_Agent

# Install the CLI
pip install -e .

# Verify installation
code-crawler --help
```

### Option 2: Install from PyPI (When Published)

```bash
# Once published to PyPI
pip install code-crawler-cli

# Verify installation
code-crawler --help
```

### Option 3: Install from Source ZIP

```bash
# Download and extract the ZIP
unzip Codebase_Agent.zip
cd Codebase_Agent

# Install
pip install -e .
```

## Quick Setup

### 1. Install Dependencies

The CLI will automatically install all required dependencies:
- `typer` - CLI framework
- `rich` - Terminal UI
- `prompt-toolkit` - Interactive prompts
- All existing codebase dependencies

### 2. Set API Key

```bash
# Option A: Environment variable (temporary)
export GOOGLE_API_KEY="your-api-key-here"

# Option B: Store in CLI config (persistent)
code-crawler config set GOOGLE_API_KEY your-api-key-here

# For Groq instead of Gemini
export GROQ_API_KEY="your-groq-key-here"
```

### 3. First Use

```bash
# Index a codebase
code-crawler index ./my-project

# Start chatting
code-crawler chat
```

## System Requirements

- **Python**: 3.9 - 3.13
- **OS**: macOS, Linux, Windows
- **Terminal**: Any modern terminal (iTerm2, Terminal.app, Windows Terminal, etc.)

## For Different Operating Systems

### macOS / Linux

```bash
# Install
pip install -e .

# Add to PATH (if needed)
export PATH="$PATH:$HOME/.local/bin"

# Set API key
export GOOGLE_API_KEY="your-key"
```

### Windows

```powershell
# Install
pip install -e .

# Set API key
$env:GOOGLE_API_KEY="your-key"

# Or permanently
setx GOOGLE_API_KEY "your-key"
```

## Configuration

The CLI stores configuration in:
- **macOS/Linux**: `~/.code-crawler/`
- **Windows**: `C:\Users\USERNAME\.code-crawler\`

Files created:
- `config.json` - User preferences
- `sessions/` - Conversation history
- `history/` - Command history

## Troubleshooting

### Command not found

```bash
# Ensure pip install location is in PATH
pip show storia-sage | grep Location

# Add to PATH
export PATH="$PATH:$(pip show storia-sage | grep Location | cut -d' ' -f2)/bin"
```

### API Key Issues

```bash
# Verify API key is set
echo $GOOGLE_API_KEY

# Or check config
code-crawler config get GOOGLE_API_KEY
```

### Import Errors

```bash
# Reinstall dependencies
pip install -e . --force-reinstall
```

## Uninstallation

```bash
# Uninstall the package
pip uninstall storia-sage

# Remove configuration (optional)
rm -rf ~/.code-crawler
```

## For Developers

### Building from Source

```bash
# Clone repository
git clone https://github.com/Asishkarthikeya/Codebase_Agent.git
cd Codebase_Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests (if available)
pytest tests/
```

### Creating a Distribution Package

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# This creates:
# - dist/storia_sage-0.1.0-py3-none-any.whl
# - dist/storia-sage-0.1.0.tar.gz

# Install from wheel
pip install dist/storia_sage-0.1.0-py3-none-any.whl
```

### Publishing to PyPI

```bash
# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Then publish to PyPI
twine upload dist/*

# Users can then install with:
# pip install code-crawler-cli
```

## Docker Installation (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . .

# Install CLI
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["code-crawler"]
CMD ["--help"]
```

Build and run:
```bash
docker build -t code-crawler .
docker run -e GOOGLE_API_KEY="your-key" code-crawler chat
```

## Network Installation (For Teams)

### Option 1: Shared Git Repository

```bash
# Team members clone and install
git clone https://your-company.com/code-crawler.git
cd code-crawler
pip install -e .
```

### Option 2: Private PyPI Server

```bash
# Upload to private PyPI
twine upload --repository-url https://pypi.your-company.com dist/*

# Team installs from private PyPI
pip install --index-url https://pypi.your-company.com code-crawler-cli
```

### Option 3: Shared Network Drive

```bash
# Install from network location
pip install -e \\\\network-drive\\shared\\code-crawler
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Install and Test CLI

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install CLI
        run: pip install -e .
      - name: Test CLI
        run: code-crawler --help
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/Asishkarthikeya/Codebase_Agent/issues
- Documentation: See CLI_README.md
- Quick Reference: See CLI_QUICKREF.md
