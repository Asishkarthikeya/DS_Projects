#!/bin/bash
# Code Crawler CLI - Quick Setup Script

set -e

echo "🕷️  Code Crawler CLI - Quick Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"
echo ""

# Install the CLI
echo "Installing Code Crawler CLI..."
pip install -e . --quiet
echo "✓ Installation complete"
echo ""

# Check if API key is set
if [ -z "$GOOGLE_API_KEY" ] && [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  No API key found!"
    echo ""
    echo "Please set your API key:"
    echo "  For Gemini: export GOOGLE_API_KEY='your-key-here'"
    echo "  For Groq:   export GROQ_API_KEY='your-key-here'"
    echo ""
    echo "Or configure it with:"
    echo "  code-crawler config set GOOGLE_API_KEY your-key-here"
    echo ""
else
    echo "✓ API key found"
    echo ""
fi

# Verify installation
echo "Verifying installation..."
if command -v code-crawler &> /dev/null; then
    echo "✓ code-crawler command is available"
    echo ""
    code-crawler version
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Index a codebase: code-crawler index ./your-project"
    echo "  2. Start chatting:   code-crawler chat"
    echo "  3. Get help:         code-crawler --help"
    echo ""
else
    echo "❌ Installation failed - command not found"
    echo ""
    echo "Try adding to PATH:"
    echo "  export PATH=\"\$PATH:\$HOME/.local/bin\""
    exit 1
fi
