# Publishing to PyPI - Step by Step Guide

## Prerequisites

1. **PyPI Account**
   - Create account at https://pypi.org/account/register/
   - Create account at https://test.pypi.org/account/register/ (for testing)

2. **Install Build Tools**
   ```bash
   pip install build twine
   ```

3. **Update Project Metadata**
   
   Edit `pyproject.toml`:
   ```toml
   [project]
   name = "code-crawler-cli"  # Change from "storia-sage"
   version = "0.1.0"
   description = "Gemini-style CLI for codebase analysis and chat"
   authors = [
       { name = "Your Name", email = "your.email@example.com" },
   ]
   ```

## Step 1: Prepare the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Update version in pyproject.toml if needed
# version = "0.1.0"
```

## Step 2: Build the Package

```bash
# Build distribution packages
python -m build

# This creates:
# - dist/code_crawler_cli-0.1.0-py3-none-any.whl
# - dist/code-crawler-cli-0.1.0.tar.gz
```

## Step 3: Test on TestPyPI

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your TestPyPI token>

# Test installation
pip install --index-url https://test.pypi.org/simple/ code-crawler-cli

# Verify it works
code-crawler --help
```

## Step 4: Publish to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your PyPI token>
```

## Step 5: Verify Publication

```bash
# Check on PyPI
# Visit: https://pypi.org/project/code-crawler-cli/

# Test installation
pip install code-crawler-cli

# Verify
code-crawler version
```

## Using API Tokens

### Create PyPI Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: "code-crawler-cli"
4. Scope: "Entire account" or specific project
5. Copy the token (starts with `pypi-`)

### Configure Twine

Create `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
```

Add `PYPI_TOKEN` to GitHub Secrets:
1. Go to repository Settings → Secrets → Actions
2. Add new secret: `PYPI_TOKEN`
3. Paste your PyPI token

## Version Management

### Semantic Versioning

- **0.1.0** - Initial release
- **0.1.1** - Bug fixes
- **0.2.0** - New features (backward compatible)
- **1.0.0** - Stable release

### Update Version

1. Edit `pyproject.toml`:
   ```toml
   version = "0.2.0"
   ```

2. Edit `code_chatbot/cli/__init__.py`:
   ```python
   __version__ = "0.2.0"
   ```

3. Create git tag:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

## Checklist Before Publishing

- [ ] Update version number
- [ ] Update CHANGELOG.md
- [ ] Run tests: `pytest tests/`
- [ ] Build locally: `python -m build`
- [ ] Test installation: `pip install dist/*.whl`
- [ ] Verify CLI works: `code-crawler --help`
- [ ] Update README.md
- [ ] Commit all changes
- [ ] Create git tag
- [ ] Test on TestPyPI first
- [ ] Publish to PyPI
- [ ] Create GitHub release

## After Publishing

Users can now install with:
```bash
pip install code-crawler-cli
```

Update documentation to reflect PyPI availability:
- Update README.md
- Update INSTALL.md
- Announce on GitHub releases
- Update project website (if any)

## Troubleshooting

### "File already exists"
- Version already published
- Increment version number and rebuild

### "Invalid distribution"
- Check `pyproject.toml` format
- Ensure all required fields are present

### "Authentication failed"
- Check token is correct
- Ensure token has correct scope
- Try regenerating token

## Resources

- PyPI: https://pypi.org
- TestPyPI: https://test.pypi.org
- Packaging Guide: https://packaging.python.org
- Twine Docs: https://twine.readthedocs.io
