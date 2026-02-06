# Code Crawler CLI - Example Usage Scenarios

## Scenario 1: Understanding a New Codebase

**Goal**: Quickly understand how a new project works

```bash
# 1. Clone and index
git clone https://github.com/some-org/project.git
cd project
code-crawler index .

# 2. Ask questions
code-crawler chat
```

Example questions:
- "What does this project do?"
- "Show me the main entry point"
- "How is authentication handled?"
- "@auth/login.py Explain this file"

## Scenario 2: Debugging an Issue

**Goal**: Find and understand a bug

```bash
# Start chat with context
code-crawler chat --file src/buggy_module.py
```

Example session:
```
You: @src/buggy_module.py:process_data What could cause a null pointer here?

You: Show me all files that call this function

You: /search "process_data"
```

## Scenario 3: Code Review

**Goal**: Review changes before merging

```bash
# Index the feature branch
git checkout feature-branch
code-crawler index .

# Review specific files
code-crawler chat
```

Example questions:
- "@new_feature.py Does this follow our patterns?"
- "Are there any security issues in @auth/new_auth.py?"
- "What tests are needed for this feature?"

## Scenario 4: Documentation Generation

**Goal**: Generate documentation for a module

```bash
code-crawler chat
```

Example:
```
You: @api/routes/users.py Generate API documentation for this file

You: Create a README for the authentication module

You: Explain the database schema in @models/
```

## Scenario 5: Team Onboarding

**Goal**: Help new team members understand the codebase

```bash
# Index the main repository
code-crawler index https://github.com/company/main-repo

# Save common Q&A sessions
code-crawler chat
```

Share saved sessions with team:
```bash
# Sessions are in ~/.code-crawler/sessions/
# Share the JSON files with team members
```

## Scenario 6: Refactoring Planning

**Goal**: Plan a large refactoring

```bash
code-crawler chat
```

Example questions:
- "Find all files that use the old API"
- "What would break if I change this interface?"
- "Show me the dependency graph for @core/engine.py"

## Scenario 7: Cross-Repository Analysis

**Goal**: Understand how multiple repos interact

```bash
# Index multiple repositories
mkdir workspace
cd workspace

code-crawler index https://github.com/org/frontend
code-crawler index https://github.com/org/backend
code-crawler index https://github.com/org/shared

# Chat about the entire workspace
code-crawler chat
```

## Scenario 8: Learning a Framework

**Goal**: Learn how a framework is used in your codebase

```bash
code-crawler chat
```

Example:
```
You: How is React used in this project?

You: Show me examples of custom hooks

You: @components/UserProfile.tsx Explain this component
```

## Scenario 9: Security Audit

**Goal**: Find potential security issues

```bash
code-crawler search "password"
code-crawler search "api_key"
code-crawler search "eval("
```

Then in chat:
```
You: Are there any SQL injection vulnerabilities?

You: Check @auth/ for security best practices

You: Find all places where user input is used
```

## Scenario 10: Performance Investigation

**Goal**: Find performance bottlenecks

```bash
code-crawler chat
```

Example:
```
You: Find all database queries in @services/

You: What could be slow in @api/heavy_endpoint.py?

You: Show me all loops that could be optimized
```

## Tips for Effective Use

### 1. Use Context Wisely
```bash
# Add specific files
You: @file1.py @file2.py Compare these implementations

# Add functions
You: @utils.py:helper_function How is this used?

# Add line ranges
You: @main.py:100-150 What does this section do?
```

### 2. Combine Commands
```bash
# Search first, then chat about results
code-crawler search "authentication"
code-crawler chat
You: Explain the search results
```

### 3. Save Important Sessions
```bash
# Sessions auto-save to ~/.code-crawler/sessions/
# Load previous session
code-crawler chat --session 2026-02-06_00-00
```

### 4. Use Slash Commands
```
/help     - Show available commands
/files    - List indexed files
/context  - Show current context
/clear    - Clear conversation history
/save     - Save current session
```

### 5. Configure for Your Workflow
```bash
# Set preferred model
code-crawler config set model gemini-2.0-flash

# Set default provider
code-crawler config set provider groq

# View all settings
code-crawler config list
```

## Common Workflows

### Daily Development
```bash
# Morning: Index latest changes
git pull
code-crawler index . --force

# During day: Quick questions
code-crawler chat
```

### Code Review
```bash
# Review PR changes
git checkout pr-branch
code-crawler index .
code-crawler chat
You: Review the changes in this branch
```

### Bug Investigation
```bash
# Start with the error
code-crawler chat
You: @error_file.py:error_line What causes this error?
You: Show me the call stack
You: Find similar issues in the codebase
```
