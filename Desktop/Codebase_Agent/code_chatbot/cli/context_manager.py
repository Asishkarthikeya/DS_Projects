"""
Context manager for handling @file.py mentions (Gemini-style)
"""
import os
import re
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class ContextManager:
    """Manages file and code context for chat sessions"""
    
    def __init__(self):
        self.active_contexts: List[Dict[str, str]] = []
        self.repo_dir = os.getcwd()
    
    def parse_mentions(self, user_input: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Parse @mentions from user input and extract contexts.
        
        Supported formats:
        - @file.py - entire file
        - @file.py:function_name - specific function
        - @file.py:10-20 - line range
        - @codebase - use full indexed codebase
        
        Returns:
            Tuple of (clean_input, list of contexts)
        """
        # Find all @mentions
        mention_pattern = r'@([\w/.:-]+)'
        mentions = re.findall(mention_pattern, user_input)
        
        # Remove mentions from input
        clean_input = re.sub(mention_pattern, '', user_input).strip()
        
        # Process each mention
        contexts = []
        for mention in mentions:
            context = self._resolve_mention(mention)
            if context:
                contexts.append(context)
                # Add to active contexts if not already there
                if not any(ctx['source'] == context['source'] for ctx in self.active_contexts):
                    self.active_contexts.append(context)
        
        return clean_input, contexts
    
    def _resolve_mention(self, mention: str) -> Optional[Dict[str, str]]:
        """Resolve a mention to actual content"""
        
        # Handle @codebase
        if mention == "codebase":
            return {
                "source": "@codebase",
                "content": "Using full indexed codebase context",
                "type": "codebase"
            }
        
        # Parse file:target format
        if ':' in mention:
            file_path, target = mention.split(':', 1)
        else:
            file_path = mention
            target = None
        
        # Try to find the file
        full_path = self._find_file(file_path)
        if not full_path:
            return {
                "source": f"@{mention}",
                "content": f"File not found: {file_path}",
                "type": "error"
            }
        
        # Read file content
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract specific target if specified
            if target:
                content = self._extract_target(content, target, full_path)
            
            return {
                "source": f"@{mention}",
                "content": content,
                "type": "file",
                "path": full_path
            }
        
        except Exception as e:
            return {
                "source": f"@{mention}",
                "content": f"Error reading file: {e}",
                "type": "error"
            }
    
    def _find_file(self, file_path: str) -> Optional[str]:
        """Find a file in the repository"""
        # Try as absolute path
        if os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path
        
        # Try relative to current directory
        rel_path = os.path.join(self.repo_dir, file_path)
        if os.path.exists(rel_path):
            return rel_path
        
        # Try to find by filename (search in repo)
        filename = os.path.basename(file_path)
        for root, dirs, files in os.walk(self.repo_dir):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', 'venv', '__pycache__']]
            
            if filename in files:
                candidate = os.path.join(root, filename)
                # If path components match, prefer this one
                if file_path in candidate:
                    return candidate
        
        return None
    
    def _extract_target(self, content: str, target: str, file_path: str) -> str:
        """Extract specific target from file content"""
        
        # Check if target is a line range (e.g., 10-20)
        if re.match(r'^\d+-\d+$', target):
            start, end = map(int, target.split('-'))
            lines = content.split('\n')
            return '\n'.join(lines[start-1:end])
        
        # Otherwise, try to find function/class definition
        # Simple heuristic: look for "def target" or "class target"
        patterns = [
            rf'^def {target}\s*\(',
            rf'^class {target}\s*[\(:]',
            rf'^\s+def {target}\s*\(',  # indented method
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern, line):
                    # Extract function/class (simple approach: until next def/class at same level)
                    result = [line]
                    indent = len(line) - len(line.lstrip())
                    
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j]
                        if not next_line.strip():  # empty line
                            result.append(next_line)
                            continue
                        
                        next_indent = len(next_line) - len(next_line.lstrip())
                        
                        # Stop at same or lower indentation level with def/class
                        if next_indent <= indent and re.match(r'^\s*(def|class)\s+', next_line):
                            break
                        
                        result.append(next_line)
                    
                    return '\n'.join(result)
        
        # If not found, return full content with a note
        return f"# Could not find '{target}' in {file_path}\n# Showing full file:\n\n{content}"
    
    def add_file_context(self, file_path: str):
        """Manually add a file to context"""
        context = self._resolve_mention(file_path)
        if context and not any(ctx['source'] == context['source'] for ctx in self.active_contexts):
            self.active_contexts.append(context)
    
    def remove_context(self, source: str):
        """Remove a context by source"""
        self.active_contexts = [ctx for ctx in self.active_contexts if source not in ctx['source']]
    
    def clear_all(self):
        """Clear all active contexts"""
        self.active_contexts = []
    
    def get_all_contexts(self) -> List[Dict[str, str]]:
        """Get all active contexts"""
        return self.active_contexts
    
    def get_context_summary(self) -> str:
        """Get a summary of active contexts"""
        if not self.active_contexts:
            return "No active contexts"
        
        summary = []
        for ctx in self.active_contexts:
            summary.append(f"{ctx['source']} ({len(ctx['content'])} chars)")
        
        return ", ".join(summary)
