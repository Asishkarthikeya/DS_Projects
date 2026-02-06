"""
Interactive chat session manager - Gemini-style conversational interface
"""
import os
import re
import json
from datetime import datetime
from typing import Optional, List, Tuple
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from code_chatbot.cli.context_manager import ContextManager
from code_chatbot.cli.output_formatter import OutputFormatter
from code_chatbot.cli.config import Config


class ChatSession:
    """Manages an interactive chat session with the codebase"""
    
    def __init__(
        self,
        initial_file: Optional[str] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.console = Console()
        self.config = Config()
        self.context_manager = ContextManager()
        self.formatter = OutputFormatter(self.console)
        
        # Session settings
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.model = model or self.config.get("model", "gemini-2.5-flash")
        self.provider = provider or self.config.get("provider", "gemini")
        
        # Initialize chat engine
        self.chat_engine = None
        self.conversation_history = []
        
        # Initial context
        if initial_file:
            self.context_manager.add_file_context(initial_file)
        
        # Setup prompt session
        history_dir = Path.home() / ".code-crawler" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_dir / "chat_history.txt")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self._create_completer(),
        )
    
    def _create_completer(self) -> WordCompleter:
        """Create auto-completer for commands and common terms"""
        commands = [
            "/help", "/exit", "/clear", "/save", "/load",
            "/context", "/remove", "/reset", "/model", "/files"
        ]
        return WordCompleter(commands, ignore_case=True)
    
    def _load_chat_engine(self):
        """Load or initialize the chat engine"""
        if self.chat_engine is not None:
            return
        
        try:
            from code_chatbot.ingestion.indexer import Indexer
            from code_chatbot.retrieval.rag import ChatEngine
            
            # Get API key
            api_key = os.getenv("GOOGLE_API_KEY") if self.provider == "gemini" else os.getenv("GROQ_API_KEY")
            if not api_key:
                self.console.print("[red]❌ API key not found. Please set GOOGLE_API_KEY or GROQ_API_KEY[/red]")
                self.console.print("[yellow]Run: code-crawler config set GOOGLE_API_KEY your-key-here[/yellow]")
                return False
            
            # Load indexer
            indexer = Indexer(provider=self.provider, api_key=api_key)
            retriever = indexer.get_retriever(vector_db_type="chroma")
            
            # Initialize chat engine
            self.chat_engine = ChatEngine(
                retriever=retriever,
                model_name=self.model,
                provider=self.provider,
                api_key=api_key,
                use_agent=True,
                repo_dir=os.getcwd()
            )
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Failed to load chat engine: {e}[/red]")
            self.console.print("[yellow]💡 Have you indexed a codebase? Run: code-crawler index <path>[/yellow]")
            return False
    
    def start(self):
        """Start the interactive chat session"""
        # Show welcome banner
        self.formatter.show_welcome(self.model, self.provider)
        
        # Load chat engine
        if not self._load_chat_engine():
            return
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = self.prompt_session.prompt("\n[bold cyan]You:[/bold cyan] ")
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break  # Exit command
                    continue
                
                # Parse context mentions (@file.py)
                clean_query, contexts = self.context_manager.parse_mentions(user_input)
                
                # Show context if added
                if contexts:
                    self.formatter.show_context_added(contexts)
                
                # Get all active contexts
                all_contexts = self.context_manager.get_all_contexts()
                
                # Stream response
                self._stream_response(clean_query, all_contexts)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]⚠️  Use /exit to quit or Ctrl+D[/yellow]")
                continue
            except EOFError:
                break
        
        # Save session and exit
        self._save_session()
        self.formatter.show_goodbye(self.session_id)
    
    def _stream_response(self, query: str, contexts: List[str]):
        """Stream the AI response with live updates"""
        try:
            # Prepare query with context
            full_query = query
            if contexts:
                context_str = "\n\n".join([f"Context from {ctx['source']}:\n{ctx['content']}" for ctx in contexts])
                full_query = f"{context_str}\n\nQuestion: {query}"
            
            # Get streaming response
            generator, sources = self.chat_engine.stream_chat(full_query)
            
            # Stream with live rendering
            self.console.print("\n[bold green]Assistant:[/bold green]")
            
            full_response = ""
            with Live(console=self.console, refresh_per_second=10) as live:
                for chunk in generator:
                    full_response += chunk
                    # Render as markdown
                    live.update(Markdown(full_response))
            
            # Show sources
            if sources:
                self.formatter.show_sources(sources)
            
            # Save to history
            self.conversation_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.console.print(f"[red]❌ Error: {e}[/red]")
    
    def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns False if should exit."""
        cmd = command.lower().strip()
        
        if cmd == "/exit" or cmd == "/quit":
            return False
        
        elif cmd == "/help":
            self._show_help()
        
        elif cmd == "/clear":
            self.chat_engine.clear_memory()
            self.conversation_history = []
            self.console.print("[green]✅ Conversation history cleared[/green]")
        
        elif cmd == "/context":
            self._show_current_context()
        
        elif cmd.startswith("/remove"):
            parts = cmd.split(maxsplit=1)
            if len(parts) > 1:
                self.context_manager.remove_context(parts[1])
                self.console.print(f"[green]✅ Removed context: {parts[1]}[/green]")
        
        elif cmd == "/reset":
            self.context_manager.clear_all()
            self.console.print("[green]✅ All contexts cleared[/green]")
        
        elif cmd.startswith("/model"):
            parts = cmd.split(maxsplit=1)
            if len(parts) > 1:
                self.model = parts[1]
                self.chat_engine = None  # Force reload
                self._load_chat_engine()
                self.console.print(f"[green]✅ Switched to model: {self.model}[/green]")
        
        elif cmd == "/save":
            self._save_session()
            self.console.print(f"[green]✅ Session saved: {self.session_id}[/green]")
        
        elif cmd == "/files":
            self._show_indexed_files()
        
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("[yellow]Type /help for available commands[/yellow]")
        
        return True
    
    def _show_help(self):
        """Show help information"""
        help_table = Table(title="Available Commands", show_header=True)
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        commands = [
            ("/help", "Show this help message"),
            ("/exit, /quit", "Exit the chat session"),
            ("/clear", "Clear conversation history"),
            ("/context", "Show current context"),
            ("/remove <file>", "Remove a file from context"),
            ("/reset", "Clear all contexts"),
            ("/model <name>", "Switch LLM model"),
            ("/save", "Save current session"),
            ("/files", "Show indexed files"),
            ("@file.py", "Add file to context (in your message)"),
            ("@file.py:func", "Add specific function to context"),
        ]
        
        for cmd, desc in commands:
            help_table.add_row(cmd, desc)
        
        self.console.print(help_table)
    
    def _show_current_context(self):
        """Show currently active contexts"""
        contexts = self.context_manager.get_all_contexts()
        if not contexts:
            self.console.print("[yellow]No active contexts[/yellow]")
            return
        
        self.console.print("\n[bold]Active Contexts:[/bold]")
        for ctx in contexts:
            self.console.print(f"  📎 {ctx['source']} ({len(ctx['content'])} chars)")
    
    def _show_indexed_files(self):
        """Show files in the indexed codebase"""
        if hasattr(self.chat_engine, 'repo_files') and self.chat_engine.repo_files:
            self.console.print("\n[bold]Indexed Files:[/bold]")
            for file in sorted(self.chat_engine.repo_files[:20]):  # Show first 20
                self.console.print(f"  📄 {file}")
            if len(self.chat_engine.repo_files) > 20:
                self.console.print(f"\n  ... and {len(self.chat_engine.repo_files) - 20} more files")
        else:
            self.console.print("[yellow]No indexed files found[/yellow]")
    
    def _save_session(self):
        """Save conversation session to disk"""
        sessions_dir = Path.home() / ".code-crawler" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = sessions_dir / f"{self.session_id}.json"
        
        session_data = {
            "session_id": self.session_id,
            "model": self.model,
            "provider": self.provider,
            "conversation": self.conversation_history,
            "created_at": datetime.now().isoformat()
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
