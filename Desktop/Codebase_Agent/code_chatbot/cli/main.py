"""
Main CLI entry point for Code Crawler
Gemini-style conversational interface
"""
import typer
from rich.console import Console
from rich.panel import Panel
from typing import Optional
import os

app = typer.Typer(
    name="code-crawler",
    help="🕷️ AI-powered codebase assistant - Chat with your code like Gemini CLI",
    add_completion=False,
)

console = Console()


@app.command()
def chat(
    file: Optional[str] = typer.Option(
        None,
        "--file",
        "-f",
        help="Start chat with a specific file context"
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Load a previous conversation session"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model to use (e.g., gemini-2.5-flash, llama-3.3-70b-versatile)"
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider (gemini or groq)"
    ),
):
    """
    Start an interactive chat session with your indexed codebase.
    
    Examples:
        code-crawler chat
        code-crawler chat --file code_chatbot/retrieval/rag.py
        code-crawler chat --session 2026-02-05_23-10
    """
    from code_chatbot.cli.chat_session import ChatSession
    
    session_manager = ChatSession(
        initial_file=file,
        session_id=session,
        model=model,
        provider=provider
    )
    session_manager.start()


@app.command()
def index(
    path: str = typer.Argument(..., help="Path to codebase (directory, GitHub URL, or ZIP file)"),
    provider: str = typer.Option(
        "gemini",
        "--provider",
        "-p",
        help="LLM provider (gemini or groq)"
    ),
    vector_db: str = typer.Option(
        "chroma",
        "--vector-db",
        "-v",
        help="Vector database (chroma, faiss, or qdrant)"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-indexing even if already indexed"
    ),
):
    """
    Index a codebase for semantic search and chat.
    
    Examples:
        code-crawler index ./my-project
        code-crawler index https://github.com/user/repo
        code-crawler index ./codebase.zip --provider groq
    """
    from code_chatbot.cli.commands.index import index_codebase
    
    index_codebase(path, provider, vector_db, force)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(
        5,
        "--limit",
        "-l",
        help="Number of results to return"
    ),
):
    """
    Search the indexed codebase for relevant code snippets.
    
    Examples:
        code-crawler search "RAG implementation"
        code-crawler search "authentication logic" --limit 10
    """
    from code_chatbot.cli.commands.search import search_codebase
    
    search_codebase(query, limit)


@app.command()
def config(
    action: str = typer.Argument(..., help="Action: set, get, list, or reset"),
    key: Optional[str] = typer.Argument(None, help="Configuration key"),
    value: Optional[str] = typer.Argument(None, help="Configuration value"),
):
    """
    Manage CLI configuration (API keys, preferences, etc.).
    
    Examples:
        code-crawler config list
        code-crawler config set GOOGLE_API_KEY sk-...
        code-crawler config get model
        code-crawler config reset
    """
    from code_chatbot.cli.commands.config_cmd import manage_config
    
    manage_config(action, key, value)


@app.command()
def version():
    """Show version information"""
    from code_chatbot.cli import __version__
    
    console.print(Panel(
        f"[bold cyan]Code Crawler CLI[/bold cyan]\n"
        f"Version: [yellow]{__version__}[/yellow]\n"
        f"Gemini-style conversational codebase assistant",
        title="🕷️ Code Crawler",
        border_style="cyan"
    ))


def main():
    """Entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()
