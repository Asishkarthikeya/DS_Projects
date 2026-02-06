"""
Search command - Search the indexed codebase
"""
import os
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax


console = Console()


def search_codebase(query: str, limit: int):
    """Search the indexed codebase for relevant code snippets"""
    
    try:
        from code_chatbot.ingestion.indexer import Indexer
        
        console.print(f"\n[bold cyan]🔍 Searching Codebase[/bold cyan]")
        console.print(f"[dim]Query: {query}[/dim]")
        console.print(f"[dim]Limit: {limit}[/dim]\n")
        
        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            console.print("[red]❌ GOOGLE_API_KEY not found[/red]")
            return
        
        # Load indexer
        indexer = Indexer(provider="gemini", api_key=api_key)
        retriever = indexer.get_retriever(vector_db_type="chroma", k=limit)
        
        # Search
        results = retriever.get_relevant_documents(query)
        
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Display results
        console.print(f"[bold green]Found {len(results)} results:[/bold green]\n")
        
        for i, doc in enumerate(results, 1):
            file_path = doc.metadata.get('file_path', 'unknown')
            score = doc.metadata.get('score', None)
            
            console.print(f"[bold cyan]{i}. {file_path}[/bold cyan]")
            if score:
                console.print(f"   [dim]Relevance: {score:.2f}[/dim]")
            
            # Show code snippet
            content = doc.page_content
            if len(content) > 500:
                content = content[:500] + "..."
            
            syntax = Syntax(
                content,
                "python",
                theme="monokai",
                line_numbers=False,
                word_wrap=True
            )
            console.print(syntax)
            console.print()
        
    except Exception as e:
        console.print(f"[red]❌ Search failed: {e}[/red]")
