"""
Output formatter for beautiful terminal display
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from typing import List, Dict, Any


class OutputFormatter:
    """Formats output for rich terminal display"""
    
    def __init__(self, console: Console):
        self.console = console
    
    def show_welcome(self, model: str, provider: str):
        """Show welcome banner"""
        welcome_text = f"""
[bold cyan]🕷️ Code Crawler - AI Codebase Assistant[/bold cyan]

[dim]Gemini-style conversational interface for your code[/dim]

[yellow]Model:[/yellow] {model}
[yellow]Provider:[/yellow] {provider}

[dim]Type your question or use /help for commands
Use @ to reference files (e.g., @file.py)
Press Ctrl+D or type /exit to quit[/dim]
        """
        
        self.console.print(Panel(
            welcome_text.strip(),
            border_style="cyan",
            padding=(1, 2)
        ))
    
    def show_goodbye(self, session_id: str):
        """Show goodbye message"""
        self.console.print(f"\n[dim]💾 Session saved to ~/.code-crawler/sessions/{session_id}.json[/dim]")
        self.console.print("[bold cyan]👋 Goodbye![/bold cyan]\n")
    
    def show_context_added(self, contexts: List[Dict[str, str]]):
        """Show contexts that were added"""
        for ctx in contexts:
            if ctx['type'] == 'error':
                self.console.print(f"[red]❌ {ctx['source']}: {ctx['content']}[/red]")
            else:
                char_count = len(ctx['content'])
                self.console.print(f"[dim]📎 Added context: {ctx['source']} ({char_count:,} chars)[/dim]")
    
    def show_sources(self, sources: List[Any]):
        """Show source documents used in response"""
        if not sources:
            return
        
        self.console.print("\n[dim]📚 Sources:[/dim]")
        
        # Create table for sources
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Icon", style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Score", style="yellow")
        
        for i, source in enumerate(sources[:5], 1):  # Show top 5
            file_path = getattr(source, 'metadata', {}).get('file_path', 'unknown')
            score = getattr(source, 'metadata', {}).get('score', None)
            
            score_str = f"{score:.2f}" if score is not None else "N/A"
            table.add_row("📄", file_path, score_str)
        
        self.console.print(table)
    
    def show_code(self, code: str, language: str = "python", line_numbers: bool = True):
        """Display syntax-highlighted code"""
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
            word_wrap=True
        )
        self.console.print(syntax)
    
    def show_diff(self, diff_text: str):
        """Display a diff"""
        syntax = Syntax(
            diff_text,
            "diff",
            theme="monokai",
            line_numbers=False
        )
        self.console.print(syntax)
    
    def show_table(self, data: List[Dict[str, Any]], title: str = None):
        """Display data as a table"""
        if not data:
            return
        
        # Get columns from first row
        columns = list(data[0].keys())
        
        table = Table(title=title, show_header=True)
        for col in columns:
            table.add_column(col, style="cyan")
        
        for row in data:
            table.add_row(*[str(row.get(col, "")) for col in columns])
        
        self.console.print(table)
    
    def show_error(self, message: str):
        """Display an error message"""
        self.console.print(f"[red]❌ Error: {message}[/red]")
    
    def show_success(self, message: str):
        """Display a success message"""
        self.console.print(f"[green]✅ {message}[/green]")
    
    def show_warning(self, message: str):
        """Display a warning message"""
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")
    
    def show_info(self, message: str):
        """Display an info message"""
        self.console.print(f"[blue]ℹ️  {message}[/blue]")
    
    def show_progress_message(self, message: str):
        """Display a progress message"""
        self.console.print(f"[dim]⏳ {message}...[/dim]")
