"""
Config command - Manage CLI configuration
"""
from rich.console import Console
from rich.table import Table
from code_chatbot.cli.config import Config


console = Console()


def manage_config(action: str, key: str = None, value: str = None):
    """Manage CLI configuration"""
    
    config = Config()
    
    if action == "list":
        # List all configuration
        all_config = config.list_all()
        
        table = Table(title="Configuration", show_header=True)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="yellow")
        
        for k, v in all_config.items():
            # Mask API keys
            if "API_KEY" in k.upper():
                v = f"{str(v)[:8]}..." if v else "Not set"
            table.add_row(k, str(v))
        
        console.print(table)
    
    elif action == "get":
        if not key:
            console.print("[red]❌ Key required for 'get' action[/red]")
            return
        
        val = config.get(key)
        if val is None:
            console.print(f"[yellow]Key '{key}' not found[/yellow]")
        else:
            # Mask API keys
            if "API_KEY" in key.upper():
                val = f"{str(val)[:8]}..."
            console.print(f"[cyan]{key}[/cyan] = [yellow]{val}[/yellow]")
    
    elif action == "set":
        if not key or value is None:
            console.print("[red]❌ Key and value required for 'set' action[/red]")
            return
        
        config.set(key, value)
        console.print(f"[green]✅ Set {key}[/green]")
    
    elif action == "delete":
        if not key:
            console.print("[red]❌ Key required for 'delete' action[/red]")
            return
        
        config.delete(key)
        console.print(f"[green]✅ Deleted {key}[/green]")
    
    elif action == "reset":
        config.reset()
        console.print("[green]✅ Configuration reset to defaults[/green]")
    
    else:
        console.print(f"[red]❌ Unknown action: {action}[/red]")
        console.print("[yellow]Valid actions: list, get, set, delete, reset[/yellow]")
