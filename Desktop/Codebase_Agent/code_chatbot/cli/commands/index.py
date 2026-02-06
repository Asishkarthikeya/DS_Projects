"""
Index command - Index a codebase for semantic search
"""
import os
import shutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from pathlib import Path


console = Console()


def index_codebase(path: str, provider: str, vector_db: str, force: bool):
    """Index a codebase from various sources"""
    
    try:
        # Import required modules
        from code_chatbot.ingestion.universal_ingestor import process_source
        from code_chatbot.analysis.ast_analysis import ASTGraphBuilder
        from code_chatbot.ingestion.indexer import Indexer
        from code_chatbot.ingestion.chunker import StructuralChunker
        from langchain_community.vectorstores import Chroma, FAISS
        from langchain_community.vectorstores.utils import filter_complex_metadata
        
        console.print(f"\n[bold cyan]🕷️ Indexing Codebase[/bold cyan]")
        console.print(f"[dim]Source: {path}[/dim]")
        console.print(f"[dim]Provider: {provider}[/dim]")
        console.print(f"[dim]Vector DB: {vector_db}[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            # Stage 1: Extract & Ingest
            task1 = progress.add_task("[cyan]Extracting source...", total=None)
            extract_to = os.path.join("data", "extracted")
            if os.path.exists(extract_to) and force:
                shutil.rmtree(extract_to)
            
            documents, local_path = process_source(path, extract_to)
            progress.update(task1, completed=True, description=f"[green]✓ Extracted {len(documents)} files")
            
            if not documents:
                console.print("[red]❌ No documents found in the source[/red]")
                return
            
            # Stage 2: AST Analysis
            task2 = progress.add_task("[cyan]Building AST graph...", total=None)
            ast_builder = ASTGraphBuilder()
            for doc in documents:
                ast_builder.add_file(doc.metadata['file_path'], doc.page_content)
            
            os.makedirs(local_path, exist_ok=True)
            graph_path = os.path.join(local_path, "ast_graph.graphml")
            ast_builder.save_graph(graph_path)
            graph_nodes = ast_builder.graph.number_of_nodes()
            progress.update(task2, completed=True, description=f"[green]✓ Built AST graph ({graph_nodes} nodes)")
            
            # Stage 3: Get API key
            api_key = os.getenv("GOOGLE_API_KEY") if provider == "gemini" else os.getenv("GROQ_API_KEY")
            if not api_key:
                console.print(f"[red]❌ API key not found for {provider}[/red]")
                console.print(f"[yellow]Set {'GOOGLE_API_KEY' if provider == 'gemini' else 'GROQ_API_KEY'} environment variable[/yellow]")
                return
            
            # Stage 4: Chunking
            task3 = progress.add_task("[cyan]Chunking documents...", total=len(documents))
            indexer = Indexer(provider=provider, api_key=api_key)
            indexer.clear_collection(collection_name="codebase")
            
            chunker = StructuralChunker()
            all_chunks = []
            for doc in documents:
                file_chunks = chunker.chunk(doc.page_content, doc.metadata["file_path"])
                all_chunks.extend(file_chunks)
                progress.advance(task3)
            
            progress.update(task3, description=f"[green]✓ Created {len(all_chunks)} chunks")
            
            # Clean metadata
            for doc in all_chunks:
                doc.metadata = {k: v for k, v in doc.metadata.items() if v is not None}
            all_chunks = filter_complex_metadata(all_chunks)
            
            # Stage 5: Index into vector store
            task4 = progress.add_task(f"[cyan]Indexing into {vector_db}...", total=None)
            
            if vector_db == "faiss":
                vectordb = FAISS.from_documents(all_chunks, indexer.embedding_function)
                vectordb.save_local(folder_path=indexer.persist_directory, index_name="codebase")
            elif vector_db == "qdrant":
                from langchain_qdrant import QdrantVectorStore
                url = os.getenv("QDRANT_URL")
                api_key_qdrant = os.getenv("QDRANT_API_KEY")
                vectordb = QdrantVectorStore.from_documents(
                    documents=all_chunks,
                    embedding=indexer.embedding_function,
                    url=url,
                    api_key=api_key_qdrant,
                    collection_name="codebase",
                    prefer_grpc=True
                )
            else:  # Chroma
                # Use shared client to avoid "different settings" error
                from code_chatbot.core.db_connection import get_chroma_client
                chroma_client = get_chroma_client(indexer.persist_directory)
                
                vectordb = Chroma(
                    client=chroma_client,
                    embedding_function=indexer.embedding_function,
                    collection_name="codebase"
                )
                vectordb.add_documents(documents=all_chunks)
            
            progress.update(task4, completed=True, description=f"[green]✓ Indexed into {vector_db}")
        
        # Success summary
        console.print(f"\n[bold green]✅ Indexing Complete![/bold green]\n")
        console.print(f"  📁 Files indexed: [cyan]{len(documents)}[/cyan]")
        console.print(f"  📦 Chunks created: [cyan]{len(all_chunks)}[/cyan]")
        console.print(f"  🌳 Graph nodes: [cyan]{graph_nodes}[/cyan]")
        console.print(f"  💾 Vector DB: [cyan]{vector_db}[/cyan]")
        console.print(f"\n[dim]Ready to chat! Run: code-crawler chat[/dim]\n")
        
    except Exception as e:
        console.print(f"\n[red]❌ Indexing failed: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
