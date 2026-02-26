import os
import yaml
from rich import print
from dilu.driver_agent.vectorStore import DrivingMemory

# 1. Load the configuration
try:
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
except FileNotFoundError:
    print("[red]Error: config.yaml not found. Make sure you are in the DiLu root directory.[/red]")
    exit(1)

# 2. Setup Environment Variables (Critical Step)
if config['OPENAI_API_TYPE'] == 'ollama':
    os.environ["OPENAI_API_TYPE"] = 'ollama'
    os.environ["OLLAMA_API_BASE"] = config.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
    os.environ["OPENAI_BASE_URL"] = os.environ["OLLAMA_API_BASE"]
    os.environ["OLLAMA_API_KEY"] = config.get("OLLAMA_API_KEY", "ollama")
    os.environ["OLLAMA_CHAT_MODEL"] = config['OLLAMA_CHAT_MODEL']
    os.environ["OLLAMA_EMBED_MODEL"] = config['OLLAMA_EMBED_MODEL']
    print(f"[yellow]Configured for Local Ollama: {config['OLLAMA_CHAT_MODEL']}[/yellow]")

elif config['OPENAI_API_TYPE'] == 'openai':
    os.environ["OPENAI_API_TYPE"] = 'openai'
    os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
    os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']

elif config['OPENAI_API_TYPE'] == 'azure':
    os.environ["OPENAI_API_TYPE"] = 'azure'
    os.environ["AZURE_EMBED_DEPLOY_NAME"] = config['AZURE_EMBED_DEPLOY_NAME']
    # Add other azure keys if needed

# 3. Initialize and Check Memory
try:
    print(f"[cyan]Loading memory from: {config['memory_path']}...[/cyan]")
    memory = DrivingMemory(db_path=config['memory_path'])

    # Check total items
    count = memory.scenario_memory._collection.count()
    print(f"[green]Total memories found: {count}[/green]")

    # Peek at the data (first item)
    if count > 0:
        print("\n[bold]Sample Memory Item:[/bold]")
        peek_data = memory.scenario_memory._collection.peek()
        print(peek_data)
    else:
        print("[yellow]The memory database is empty.[/yellow]")

except Exception as e:
    print(f"[red]Error loading memory:[/red] {e}")
    print("[yellow]Hint: Embedding dimension mismatches are common when switching models/providers. Use a separate memory folder per embedding model (for example, `memories/qwen3_embed_8b`).[/yellow]")
