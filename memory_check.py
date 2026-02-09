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
    # Trick the system into thinking it's OpenAI, but point to localhost
    os.environ["OPENAI_API_TYPE"] = 'openai'
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"
    os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
    print(f"[yellow]Configured for Local Ollama: {config['OPENAI_CHAT_MODEL']}[/yellow]")

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
    print("[yellow]Hint: If you are using Ollama with the default '20_mem' folder, it will likely fail due to dimension mismatch (1536 vs 5120). You should delete the 'memories' folder to let Ollama create a fresh one.[/yellow]")