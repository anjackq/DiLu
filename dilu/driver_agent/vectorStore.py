import os
import textwrap
from rich import print

# UPDATED IMPORTS: Adapting to LangChain v0.1+
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from dilu.scenario.envScenario import EnvScenario


class DrivingMemory:

    def __init__(self, encode_type='sce_language', db_path=None) -> None:
        self.encode_type = encode_type

        if encode_type == 'sce_encode':
            raise ValueError("encode_type sce_encode is deprecated for now.")

        elif encode_type == 'sce_language':
            # --- 1. Handle Azure ---
            if os.environ.get("OPENAI_API_TYPE") == 'azure':
                self.embedding = OpenAIEmbeddings(
                    deployment=os.environ['AZURE_EMBED_DEPLOY_NAME'],
                    chunk_size=1
                )

            # --- 2. Handle OpenAI & Ollama ---
            elif os.environ.get("OPENAI_API_TYPE") == 'openai':
                base_url = os.environ.get("OPENAI_API_BASE", "")

                # Check if we are running locally (Ollama)
                if "localhost" in base_url or "127.0.0.1" in base_url:
                    # CRITICAL: Ollama needs the specific model name.
                    # Default 'text-embedding-ada-002' will fail locally.
                    # We reuse the chat model (e.g. qwen2.5:14b) for embeddings.
                    model_name = os.environ.get("OPENAI_CHAT_MODEL", "qwen2.5:14b")

                    self.embedding = OpenAIEmbeddings(
                        model=model_name,
                        openai_api_base=base_url,
                        openai_api_key="ollama",  # Dummy key
                        check_embedding_ctx_length=False  # Prevents errors with local models
                    )
                    print(f"[yellow]Using Local Ollama Embeddings with model: {model_name}[/yellow]")
                else:
                    # Standard OpenAI (uses default ada-002)
                    self.embedding = OpenAIEmbeddings()

            # --- ADDED OLLAMA SUPPORT ---
            elif os.environ.get("OPENAI_API_TYPE") == 'ollama':
                model_name = os.getenv('OLLAMA_EMBED_MODEL')
                print(f"[green]Using Ollama Embeddings[/green] with model: {model_name}")
                # Note: We use the OpenAI-compatible endpoint provided by Ollama
                self.embedding = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_base=os.getenv("OLLAMA_API_BASE"),  # http://localhost:11434/v1
                    openai_api_key=os.getenv("OLLAMA_API_KEY"),  # 'ollama'
                    check_embedding_ctx_length=False  # Necessary for some local models
                )

            else:
                raise ValueError("Unknown OPENAI_API_TYPE: should be azure or openai")

            # Define DB path
            db_path = os.path.join('./db', 'chroma_5_shot_20_mem/') if db_path is None else db_path

            # Initialize Chroma
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path
            )
        else:
            raise ValueError("Unknown ENCODE_TYPE: should be sce_encode or sce_language")

        # Safety check for collection
        try:
            count = self.scenario_memory._collection.count()
            print("==========Loaded ", db_path, " Memory. Total items: ", count, "==========")
        except Exception as e:
            print(f"[red]Warning loading memory count: {e}[/red]")

    def retriveMemory(self, driving_scenario: EnvScenario, frame_id: int, top_k: int = 5):
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            query_scenario = driving_scenario.describe(frame_id)
            # Perform similarity search
            similarity_results = self.scenario_memory.similarity_search_with_score(
                query_scenario, k=top_k)
            fewshot_results = []
            for res, score in similarity_results:
                fewshot_results.append(res.metadata)
            return fewshot_results
        return []

    def addMemory(self, sce_descrip: str, human_question: str, response: str, action: int, sce: EnvScenario = None,
                  comments: str = ""):
        if self.encode_type == 'sce_language':
            sce_descrip = sce_descrip.replace("'", '')

        # Access raw collection to check for duplicates
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": sce_descrip
            }
        )

        if len(get_results['ids']) > 0:
            # Update existing memory
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id,
                metadatas={"human_question": human_question,
                           'LLM_response': response, 'action': action, 'comments': comments}
            )
            print("Modify a memory item.")
        else:
            # Add new memory
            doc = Document(
                page_content=sce_descrip,
                metadata={"human_question": human_question,
                          'LLM_response': response, 'action': action, 'comments': comments}
            )
            self.scenario_memory.add_documents([doc])
            print("Add a memory item.")

    def deleteMemory(self, ids):
        self.scenario_memory.delete(ids=ids)
        print("Delete", len(ids), "memory items.")

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])

        # --- FIX START: Safe handling of embeddings ---
        current_embeddings = current_documents.get('embeddings')
        if current_embeddings is None:
            current_embeddings = []
        # Convert to list if it's a numpy array to prevent comparison errors
        elif hasattr(current_embeddings, 'tolist'):
            current_embeddings = current_embeddings.tolist()

        other_embeddings = other_documents.get('embeddings')
        if other_embeddings is None:
            other_embeddings = []
        elif hasattr(other_embeddings, 'tolist'):
            other_embeddings = other_embeddings.tolist()
        # --- FIX END ---

        # Iterate and merge
        for i in range(len(other_embeddings)):
            # Check if embedding already exists in current memory
            if other_embeddings[i] in current_embeddings:
                # Optional: reduce verbosity if needed
                # print("Already have one memory item, skip.")
                pass
            else:
                self.scenario_memory._collection.add(
                    embeddings=[other_embeddings[i]],  # Must be a list
                    metadatas=[other_documents['metadatas'][i]],
                    documents=[other_documents['documents'][i]],
                    ids=[other_documents['ids'][i]]
                )

        # Safe count check
        count = self.scenario_memory._collection.count()
        print("Merge complete. Now the database has ", count, " items.")


if __name__ == "__main__":
    pass