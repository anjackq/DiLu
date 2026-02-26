import os
import textwrap
import time
from rich import print

# UPDATED IMPORTS
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class ReflectionAgent:
    def __init__(
            self, temperature: float = 0.0, verbose: bool = False
    ) -> None:
        oai_api_type = os.getenv("OPENAI_API_TYPE")

        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=1000,
                request_timeout=60,
            )
        elif oai_api_type == "openai":
            # Check if we are using Ollama (localhost) to avoid hardcoded GPT-4
            base_url = os.getenv("OPENAI_API_BASE", "")

            # FORCE the Reflection Agent to use a smarter model
            # distinct from the driving model
            if "localhost" in base_url or "127.0.0.1" in base_url:
                model_name = os.getenv("OPENAI_REFLECTION_MODEL", "deepseek-r1:14b")
                print(f"[yellow]Reflection Agent using Local Ollama: {model_name}[/yellow]")
            else:
                # Default to strong GPT-4 for OpenAI reflection
                model_name = os.getenv("OPENAI_REFLECTION_MODEL", 'gpt-4-1106-preview')
                print("[red]Cautious: Reflection mode uses OpenAI GPT-4, may cost a lot of money![/red]")

            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                max_tokens=1000,
                request_timeout=60,
            )

        elif oai_api_type == "ollama":
            # Check if we are using Ollama (localhost) to avoid hardcoded GPT-4
            base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")

            # FORCE the Reflection Agent to use a smarter model
            # distinct from the driving model
            if "localhost" in base_url or "127.0.0.1" in base_url:
                model_name = os.getenv("OLLAMA_REFLECTION_MODEL", "deepseek-r1:14b")
                print(f"[yellow]Reflection Agent using Local Ollama: {model_name}[/yellow]")
            else:
                # Default to strong GPT-4 for OpenAI reflection
                model_name = os.getenv("OPENAI_REFLECTION_MODEL", 'gpt-4-1106-preview')
                print("[red]Cautious: Reflection mode uses OpenAI GPT-4, may cost a lot of money![/red]")

            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                openai_api_base=base_url,  # <--- ADD THIS
                openai_api_key=os.getenv("OLLAMA_API_KEY", "ollama"),  # <--- ADD THIS
                max_tokens=1000,
                request_timeout=60,
            )
        else:
            raise ValueError(f"Unknown OPENAI_API_TYPE: {oai_api_type}")

    def reflection(self, human_message: str, llm_response: str) -> str:
        delimiter = "####"

        # [UPDATED] Removed "ChatGPT" persona and made rules explicit for SLMs
        system_message = textwrap.dedent(f"""\
                You are an expert autonomous driving reflection engine. Your task is to analyze incorrect driving decisions that led to collisions and provide safety corrections.
                You will be given a detailed description of a driving scenario and the agent's incorrect response.

                Your answer MUST strictly follow this exact format:
                {delimiter} Analysis of the mistake:
                <Your analysis of the mistake in the previous reasoning>
                {delimiter} What should be done to avoid such errors in the future:
                <Your advice for safer driving>
                {delimiter} Corrected version of response:
                <Corrected reasoning>
                Response to user:{delimiter} <Corrected Action_id>
                """)

        # [UPDATED] Removed ChatGPT references from the human prompt
        human_message = textwrap.dedent(f"""\
                    ``` Human Message ```
                    {human_message}
                    ``` Previous Agent Response ```
                    {llm_response}

                    The action chosen in the Previous Agent Response resulted in a collision. This means there is a critical mistake in the reasoning process.    
                    Please carefully check the reasoning in the previous response, find the mistake, and output a corrected version of the decision.

                    Strictly follow the output format defined in the system prompt.
                """)

        print("Self-reflection is running, may take time...")
        start_time = time.time()
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]

        # UPDATED: Use .invoke() instead of __call__
        response = self.llm.invoke(messages)

        # [UPDATED] The target phrase must match the new prompt exactly!
        target_phrase = f"{delimiter} What should be done to avoid such errors in the future:"

        # Add safety check if target phrase is missing (common with smaller local models)
        if target_phrase in response.content:
            substring = response.content[response.content.find(
                target_phrase) + len(target_phrase):].strip()
        else:
            # Fallback if model didn't follow strict formatting
            print("[yellow]Warning: Reflection checkpoints format mismatch. Saving full content.[/yellow]")
            substring = response.content

        corrected_memory = f"{delimiter} I have made a misake before and below is my self-reflection:\n{substring}"
        print("Reflection done. Time taken: {:.2f}s".format(
            time.time() - start_time))
        # print("corrected_memory:", corrected_memory)

        return corrected_memory
