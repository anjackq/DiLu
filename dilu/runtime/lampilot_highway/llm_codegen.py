from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None

from .prompts import render_system_prompt, render_user_prompt


@dataclass(frozen=True)
class CodegenResponse:
    content: str


class HighwayCodegenAgent:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        model_name: str,
        request_timeout: float = 60.0,
        few_shot: bool = False,
        temperature: float = 0.0,
    ) -> None:
        self.config = dict(config)
        self.model_name = model_name
        self.request_timeout = float(request_timeout)
        self.few_shot = bool(few_shot)
        self.temperature = float(temperature)
        self.client = self._build_client()

    def _build_client(self):
        api_type = str(self.config["OPENAI_API_TYPE"]).strip().lower()
        if api_type == "ollama":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=str(self.config.get("OLLAMA_API_KEY", "ollama")),
                base_url=str(self.config.get("OLLAMA_API_BASE", "http://localhost:11434/v1")),
            )
        if api_type == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=str(self.config["OPENAI_KEY"]),
            )
        if api_type == "azure":
            return AzureChatOpenAI(
                azure_deployment=str(self.config["AZURE_CHAT_DEPLOY_NAME"]),
                api_version=str(self.config["AZURE_API_VERSION"]),
                azure_endpoint=str(self.config["AZURE_API_BASE"]),
                api_key=str(self.config["AZURE_API_KEY"]),
                temperature=self.temperature,
                timeout=self.request_timeout,
            )
        if api_type == "gemini":
            if ChatGoogleGenerativeAI is None:
                raise ImportError("langchain_google_genai is required for Gemini code generation.")
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=str(self.config["GEMINI_API_KEY"]),
                temperature=self.temperature,
                timeout=self.request_timeout,
            )
        raise ValueError(f"Unsupported OPENAI_API_TYPE for highway codegen: {api_type}")

    def generate(self, *, command: str, context_info: str) -> CodegenResponse:
        messages = [
            SystemMessage(content=render_system_prompt(few_shot=self.few_shot)),
            HumanMessage(content=render_user_prompt(command, context_info)),
        ]
        response = self.client.invoke(messages)
        return CodegenResponse(content=str(getattr(response, "content", response)))
