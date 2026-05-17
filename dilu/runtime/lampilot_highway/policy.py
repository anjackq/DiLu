from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable


SAFE_BUILTINS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "sum": sum,
    "tuple": tuple,
}
FORBIDDEN_CALLS = {"exec", "eval", "compile", "open", "__import__", "input", "globals", "locals", "vars", "dir"}
CODE_BLOCK_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL | re.IGNORECASE)


class PolicyCompilationError(ValueError):
    pass


class _PolicyValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_names: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        raise PolicyCompilationError("Imports are not allowed in benchmark policies.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise PolicyCompilationError("Imports are not allowed in benchmark policies.")

    def visit_With(self, node: ast.With) -> None:
        raise PolicyCompilationError("Context managers are not allowed in benchmark policies.")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        raise PolicyCompilationError("Async constructs are not allowed in benchmark policies.")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise PolicyCompilationError("Classes are not allowed in benchmark policies.")

    def visit_Delete(self, node: ast.Delete) -> None:
        raise PolicyCompilationError("Delete statements are not allowed in benchmark policies.")

    def visit_Global(self, node: ast.Global) -> None:
        raise PolicyCompilationError("Global declarations are not allowed in benchmark policies.")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise PolicyCompilationError("Nonlocal declarations are not allowed in benchmark policies.")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise PolicyCompilationError("Lambda expressions are not allowed in benchmark policies.")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise PolicyCompilationError("Attribute access is not allowed in benchmark policies.")

    def visit_Name(self, node: ast.Name) -> None:
        if str(node.id).startswith("__"):
            raise PolicyCompilationError("Dunder names are not allowed in benchmark policies.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise PolicyCompilationError("Only direct API and safe builtin calls are allowed.")
        func_name = str(node.func.id)
        if func_name.startswith("__") or func_name in FORBIDDEN_CALLS:
            raise PolicyCompilationError(f"Forbidden call in benchmark policy: {func_name}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.args.args or node.args.kwonlyargs or node.args.vararg or node.args.kwarg:
            raise PolicyCompilationError("Benchmark policy function must not take arguments.")
        if str(node.name).startswith("__"):
            raise PolicyCompilationError("Policy function name cannot use dunder syntax.")
        self.function_names.append(node.name)
        self.generic_visit(node)


@dataclass(frozen=True)
class CompiledPolicy:
    function_name: str
    source_code: str
    raw_response: str

    def instantiate(self, bindings: dict[str, Callable[..., Any]]) -> Iterable[Any]:
        safe_globals: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        safe_globals.update(bindings)
        local_vars: dict[str, Any] = {}
        try:
            exec(self.source_code, safe_globals, local_vars)
        except Exception as exc:  # noqa: S102 - sandboxed globals are explicit
            raise PolicyCompilationError(f"Failed to initialize benchmark policy: {exc}") from exc
        policy_fn = local_vars.get(self.function_name) or safe_globals.get(self.function_name)
        if not callable(policy_fn):
            raise PolicyCompilationError(f"Policy function not found after compilation: {self.function_name}")
        result = policy_fn()
        if result is None:
            return iter(())
        if hasattr(result, "__next__") or hasattr(result, "__iter__"):
            return iter(result)
        raise PolicyCompilationError("Policy function must return a generator or iterable.")


def _extract_policy_source(response_text: str) -> str:
    text = str(response_text or "").strip()
    if not text:
        raise PolicyCompilationError("Empty policy response.")
    match = CODE_BLOCK_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text


def compile_policy_response(response_text: str) -> CompiledPolicy:
    source_code = _extract_policy_source(response_text)
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        raise PolicyCompilationError(f"Syntax error in benchmark policy: {exc}") from exc
    validator = _PolicyValidator()
    validator.visit(tree)
    function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) != 1:
        raise PolicyCompilationError("Benchmark policy response must define exactly one function.")
    function_name = function_defs[0].name
    return CompiledPolicy(function_name=function_name, source_code=source_code, raw_response=response_text)
