"""Typed tool abstraction.

Two ways to define a tool:

1. Subclass :class:`BaseTool` for full control.
2. Decorate a plain function with :func:`tool` -- its signature and docstring are
   introspected to auto-generate the JSON schema used for tool-calling.

A :class:`ToolRegistry` collects tools and emits the schema list that the LLM
client expects, and dispatches calls by name.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, get_args, get_origin, get_type_hints


# Tool risk vocabulary — from Harness-Bench (2025) harness engineering research.
# Labels inform the harness of each tool's impact level; WRITE/DESTRUCTIVE
# annotations are surfaced in the LLM description so the model can reason about
# side-effects before calling.
RISK_READ = "read"          # safe, idempotent, no side-effects
RISK_WRITE = "write"        # modifies state, but reversible
RISK_DESTRUCTIVE = "destructive"  # irreversible or high-impact


# Map Python types to JSON-schema primitive types.
_PY_TO_JSON: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    """Best-effort mapping from a type annotation to a JSON-schema type."""

    if annotation is inspect.Parameter.empty:
        return "string"
    origin = get_origin(annotation)
    if origin is not None:
        # Optional[X] / Union[...] -> use the first non-None arg.
        if origin is type(None):
            return "string"
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return _json_type(args[0])
        return "string"
    return _PY_TO_JSON.get(annotation, "string")


def _parse_docstring(doc: Optional[str]) -> tuple[str, Dict[str, str]]:
    """Split a docstring into a summary and per-parameter descriptions.

    Recognizes a simple ``Args:`` block with ``name: description`` lines.
    """

    if not doc:
        return "", {}
    lines = [ln.rstrip() for ln in doc.strip().splitlines()]
    summary_parts: List[str] = []
    params: Dict[str, str] = {}
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "params:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if ":" in stripped and stripped:
                name, _, desc = stripped.partition(":")
                params[name.strip()] = desc.strip()
            elif not stripped:
                in_args = False
        else:
            summary_parts.append(stripped)
    return " ".join(p for p in summary_parts if p).strip(), params


class BaseTool(ABC):
    """Abstract base class for tools.

    Concrete tools must set :attr:`name`/:attr:`description`, implement
    :meth:`run`, and provide a JSON-schema ``parameters`` block via
    :meth:`parameters_schema`.

    Set :attr:`risk` to one of :data:`RISK_READ`, :data:`RISK_WRITE`, or
    :data:`RISK_DESTRUCTIVE` to annotate the tool's side-effect profile.
    """

    name: str = ""
    description: str = ""
    risk: str = RISK_READ

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return an observation string."""

    def parameters_schema(self) -> Dict[str, Any]:
        """Return the JSON-schema ``parameters`` object for this tool."""

        return {"type": "object", "properties": {}, "required": []}

    def to_schema(self) -> Dict[str, Any]:
        """Return the OpenAI-style ``function`` tool schema."""

        desc = self.description
        if self.risk in (RISK_WRITE, RISK_DESTRUCTIVE):
            desc = f"[{self.risk.upper()}] {desc}"
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": self.parameters_schema(),
                "x_risk": self.risk,
            },
        }


class FunctionTool(BaseTool):
    """A :class:`BaseTool` backed by a plain Python function.

    Produced by the :func:`tool` decorator; the function signature and docstring
    drive the auto-generated JSON schema.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        risk: str = RISK_READ,
    ) -> None:
        self._func = func
        self.name = name or func.__name__
        summary, param_docs = _parse_docstring(func.__doc__)
        self.description = summary or self.name
        self.risk = risk
        self._param_docs = param_docs
        self._signature = inspect.signature(func)
        try:
            self._hints = get_type_hints(func)
        except Exception:  # pragma: no cover - exotic annotations
            self._hints = {}

    def parameters_schema(self) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for pname, param in self._signature.parameters.items():
            if pname == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = self._hints.get(pname, param.annotation)
            prop: Dict[str, Any] = {"type": _json_type(annotation)}
            if pname in self._param_docs:
                prop["description"] = self._param_docs[pname]
            properties[pname] = prop
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": properties, "required": required}

    def run(self, **kwargs: Any) -> str:
        return str(self._func(**kwargs))


def tool(name_or_func: Any = None, *, risk: str = RISK_READ) -> Any:
    """Decorator that turns a function into a :class:`FunctionTool`.

    Usage::

        @tool
        def calculator(expression: str) -> str:
            '''Evaluate an arithmetic expression.'''
            ...

        @tool("web_search")
        def search(query: str) -> str:
            ...

        @tool(risk=RISK_WRITE)
        def write_file(path: str, content: str) -> str:
            ...
    """

    if callable(name_or_func):
        return FunctionTool(name_or_func, risk=risk)

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name_or_func, risk=risk)

    return decorator


class ToolRegistry:
    """Holds tools and provides schema export + dispatch."""

    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: BaseTool) -> BaseTool:
        if not t.name:
            raise ValueError("Tool must have a non-empty name.")
        self._tools[t.name] = t
        return t

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        """Return the list of tool schemas for the LLM client."""

        return [t.to_schema() for t in self._tools.values()]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute the named tool. Raises :class:`KeyError` if unknown."""

        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name].run(**arguments)
