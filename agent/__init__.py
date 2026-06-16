"""agent-harness-from-scratch: a production-shaped ReAct agent framework.

Public surface:

* :class:`~agent.agent.ReActAgent` / :class:`~agent.agent.AgentResult`
* :class:`~agent.context.ExecutionContext`
* :class:`~agent.tools.BaseTool`, :func:`~agent.tools.tool`,
  :class:`~agent.tools.ToolRegistry`
* :class:`~agent.memory.ShortTermMemory`, :class:`~agent.memory.LongTermMemory`
* :class:`~agent.llm.BaseLLM`, :class:`~agent.llm.MockLLM`,
  :class:`~agent.llm.OpenAILLM`
"""

from .agent import AgentResult, ReActAgent
from .context import ExecutionContext, Step
from .llm import BaseLLM, LLMResponse, MockLLM, OpenAILLM, ToolCall, Usage
from .memory import LongTermMemory, ShortTermMemory
from .tools import BaseTool, FunctionTool, ToolRegistry, tool

__version__ = "0.1.0"

__all__ = [
    "ReActAgent",
    "AgentResult",
    "ExecutionContext",
    "Step",
    "BaseLLM",
    "LLMResponse",
    "MockLLM",
    "OpenAILLM",
    "ToolCall",
    "Usage",
    "ShortTermMemory",
    "LongTermMemory",
    "BaseTool",
    "FunctionTool",
    "ToolRegistry",
    "tool",
]
