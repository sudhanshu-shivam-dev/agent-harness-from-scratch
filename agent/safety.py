"""Defenses for untrusted tool output (indirect prompt injection).

Tool results are *untrusted input*: a web page, file, or API response can carry
text like "ignore your previous instructions and ...". If that flows verbatim
into the model's context, it becomes an attack surface (indirect prompt
injection). :class:`ToolOutputGuard` scans observations for known injection
patterns and neutralizes the directive before it reaches the model, while keeping
the surrounding (legitimate) content.

This is a lightweight, pattern-based guard -- not a complete defense -- but it
makes the threat explicit and gives the agent a single, testable choke point for
tool output, which is the right place to harden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Pattern

# Common indirect-prompt-injection directives seen in tool/web output.
# Each pattern consumes to the end of the sentence/line ([^.\n]*) so the entire
# injected directive -- not just its trigger phrase -- is redacted.
_INJECTION_PATTERNS: List[Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:your\s+)?(?:previous|prior|above)\s+instructions[^.\n]*", re.I),
    re.compile(r"disregard\s+(?:the\s+)?(?:above|previous|prior|all)\b[^.\n]*", re.I),
    re.compile(r"forget\s+(?:everything|all|your\s+instructions)\b[^.\n]*", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\b[^.\n]*", re.I),
    re.compile(r"(?:reveal|print|show|repeat)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)[^.\n]*", re.I),
    re.compile(r"(?:new|updated)\s+(?:system\s+)?(?:instructions?|directive)s?\s*:[^.\n]*", re.I),
    re.compile(r"\boverride\s+(?:your\s+)?(?:safety|guardrails|instructions)\b[^.\n]*", re.I),
]

_REDACTION = "[redacted: possible prompt-injection]"


@dataclass
class ScanResult:
    """Outcome of scanning one piece of tool output."""

    suspicious: bool
    sanitized: str
    matches: List[str] = field(default_factory=list)


class ToolOutputGuard:
    """Scans and sanitizes untrusted tool output for injection directives."""

    def __init__(self, patterns: List[Pattern[str]] | None = None) -> None:
        self._patterns = patterns or _INJECTION_PATTERNS

    def scan(self, text: str) -> ScanResult:
        """Return a :class:`ScanResult` with injected directives redacted."""

        if not text:
            return ScanResult(suspicious=False, sanitized=text)

        matches: List[str] = []
        sanitized = text
        for pattern in self._patterns:
            for m in pattern.finditer(sanitized):
                matches.append(m.group(0))
            sanitized = pattern.sub(_REDACTION, sanitized)

        return ScanResult(suspicious=bool(matches), sanitized=sanitized, matches=matches)
