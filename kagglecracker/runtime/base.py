"""Agent runtime interface: prompt in, a train.py out.

    AgentRuntime.generate(prompt, action) -> GenerateResult
         │
         ├── OpenAICompatibleRuntime   openai + deepseek (same wire shape)
         ├── AnthropicRuntime          native anthropic SDK
         └── FakeRuntime               tests

The engine never learns which provider it is on. Both real adapters normalize
to one `GenerateResult`, which is what makes `max_usd` accurate: token
accounting has to be exact per provider, and that is why the Anthropic
OpenAI-compatibility endpoint was rejected in favour of a second adapter.

Why the status enum has more than ok/error
------------------------------------------
`TRUNCATED` is not a hypothetical. Every current frontier model is a reasoning
model, and `max_tokens` caps reasoning *plus* visible output. Set it too low and
the call succeeds, bills you, and returns an empty string. Verified on the Day-2
probe: gpt-5-nano at 64 tokens spent all 64 on reasoning and returned `''` with
`finish_reason: length`.

If that were filed as "the model returned no code", the debug action would spend
the rest of the night trying to repair a config value. So it gets its own status
and its own message, and the fix it suggests is raising `max_tokens` — not
editing the code.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

Action = str  # "draft" | "improve" | "debug"


class GenerateStatus(StrEnum):
    OK = "ok"
    #: Hit the token ceiling. A config problem, never a code problem.
    TRUNCATED = "truncated"
    #: Returned prose with no code block at all.
    NO_CODE = "no_code"
    #: Returned something code-shaped that will not parse as Python.
    INVALID_PYTHON = "invalid_python"
    #: The provider's safety classifiers declined the request.
    REFUSAL = "refusal"

    @property
    def is_config_fault(self) -> bool:
        """TRUNCATED means *we* set max_tokens too low. Do not ask the model to fix it."""
        return self is GenerateStatus.TRUNCATED


@dataclass(frozen=True, slots=True)
class TokenUsage:
    tokens_in: int = 0
    tokens_out: int = 0
    cached_in: int = 0

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass(frozen=True, slots=True)
class GenerateResult:
    status: GenerateStatus
    code: str | None
    raw_text: str
    usage: TokenUsage
    provider: str
    model: str
    usd: float = 0.0
    error_text: str = ""
    prompt_chars: int = 0
    stop_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is GenerateStatus.OK


class AgentRuntime(Protocol):
    def generate(self, prompt: str, *, action: Action) -> GenerateResult: ...


# --------------------------------------------------------------------------
# turning a model response into runnable code
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    """Pull a Python file out of a model response.

    Models answer in three shapes: a bare file, one fenced block, or prose
    wrapped around several blocks. When there are several, take the longest —
    the others are almost always illustrative snippets, and the real file is
    the big one.
    """
    if not text or not text.strip():
        return None

    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()

    # No fences. If the whole response parses as Python, it *is* the file.
    stripped = text.strip()
    try:
        ast.parse(stripped)
    except SyntaxError:
        return None
    return stripped


def classify_output(
    text: str, *, hit_token_ceiling: bool
) -> tuple[GenerateStatus, str | None, str]:
    """Map a raw model response onto (status, code, error_text).

    Order matters: the token ceiling is checked first. A truncated response
    usually *also* fails to parse, and reporting that as a syntax error would
    send the debug action after the symptom instead of the cause.
    """
    if hit_token_ceiling:
        return (
            GenerateStatus.TRUNCATED,
            None,
            "The response hit the max_tokens ceiling and was cut off. This is a "
            "configuration limit, not a defect in the generated code — raise "
            "KC_MAX_TOKENS. Note that on reasoning models max_tokens covers "
            "reasoning as well as visible output.",
        )

    code = extract_code(text)
    if code is None:
        preview = text.strip()[:400] if text and text.strip() else "(empty response)"
        return (
            GenerateStatus.NO_CODE,
            None,
            f"The response contained no Python code block. Got: {preview}",
        )

    try:
        ast.parse(code)
    except SyntaxError as exc:
        return (
            GenerateStatus.INVALID_PYTHON,
            None,
            f"The returned code is not valid Python: {exc.msg} at line {exc.lineno}.",
        )

    return GenerateStatus.OK, code, ""


@dataclass
class RetryPolicy:
    """Retries live in the provider SDKs; this only records the intent."""

    max_retries: int = 3
    extra: dict = field(default_factory=dict)
