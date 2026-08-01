"""Canned agent runtime for tests.

Together with FakeExecutor this lets the whole search loop run with no network
and no Docker, which is what turns "the interface is swappable" into something
a test can prove.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kagglecracker.runtime.base import (
    Action,
    GenerateResult,
    GenerateStatus,
    TokenUsage,
    classify_output,
)


@dataclass
class FakeResponse:
    text: str = "```python\nprint('hello')\n```"
    hit_token_ceiling: bool = False
    status: GenerateStatus | None = None
    error_text: str = ""
    tokens_in: int = 100
    tokens_out: int = 50
    cached_in: int = 0
    usd: float = 0.001

    @classmethod
    def code(cls, source: str, **kw) -> FakeResponse:
        return cls(text=f"```python\n{source}\n```", **kw)

    @classmethod
    def truncated(cls) -> FakeResponse:
        return cls(text="import pandas as pd\ndef main(", hit_token_ceiling=True)

    @classmethod
    def prose(cls) -> FakeResponse:
        return cls(text="Sure! Here is how I would approach this problem.")

    @classmethod
    def refusal(cls) -> FakeResponse:
        return cls(
            text="",
            status=GenerateStatus.REFUSAL,
            error_text="The provider declined this request.",
        )


@dataclass
class FakeRuntime:
    scripted: list[FakeResponse] = field(default_factory=list)
    default: FakeResponse = field(default_factory=FakeResponse)
    provider: str = "fake"
    model: str = "fake-model"
    calls: list[tuple[Action, str]] = field(default_factory=list)

    def generate(self, prompt: str, *, action: Action = "draft") -> GenerateResult:
        self.calls.append((action, prompt))
        r = self.scripted.pop(0) if self.scripted else self.default

        if r.status is not None:
            status, code, error_text = r.status, None, r.error_text
        else:
            status, code, error_text = classify_output(
                r.text, hit_token_ceiling=r.hit_token_ceiling
            )

        return GenerateResult(
            status=status,
            code=code,
            raw_text=r.text,
            usage=TokenUsage(r.tokens_in, r.tokens_out, r.cached_in),
            provider=self.provider,
            model=self.model,
            usd=r.usd,
            error_text=error_text,
            prompt_chars=len(prompt),
        )
