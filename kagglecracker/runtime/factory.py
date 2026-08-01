"""The one place that branches on provider name.

Everything downstream holds an `AgentRuntime` and never asks which provider it
is talking to.
"""

from __future__ import annotations

from kagglecracker.config import Settings
from kagglecracker.runtime.base import AgentRuntime
from kagglecracker.runtime.pricing import assert_all_priced


def build_runtime(settings: Settings) -> AgentRuntime:
    # Fail here rather than three hours into an overnight run: an unpriced model
    # computes $0.00 per node, so max_usd would never trip.
    assert_all_priced(settings.provider, settings.models_in_use())

    if settings.provider == "anthropic":
        from kagglecracker.runtime.anthropic_rt import AnthropicRuntime

        return AnthropicRuntime(settings)

    from kagglecracker.runtime.openai_compat import OpenAICompatibleRuntime

    return OpenAICompatibleRuntime(settings)
