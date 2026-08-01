"""Project-wide settings. Every knob lives here; nothing reads os.environ directly.

Loaded once at process start and passed down. The worker and the web process each
build their own Settings from the same .env, so they cannot drift.

    .env / environment
           │
           ▼
       Settings ──┬──► AgentRuntime   (provider, model, max_tokens, api key)
                  ├──► SearchLoop     (budgets, circuit breaker)
                  ├──► Executor       (docker flags, timeout, paths)
                  └──► preflight      (validates all of the above before anything runs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "deepseek", "anthropic"]
Action = Literal["draft", "improve", "debug"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DeepSeek is deliberately OpenAI-shaped, so it rides the same adapter and differs
# only by base URL. Anthropic gets its own adapter and never appears here.
OPENAI_COMPATIBLE_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KC_",
        extra="ignore",
    )

    # ---- competition -------------------------------------------------------
    competition: str = Field(
        default="",
        description="Kaggle competition slug, e.g. 'playground-series-s5e7'. Set on Day 0.",
    )

    # ---- LLM provider ------------------------------------------------------
    # DeepSeek by default: a 40-node search on deepseek-v4-flash costs cents,
    # which is what makes a big overnight tree affordable. Override per action
    # below to put a stronger model on `debug`, where reasoning quality pays.
    provider: Provider = "deepseek"
    model: str = "deepseek-v4-flash"

    # Per-action overrides. None means "use `model`". Lets the bulk of the search
    # run on something cheap while debug — where reasoning quality actually changes
    # the outcome — uses a stronger model.
    model_draft: str | None = None
    model_improve: str | None = None
    model_debug: str | None = None

    # Anthropic REQUIRES this; OpenAI-compatible providers default it.
    #
    # Every current frontier model is a reasoning model, and this ceiling covers
    # reasoning AND visible output. Measured on the first real draft: 8192 was
    # entirely consumed before a single line of train.py appeared
    # (stop_reason=length, empty text). A full training script plus the
    # reasoning to design it needs room — do not shave this to save money. The
    # cost of a truncated node is the whole node, not the tokens you saved.
    max_tokens: int = 32000

    # Reasoning depth. Measured: deepseek-v4-flash at its default effort consumed
    # all 32000 output tokens on reasoning alone — twice — and never emitted a
    # line of train.py, at ~5 minutes and $0.009 per wasted attempt. At high/max
    # effort these models expect an output budget in the hundreds of thousands of
    # tokens, which is not the shape of this task: writing one training script is
    # a well-specified job, not an open research problem.
    #
    # None means "do not send the parameter" — required for non-reasoning models,
    # which reject it outright.
    reasoning_effort: str | None = "low"

    max_retries: int = 3
    request_timeout_s: int = 600

    openai_api_key: str = ""
    deepseek_api_key: str = ""
    anthropic_api_key: str = ""

    # ---- budgets -----------------------------------------------------------
    max_nodes: int = 20
    max_wall_clock_s: int = 8 * 3600
    max_usd: float = 5.0

    # Circuit breaker: budget caps alone let the loop grind through one failure
    # mode 20 times and report a completed run.
    max_consecutive_failures: int = 5
    max_consecutive_infra_errors: int = 3

    # How stale a worker's heartbeat may get before another worker may steal the
    # lock. Long enough to outlast one slow node, short enough that a crash at
    # 3am does not leave the project locked until morning.
    worker_lock_ttl_s: int = 90

    # Search policy. Frozen for week 1 — tuning it is the project's named rabbit hole.
    policy_max_depth: int = 5
    policy_seed_drafts: int = 3

    # ---- prompt ------------------------------------------------------------
    # Context blocks are truncated to this budget, oldest discussion summaries
    # dropped first. This is the real cost curve as the tree grows.
    context_char_budget: int = 20_000
    tree_summary_max_nodes: int = 40

    # Whether the feeders' output is injected into every node prompt. Separate
    # flags because experiment C enabled both at once and therefore cannot say
    # which of them caused the regression it measured (mean cv 0.81607 -> 0.81000,
    # +52% prompt). Toggling one at a time is the experiment that would.
    #
    # Both default off: the bundle was measured harmful, so neither is presumed
    # innocent until it has been tested on its own. Generating the documents is
    # unaffected — `kagglecracker eda` and `discussions` always write their files.
    inject_eda: bool = False
    inject_discussions: bool = False

    # Append observed API-drift notes to the output contract. A flag rather than
    # a constant so the two arms of the experiment are reproducible: the notes
    # cost context on every node, so they have to earn their place against a
    # measured failure rate rather than a plausible story.
    contract_api_notes: bool = False

    # ---- executor ----------------------------------------------------------
    docker_image: str = "kagglecracker-sandbox:latest"
    node_timeout_s: int = 900
    docker_cpus: float = 4.0
    docker_memory: str = "8g"
    docker_pids_limit: int = 256

    # ---- paths -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    runs_dir: Path = PROJECT_ROOT / "runs"
    context_dir: Path = PROJECT_ROOT / "context"
    db_path: Path = PROJECT_ROOT / "kagglecracker.db"

    # ---- derived -----------------------------------------------------------
    @property
    def api_key(self) -> str:
        return {
            "openai": self.openai_api_key,
            "deepseek": self.deepseek_api_key,
            "anthropic": self.anthropic_api_key,
        }[self.provider]

    @property
    def base_url(self) -> str | None:
        """None for Anthropic — its SDK owns its own endpoint."""
        return OPENAI_COMPATIBLE_BASE_URLS.get(self.provider)

    def model_for(self, action: Action) -> str:
        override = {
            "draft": self.model_draft,
            "improve": self.model_improve,
            "debug": self.model_debug,
        }[action]
        return override or self.model

    def models_in_use(self) -> set[str]:
        """Every model this config can actually call. Preflight prices all of them."""
        return {self.model_for(a) for a in ("draft", "improve", "debug")}

    @model_validator(mode="after")
    def _fall_back_to_standard_env_names(self) -> Settings:
        """Accept the conventional OPENAI_API_KEY / DEEPSEEK_API_KEY / ANTHROPIC_API_KEY.

        pydantic-settings only sees KC_-prefixed vars. Everyone already has the
        unprefixed ones exported, and making them retype the key under a new name
        is exactly the kind of friction that gets worked around with a hardcoded
        literal.
        """
        import os

        for field, env_name in (
            ("openai_api_key", "OPENAI_API_KEY"),
            ("deepseek_api_key", "DEEPSEEK_API_KEY"),
            ("anthropic_api_key", "ANTHROPIC_API_KEY"),
        ):
            if not getattr(self, field):
                object.__setattr__(self, field, os.environ.get(env_name, ""))
        return self


def load_settings() -> Settings:
    return Settings()
