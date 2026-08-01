"""Day-2 risk check: does each provider actually report token usage?

`max_usd` is computed from the usage object. If a provider returns it empty or
zeroed, the budget cap silently computes $0.00 and never trips — and you find
out from a bill rather than from the run journal. So this runs once, per
provider, before any budget logic is written, and prints the raw object.

    uv run python scripts/day2_usage_probe.py            # all configured providers
    uv run python scripts/day2_usage_probe.py deepseek   # just one
"""

from __future__ import annotations

import os
import sys

PROMPT = "Reply with exactly the word: ok"

PROBES = {
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-v4-flash"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-5-nano"),
}


def probe_openai_compatible(provider: str) -> dict | None:
    from openai import OpenAI

    env, base_url, model = PROBES[provider]
    key = os.environ.get(env)
    if not key:
        print(f"[{provider}] {env} unset — skipped")
        return None

    client = OpenAI(api_key=key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_completion_tokens=64,
    )
    usage = resp.usage
    print(f"\n[{provider}] model={model}")
    print(f"  text          : {resp.choices[0].message.content!r}")
    print(f"  finish_reason : {resp.choices[0].finish_reason}")
    print(f"  usage (raw)   : {usage.model_dump() if usage else None}")
    if usage is None:
        print("  !! usage is None — max_usd cannot be enforced on this provider")
        return None
    return {
        "tokens_in": usage.prompt_tokens,
        "tokens_out": usage.completion_tokens,
        "cached_in": getattr(usage, "prompt_cache_hit_tokens", None)
        or getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0)
        or 0,
    }


def probe_anthropic() -> dict | None:
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[anthropic] ANTHROPIC_API_KEY unset — skipped")
        return None

    model = "claude-opus-5"
    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model,
        max_tokens=1024,  # required on Anthropic; caps thinking + text together
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": PROMPT}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    print(f"\n[anthropic] model={model}")
    print(f"  text          : {text!r}")
    print(f"  stop_reason   : {resp.stop_reason}")
    print(f"  usage (raw)   : {resp.usage.model_dump()}")
    return {
        "tokens_in": resp.usage.input_tokens,
        "tokens_out": resp.usage.output_tokens,
        "cached_in": (resp.usage.cache_read_input_tokens or 0),
    }


def main() -> int:
    wanted = sys.argv[1:] or ["deepseek", "openai", "anthropic"]
    results: dict[str, dict | None] = {}

    for provider in wanted:
        try:
            if provider == "anthropic":
                results[provider] = probe_anthropic()
            else:
                results[provider] = probe_openai_compatible(provider)
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
            print(f"\n[{provider}] FAILED: {type(exc).__name__}: {exc}")
            results[provider] = None

    print("\n=== normalized ===")
    for provider, r in results.items():
        if r is None:
            print(f"  {provider:10s} unusable for budget accounting")
        else:
            usable = r["tokens_in"] > 0 and r["tokens_out"] > 0
            print(
                f"  {provider:10s} in={r['tokens_in']:5d} out={r['tokens_out']:5d} "
                f"cached={r['cached_in']:5d}  {'OK' if usable else 'ZEROED — max_usd unsafe'}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
