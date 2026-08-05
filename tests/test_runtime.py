"""Runtime output classification and cost accounting.

The classification order is the load-bearing part: a truncated response usually
*also* fails to parse, and reporting that as a syntax error sends the debug
action after the symptom instead of the cause.
"""

from __future__ import annotations

import pytest

from kagglecracker.runtime.base import (
    GenerateStatus,
    TokenUsage,
    classify_output,
    extract_code,
)
from kagglecracker.runtime.fake import FakeResponse, FakeRuntime
from kagglecracker.runtime.pricing import (
    UnpricedModelError,
    assert_all_priced,
    estimate_usd,
    price_for,
)

# --------------------------------------------------------------------------
# code extraction
# --------------------------------------------------------------------------


def test_extracts_fenced_block():
    assert extract_code("Here you go:\n```python\nx = 1\n```\nDone.") == "x = 1"


def test_extracts_unlabelled_fence():
    assert extract_code("```\nx = 1\n```") == "x = 1"


def test_bare_python_is_the_file():
    assert extract_code("import pandas as pd\nx = 1\n") == "import pandas as pd\nx = 1"


def test_picks_the_longest_block_when_several():
    """Short blocks are illustrative snippets; the real file is the big one."""
    text = (
        "```python\n# nope\n```\nand the actual file:\n"
        "```python\nimport pandas\nx = 1\ny = 2\n```"
    )
    assert "import pandas" in extract_code(text)


def test_prose_yields_no_code():
    assert extract_code("I would start by exploring the data.") is None


def test_empty_yields_no_code():
    assert extract_code("") is None
    assert extract_code("   \n  ") is None


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def test_valid_code_is_ok():
    status, code, err = classify_output("```python\nx = 1\n```", hit_token_ceiling=False)
    assert status is GenerateStatus.OK
    assert code == "x = 1"
    assert err == ""


def test_truncation_wins_over_syntax_error():
    """A cut-off response is also unparseable. Reporting the syntax error would
    point the debug action at the code instead of at max_tokens."""
    status, code, err = classify_output("def main(", hit_token_ceiling=True)
    assert status is GenerateStatus.TRUNCATED
    assert code is None
    assert "max_tokens" in err
    assert "not a defect in the generated code" in err


def test_truncated_is_a_config_fault_not_an_agent_fault():
    assert GenerateStatus.TRUNCATED.is_config_fault
    assert not GenerateStatus.NO_CODE.is_config_fault
    assert not GenerateStatus.INVALID_PYTHON.is_config_fault


def test_prose_only_is_no_code():
    status, code, err = classify_output("Sure, here's my plan.", hit_token_ceiling=False)
    assert status is GenerateStatus.NO_CODE
    assert code is None
    assert "no Python code block" in err


def test_unparseable_code_is_reported_with_a_line_number():
    status, code, err = classify_output(
        "```python\ndef broken(:\n    pass\n```", hit_token_ceiling=False
    )
    assert status is GenerateStatus.INVALID_PYTHON
    assert code is None
    assert "line" in err


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------


def test_unpriced_model_raises_rather_than_costing_zero():
    """The failure this prevents is silent: $0.00 per node means max_usd never trips."""
    with pytest.raises(UnpricedModelError, match="No price entry"):
        price_for("deepseek", "deepseek-imaginary-9000")


def test_unpriced_error_names_the_known_models():
    with pytest.raises(UnpricedModelError, match="deepseek-v4-flash"):
        price_for("deepseek", "nope")


def test_assert_all_priced_checks_every_configured_model():
    assert_all_priced("deepseek", {"deepseek-v4-flash", "deepseek-v4-pro"})
    with pytest.raises(UnpricedModelError):
        assert_all_priced("deepseek", {"deepseek-v4-flash", "nope"})


def test_cost_is_per_million_tokens():
    # deepseek-v4-flash: $0.14 in / $0.28 out per 1M
    usd = estimate_usd("deepseek", "deepseek-v4-flash", 1_000_000, 1_000_000)
    assert usd == pytest.approx(0.42)


def test_cached_input_is_billed_at_the_cache_rate():
    """DeepSeek cache hits are ~50x cheaper; missing them would make a long tree
    look far more expensive than it is."""
    fresh = estimate_usd("deepseek", "deepseek-v4-flash", 1_000_000, 0, cached_tokens_in=0)
    cached = estimate_usd(
        "deepseek", "deepseek-v4-flash", 1_000_000, 0, cached_tokens_in=1_000_000
    )
    assert cached < fresh / 10


def test_anthropic_cache_read_defaults_to_a_tenth_of_input():
    p = price_for("anthropic", "claude-opus-5")
    assert p.cache_read_usd == pytest.approx(p.input_usd * 0.1)


# --------------------------------------------------------------------------
# FakeRuntime
# --------------------------------------------------------------------------


def test_fake_runtime_scripts_a_sequence():
    rt = FakeRuntime(
        scripted=[FakeResponse.truncated(), FakeResponse.prose(), FakeResponse.code("x = 1")]
    )
    assert rt.generate("p", action="draft").status is GenerateStatus.TRUNCATED
    assert rt.generate("p", action="draft").status is GenerateStatus.NO_CODE
    third = rt.generate("p", action="draft")
    assert third.status is GenerateStatus.OK
    assert third.code == "x = 1"


def test_fake_runtime_records_action_and_prompt():
    rt = FakeRuntime()
    rt.generate("the prompt", action="improve")
    assert rt.calls == [("improve", "the prompt")]


def test_fake_runtime_refusal_carries_no_code():
    rt = FakeRuntime(scripted=[FakeResponse.refusal()])
    r = rt.generate("p", action="draft")
    assert r.status is GenerateStatus.REFUSAL
    assert r.code is None
    assert r.error_text


def test_token_usage_total():
    assert TokenUsage(tokens_in=10, tokens_out=5).total == 15
