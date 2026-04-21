"""Tests for the evaluator module: contains, regex, llm evaluation, and aggregation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.markers import requires_evaluator

pytestmark = requires_evaluator

from agentic_swarm_bench.scenarios.evaluator import (
    EvalResult,
    aggregate_task_evals,
    evaluate_response,
    evaluate_response_llm,
)

# ---------------------------------------------------------------------------
# contains evaluation
# ---------------------------------------------------------------------------


class TestContains:
    def test_exact_match(self):
        results = evaluate_response(
            [{"type": "contains", "value": "Paris"}],
            "The capital of France is Paris.",
            seq=1,
        )
        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].directive_type == "contains"

    def test_no_match(self):
        results = evaluate_response(
            [{"type": "contains", "value": "Berlin"}],
            "The capital of France is Paris.",
            seq=1,
        )
        assert len(results) == 1
        assert results[0].passed is False

    def test_case_sensitive_default(self):
        results = evaluate_response(
            [{"type": "contains", "value": "paris"}],
            "The capital of France is Paris.",
            seq=1,
        )
        assert results[0].passed is False

    def test_case_insensitive(self):
        results = evaluate_response(
            [{"type": "contains", "value": "paris", "case_sensitive": False}],
            "The capital of France is Paris.",
            seq=1,
        )
        assert results[0].passed is True

    def test_empty_value_always_matches(self):
        results = evaluate_response(
            [{"type": "contains", "value": ""}],
            "anything",
            seq=1,
        )
        assert results[0].passed is True

    def test_empty_response(self):
        results = evaluate_response(
            [{"type": "contains", "value": "something"}],
            "",
            seq=1,
        )
        assert results[0].passed is False

    def test_matched_seq_is_set(self):
        results = evaluate_response(
            [{"type": "contains", "value": "Jupiter"}],
            "Jupiter is the largest planet.",
            seq=3,
        )
        assert results[0].matched_seq == 3


# ---------------------------------------------------------------------------
# regex evaluation
# ---------------------------------------------------------------------------


class TestRegex:
    def test_simple_match(self):
        results = evaluate_response(
            [{"type": "regex", "pattern": r"\d+"}],
            "The answer is 42.",
            seq=1,
        )
        assert results[0].passed is True

    def test_no_match(self):
        results = evaluate_response(
            [{"type": "regex", "pattern": r"^\d+$"}],
            "not just numbers here",
            seq=1,
        )
        assert results[0].passed is False

    def test_complex_pattern(self):
        results = evaluate_response(
            [{"type": "regex", "pattern": r"3\s*[×x*]\s*10|299"}],
            "The speed of light is approximately 3 × 10^8 m/s.",
            seq=1,
        )
        assert results[0].passed is True

    def test_invalid_regex(self):
        results = evaluate_response(
            [{"type": "regex", "pattern": r"[invalid"}],
            "anything",
            seq=1,
        )
        assert results[0].passed is False
        assert "invalid regex" in results[0].detail

    def test_matched_seq_is_set(self):
        results = evaluate_response(
            [{"type": "regex", "pattern": r"def\s+fib"}],
            "def fibonacci(n):",
            seq=5,
        )
        assert results[0].passed is True
        assert results[0].matched_seq == 5


# ---------------------------------------------------------------------------
# seq targeting
# ---------------------------------------------------------------------------


class TestSeqTargeting:
    def test_directive_with_seq_only_runs_on_matching_seq(self):
        results = evaluate_response(
            [{"type": "contains", "value": "Paris", "seq": 3}],
            "Paris is great",
            seq=1,
        )
        assert len(results) == 0

    def test_directive_with_seq_runs_on_match(self):
        results = evaluate_response(
            [{"type": "contains", "value": "Paris", "seq": 3}],
            "Paris is great",
            seq=3,
        )
        assert len(results) == 1
        assert results[0].passed is True

    def test_directive_without_seq_runs_on_all(self):
        r1 = evaluate_response(
            [{"type": "contains", "value": "hello"}],
            "hello world",
            seq=1,
        )
        r2 = evaluate_response(
            [{"type": "contains", "value": "hello"}],
            "hello world",
            seq=99,
        )
        assert len(r1) == 1
        assert len(r2) == 1

    def test_llm_directive_skipped_in_sync_eval(self):
        results = evaluate_response(
            [{"type": "llm", "prompt": "Does this answer the question?"}],
            "some response",
            seq=1,
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# unknown directive type
# ---------------------------------------------------------------------------


def test_unknown_type():
    results = evaluate_response(
        [{"type": "hamming_distance", "value": "abc"}],
        "some text",
        seq=1,
    )
    assert len(results) == 1
    assert results[0].passed is False
    assert "unknown" in results[0].detail


# ---------------------------------------------------------------------------
# multiple directives
# ---------------------------------------------------------------------------


def test_multiple_directives():
    results = evaluate_response(
        [
            {"type": "contains", "value": "Paris"},
            {"type": "regex", "pattern": r"capital"},
        ],
        "The capital of France is Paris.",
        seq=1,
    )
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_mixed_pass_fail():
    results = evaluate_response(
        [
            {"type": "contains", "value": "Paris"},
            {"type": "contains", "value": "Berlin"},
        ],
        "Paris is nice",
        seq=1,
    )
    assert results[0].passed is True
    assert results[1].passed is False


# ---------------------------------------------------------------------------
# aggregate_task_evals
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_passes_if_any_turn_matches(self):
        directives = [{"type": "contains", "value": "Paris"}]
        fail = EvalResult(
            directive_type="contains", passed=False,
            detail="no match", directive_index=0,
        )
        ok = EvalResult(
            directive_type="contains", passed=True,
            detail="matched", matched_seq=2, directive_index=0,
        )
        per_turn = [[fail], [ok]]
        agg = aggregate_task_evals(per_turn, directives)
        assert len(agg) == 1
        assert agg[0].passed is True

    def test_fails_if_no_turn_matches(self):
        directives = [{"type": "contains", "value": "Berlin"}]
        fail = EvalResult(
            directive_type="contains", passed=False,
            detail="no match", directive_index=0,
        )
        per_turn = [[fail], [fail]]
        agg = aggregate_task_evals(per_turn, directives)
        assert len(agg) == 1
        assert agg[0].passed is False

    def test_empty_directives(self):
        agg = aggregate_task_evals([], [])
        assert agg == []

    def test_no_applicable_responses(self):
        directives = [{"type": "regex", "pattern": ".*", "seq": 99}]
        per_turn: list[list] = [[]]
        agg = aggregate_task_evals(per_turn, directives)
        assert len(agg) == 1
        assert agg[0].passed is False

    def test_multiple_directives_aggregated_independently(self):
        directives = [
            {"type": "contains", "value": "Paris"},
            {"type": "contains", "value": "Berlin"},
        ]
        per_turn = [
            [
                EvalResult(
                    directive_type="contains", passed=True,
                    detail="Paris matched", directive_index=0,
                ),
                EvalResult(
                    directive_type="contains", passed=False,
                    detail="Berlin no", directive_index=1,
                ),
            ],
        ]
        agg = aggregate_task_evals(per_turn, directives)
        assert len(agg) == 2
        assert agg[0].passed is True
        assert agg[1].passed is False


# ---------------------------------------------------------------------------
# LLM evaluation
# ---------------------------------------------------------------------------


class TestLLMEval:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_llm_eval_passes_on_yes(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "YES"}}],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        results = self._run(
            evaluate_response_llm(
                [{"type": "llm", "prompt": "Is this about France?"}],
                "Paris is in France.",
                seq=1,
                client=mock_client,
                url="http://test/v1/chat/completions",
                model="test-model",
                headers={},
            )
        )
        assert len(results) == 1
        assert results[0].passed is True

    def test_llm_eval_fails_on_no(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "NO"}}],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        results = self._run(
            evaluate_response_llm(
                [{"type": "llm", "prompt": "Is this about Germany?"}],
                "Paris is in France.",
                seq=1,
                client=mock_client,
                url="http://test/v1/chat/completions",
                model="test-model",
                headers={},
            )
        )
        assert len(results) == 1
        assert results[0].passed is False

    def test_llm_eval_skips_non_llm_directives(self):
        mock_client = AsyncMock()

        results = self._run(
            evaluate_response_llm(
                [{"type": "contains", "value": "Paris"}],
                "Paris is great",
                seq=1,
                client=mock_client,
                url="http://test/v1",
                model="test",
                headers={},
            )
        )
        assert len(results) == 0
        mock_client.post.assert_not_called()

    def test_llm_eval_handles_http_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        results = self._run(
            evaluate_response_llm(
                [{"type": "llm", "prompt": "test"}],
                "response",
                seq=1,
                client=mock_client,
                url="http://test/v1",
                model="test",
                headers={},
            )
        )
        assert len(results) == 1
        assert results[0].passed is False
        assert "500" in results[0].detail

    def test_llm_eval_handles_exception(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("refused"))

        results = self._run(
            evaluate_response_llm(
                [{"type": "llm", "prompt": "test"}],
                "response",
                seq=1,
                client=mock_client,
                url="http://test/v1",
                model="test",
                headers={},
            )
        )
        assert len(results) == 1
        assert results[0].passed is False
        assert "ConnectionError" in results[0].detail

    def test_llm_eval_respects_seq(self):
        mock_client = AsyncMock()

        results = self._run(
            evaluate_response_llm(
                [{"type": "llm", "prompt": "test", "seq": 5}],
                "response",
                seq=1,
                client=mock_client,
                url="http://test/v1",
                model="test",
                headers={},
            )
        )
        assert len(results) == 0
        mock_client.post.assert_not_called()
