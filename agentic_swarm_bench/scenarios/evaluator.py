"""Evaluate model responses against directives from scenario.json.

Directive types:

  contains     Substring match (case_sensitive defaults to true).
  regex        Python ``re.search`` against response text.
  llm          Send response + prompt to an LLM and parse YES/NO.
               Only runs when ``evaluate_llm=True`` is passed.

Each directive may optionally include ``"seq": N`` to target a specific
entry's response.  Without ``seq``, the directive is checked against
every turn's response and passes if ANY turn matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx


@dataclass
class EvalResult:
    """Result of one evaluation directive against one or more responses."""

    directive_type: str = ""
    passed: bool = False
    detail: str = ""
    matched_seq: int | None = None
    directive_index: int | None = None


def evaluate_response(
    directives: list[dict],
    response_text: str,
    seq: int,
    *,
    evaluate_llm: bool = False,
) -> list[EvalResult]:
    """Run non-LLM directives against a single turn's response.

    Returns one ``EvalResult`` per applicable directive (directives
    whose ``seq`` matches or that have no ``seq``).
    """
    results: list[EvalResult] = []
    for di, directive in enumerate(directives):
        target_seq = directive.get("seq")
        if target_seq is not None and target_seq != seq:
            continue

        dtype = directive.get("type", "")

        if dtype == "contains":
            r = _eval_contains(directive, response_text, seq)
        elif dtype == "regex":
            r = _eval_regex(directive, response_text, seq)
        elif dtype == "llm":
            continue
        else:
            r = EvalResult(
                directive_type=dtype,
                passed=False,
                detail=f"unknown directive type: {dtype!r}",
            )

        r.directive_index = di
        results.append(r)

    return results


async def evaluate_response_llm(
    directives: list[dict],
    response_text: str,
    seq: int,
    *,
    client: httpx.AsyncClient,
    url: str,
    model: str,
    headers: dict,
    timeout: float = 30.0,
) -> list[EvalResult]:
    """Run LLM-type directives against a single turn's response."""
    results: list[EvalResult] = []
    for di, directive in enumerate(directives):
        if directive.get("type") != "llm":
            continue
        target_seq = directive.get("seq")
        if target_seq is not None and target_seq != seq:
            continue

        prompt = directive.get("prompt", "")
        result = await _eval_llm_single(
            prompt, response_text, client=client, url=url,
            model=model, headers=headers, timeout=timeout,
        )
        result.directive_index = di
        results.append(result)

    return results


def aggregate_task_evals(
    per_turn_results: list[list[EvalResult]],
    directives: list[dict],
) -> list[EvalResult]:
    """Aggregate per-turn results into per-directive task-level results.

    A directive passes the task if ANY turn's response matched it.
    Returns one ``EvalResult`` per directive.
    """
    if not directives:
        return []

    aggregated: list[EvalResult] = []
    for di, directive in enumerate(directives):
        dtype = directive.get("type", "")
        target_seq = directive.get("seq")

        best: EvalResult | None = None
        for turn_results in per_turn_results:
            for result in turn_results:
                if result.directive_index != di:
                    continue
                if result.passed:
                    best = result
                    break
                if best is None:
                    best = result
            if best and best.passed:
                break

        if best is not None:
            aggregated.append(best)
        else:
            value = directive.get("value", directive.get("pattern", directive.get("prompt", "")))
            detail = "no applicable responses"
            if target_seq is not None:
                detail = f"seq {target_seq} not seen"
            aggregated.append(EvalResult(
                directive_type=dtype,
                passed=False,
                detail=f"{dtype}({_truncate(value, 30)}): {detail}",
                directive_index=di,
            ))

    return aggregated


def _eval_contains(directive: dict, text: str, seq: int) -> EvalResult:
    value = directive.get("value", "")
    case_sensitive = directive.get("case_sensitive", True)

    if case_sensitive:
        matched = value in text
    else:
        matched = value.lower() in text.lower()

    detail_prefix = f"contains({_truncate(value, 40)})"
    if matched:
        return EvalResult(
            directive_type="contains",
            passed=True,
            detail=f"{detail_prefix}: matched at seq {seq}",
            matched_seq=seq,
        )
    return EvalResult(
        directive_type="contains",
        passed=False,
        detail=f"{detail_prefix}: no match",
    )


def _eval_regex(directive: dict, text: str, seq: int) -> EvalResult:
    pattern = directive.get("pattern", "")
    detail_prefix = f"regex({_truncate(pattern, 40)})"

    try:
        match = re.search(pattern, text)
    except re.error as e:
        return EvalResult(
            directive_type="regex",
            passed=False,
            detail=f"{detail_prefix}: invalid regex: {e}",
        )

    if match:
        return EvalResult(
            directive_type="regex",
            passed=True,
            detail=f"{detail_prefix}: matched at seq {seq}",
            matched_seq=seq,
        )
    return EvalResult(
        directive_type="regex",
        passed=False,
        detail=f"{detail_prefix}: no match",
    )


async def _eval_llm_single(
    prompt: str,
    response_text: str,
    *,
    client: httpx.AsyncClient,
    url: str,
    model: str,
    headers: dict,
    timeout: float = 30.0,
) -> EvalResult:
    """Ask an LLM whether the response satisfies the prompt."""
    detail_prefix = f"llm({_truncate(prompt, 40)})"
    judge_messages = [
        {"role": "system", "content": "You are an evaluation judge. Answer YES or NO only."},
        {
            "role": "user",
            "content": (
                f"Given this model response:\n\n{response_text}\n\n"
                f"{prompt}\n\nAnswer YES or NO."
            ),
        },
    ]
    payload = {
        "model": model,
        "messages": judge_messages,
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return EvalResult(
                directive_type="llm",
                passed=False,
                detail=f"{detail_prefix}: HTTP {resp.status_code}",
            )

        data = resp.json()
        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .upper()
        )

        passed = answer.startswith("YES")
        return EvalResult(
            directive_type="llm",
            passed=passed,
            detail=f"{detail_prefix}: {answer[:20]}",
        )
    except Exception as e:
        return EvalResult(
            directive_type="llm",
            passed=False,
            detail=f"{detail_prefix}: {type(e).__name__}: {str(e)[:100]}",
        )


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
