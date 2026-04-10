"""Recording proxy server for agentic benchmarks.

Sits between a coding agent (Claude Code, etc.) and an OpenAI-compatible endpoint.
Translates Anthropic Messages API to OpenAI Chat Completions API, records timing.

Usage:
    python -m agentic_coding_bench.proxy.server --upstream http://localhost:8000 --port 19000
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from agentic_coding_bench.proxy.padding import pad_messages_to_target
from agentic_coding_bench.proxy.translators import (
    anthropic_to_openai,
    make_anthropic_stream_events,
    openai_to_anthropic_response,
)


def create_app(
    upstream_url: str,
    model: str,
    api_key: str = "",
    context_target_tokens: int = 0,
    defeat_cache: bool = True,
    log_dir: str = "./traces",
) -> "FastAPI":
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI and uvicorn are required for the proxy. "
            "Install with: pip install agentic-coding-bench[proxy]"
        )

    app = FastAPI(title="agentic-coding-bench Recording Proxy")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_log = log_path / "metrics.jsonl"

    state = {"request_counter": 0}

    @app.get("/benchmark/metrics")
    async def get_metrics():
        if not metrics_log.exists():
            return {"metrics": []}
        lines = metrics_log.read_text().strip().split("\n")
        return {"metrics": [json.loads(line) for line in lines if line]}

    @app.get("/benchmark/summary")
    async def get_summary():
        if not metrics_log.exists():
            return {"error": "No metrics yet"}
        lines = metrics_log.read_text().strip().split("\n")
        entries = [json.loads(line) for line in lines if line]
        streaming = [e for e in entries if e.get("stream")]
        if not streaming:
            return {"total_requests": len(entries), "streaming_requests": 0}

        def _stats(vals):
            if not vals:
                return {}
            vals = sorted(vals)
            n = len(vals)
            return {
                "count": n,
                "min": round(vals[0], 2),
                "max": round(vals[-1], 2),
                "mean": round(sum(vals) / n, 2),
                "median": round(vals[n // 2], 2),
                "p95": round(vals[int(n * 0.95)], 2) if n > 1 else round(vals[0], 2),
            }

        return {
            "total_requests": len(entries),
            "streaming_requests": len(streaming),
            "ttft_ms": _stats([e["ttft_ms"] for e in streaming if e.get("ttft_ms")]),
            "tok_per_sec": _stats([e["tok_per_sec"] for e in streaming if e.get("tok_per_sec")]),
            "prefill_tok_per_sec": _stats(
                [e["prefill_tok_per_sec"] for e in streaming if e.get("prefill_tok_per_sec")]
            ),
        }

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def proxy(request: Request, path: str):
        state["request_counter"] += 1
        req_id = state["request_counter"]

        body = await request.body()
        body_json = None
        is_streaming = False
        is_messages_api = path.rstrip("/") in ("v1/messages",)

        try:
            body_json = json.loads(body)
            is_streaming = body_json.get("stream", False)
        except Exception:
            pass

        t_start = time.perf_counter()

        if is_messages_api and body_json:
            return await _handle_messages(
                req_id,
                body_json,
                is_streaming,
                t_start,
                upstream_url,
                model,
                api_key,
                context_target_tokens,
                defeat_cache,
                metrics_log,
                log_path,
            )

        target = f"{upstream_url}/{path}"
        if request.url.query:
            target += f"?{request.url.query}"
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=headers,
                content=body,
            )

        fwd_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
        }
        return Response(content=resp.content, status_code=resp.status_code, headers=fwd_headers)

    return app


async def _handle_messages(
    req_id,
    body_json,
    is_streaming,
    t_start,
    upstream_url,
    model,
    api_key,
    context_target_tokens,
    defeat_cache,
    metrics_log,
    log_path,
):
    """Handle /v1/messages: translate Anthropic -> OpenAI, record metrics."""
    anthropic_model = body_json.get("model", "unknown")
    oai_body = anthropic_to_openai(body_json, model)

    if context_target_tokens > 0:
        oai_body["messages"] = pad_messages_to_target(
            oai_body["messages"],
            context_target_tokens,
            defeat_cache=defeat_cache,
        )

    upstream = f"{upstream_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if not is_streaming:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            response = await client.post(upstream, json=oai_body, headers=headers)
        t_end = time.perf_counter()
        oai_resp = response.json() if response.status_code == 200 else {}
        if "choices" in oai_resp:
            anthropic_resp = openai_to_anthropic_response(oai_resp, anthropic_model)
        else:
            anthropic_resp = oai_resp
        metrics = {"req_id": req_id, "stream": False, "total_time_s": round(t_end - t_start, 3)}
        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        return JSONResponse(content=anthropic_resp, status_code=response.status_code)

    msg_id = "msg_" + uuid.uuid4().hex[:24]
    metrics = {"req_id": req_id, "stream": True, "timestamp": datetime.now().isoformat()}

    async def _stream():
        ttft = None
        token_count = 0
        first_content_time = None
        last_content_time = None
        input_tokens_actual = 0
        output_tokens_actual = 0

        for event in make_anthropic_stream_events(anthropic_model, msg_id):
            yield event.encode()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream("POST", upstream, json=oai_body, headers=headers) as response:
                async for chunk in response.aiter_lines():
                    if not chunk.startswith("data: "):
                        continue
                    data_str = chunk[6:].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    now = time.perf_counter()

                    usage = event.get("usage")
                    if usage:
                        input_tokens_actual = usage.get("prompt_tokens", input_tokens_actual)
                        output_tokens_actual = usage.get("completion_tokens", output_tokens_actual)

                    for choice in event.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if not content:
                            continue
                        if first_content_time is None:
                            first_content_time = now
                            ttft = (now - t_start) * 1000
                        last_content_time = now
                        token_count += 1

                        delta_event = {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": content},
                        }
                        sse = f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                        yield sse.encode()

        if output_tokens_actual > 0:
            token_count = output_tokens_actual

        block_stop = {"type": "content_block_stop", "index": 0}
        yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n".encode()

        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": token_count},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode()
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'.encode()

        t_end = time.perf_counter()
        metrics["ttft_ms"] = round(ttft, 2) if ttft else None
        metrics["total_time_s"] = round(t_end - t_start, 3)
        metrics["output_tokens"] = token_count
        if input_tokens_actual:
            metrics["input_tokens_actual"] = input_tokens_actual

        if first_content_time and last_content_time and token_count > 1:
            decode_time = last_content_time - first_content_time
            metrics["decode_time_s"] = round(decode_time, 3)
            metrics["tok_per_sec"] = round(token_count / decode_time, 2) if decode_time > 0 else 0
        else:
            metrics["tok_per_sec"] = 0

        input_for_prefill = input_tokens_actual or 0
        if ttft and input_for_prefill:
            metrics["prefill_tok_per_sec"] = round(input_for_prefill / (ttft / 1000), 2)

        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    return StreamingResponse(_stream(), media_type="text/event-stream")


def run_proxy(
    upstream_url: str = "http://localhost:8000",
    port: int = 19000,
    model: str = "default",
    api_key: str = "",
    context_target_tokens: int = 0,
    defeat_cache: bool = True,
    log_dir: str = "./traces",
):
    """Start the recording proxy server."""
    if not HAS_FASTAPI:
        raise ImportError("Install proxy deps: pip install agentic-coding-bench[proxy]")

    app = create_app(
        upstream_url=upstream_url,
        model=model,
        api_key=api_key,
        context_target_tokens=context_target_tokens,
        defeat_cache=defeat_cache,
        log_dir=log_dir,
    )
    print(f"agentic-coding-bench proxy on :{port} -> {upstream_url}")
    print(f"  Model: {model}")
    print(f"  Context target: {context_target_tokens} tokens (0 = no padding)")
    print(f"  Defeat cache: {defeat_cache}")
    print(f"  Traces: {log_dir}")
    print(f"  Metrics: http://localhost:{port}/benchmark/summary")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
