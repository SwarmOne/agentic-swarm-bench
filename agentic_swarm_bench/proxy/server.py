"""Recording proxy server for agentic benchmarks.

Sits between an agent (Claude Code, etc.) and an OpenAI-compatible endpoint.
Translates Anthropic Messages API to OpenAI Chat Completions API, records timing.

Usage:
    python -m agentic_swarm_bench.proxy.server --upstream http://localhost:8000 --port 19000
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

from agentic_swarm_bench.proxy.context import pad_messages_to_target
from agentic_swarm_bench.proxy.translators import (
    anthropic_to_openai,
    make_anthropic_stream_events,
    openai_to_anthropic_response,
)
from agentic_swarm_bench.proxy.utils import _detect_upstream_api


def create_app(
    upstream_url: str,
    model: str,
    api_key: str = "",
    api_key_header: str = "Authorization",
    context_target_tokens: int = 0,
    log_dir: str = "./traces",
    upstream_api: str | None = None,
) -> "FastAPI":
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI and uvicorn are required for the proxy. "
            "Install with: pip install agentic-swarm-bench[proxy]"
        )

    app = FastAPI(title="agentic-swarm-bench Recording Proxy")
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_log = log_path / "metrics.jsonl"

    is_anthropic_upstream = _detect_upstream_api(upstream_url, upstream_api) == "anthropic"

    state = {"request_counter": 0}

    def _upstream_headers() -> dict:
        headers = {"Content-Type": "application/json"}
        if not api_key:
            return headers
        if api_key_header.lower() == "authorization":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers[api_key_header] = api_key
        return headers

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

        if is_messages_api and body_json and is_anthropic_upstream:
            return await _handle_anthropic_passthrough(
                request, req_id, body_json, is_streaming, t_start, metrics_log,
            )

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

    _ANTHROPIC_FORWARD_HEADERS = {"anthropic-version", "anthropic-beta"}

    async def _handle_anthropic_passthrough(
        request: Request, req_id: int, body_json: dict, is_streaming: bool,
        t_start: float, metrics_log_path: Path,
    ):
        """Forward Anthropic requests natively to an Anthropic upstream, capture metrics."""
        target_url = upstream_url.rstrip("/") + "/v1/messages"
        headers = _upstream_headers()
        for h in _ANTHROPIC_FORWARD_HEADERS:
            val = request.headers.get(h)
            if val:
                headers[h] = val
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"

        if model:
            body_json["model"] = model

        metrics = {
            "req_id": req_id,
            "stream": is_streaming,
            "timestamp": datetime.now().isoformat(),
        }

        if not is_streaming:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                resp = await client.post(target_url, json=body_json, headers=headers)
            t_end = time.perf_counter()
            metrics["total_time_s"] = round(t_end - t_start, 3)
            if resp.status_code != 200:
                metrics["error"] = f"HTTP {resp.status_code}: {resp.text[:500]}"
                print(f"[proxy] ERROR: Upstream returned HTTP {resp.status_code}")
            with open(metrics_log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            fwd_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
            }
            return Response(content=resp.content, status_code=resp.status_code, headers=fwd_headers)

        async def _stream_anthropic():
            ttft = None
            token_count = 0
            first_time = None
            last_time = None
            upstream_error = None

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                async with client.stream(
                    "POST", target_url, json=body_json, headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        error_body = body.decode(errors='replace')[:500]
                        upstream_error = f"HTTP {resp.status_code}: {error_body}"
                        print(f"[proxy] ERROR: Upstream returned {upstream_error}")
                        yield body
                    else:
                        buf = b""
                        async for chunk_bytes in resp.aiter_bytes():
                            yield chunk_bytes

                            buf += chunk_bytes
                            while b"\n" in buf:
                                raw_line, buf = buf.split(b"\n", 1)
                                line = raw_line.decode("utf-8", errors="replace").strip()
                                if not line.startswith("data: "):
                                    continue
                                data_str = line[6:].strip()
                                try:
                                    data_obj = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue

                                now = time.perf_counter()
                                event_type = data_obj.get("type", "")

                                if event_type == "content_block_delta":
                                    delta = data_obj.get("delta", {})
                                    delta_type = delta.get("type", "")
                                    dt = delta_type
                                    is_text = dt == "text_delta" and delta.get("text")
                                    is_json = dt == "input_json_delta" and delta.get("partial_json")
                                    is_think = dt == "thinking_delta" and delta.get("thinking")
                                    has_content = is_text or is_json or is_think
                                    if has_content:
                                        if first_time is None:
                                            first_time = now
                                            ttft = (now - t_start) * 1000
                                        last_time = now
                                        token_count += 1

                                elif event_type == "message_delta":
                                    usage = data_obj.get("usage", {})
                                    if usage.get("output_tokens"):
                                        token_count = usage["output_tokens"]

                                elif event_type == "message_start":
                                    msg = data_obj.get("message", {})
                                    usage = msg.get("usage", {})
                                    metrics["input_tokens_actual"] = (
                                        usage.get("input_tokens", 0)
                                        + usage.get("cache_read_input_tokens", 0)
                                        + usage.get("cache_creation_input_tokens", 0)
                                    )

            t_end = time.perf_counter()
            metrics["ttft_ms"] = round(ttft, 2) if ttft else None
            metrics["total_time_s"] = round(t_end - t_start, 3)
            metrics["output_tokens"] = token_count
            if upstream_error:
                metrics["error"] = upstream_error

            if first_time and last_time and token_count > 1:
                decode_time = last_time - first_time
                metrics["decode_time_s"] = round(decode_time, 3)
                if decode_time > 0:
                    metrics["tok_per_sec"] = round(token_count / decode_time, 2)
                else:
                    metrics["tok_per_sec"] = 0
            else:
                metrics["tok_per_sec"] = 0

            input_for_prefill = metrics.get("input_tokens_actual", 0)
            if ttft and input_for_prefill:
                metrics["prefill_tok_per_sec"] = round(input_for_prefill / (ttft / 1000), 2)

            with open(metrics_log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

        return StreamingResponse(_stream_anthropic(), media_type="text/event-stream")

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
        )

    upstream = f"{upstream_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if not is_streaming:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            response = await client.post(upstream, json=oai_body, headers=headers)
        t_end = time.perf_counter()
        metrics = {"req_id": req_id, "stream": False, "total_time_s": round(t_end - t_start, 3)}

        if response.status_code != 200:
            error_msg = (
                f"Upstream returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
            print(f"[proxy] ERROR: {error_msg}")
            metrics["error"] = error_msg
            anthropic_resp = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": (
                        f"Upstream at {upstream} returned HTTP {response.status_code}. "
                        f"The endpoint may not support OpenAI-compatible /v1/chat/completions."
                    ),
                },
            }
        else:
            oai_resp = response.json()
            if "choices" in oai_resp:
                anthropic_resp = openai_to_anthropic_response(oai_resp, anthropic_model)
            else:
                anthropic_resp = oai_resp

        with open(metrics_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        return JSONResponse(
            content=anthropic_resp,
            status_code=200 if response.status_code == 200 else 502,
        )

    msg_id = "msg_" + uuid.uuid4().hex[:24]
    metrics = {"req_id": req_id, "stream": True, "timestamp": datetime.now().isoformat()}

    async def _stream():
        ttft = None
        token_count = 0
        first_content_time = None
        last_content_time = None
        input_tokens_actual = 0
        output_tokens_actual = 0
        upstream_error = None

        for event in make_anthropic_stream_events(anthropic_model, msg_id):
            yield event.encode()

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream("POST", upstream, json=oai_body, headers=headers) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    upstream_error = (
                        f"Upstream returned HTTP {response.status_code}: "
                        f"{body.decode(errors='replace')[:500]}"
                    )
                    print(f"[proxy] ERROR: {upstream_error}")
                    error_text = (
                        f"[Proxy error] Upstream at {upstream} returned HTTP "
                        f"{response.status_code}. The endpoint may not support "
                        f"OpenAI-compatible /v1/chat/completions."
                    )
                    delta_event = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": error_text},
                    }
                    sse = f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                    yield sse.encode()
                else:
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
                            input_tokens_actual = usage.get(
                                "prompt_tokens", input_tokens_actual,
                            )
                            output_tokens_actual = usage.get(
                                "completion_tokens", output_tokens_actual,
                            )

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
        if upstream_error:
            metrics["error"] = upstream_error
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
    api_key_header: str = "Authorization",
    context_target_tokens: int = 0,
    log_dir: str = "./traces",
    upstream_api: str | None = None,
):
    """Start the recording proxy server."""
    if not HAS_FASTAPI:
        raise ImportError("Install proxy deps: pip install agentic-swarm-bench[proxy]")

    detected_api = _detect_upstream_api(upstream_url, upstream_api)

    app = create_app(
        upstream_url=upstream_url,
        model=model,
        api_key=api_key,
        api_key_header=api_key_header,
        context_target_tokens=context_target_tokens,
        log_dir=log_dir,
        upstream_api=upstream_api,
    )
    print(f"agentic-swarm-bench proxy on :{port} -> {upstream_url}")
    print(f"  Model: {model}")
    print(f"  Upstream API: {detected_api}")
    print(f"  Context target: {context_target_tokens} tokens (0 = no padding)")
    print(f"  Traces: {log_dir}")
    print(f"  Metrics: http://localhost:{port}/benchmark/summary")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
