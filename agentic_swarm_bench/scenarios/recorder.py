"""Recording proxy that captures real coding sessions as JSONL recordings.

Sits between an agent (Claude Code, Cursor, etc.) and any LLM
endpoint. Every request/response pair is saved as a JSONL line, creating
a replayable scenario recording.

Supports two upstream modes:
  - OpenAI-compatible (default): translates Anthropic → OpenAI if needed
  - Anthropic passthrough: forwards native Anthropic requests as-is,
    converts messages to OpenAI format only for the JSONL recording

The recording file captures the exact messages, model, timing, and token
counts from a real session -- so anyone can replay it later against a
different endpoint to compare performance.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


_ANTHROPIC_HOSTS = ("api.anthropic.com", "anthropic.com")


def _detect_upstream_api(upstream_url: str, explicit: str | None) -> str:
    """Return 'anthropic' or 'openai' based on explicit flag or URL heuristic."""
    if explicit:
        return explicit

    from urllib.parse import urlparse

    host = urlparse(upstream_url).hostname or ""
    if any(host.endswith(h) for h in _ANTHROPIC_HOSTS):
        return "anthropic"
    return "openai"


def create_recording_app(
    upstream_url: str,
    model: str,
    api_key: str = "",
    api_key_header: str = "Authorization",
    output_file: str = "recording.jsonl",
    upstream_api: str | None = None,
) -> "FastAPI":
    """Create a FastAPI app that records all requests to a JSONL recording file."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI and uvicorn are required for recording. "
            "Install with: pip install agentic-swarm-bench[proxy]"
        )

    app = FastAPI(title="agentic-swarm-bench Recording Proxy")
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_anthropic_upstream = _detect_upstream_api(upstream_url, upstream_api) == "anthropic"

    state = {
        "experiment_id": uuid.uuid4().hex[:12],
        "seq": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def _upstream_headers() -> dict:
        headers = {"Content-Type": "application/json"}
        if not api_key:
            return headers
        if api_key_header.lower() == "authorization":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers[api_key_header] = api_key
        return headers

    def _resolve_upstream(path: str) -> str:
        base = upstream_url.rstrip("/")
        if path.startswith("/"):
            return base + path
        return base + "/" + path

    def _write_entry(entry: dict) -> None:
        with open(out_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _anthropic_messages_to_openai(body: dict) -> list[dict]:
        """Convert Anthropic message format to OpenAI for the JSONL recording."""
        from agentic_swarm_bench.proxy.translators import anthropic_to_openai

        oai_body = anthropic_to_openai(body, model)
        return oai_body.get("messages", [])

    @app.get("/recording/status")
    async def recording_status():
        return {
            "experiment_id": state["experiment_id"],
            "requests_recorded": state["seq"],
            "output_file": str(out_path),
            "started_at": state["started_at"],
            "upstream_api": "anthropic" if is_anthropic_upstream else "openai",
        }

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def record_and_forward(request: Request, path: str):
        body = await request.body()
        body_json = None
        is_chat_completion = "chat/completions" in path
        is_messages_api = path.rstrip("/") in ("v1/messages",)
        is_streaming = False

        try:
            body_json = json.loads(body)
            is_streaming = body_json.get("stream", False)
        except Exception:
            pass

        if not is_chat_completion and not is_messages_api:
            return await _passthrough(request, path, body)

        if body_json is None:
            return await _passthrough(request, path, body)

        # Anthropic passthrough: forward native Anthropic requests to Anthropic
        if is_messages_api and is_anthropic_upstream:
            return await _handle_anthropic_passthrough(request, body_json, is_streaming, path)

        # Default: translate Anthropic → OpenAI, forward to OpenAI endpoint
        state["seq"] += 1
        seq = state["seq"]
        t_start = time.perf_counter()

        if is_messages_api and body_json:
            from agentic_swarm_bench.proxy.translators import anthropic_to_openai

            oai_body = anthropic_to_openai(body_json, model)
            target_url = _resolve_upstream("v1/chat/completions")
        else:
            oai_body = body_json or {}
            if model:
                oai_body["model"] = model
            target_url = _resolve_upstream(path)

        headers = _upstream_headers()
        messages_snapshot = oai_body.get("messages", [])

        entry = {
            "seq": seq,
            "experiment_id": state["experiment_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages": messages_snapshot,
            "model": oai_body.get("model", model),
            "max_tokens": oai_body.get("max_tokens", 4096),
            "temperature": oai_body.get("temperature", 1.0),
            "stream": is_streaming,
            "source_api": "anthropic" if is_messages_api else "openai",
        }

        if is_streaming:
            return await _handle_streaming(
                entry, oai_body, target_url, headers, t_start, is_messages_api
            )

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(target_url, json=oai_body, headers=headers)

        t_end = time.perf_counter()
        entry["total_time_s"] = round(t_end - t_start, 3)
        entry["status_code"] = resp.status_code

        resp_json = None
        if resp.status_code == 200:
            resp_json = resp.json()
            usage = resp_json.get("usage", {})
            entry["prompt_tokens"] = usage.get("prompt_tokens", 0)
            entry["completion_tokens"] = usage.get("completion_tokens", 0)
            content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            entry["response_length"] = len(content)

        _write_entry(entry)

        if is_messages_api and resp_json is not None:
            from agentic_swarm_bench.proxy.translators import openai_to_anthropic_response

            anthropic_resp = openai_to_anthropic_response(
                resp_json, body_json.get("model", "unknown")
            )
            return JSONResponse(content=anthropic_resp)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
            },
        )

    _ANTHROPIC_FORWARD_HEADERS = {
        "anthropic-version",
        "anthropic-beta",
        "anthropic-dangerous-direct-browser-access",
    }

    async def _handle_anthropic_passthrough(
        request: Request, body_json: dict, is_streaming: bool, path: str
    ):
        """Forward Anthropic requests natively, record as OpenAI for replay."""
        state["seq"] += 1
        seq = state["seq"]
        t_start = time.perf_counter()

        target_url = _resolve_upstream(path)
        headers = _upstream_headers()
        for h in _ANTHROPIC_FORWARD_HEADERS:
            val = request.headers.get(h)
            if val:
                headers[h] = val

        oai_messages = _anthropic_messages_to_openai(body_json)

        entry = {
            "seq": seq,
            "experiment_id": state["experiment_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages": oai_messages,
            "model": body_json.get("model", model),
            "max_tokens": body_json.get("max_tokens", 4096),
            "temperature": body_json.get("temperature", 1.0),
            "stream": is_streaming,
            "source_api": "anthropic",
        }

        if is_streaming:
            return await _handle_anthropic_streaming(entry, body_json, target_url, headers, t_start)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(target_url, json=body_json, headers=headers)

        t_end = time.perf_counter()
        entry["total_time_s"] = round(t_end - t_start, 3)
        entry["status_code"] = resp.status_code

        if resp.status_code == 200:
            resp_json = resp.json()
            usage = resp_json.get("usage", {})
            entry["prompt_tokens"] = usage.get("input_tokens", 0)
            entry["completion_tokens"] = usage.get("output_tokens", 0)
            content_blocks = resp_json.get("content", [])
            text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
            entry["response_length"] = len(text)

        _write_entry(entry)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
            },
        )

    async def _handle_anthropic_streaming(
        entry: dict, body_json: dict, target_url: str, headers: dict, t_start: float
    ):
        """Stream from Anthropic, pass through to client, record metrics."""

        async def _stream():
            ttft = None
            token_count = 0
            first_time = None
            last_time = None

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                async with client.stream(
                    "POST", target_url, json=body_json, headers=headers
                ) as resp:
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
                                has_content = (
                                    (delta_type == "text_delta" and delta.get("text"))
                                    or (
                                        delta_type == "input_json_delta"
                                        and delta.get("partial_json")
                                    )
                                    or (delta_type == "thinking_delta" and delta.get("thinking"))
                                )
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
                                entry["prompt_tokens"] = (
                                    usage.get("input_tokens", 0)
                                    + usage.get("cache_read_input_tokens", 0)
                                    + usage.get("cache_creation_input_tokens", 0)
                                )

            t_end = time.perf_counter()
            entry["ttft_ms"] = round(ttft, 2) if ttft else None
            entry["total_time_s"] = round(t_end - t_start, 3)
            entry["completion_tokens"] = token_count

            if first_time and last_time and token_count > 1:
                decode_time = last_time - first_time
                entry["decode_time_s"] = round(decode_time, 3)
                entry["tok_per_sec"] = round(token_count / decode_time, 2) if decode_time > 0 else 0

            _write_entry(entry)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    async def _handle_streaming(entry, oai_body, target_url, headers, t_start, is_messages_api):
        """Handle streaming request via OpenAI upstream, record metrics, forward response."""

        async def _stream():
            ttft = None
            token_count = 0
            first_time = None
            last_time = None

            if is_messages_api:
                from agentic_swarm_bench.proxy.translators import make_anthropic_stream_events

                msg_id = "msg_" + uuid.uuid4().hex[:24]
                anth_model = entry.get("model", "unknown")
                for evt in make_anthropic_stream_events(anth_model, msg_id):
                    yield evt.encode()

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                async with client.stream(
                    "POST", target_url, json=oai_body, headers=headers
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        now = time.perf_counter()
                        for choice in chunk.get("choices", []):
                            content = choice.get("delta", {}).get("content")
                            if not content:
                                continue
                            if first_time is None:
                                first_time = now
                                ttft = (now - t_start) * 1000
                            last_time = now
                            token_count += 1

                            if is_messages_api:
                                delta_event = {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": content},
                                }
                                sse = (
                                    f"event: content_block_delta\n"
                                    f"data: {json.dumps(delta_event)}\n\n"
                                )
                                yield sse.encode()
                            else:
                                yield f"data: {data_str}\n\n".encode()

            if is_messages_api:
                block_stop = {"type": "content_block_stop", "index": 0}
                yield (f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n").encode()
                msg_delta = {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": token_count},
                }
                yield (f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n").encode()
                yield ('event: message_stop\ndata: {"type": "message_stop"}\n\n').encode()
            else:
                yield b"data: [DONE]\n\n"

            t_end = time.perf_counter()
            entry["ttft_ms"] = round(ttft, 2) if ttft else None
            entry["total_time_s"] = round(t_end - t_start, 3)
            entry["completion_tokens"] = token_count

            if first_time and last_time and token_count > 1:
                decode_time = last_time - first_time
                entry["decode_time_s"] = round(decode_time, 3)
                entry["tok_per_sec"] = round(token_count / decode_time, 2) if decode_time > 0 else 0

            _write_entry(entry)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    async def _passthrough(request: Request, path: str, body: bytes):
        """Forward non-API requests without recording."""
        target = _resolve_upstream(path)
        if request.url.query:
            target += f"?{request.url.query}"
        fwd_headers = dict(request.headers)
        fwd_headers.pop("host", None)
        fwd_headers.pop("content-length", None)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=fwd_headers,
                content=body,
            )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
            },
        )

    return app


def run_recorder(
    upstream_url: str,
    model: str,
    api_key: str = "",
    api_key_header: str = "Authorization",
    port: int = 19000,
    output_file: str = "recording.jsonl",
    upstream_api: str | None = None,
) -> None:
    """Start the recording proxy server."""
    if not HAS_FASTAPI:
        raise ImportError("Install proxy deps: pip install agentic-swarm-bench[proxy]")

    detected_api = _detect_upstream_api(upstream_url, upstream_api)

    app = create_recording_app(
        upstream_url=upstream_url,
        model=model,
        api_key=api_key,
        api_key_header=api_key_header,
        output_file=output_file,
        upstream_api=upstream_api,
    )
    print(f"\nagentic-swarm-bench recorder on :{port} -> {upstream_url}")
    print(f"  Model: {model}")
    print(f"  Upstream API: {detected_api}")
    print(f"  Output: {output_file}")
    print(f"  Status: http://localhost:{port}/recording/status")
    print()
    print("  Point your agent (Claude Code, Cursor, etc.) at:")
    print(f"    http://localhost:{port}")
    if detected_api == "anthropic":
        print()
        print("  Anthropic passthrough mode - requests forwarded natively.")
        print("  Launch Claude Code with:")
        print(f"    ANTHROPIC_BASE_URL=http://localhost:{port} claude")
    print()
    print("  Press Ctrl+C to stop recording.\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
