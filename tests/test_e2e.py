"""End-to-end test: run speed benchmark against a mock OpenAI server."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from agentic_swarm_bench.config import BenchmarkConfig
from agentic_swarm_bench.runner.direct import run_speed_benchmark

_mock_server = None
_mock_port = None


class MockOpenAIHandler(BaseHTTPRequestHandler):
    """Simulates an OpenAI-compatible streaming endpoint."""

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_len)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        tokens = ["Hello", " from", " mock", " server", "!"]
        for i, tok in enumerate(tokens):
            chunk = {
                "id": "chatcmpl-mock",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": tok},
                        "finish_reason": None,
                    }
                ],
            }
            if i == 0:
                chunk["usage"] = {"prompt_tokens": 100}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)

        final = {
            "id": "chatcmpl-mock",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


def _get_mock_server():
    """Start mock server once, reuse across tests."""
    global _mock_server, _mock_port
    if _mock_server is not None:
        return _mock_port

    server = HTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    _mock_port = server.server_address[1]
    _mock_server = server
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    return _mock_port


def test_speed_benchmark_e2e():
    """Run a full speed benchmark against a local mock server."""
    port = _get_mock_server()

    cfg = BenchmarkConfig(
        endpoint=f"http://127.0.0.1:{port}",
        model="mock-model",
        context_tokens=100,
        users=2,
        cache_mode="allwarm",
        timeout=10.0,
        max_output_tokens=64,
    )

    run = asyncio.run(run_speed_benchmark(cfg))

    assert run.model == "mock-model"
    assert len(run.scenarios) == 1

    scenario = run.scenarios[0]
    assert scenario.num_users == 2
    assert len(scenario.requests) == 2

    for req in scenario.requests:
        assert req.succeeded, f"Request failed: {req.error}"
        assert req.completion_tokens == 5
        assert req.ttft_ms > 0
        assert req.tok_per_sec > 0
        assert req.total_time_s > 0


def test_speed_benchmark_report_generation(tmp_path):
    """Run benchmark and generate a markdown report."""
    port = _get_mock_server()

    output_path = str(tmp_path / "report.md")
    cfg = BenchmarkConfig(
        endpoint=f"http://127.0.0.1:{port}",
        model="mock-model",
        context_tokens=100,
        users=1,
        cache_mode="allwarm",
        timeout=10.0,
        output=output_path,
    )

    run = asyncio.run(run_speed_benchmark(cfg))

    from agentic_swarm_bench.report.markdown import generate_report

    report = generate_report(run)
    assert "mock-model" in report
    assert "SwarmOne" in report
    assert "TTFT" in report

    json_path = str(tmp_path / "report.json")
    run.save(json_path)

    from agentic_swarm_bench.metrics.collector import BenchmarkRun

    loaded = BenchmarkRun.load(json_path)
    assert loaded.model == "mock-model"
    assert len(loaded.scenarios) == 1
    assert loaded.scenarios[0].requests[0].completion_tokens == 5
