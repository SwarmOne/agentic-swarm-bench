"""Tests for eval_runner: code extraction and validation."""

from __future__ import annotations

from agentic_swarm_bench.runner.eval_runner import (
    _extract_code,
    validate_execution,
    validate_syntax,
)

# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_fenced_python_block(self):
        text = "Here is the solution:\n```python\nprint('hello')\n```"
        assert _extract_code(text) == "print('hello')"

    def test_fenced_py_shorthand(self):
        text = "```py\nx = 1 + 1\n```"
        assert _extract_code(text) == "x = 1 + 1"

    def test_fenced_no_lang_tag(self):
        text = "```\ndef foo():\n    return 42\n```"
        assert _extract_code(text) == "def foo():\n    return 42"

    def test_indented_block(self):
        text = "Sure, here:\n    x = 1\n    y = 2\nThat's it."
        result = _extract_code(text)
        assert "x = 1" in result
        assert "y = 2" in result

    def test_prose_prefix_stripped_def(self):
        # One-liner so the indented-body branch doesn't fire before prose-prefix
        text = "Let me write a function:\ndef add(a, b): return a + b"
        result = _extract_code(text)
        assert result.startswith("def add")

    def test_prose_prefix_stripped_import(self):
        text = "Here you go:\nimport os\nprint(os.getcwd())"
        result = _extract_code(text)
        assert result.startswith("import os")

    def test_prose_prefix_stripped_assignment(self):
        text = "The answer is:\nresult = 42\nprint(result)"
        result = _extract_code(text)
        assert "result = 42" in result

    def test_fallback_plain_text(self):
        text = "just some text with no code patterns"
        result = _extract_code(text)
        assert result == text.strip()

    def test_empty_string(self):
        assert _extract_code("") == ""

    def test_fenced_takes_first_block(self):
        text = "```python\nblock_one()\n```\n\n```python\nblock_two()\n```"
        assert _extract_code(text) == "block_one()"


# ---------------------------------------------------------------------------
# validate_syntax
# ---------------------------------------------------------------------------


class TestValidateSyntax:
    def test_valid_code_passes(self):
        ok, msg = validate_syntax("x = 1 + 1")
        assert ok is True
        assert msg == "OK"

    def test_valid_function_passes(self):
        ok, msg = validate_syntax("def foo():\n    return 42")
        assert ok is True

    def test_syntax_error_caught(self):
        ok, msg = validate_syntax("def broken(:\n    pass")
        assert ok is False
        assert "SyntaxError" in msg

    def test_empty_code_passes(self):
        ok, msg = validate_syntax("")
        assert ok is True

    def test_bad_indent_caught(self):
        ok, msg = validate_syntax("if True:\nprint('bad')")
        assert ok is False


# ---------------------------------------------------------------------------
# validate_execution
# ---------------------------------------------------------------------------


class TestValidateExecution:
    def test_clean_code_passes(self):
        ok, msg = validate_execution("x = 1 + 1")
        assert ok is True
        assert msg == "OK"

    def test_runtime_error_caught(self):
        ok, msg = validate_execution("raise ValueError('boom')")
        assert ok is False
        assert "Exit" in msg or "ValueError" in msg.lower() or "1" in msg

    def test_timeout_caught(self):
        ok, msg = validate_execution("while True: pass", timeout=0.5)
        assert ok is False
        assert "Timed out" in msg

    def test_import_error_caught(self):
        ok, msg = validate_execution("import nonexistent_module_xyz")
        assert ok is False
