"""Generate realistic agentic assistant context for benchmark requests.

Produces content that looks like real Claude Code / Cursor / Copilot sessions:
- System prompt with tool definitions
- Prior conversation turns with code files, tool calls, and results
"""

from __future__ import annotations

import random

SYSTEM_PROMPT = (
    "You are an expert software engineer assistant integrated into a code editor. "
    "You have access to the user's full project codebase. Answer concisely with "
    "working code. When modifying existing code, show only the changed functions. "
    "Use type hints, follow existing code style, and handle edge cases."
)

TOOL_DEFINITIONS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Glob",
    "Grep",
    "ListDir",
    "SearchReplace",
    "InsertCodeBlock",
    "CreateFile",
    "DeleteFile",
]

FILE_NAMES = [
    "src/auth/middleware.py",
    "src/api/routes.py",
    "src/models/user.py",
    "src/services/payment.py",
    "src/utils/crypto.py",
    "tests/test_auth.py",
    "src/config/settings.py",
    "src/db/migrations/001_init.py",
    "frontend/src/App.tsx",
    "frontend/src/hooks/useAuth.ts",
    "src/services/order_service.py",
    "src/services/inventory.py",
    "src/api/middleware/rate_limit.py",
    "src/workers/email_worker.py",
    "tests/test_orders.py",
    "src/models/order.py",
]

KEYWORDS = [
    "def ",
    "class ",
    "if ",
    "for ",
    "return ",
    "import ",
    "from ",
    "async def ",
    "await ",
    "try:",
    "except ",
    "raise ",
    "with ",
    "self.",
    "result = ",
    "data = ",
    "response = ",
    "logger.",
    "assert ",
    "yield ",
    "lambda ",
    "elif ",
    "else:",
]

_CACHE: dict[int, str] = {}


def _build_tool_block() -> str:
    """Generate tool definition XML that mimics Claude Code's system prompt."""
    chunks = []
    for tool in TOOL_DEFINITIONS:
        chunks.append(
            f'<tool name="{tool}">\n'
            f"  <description>Tool for performing {tool.lower()} operations on the codebase. "
            f"This tool accepts parameters for path, content, regex patterns, and flags. "
            f"Use when you need to {tool.lower()} files in the project workspace. "
            f"Always verify paths exist before operating.</description>\n"
            f"  <parameters>\n"
            f'    <param name="path" type="string" required="true">Absolute file path</param>\n'
            f'    <param name="content" type="string" required="false">Content to write</param>\n'
            f'    <param name="regex" type="string" required="false">Pattern to match</param>\n'
            f'    <param name="recursive" type="boolean" required="false">Recurse subdirs</param>\n'
            f"  </parameters>\n"
            f"</tool>"
        )
    return "\n".join(chunks)


def _build_conversation_turn(rng: random.Random, index: int) -> str:
    """Generate one realistic conversation turn (user question + assistant response)."""
    fname = FILE_NAMES[index % len(FILE_NAMES)]
    n_lines = rng.randint(40, 120)

    code_lines = []
    for ln in range(1, n_lines + 1):
        indent = "    " * rng.randint(0, 3)
        kw = rng.choice(KEYWORDS)
        var = f"var_{rng.randint(1, 999)}"
        code_lines.append(f"{indent}{kw}{var}_function(param_{ln}, config=True)")

    code_block = "\n".join(code_lines)
    error_line = rng.randint(10, n_lines)
    commit_hash = rng.randint(1000, 9999)

    return (
        f'<user_turn index="{index}">\n'
        f"I need you to review and fix the bug in `{fname}`. Here is the current file:\n\n"
        f"```python\n"
        f"# {fname}\n"
        f"# Auto-generated module for {fname.split('/')[-1].replace('.py', '')} functionality\n"
        f"# Last modified: 2026-03-{rng.randint(1, 28):02d}\n\n"
        f"{code_block}\n"
        f"```\n\n"
        f"The error trace shows:\n"
        f'  File "{fname}", line {error_line}, in handle_request\n'
        f"    result = process_data(validated_input, context=request.context)\n"
        f"  TypeError: unexpected keyword argument 'context'\n\n"
        f"Please analyze the code and fix the issue.\n"
        f"</user_turn>\n\n"
        f'<assistant_turn index="{index}">\n'
        f"I can see the issue. The `process_data` function in `{fname}` was updated to accept\n"
        f"a `context` parameter, but the signature on line "
        f"{rng.randint(5, 20)} still uses\n"
        f"the old parameter names. Let me trace through the call chain:\n\n"
        f"1. `handle_request` calls `validate_input` which returns a dict\n"
        f"2. The validated dict is passed to `process_data` with the request context\n"
        f"3. But `process_data` expects `ctx` not `context` "
        f"(renamed in commit abc{commit_hash})\n\n"
        f"Here is the fix:\n"
        f"```python\n"
        f"{chr(10).join(code_lines[:5])}\n"
        f"```\n"
        f"</assistant_turn>"
    )


def build_context_block(target_chars: int) -> str:
    """Build a block of realistic agentic assistant context.

    Uses a cache keyed by target size to avoid regenerating identical blocks.
    """
    if target_chars in _CACHE and len(_CACHE[target_chars]) >= target_chars:
        return _CACHE[target_chars][:target_chars]

    rng = random.Random(42)
    chunks = [_build_tool_block()]

    for i in range(60):
        chunks.append(_build_conversation_turn(rng, i))

    combined = "\n\n".join(chunks)

    while len(combined) < target_chars:
        combined = combined + "\n" + combined

    result = combined[:target_chars]
    _CACHE[target_chars] = result
    return result


def build_messages(
    task_prompt: str,
    target_tokens: int,
    random_seed: float | int | None = None,
) -> list[dict]:
    """Build a complete message list for one benchmark request.

    Structure:
      [system] system prompt
      [user]   context padding + task prompt

    The context padding simulates a real coding session. The task prompt
    is appended at the end so it's the last thing the model sees.

    If random_seed is set, each request gets unique context conversation
    turns (not just unique salt). This tests how well the endpoint handles
    diverse prefill patterns.
    """
    task_chars = len(task_prompt)
    system_chars = len(SYSTEM_PROMPT)
    current_tokens_approx = (task_chars + system_chars) // 4

    if current_tokens_approx >= target_tokens:
        padding = ""
    else:
        needed_chars = (target_tokens - current_tokens_approx) * 4
        if random_seed is not None:
            padding = _build_random_context_block(needed_chars, random_seed)
        else:
            padding = build_context_block(needed_chars)

    user_content = (
        f"Here is my current project codebase for context:\n\n"
        f"{padding}\n\n"
        f"---\n\n"
        f"Based on the codebase above, {task_prompt}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_random_context_block(target_chars: int, seed: float | int) -> str:
    """Build context with a unique random seed for varied prefill patterns."""
    rng = random.Random(seed)
    chunks = [_build_tool_block()]

    for i in range(60):
        chunks.append(_build_conversation_turn(rng, i))

    combined = "\n\n".join(chunks)
    while len(combined) < target_chars:
        combined = combined + "\n" + combined

    return combined[:target_chars]
