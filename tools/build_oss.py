#!/usr/bin/env python3
"""Build the public OSS distribution from the private monorepo.

Copies the entire repo into a staging directory, then:

  1. Deletes private directories and files (modules/, skill/, etc.)
  2. Strips code between ``# --- PRIVATE ---`` / ``# --- /PRIVATE ---``
     markers in .py files
  3. Strips content between ``<!-- PRIVATE -->`` / ``<!-- /PRIVATE -->``
     markers in .md files
  4. Uncomments OSS-only content between ``<!-- OSS`` / ``OSS -->`` in .md files

Everything that remains is the open-source package. No private module
names, no private function bodies, no skip markers.

Usage:
    python tools/build_oss.py [--output-dir ./oss-staging]
    python tools/build_oss.py --test  # strip + run public test suite
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STRIP_DIRS = [
    "modules",
    "skill",
]

STRIP_FILES = [
    "agentic_swarm_bench/scenarios/evaluator.py",
    "tests/test_evaluator.py",
    "tests/markers.py",
    "tools/build_oss.py",
    ".github/workflows/publish-oss.yml",
]

IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    "*.egg-info",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
]

PY_PRIVATE_START = re.compile(r"^\s*# --- PRIVATE ---\s*$")
PY_PRIVATE_END = re.compile(r"^\s*# --- /PRIVATE ---\s*$")

MD_PRIVATE_START = re.compile(r"^\s*<!-- PRIVATE -->\s*$")
MD_PRIVATE_END = re.compile(r"^\s*<!-- /PRIVATE -->\s*$")

MD_OSS_START = re.compile(r"^\s*<!-- OSS\s*$")
MD_OSS_END = re.compile(r"^\s*OSS -->\s*$")


def copy_repo(src: Path, dst: Path) -> None:
    """Copy the repo, excluding common build artifacts."""
    if dst.exists():
        shutil.rmtree(dst)

    def _ignore(directory: str, contents: list[str]) -> list[str]:
        ignored = []
        for c in contents:
            for pattern in IGNORE_PATTERNS:
                if pattern.startswith("*"):
                    if c.endswith(pattern[1:]):
                        ignored.append(c)
                        break
                elif c == pattern:
                    ignored.append(c)
                    break
        return ignored

    shutil.copytree(src, dst, ignore=_ignore)


def strip_private(staging: Path) -> list[str]:
    """Remove private files and directories from the staging area."""
    removed: list[str] = []

    for d in STRIP_DIRS:
        target = staging / d
        if target.exists():
            shutil.rmtree(target)
            removed.append(f"dir:  {d}/")

    for f in STRIP_FILES:
        target = staging / f
        if target.exists():
            target.unlink()
            removed.append(f"file: {f}")

    return removed


def strip_py_markers(staging: Path) -> int:
    """Strip lines between # --- PRIVATE --- / # --- /PRIVATE --- in .py files."""
    count = 0
    for py_file in staging.rglob("*.py"):
        lines = py_file.read_text().splitlines(keepends=True)
        out: list[str] = []
        inside = False
        changed = False

        for line in lines:
            if PY_PRIVATE_START.match(line):
                inside = True
                changed = True
                continue
            if PY_PRIVATE_END.match(line):
                inside = False
                continue
            if not inside:
                out.append(line)

        if changed:
            py_file.write_text("".join(out))
            count += 1

    return count


def strip_md_markers(staging: Path) -> int:
    """Strip PRIVATE blocks and uncomment OSS blocks in .md files."""
    count = 0
    for md_file in staging.rglob("*.md"):
        lines = md_file.read_text().splitlines(keepends=True)
        out: list[str] = []
        inside_private = False
        changed = False

        for line in lines:
            if MD_PRIVATE_START.match(line):
                inside_private = True
                changed = True
                continue
            if MD_PRIVATE_END.match(line):
                inside_private = False
                continue
            if inside_private:
                continue

            if MD_OSS_START.match(line):
                changed = True
                continue
            if MD_OSS_END.match(line):
                continue

            out.append(line)

        if changed:
            md_file.write_text("".join(out))
            count += 1

    return count


def run_tests(staging: Path) -> bool:
    """Install and run the test suite in the stripped staging area."""
    print("\n--- Installing stripped package ---")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", f"{staging}[dev]"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Install failed:")
        print(result.stderr)
        return False

    print("\n--- Running public test suite ---")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(staging / "tests"), "-x", "-q"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OSS distribution")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "oss-staging"),
        help="Staging directory for the stripped repo (default: ./oss-staging)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the public test suite after stripping",
    )
    args = parser.parse_args()

    staging = Path(args.output_dir).resolve()
    print(f"Source:  {REPO_ROOT}")
    print(f"Staging: {staging}")

    print("\n--- Copying repository ---")
    copy_repo(REPO_ROOT, staging)

    print("\n--- Stripping private files ---")
    removed = strip_private(staging)
    for r in removed:
        print(f"  removed {r}")
    if not removed:
        print("  (nothing to strip)")

    print("\n--- Stripping PRIVATE markers from .py files ---")
    py_count = strip_py_markers(staging)
    print(f"  patched {py_count} file(s)")

    print("\n--- Stripping PRIVATE / uncommenting OSS markers from .md files ---")
    md_count = strip_md_markers(staging)
    print(f"  patched {md_count} file(s)")

    print(f"\nOSS staging ready at: {staging}")

    if args.test:
        ok = run_tests(staging)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
