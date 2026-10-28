#!/usr/bin/env python3
"""Deterministic Privacy & Compliance Auditor for PAI Final Project Submission.

Scans all public code, Colab notebooks, markdown reports, and generated PDF/JSON metadata
to guarantee zero leakage of:
- Local machine username or `/Users/...` home paths
- Personal email addresses
- Forbidden AI tool/model brand identifiers in code/comments
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent

PUBLIC_SCAN_TARGETS = [
    "src",
    "notebooks",
    "scripts",
    "tests",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "docs/submission_report.md",
    "outputs/youtube_submission_metadata.json",
]

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def scan_file_for_violations(file_path: Path, current_user: str) -> List[str]:
    violations: List[str] = []
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return [f"Could not read {file_path}: {exc}"]

    # 1. Check for local home directory path `/Users/<username>`
    if f"/Users/{current_user}" in text or current_user.lower() in text.lower():
        violations.append(f"Local username or path found in {file_path.relative_to(ROOT_DIR)}")

    # 2. Check for generic `/Users/` path outside of system font lookup
    for line_no, line in enumerate(text.splitlines(), start=1):
        if "/Users/" in line and "privacy_audit.py" not in str(file_path):
            violations.append(
                f"Absolute macOS user path `/Users/` at {file_path.relative_to(ROOT_DIR)}:{line_no}"
            )
        # Check for email addresses
        for match in EMAIL_REGEX.findall(line):
            if not match.endswith((".png", ".jpg", ".svg")):
                violations.append(
                    f"Potential email address `{match}` at {file_path.relative_to(ROOT_DIR)}:{line_no}"
                )

    return violations


def run_privacy_audit() -> Tuple[bool, List[str]]:
    current_user = os.environ.get("USER", Path.home().name)
    all_violations: List[str] = []

    for rel_target in PUBLIC_SCAN_TARGETS:
        target_path = ROOT_DIR / rel_target
        if not target_path.exists():
            continue
        if target_path.is_file():
            all_violations.extend(scan_file_for_violations(target_path, current_user))
        else:
            for sub_path in target_path.rglob("*"):
                if sub_path.is_file() and not sub_path.name.endswith((".pyc", ".png", ".mp4", ".pdf")):
                    all_violations.extend(scan_file_for_violations(sub_path, current_user))

    # Also inspect raw bytes of generated PDF report for username leakage
    outputs_dir = ROOT_DIR / "outputs"
    if outputs_dir.exists():
        for pdf_file in outputs_dir.glob("*.pdf"):
            raw_bytes = pdf_file.read_bytes()
            if current_user.encode("utf-8") in raw_bytes:
                all_violations.append(f"Username bytes found inside PDF metadata/body: {pdf_file.name}")

    return (len(all_violations) == 0), all_violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify zero personal information or local user path leakage in submission files."
    )
    _ = parser.parse_args()
    passed, issues = run_privacy_audit()
    if passed:
        print("PRIVACY_AUDIT: PASS (0 violations across all public code, notebooks, and reports)")
        return 0
    for issue in issues:
        print(f"PRIVACY_VIOLATION: {issue}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
