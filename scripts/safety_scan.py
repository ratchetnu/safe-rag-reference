"""Pre-publication scan of tracked files.

Fails (exit 1) if it finds:

* strings shaped like real credentials (cloud keys, API tokens, private keys);
* e-mail addresses outside reserved example domains;
* any term from an optional, never-committed denylist file (default
  ``.safety-denylist``, one term per line). Use it for names of private
  projects, employers, customers or internal hosts that must not appear in a
  public repository. Keeping the list out of the repository means the scan
  itself never discloses what it is looking for.

Usage: python scripts/safety_scan.py [--denylist PATH]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "private_key": re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "connection_string_with_password": re.compile(
        r"\b(postgres(ql)?|mysql|mongodb(\+srv)?)://[^:/\s]+:(?!postgres@)[^@\s]{6,}@(?!localhost)"
    ),
}
EMAIL_RE = re.compile(r"\b[\w.+-]+@([\w-]+\.)+[A-Za-z]{2,}\b")
ALLOWED_EMAIL_DOMAINS = ("example.com", "example.net", "example.org", "anthropic.com")


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [ROOT / line for line in out.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denylist", default=str(ROOT / ".safety-denylist"))
    args = parser.parse_args()

    deny_path = Path(args.denylist)
    deny_terms: list[str] = []
    if deny_path.is_file():
        deny_terms = [
            t.strip()
            for t in deny_path.read_text("utf-8").splitlines()
            if t.strip() and not t.startswith("#")
        ]

    findings: list[str] = []
    for path in tracked_files():
        if not path.is_file() or path.resolve() == deny_path.resolve():
            continue
        try:
            text = path.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(ROOT)
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{rel}:{lineno}: possible {name}")
            for match in EMAIL_RE.finditer(line):
                domain = match.group(0).rsplit("@", 1)[1].lower()
                if not domain.endswith(ALLOWED_EMAIL_DOMAINS):
                    findings.append(f"{rel}:{lineno}: e-mail outside example domains")
            lowered = line.lower()
            for i, term in enumerate(deny_terms):
                if term.lower() in lowered:
                    # Report the term's index, not the term, so CI logs stay clean.
                    findings.append(f"{rel}:{lineno}: denylisted term #{i + 1}")

    scanned = "with" if deny_terms else "without"
    if findings:
        print("\n".join(findings))
        print(f"safety scan FAILED ({len(findings)} finding(s), {scanned} denylist)")
        return 1
    print(f"safety scan passed ({scanned} denylist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
