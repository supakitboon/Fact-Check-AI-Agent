"""
Sandboxed Python execution for claim verification.
Runs LLM-generated code in a subprocess with a timeout so a buggy or
infinite-loop script cannot hang the main process.
"""

import os
import subprocess
import sys
import tempfile

TIMEOUT_SECONDS = 10
_VALID_VERDICTS = {"supported", "contradicted", "insufficient_evidence", "partially_supported"}


def execute_verification_code(code: str) -> tuple[str, str]:
    """
    Execute verification code in a subprocess.
    Returns (verdict, raw_output).
    verdict is one of the _VALID_VERDICTS strings, or "" if execution failed.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        output = proc.stdout.strip()
        for line in reversed(output.splitlines()):
            if line.strip().lower() in _VALID_VERDICTS:
                return line.strip().lower(), output
        stderr = proc.stderr.strip()
        return "", (output or stderr)[:300]
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except Exception as exc:
        return "", str(exc)[:300]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
