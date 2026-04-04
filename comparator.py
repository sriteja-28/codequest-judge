"""
comparator.py — Compare actual vs expected output.

Normalisation rules (matching most online judges):
- Strip leading/trailing whitespace from entire output
- Normalise line endings (CRLF → LF)
- Strip trailing whitespace from each line
- Ignore trailing newlines
"""

import re


def normalize(text: str) -> str:
    """Normalize output for comparison."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    # Remove trailing empty lines
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


# def compare(actual: str, expected: str) -> bool:
#     """
#     Return True if actual output matches expected output after normalization.
#     """
#     return normalize(actual) == normalize(expected)


def compare(actual: str, expected: str) -> bool:
    import json

    a = normalize(actual)
    e = normalize(expected)

    # trying json comparison
    try:
        a_json = json.loads(a)
        e_json = json.loads(e)

        # handling list comparison
        if isinstance(a_json, list) and isinstance(e_json, list):
            return a_json == e_json  # or sorted(...) if order doesn't matter

        return a_json == e_json

    except Exception:
        pass

    # fallback to string compare
    return a == e


def compare_float(actual: str, expected: str, tolerance: float = 1e-5) -> bool:
    """
    For problems where float comparison is needed.
    Tries exact match first, then tolerance-based comparison.
    """
    if compare(actual, expected):
        return True
    try:
        a_val = float(normalize(actual))
        e_val = float(normalize(expected))
        return abs(a_val - e_val) <= tolerance
    except ValueError:
        return False