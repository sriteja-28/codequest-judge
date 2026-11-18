"""
comparator.py — Compare actual vs expected output.

Normalisation rules (matching most online judges):
- Normalise line endings (CRLF / CR → LF)
- Strip trailing whitespace from each line
- Ignore trailing empty lines

Comparison:
1. If both normalised outputs parse as JSON, compare the values structurally:
   whitespace inside JSON is irrelevant ("[0, 1]" == "[0,1]"), numbers compare
   by value (1 == 1.0), but booleans are NOT numbers (true != 1) and container
   types must match. List order matters.
2. Otherwise compare the normalised text exactly.

Modes (stored per problem):
- EXACT: the structural/text comparison above.
- FLOAT: exact first, then a relative/absolute tolerance for a single number.
- UNORDERED: both outputs must be JSON arrays with the same elements in any
  order (nested values are compared canonically).
"""

import json
import math


def normalize(text: str) -> str:
    """Normalize output for comparison."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    # Remove trailing empty lines
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _json_equal(a, b) -> bool:
    # bool is a subclass of int in Python: keep them distinct.
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def compare(actual: str, expected: str) -> bool:
    """Return True if actual output matches expected output."""
    a = normalize(actual)
    e = normalize(expected)
    if a == e:
        return True
    try:
        a_json = json.loads(a)
        e_json = json.loads(e)
    except (ValueError, TypeError):
        return False
    return _json_equal(a_json, e_json)


def compare_float(actual: str, expected: str, tolerance: float = 1e-5) -> bool:
    """
    For problems where float comparison is needed.
    Tries exact match first, then a relative/absolute tolerance comparison.
    """
    if compare(actual, expected):
        return True
    try:
        a_val = float(normalize(actual))
        e_val = float(normalize(expected))
    except (ValueError, TypeError):
        return False
    if not (math.isfinite(a_val) and math.isfinite(e_val)):
        return False
    return abs(a_val - e_val) <= tolerance * max(1.0, abs(e_val))


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compare_unordered(actual: str, expected: str) -> bool:
    """Both sides must be JSON arrays containing the same elements in any order."""
    try:
        a_value = json.loads(normalize(actual))
        e_value = json.loads(normalize(expected))
    except (ValueError, TypeError):
        return False
    if not isinstance(a_value, list) or not isinstance(e_value, list):
        return False
    if len(a_value) != len(e_value):
        return False
    return sorted(map(_canonical, a_value)) == sorted(map(_canonical, e_value))


def compare_output(actual: str, expected: str, mode: str = "EXACT", tolerance: float = 1e-5) -> bool:
    """Apply the problem's compare mode to one test case."""
    mode = (mode or "EXACT").upper()
    if mode == "FLOAT":
        return compare_float(actual, expected, tolerance)
    if mode == "UNORDERED":
        return compare_unordered(actual, expected)
    return compare(actual, expected)
