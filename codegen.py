"""
Function-only judging: signature validation, input encoding and harness generation.

The backend stores a language-neutral ``function_signature`` on each problem:

    {
      "name": "twoSum",
      "params": [
        {"name": "nums", "type": "int[]"},
        {"name": "target", "type": "int"}
      ],
      "return_type": "int[]"
    }

For function-mode problems the trusted judge:

1. Parses the test case's JSON object (``{"nums": [2, 7], "target": 9}``).
2. Re-encodes every value as a simple line-based token stream so C++/Java can
   read it without a JSON library.
3. Appends a language-specific harness to the user's function/class. The
   harness calls the function and prints the canonical result on a line tagged
   with :data:`RESULT_MARKER`; the runner splits that line out before comparing.

Users therefore write only ``class Solution`` / the bare function; they never
write stdin parsing, stdout printing or a ``main``. Any debug output they print
is ignored, and strings are compared exactly (except when the problem selects
the unordered or float compare mode).

Keep the type table in sync with ``backend/apps/problems/codegen.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any

RESULT_MARKER = "__CQ_RESULT__"
LANGUAGES = ("python", "cpp", "java", "javascript")
BASE_TYPES = ("int", "long", "double", "bool", "string")
MAX_DIMENSIONS = 2
NESTED_BASES = ("int", "string")
MAX_PARAMS = 10

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SignatureError(ValueError):
    """The problem's function signature is missing or unsupported."""


# ─── Signature validation ─────────────────────────────────────────────────────

def parse_type(type_name: Any) -> tuple[str, int]:
    if not isinstance(type_name, str) or not type_name:
        raise SignatureError("Each parameter and the return type need a type.")
    base = type_name
    dimensions = 0
    while base.endswith("[]"):
        base = base[:-2]
        dimensions += 1
    if base not in BASE_TYPES:
        raise SignatureError(f"Unknown type {type_name!r}.")
    if dimensions > MAX_DIMENSIONS:
        raise SignatureError(f"Type {type_name!r} has too many dimensions.")
    if dimensions == MAX_DIMENSIONS and base not in NESTED_BASES:
        raise SignatureError(f"Nested arrays are limited to {', '.join(NESTED_BASES)}.")
    return base, dimensions


def validate_signature(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise SignatureError("The function signature must be a JSON object.")
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise SignatureError("The function name must be a valid identifier.")
    params = raw.get("params")
    if not isinstance(params, list):
        raise SignatureError("The function signature needs a 'params' list.")
    if len(params) > MAX_PARAMS:
        raise SignatureError(f"At most {MAX_PARAMS} parameters are supported.")

    seen: set[str] = set()
    normalized = []
    for index, param in enumerate(params):
        if not isinstance(param, dict):
            raise SignatureError(f"Parameter {index + 1} must be an object.")
        param_name = param.get("name")
        if not isinstance(param_name, str) or not _NAME_RE.match(param_name):
            raise SignatureError(f"Parameter {index + 1} needs a valid identifier name.")
        if param_name in seen:
            raise SignatureError(f"Duplicate parameter name {param_name!r}.")
        seen.add(param_name)
        base, dimensions = parse_type(param.get("type"))
        normalized.append({"name": param_name, "type": base + "[]" * dimensions})

    base, dimensions = parse_type(raw.get("return_type"))
    return {
        "name": name,
        "params": normalized,
        "return_type": base + "[]" * dimensions,
    }


# ─── Input encoding ───────────────────────────────────────────────────────────

def _encode_value(type_name: str, value: Any) -> list[str]:
    base, dimensions = parse_type(type_name)
    if dimensions:
        if not isinstance(value, list):
            raise ValueError(f"Expected an array for {type_name!r}.")
        inner = base + "[]" * (dimensions - 1)
        tokens = [str(len(value))]
        for item in value:
            tokens.extend(_encode_value(inner, item))
        return tokens

    if base == "bool":
        if not isinstance(value, bool):
            raise ValueError("Expected true/false for a bool parameter.")
        return ["true" if value else "false"]
    if base in ("int", "long"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Expected an integer for {base}.")
        return [str(value)]
    if base == "double":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Expected a number for double.")
        return [repr(float(value))]
    if not isinstance(value, str):
        raise ValueError("Expected a string parameter.")
    if "\n" in value or "\r" in value:
        raise ValueError("Test-case strings may not contain newlines.")
    return [value]


def encode_input(signature: dict, input_data: str) -> str:
    """Turn ``{"nums": [2, 7], "target": 9}`` into the harness token stream."""
    sig = validate_signature(signature)
    try:
        payload = json.loads(input_data)
    except (TypeError, ValueError) as exc:
        raise ValueError("Function-mode test-case input must be a JSON object.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Function-mode test-case input must be a JSON object.")

    tokens: list[str] = []
    for param in sig["params"]:
        if param["name"] not in payload:
            raise ValueError(f"Test-case input is missing parameter {param['name']!r}.")
        tokens.extend(_encode_value(param["type"], payload[param["name"]]))
    return "\n".join(tokens) + ("\n" if tokens else "")


def split_result(stdout: str) -> tuple[str, str | None]:
    """Split the harness result line from the program's own stdout."""
    marker_at = -1
    search_from = 0
    while True:
        index = stdout.find(RESULT_MARKER, search_from)
        if index < 0:
            break
        if index == 0 or stdout[index - 1] == "\n":
            marker_at = index
        search_from = index + 1
    if marker_at < 0:
        return stdout, None

    line_end = stdout.find("\n", marker_at)
    if line_end < 0:
        line_end = len(stdout)
    value = stdout[marker_at + len(RESULT_MARKER):line_end]
    if value.startswith(" "):
        value = value[1:]
    if value.endswith("\r"):
        value = value[:-1]

    clean = stdout[:marker_at]
    if clean.endswith("\n"):
        clean = clean[:-1]
    return clean, value


# ─── Harness generation ───────────────────────────────────────────────────────

def wrap(language: str, signature: dict, user_code: str) -> str:
    """Return the compilable source: user code + generated harness."""
    sig = validate_signature(signature)
    if language == "python":
        return _python(sig, user_code)
    if language == "cpp":
        return _cpp(sig, user_code)
    if language == "java":
        return _java(sig, user_code)
    if language == "javascript":
        return _javascript(sig, user_code)
    raise SignatureError(f"Unsupported language {language!r}.")


def _python(sig: dict, user_code: str) -> str:
    reads = "\n".join(
        f"    _cq_args.append(_cq_read({param['type']!r}))" for param in sig["params"]
    )
    return f"""{user_code.rstrip()}

# ─── CodeQuest harness (generated, do not edit) ───
import json as _cq_json

_cq_tokens = None


def _cq_line():
    global _cq_tokens
    if _cq_tokens is None:
        _cq_tokens = iter(__import__("sys").stdin.read().split("\\n"))
    return next(_cq_tokens).rstrip("\\r")


def _cq_read(type_name):
    if type_name.endswith("[]"):
        inner = type_name[:-2]
        return [_cq_read(inner) for _ in range(int(_cq_line()))]
    if type_name in ("int", "long"):
        return int(_cq_line())
    if type_name == "double":
        return float(_cq_line())
    if type_name == "bool":
        return _cq_line() == "true"
    return _cq_line()


def _cq_write(type_name, value):
    if value is None:
        return "null"
    if type_name.endswith("[]"):
        inner = type_name[:-2]
        if inner == "string":
            return "[" + ",".join(_cq_json.dumps(str(item)) for item in value) + "]"
        return "[" + ",".join(_cq_write(inner, item) for item in value) + "]"
    if type_name == "string":
        return value
    if type_name == "bool":
        return "true" if value else "false"
    if type_name == "double":
        return repr(float(value))
    return str(int(value))


def _cq_main():
    _cq_args = []
{reads}
    _cq_fn = None
    if "Solution" in globals():
        _cq_fn = getattr(Solution(), {sig['name']!r}, None)
    if _cq_fn is None:
        _cq_fn = globals().get({sig['name']!r})
    if _cq_fn is None:
        raise RuntimeError("Define class Solution.{sig['name']} or function {sig['name']}.")
    _cq_result = _cq_fn(*_cq_args)
    print("{RESULT_MARKER} " + _cq_write({sig['return_type']!r}, _cq_result), flush=True)


_cq_main()
"""


def _cpp_type(base: str, dimensions: int) -> str:
    scalar = {"int": "int", "long": "long long", "double": "double", "bool": "bool", "string": "string"}[base]
    for _ in range(dimensions):
        scalar = f"vector<{scalar}>"
    return scalar


def _cpp(sig: dict, user_code: str) -> str:
    arg_decls = "\n".join(
        f"    {_cpp_type(*parse_type(p['type']))} _a{i} = _cq::Reader<{_cpp_type(*parse_type(p['type']))}>::read();"
        for i, p in enumerate(sig["params"])
    )
    call_args = ", ".join(f"_a{i}" for i in range(len(sig["params"])))
    return_type = _cpp_type(*parse_type(sig["return_type"]))
    return f"""#include <bits/stdc++.h>
using namespace std;

{user_code.rstrip()}

// ─── CodeQuest harness (generated, do not edit) ───
namespace _cq {{
static string line() {{
    string s;
    if (!getline(cin, s)) return "";
    if (!s.empty() && s.back() == '\\r') s.pop_back();
    return s;
}}

template <class T> struct Reader;
template <> struct Reader<int> {{ static int read() {{ return stoi(line()); }} }};
template <> struct Reader<long long> {{ static long long read() {{ return stoll(line()); }} }};
template <> struct Reader<double> {{ static double read() {{ return stod(line()); }} }};
template <> struct Reader<bool> {{ static bool read() {{ return line() == "true"; }} }};
template <> struct Reader<string> {{ static string read() {{ return line(); }} }};
template <class T> struct Reader<vector<T>> {{
    static vector<T> read() {{
        int n = stoi(line());
        vector<T> values;
        values.reserve(n);
        for (int i = 0; i < n; ++i) values.push_back(Reader<T>::read());
        return values;
    }}
}};

static string json_escape(const string& value) {{
    string out = "\\"";
    for (char c : value) {{
        switch (c) {{
            case '"': out += "\\\\\\""; break;
            case '\\\\': out += "\\\\\\\\"; break;
            case '\\n': out += "\\\\n"; break;
            case '\\r': out += "\\\\r"; break;
            case '\\t': out += "\\\\t"; break;
            default: out += c;
        }}
    }}
    return out + "\\"";
}}

template <class T> struct Writer;
template <> struct Writer<int> {{ static string write(int v) {{ return to_string(v); }} }};
template <> struct Writer<long long> {{ static string write(long long v) {{ return to_string(v); }} }};
template <> struct Writer<double> {{
    static string write(double v) {{
        ostringstream out;
        out << setprecision(15) << v;
        return out.str();
    }}
}};
template <> struct Writer<bool> {{ static string write(bool v) {{ return v ? "true" : "false"; }} }};
template <> struct Writer<string> {{ static string write(const string& v) {{ return v; }} }};
template <class T> struct Writer<vector<T>> {{
    static string write(const vector<T>& value) {{
        string out = "[";
        for (size_t i = 0; i < value.size(); ++i) {{
            if (i) out += ",";
            out += Writer<T>::write(value[i]);
        }}
        return out + "]";
    }}
}};
template <> struct Writer<vector<string>> {{
    static string write(const vector<string>& value) {{
        string out = "[";
        for (size_t i = 0; i < value.size(); ++i) {{
            if (i) out += ",";
            out += json_escape(value[i]);
        }}
        return out + "]";
    }}
}};
}}  // namespace _cq

int main() {{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
{arg_decls or "    // no parameters"}
    auto _result = Solution().{sig['name']}({call_args});
    cout << "\\n{RESULT_MARKER} " << _cq::Writer<{return_type}>::write(_result) << "\\n";
    cout.flush();
    return 0;
}}
"""


def _java_type(base: str, dimensions: int) -> str:
    scalar = {"int": "int", "long": "long", "double": "double", "bool": "boolean", "string": "String"}[base]
    return scalar + "[]" * dimensions


_JAVA_READER = {
    "int": "_cqReadInt()",
    "long": "_cqReadLong()",
    "double": "_cqReadDouble()",
    "bool": "_cqReadBool()",
    "string": "_cqReadLine()",
    "int[]": "_cqReadIntArray()",
    "long[]": "_cqReadLongArray()",
    "double[]": "_cqReadDoubleArray()",
    "bool[]": "_cqReadBoolArray()",
    "string[]": "_cqReadStringArray()",
    "int[][]": "_cqReadIntMatrix()",
    "string[][]": "_cqReadStringMatrix()",
}


def _java(sig: dict, user_code: str) -> str:
    # The editor file is Solution.java. Rename the user's class so the generated
    # public Solution class can hold main() without a second public class.
    code = re.sub(r"\bpublic\s+(?:final\s+)?class\s+Solution\b", "class UserSolution", user_code)
    code = re.sub(r"\b(?:final\s+)?class\s+Solution\b", "class UserSolution", code)
    code = code.replace("new Solution(", "new UserSolution(")

    reads = "\n".join(
        f"        {_java_type(*parse_type(p['type']))} _a{i} = {_JAVA_READER[p['type']]};"
        for i, p in enumerate(sig["params"])
    )
    call_args = ", ".join(f"_a{i}" for i in range(len(sig["params"])))
    return f"""import java.io.*;
import java.util.*;

{code.rstrip()}

// ─── CodeQuest harness (generated, do not edit) ───
public class Solution {{
    private static final BufferedReader _cqIn = new BufferedReader(new InputStreamReader(System.in));

    private static String _cqReadLine() throws IOException {{
        String line = _cqIn.readLine();
        return line == null ? "" : line;
    }}

    private static int _cqReadInt() throws IOException {{ return Integer.parseInt(_cqReadLine()); }}
    private static long _cqReadLong() throws IOException {{ return Long.parseLong(_cqReadLine()); }}
    private static double _cqReadDouble() throws IOException {{ return Double.parseDouble(_cqReadLine()); }}
    private static boolean _cqReadBool() throws IOException {{ return Boolean.parseBoolean(_cqReadLine()); }}

    private static int[] _cqReadIntArray() throws IOException {{
        int n = _cqReadInt();
        int[] values = new int[n];
        for (int i = 0; i < n; i++) values[i] = _cqReadInt();
        return values;
    }}

    private static long[] _cqReadLongArray() throws IOException {{
        int n = _cqReadInt();
        long[] values = new long[n];
        for (int i = 0; i < n; i++) values[i] = _cqReadLong();
        return values;
    }}

    private static double[] _cqReadDoubleArray() throws IOException {{
        int n = _cqReadInt();
        double[] values = new double[n];
        for (int i = 0; i < n; i++) values[i] = _cqReadDouble();
        return values;
    }}

    private static boolean[] _cqReadBoolArray() throws IOException {{
        int n = _cqReadInt();
        boolean[] values = new boolean[n];
        for (int i = 0; i < n; i++) values[i] = _cqReadBool();
        return values;
    }}

    private static String[] _cqReadStringArray() throws IOException {{
        int n = _cqReadInt();
        String[] values = new String[n];
        for (int i = 0; i < n; i++) values[i] = _cqReadLine();
        return values;
    }}

    private static int[][] _cqReadIntMatrix() throws IOException {{
        int n = _cqReadInt();
        int[][] values = new int[n][];
        for (int i = 0; i < n; i++) values[i] = _cqReadIntArray();
        return values;
    }}

    private static String[][] _cqReadStringMatrix() throws IOException {{
        int n = _cqReadInt();
        String[][] values = new String[n][];
        for (int i = 0; i < n; i++) values[i] = _cqReadStringArray();
        return values;
    }}

    private static String _cqQuote(String value) {{
        StringBuilder out = new StringBuilder("\\"");
        for (int i = 0; i < value.length(); i++) {{
            char c = value.charAt(i);
            switch (c) {{
                case '"': out.append("\\\\\\""); break;
                case '\\\\': out.append("\\\\\\\\"); break;
                case '\\n': out.append("\\\\n"); break;
                case '\\r': out.append("\\\\r"); break;
                case '\\t': out.append("\\\\t"); break;
                default:
                    if (c < 0x20) out.append(String.format("\\\\u%04x", (int) c));
                    else out.append(c);
            }}
        }}
        return out.append('"').toString();
    }}

    private static String _cqWrite(int v) {{ return Integer.toString(v); }}
    private static String _cqWrite(long v) {{ return Long.toString(v); }}
    private static String _cqWrite(double v) {{ return Double.toString(v); }}
    private static String _cqWrite(boolean v) {{ return v ? "true" : "false"; }}
    private static String _cqWrite(String v) {{ return v == null ? "null" : v; }}

    private static String _cqWrite(int[] values) {{
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {{
            if (i > 0) out.append(',');
            out.append(_cqWrite(values[i]));
        }}
        return out.append(']').toString();
    }}

    private static String _cqWrite(long[] values) {{
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {{
            if (i > 0) out.append(',');
            out.append(_cqWrite(values[i]));
        }}
        return out.append(']').toString();
    }}

    private static String _cqWrite(double[] values) {{
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {{
            if (i > 0) out.append(',');
            out.append(_cqWrite(values[i]));
        }}
        return out.append(']').toString();
    }}

    private static String _cqWrite(boolean[] values) {{
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {{
            if (i > 0) out.append(',');
            out.append(_cqWrite(values[i]));
        }}
        return out.append(']').toString();
    }}

    private static String _cqWrite(String[] values) {{
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {{
            if (i > 0) out.append(',');
            out.append(_cqQuote(values[i]));
        }}
        return out.append(']').toString();
    }}

    private static String _cqWrite(int[][] values) {{
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {{
            if (i > 0) out.append(',');
            out.append(_cqWrite(values[i]));
        }}
        return out.append(']').toString();
    }}

    private static String _cqWrite(String[][] values) {{
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {{
            if (i > 0) out.append(',');
            out.append(_cqWrite(values[i]));
        }}
        return out.append(']').toString();
    }}

    public static void main(String[] args) throws Exception {{
{reads or "        // no parameters"}
        {_java_type(*parse_type(sig['return_type']))} _result = new UserSolution().{sig['name']}({call_args});
        System.out.println();
        System.out.println("{RESULT_MARKER} " + _cqWrite(_result));
    }}
}}
"""


def _javascript(sig: dict, user_code: str) -> str:
    reads = "\n".join(
        f"const _cqA{i} = _cqRead({param['type']!r});" for i, param in enumerate(sig["params"])
    )
    call_args = ", ".join(f"_cqA{i}" for i in range(len(sig["params"])))
    return f"""{user_code.rstrip()}

// ─── CodeQuest harness (generated, do not edit) ───
const _cqFs = require('fs');
const _cqTokens = _cqFs.readFileSync(0, 'utf8').split('\\n');
let _cqIndex = 0;

function _cqLine() {{
  let line = _cqTokens[_cqIndex++];
  if (line === undefined) return '';
  if (line.endsWith('\\r')) line = line.slice(0, -1);
  return line;
}}

function _cqRead(typeName) {{
  if (typeName.endsWith('[]')) {{
    const inner = typeName.slice(0, -2);
    const count = parseInt(_cqLine(), 10);
    const values = [];
    for (let i = 0; i < count; i++) values.push(_cqRead(inner));
    return values;
  }}
  if (typeName === 'int' || typeName === 'long') return parseInt(_cqLine(), 10);
  if (typeName === 'double') return parseFloat(_cqLine());
  if (typeName === 'bool') return _cqLine() === 'true';
  return _cqLine();
}}

function _cqWrite(typeName, value) {{
  if (value === null || value === undefined) return 'null';
  if (typeName.endsWith('[]')) {{
    const inner = typeName.slice(0, -2);
    return '[' + value.map((item) =>
      inner === 'string' ? JSON.stringify(String(item)) : _cqWrite(inner, item)
    ).join(',') + ']';
  }}
  if (typeName === 'string') return String(value);
  if (typeName === 'bool') return value ? 'true' : 'false';
  return String(value);
}}

{reads or "// no parameters"}
let _cqFn = null;
if (typeof Solution === 'function') {{
  try {{ _cqFn = new Solution()[{sig['name']!r}]; }} catch (_cqError) {{ _cqFn = null; }}
}}
if (typeof _cqFn !== 'function' && typeof {sig['name']} === 'function') {{
  _cqFn = {sig['name']};
}}
if (typeof _cqFn !== 'function') {{
  throw new Error("Define class Solution.{sig['name']} or function {sig['name']}.");
}}
const _cqResult = _cqFn({call_args});
process.stdout.write('\\n{RESULT_MARKER} ' + _cqWrite({sig['return_type']!r}, _cqResult) + '\\n');
"""
