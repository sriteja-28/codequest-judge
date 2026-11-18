"""Function-mode signature, input encoding and harness generation tests."""

import subprocess
import sys

import pytest

from codegen import (
    RESULT_MARKER,
    SignatureError,
    encode_input,
    split_result,
    validate_signature,
    wrap,
)

SIGNATURE = {
    "name": "twoSum",
    "params": [
        {"name": "nums", "type": "int[]"},
        {"name": "target", "type": "int"},
    ],
    "return_type": "int[]",
}

MIXED = {
    "name": "mix",
    "params": [
        {"name": "n", "type": "int"},
        {"name": "big", "type": "long"},
        {"name": "ratio", "type": "double"},
        {"name": "flag", "type": "bool"},
        {"name": "name", "type": "string"},
        {"name": "words", "type": "string[]"},
        {"name": "grid", "type": "int[][]"},
    ],
    "return_type": "string[]",
}


def test_validate_signature_normalises_types():
    sig = validate_signature({
        "name": "f",
        "params": [{"name": "x", "type": "int"}],
        "return_type": "string[]",
    })
    assert sig == {"name": "f", "params": [{"name": "x", "type": "int"}], "return_type": "string[]"}


@pytest.mark.parametrize("bad", [
    None,
    {"name": "1bad", "params": [], "return_type": "int"},
    {"name": "f", "params": "x", "return_type": "int"},
    {"name": "f", "params": [{"name": "x", "type": "float"}], "return_type": "int"},
    {"name": "f", "params": [{"name": "x", "type": "long[][]"}], "return_type": "int"},
    {"name": "f", "params": [{"name": "x", "type": "int"}, {"name": "x", "type": "int"}], "return_type": "int"},
    {"name": "f", "params": [], "return_type": "void"},
])
def test_validate_signature_rejects_unsupported_shapes(bad):
    with pytest.raises(SignatureError):
        validate_signature(bad)


def test_encode_input_uses_one_token_per_value():
    assert encode_input(SIGNATURE, '{"nums": [2, 7, 11, 15], "target": 9}') == "4\n2\n7\n11\n15\n9\n"
    assert encode_input(
        MIXED,
        '{"n": 3, "big": 9007199254740993, "ratio": 0.5, "flag": true, "name": "Ada Lovelace", '
        '"words": ["a", "b c"], "grid": [[1, 2], [3]]}',
    ) == "3\n9007199254740993\n0.5\ntrue\nAda Lovelace\n2\na\nb c\n2\n2\n1\n2\n1\n3\n"


def test_encode_input_rejects_bad_payloads():
    with pytest.raises(ValueError):
        encode_input(SIGNATURE, "not json")
    with pytest.raises(ValueError):
        encode_input(SIGNATURE, '{"nums": [1, 2]}')
    with pytest.raises(ValueError):
        encode_input(SIGNATURE, '{"nums": [1, true], "target": 3}')
    with pytest.raises(ValueError):
        encode_input(MIXED, '{"n": 1, "big": 1, "ratio": 1, "flag": true, "name": "a\\nb", '
                            '"words": [], "grid": []}')


def test_split_result_uses_last_marker_line():
    stdout = "debug\n__CQ_RESULT__ fake\nmore debug\n__CQ_RESULT__ [0,1]\n"
    assert split_result(stdout) == ("debug\n__CQ_RESULT__ fake\nmore debug", "[0,1]")
    assert split_result("no result") == ("no result", None)


def test_split_result_keeps_string_spaces():
    assert split_result(f"{RESULT_MARKER}  two  spaces ") == ("", " two  spaces ")


@pytest.mark.parametrize("language,call", [
    ("python", "getattr(Solution(), 'twoSum'"),
    ("cpp", "Solution().twoSum(_a0, _a1)"),
    ("java", "new UserSolution().twoSum(_a0, _a1)"),
    ("javascript", "_cqFn(_cqA0, _cqA1)"),
])
def test_wrap_adds_harness_for_each_language(language, call):
    user_code = {
        "python": "class Solution:\\n    def twoSum(self, nums, target):\\n        return [0, 1]\\n",
        "cpp": "class Solution { public: vector<int> twoSum(vector<int>& n, int t) { return {0,1}; } };\\n",
        "java": "class Solution { public int[] twoSum(int[] n, int t) { return new int[]{0,1}; } }\\n",
        "javascript": "var twoSum = function(nums, target) { return [0,1]; };\\n",
    }[language]
    source = wrap(language, SIGNATURE, user_code)
    assert call in source
    assert RESULT_MARKER in source
    assert "stdin" not in user_code


def test_java_harness_renames_user_class_and_keeps_public_harness():
    source = wrap("java", SIGNATURE, "public class Solution { public int[] twoSum(int[] n, int t) { return new int[]{}; } }")
    assert "public class Solution" in source
    assert "class UserSolution" in source
    assert source.count("public class") == 1


def test_wrap_rejects_unknown_language():
    with pytest.raises(SignatureError):
        wrap("rust", SIGNATURE, "fn main() {}")


def test_python_harness_executes_with_all_supported_types():
    # Regression: the generated _cq_main() used _cq_args without initialising it.
    signature = {
        "name": "mix",
        "params": [
            {"name": "n", "type": "int"},
            {"name": "ratio", "type": "double"},
            {"name": "flag", "type": "bool"},
            {"name": "name", "type": "string"},
            {"name": "words", "type": "string[]"},
            {"name": "grid", "type": "int[][]"},
        ],
        "return_type": "string",
    }
    user_code = """class Solution:
    def mix(self, n, ratio, flag, name, words, grid):
        print("debug output")
        return f"{name}|{n}|{ratio:.1f}|{str(flag).lower()}|{'-'.join(words)}|{sum(map(sum, grid))}"
"""
    source = wrap("python", signature, user_code)
    stdin_data = encode_input(
        signature,
        '{"n": 3, "ratio": 0.5, "flag": true, "name": "Ada Lovelace", '
        '"words": ["a", "b c"], "grid": [[1, 2], [3]]}',
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    clean, answer = split_result(completed.stdout)
    assert clean == "debug output"
    assert answer == "Ada Lovelace|3|0.5|true|a-b c|6"
