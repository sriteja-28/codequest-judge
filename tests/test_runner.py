import pytest

import runner
from sandbox import Prepared, SandboxResult, LANGUAGE_CONFIG

TESTS = [
    {"id": 11, "input_data": "1 2", "expected_output": "3", "is_hidden": False, "is_sample": True},
    {"id": 12, "input_data": "5 5", "expected_output": "10", "is_hidden": True, "is_sample": False},
    {"id": 13, "input_data": "7 1", "expected_output": "8", "is_hidden": True, "is_sample": False},
]

FUNCTION_TESTS = [
    {"id": 21, "input_data": '{"a": 1, "b": 2}', "expected_output": "3", "is_hidden": False, "is_sample": True},
    {"id": 22, "input_data": '{"a": 10, "b": 20}', "expected_output": "30", "is_hidden": True, "is_sample": False},
]

FUNCTION_SIGNATURE = {
    "name": "add",
    "params": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
    "return_type": "int",
}


def ok(stdout, time_ms=5, memory_kb=1000):
    return SandboxResult(stdout, "", 0, time_ms, memory_kb, False)


class FakeSandbox:
    cpu_time_limit, memory_limit_mb = 2, 256

    def __init__(self, outputs, compile_result=None):
        self.outputs = list(outputs)
        self.compile_result = compile_result
        self.executed, self.codes, self.cleaned = [], [], False

    def prepare(self, language, code):
        self.codes.append(code)
        return Prepared(language, LANGUAGE_CONFIG[language], "/tmp/x", "/tmp/x/code", "/tmp/x/build", self.compile_result)

    def execute(self, prepared, stdin):
        self.executed.append(stdin)
        return self.outputs.pop(0)

    def cleanup(self, prepared):
        self.cleaned = True


def problem_payload(test_cases=None, signature=None, compare_mode="EXACT"):
    return {
        "problem_id": 1,
        "function_signature": signature,
        "compare_mode": compare_mode,
        "float_tolerance": 1e-5,
        "test_cases": list(test_cases or TESTS),
    }


@pytest.fixture
def callbacks(monkeypatch):
    sent = []
    monkeypatch.setattr(runner, "_post_callback", lambda sid, payload, retries=4: sent.append(payload) or True)

    def fetch(pid, sample_only=False):
        payload = problem_payload()
        if sample_only:
            payload["test_cases"] = [t for t in TESTS if t["is_sample"]]
        return payload

    monkeypatch.setattr(runner, "_fetch_problem_tests", fetch)
    return sent


def test_accepted_reports_all_results(callbacks):
    sb = FakeSandbox([ok("3"), ok("10", 9, 1200), ok("8")])
    assert runner.run_submission("sid", 1, "python", "code", sandbox=sb) == "ACCEPTED"
    final = callbacks[-1]
    assert callbacks[0]["status"] == "RUNNING"
    assert final["status"] == "ACCEPTED" and final["runtime_ms"] == 9 and final["memory_kb"] == 1200
    assert [r["test_case_id"] for r in final["results"]] == [11, 12, 13]
    assert sb.cleaned


def test_stops_at_first_failure(callbacks):
    sb = FakeSandbox([ok("3"), ok("11"), ok("8")])
    assert runner.run_submission("sid", 1, "python", "code", sandbox=sb) == "WRONG_ANSWER"
    assert len(callbacks[-1]["results"]) == 2 and len(sb.executed) == 2


def test_sample_run_only_uses_samples(callbacks):
    sb = FakeSandbox([ok("3")])
    runner.run_submission("sid", 1, "python", "code", run_sample=True, sandbox=sb)
    assert sb.executed == ["1 2"]


def test_compile_error_reported_once(callbacks):
    ce = SandboxResult("", "err: x", 1, 0, None, False, compile_error=True, compile_stderr="main.cpp:1: error: x")
    sb = FakeSandbox([], compile_result=ce)
    assert runner.run_submission("sid", 1, "cpp", "code", sandbox=sb) == "COMPILE_ERROR"
    final = callbacks[-1]
    assert final["error_message"].startswith("main.cpp:1: error")
    assert final["results"][0]["status"] == "COMPILE_ERROR" and sb.executed == []


@pytest.mark.parametrize("result,status", [
    (SandboxResult("", "", 124, 2000, None, True), "TIME_LIMIT"),
    (SandboxResult("", "MemoryError", 1, 10, 260000, False, memory_exceeded=True), "MEMORY_LIMIT"),
    (SandboxResult("x" * 10, "", 0, 10, None, False, output_exceeded=True), "RUNTIME_ERROR"),
    (SandboxResult("", "", 139, 10, None, False), "RUNTIME_ERROR"),
])
def test_failure_classification(callbacks, result, status):
    sb = FakeSandbox([result])
    assert runner.run_submission("sid", 1, "python", "code", sandbox=sb) == status
    assert callbacks[-1]["error_message"]


def test_segfault_message(callbacks):
    runner.run_submission("sid", 1, "cpp", "code", sandbox=FakeSandbox([SandboxResult("", "", 139, 1, None, False)]))
    assert "SIGSEGV" in callbacks[-1]["error_message"]


def test_infrastructure_errors_are_retryable(callbacks):
    sb = FakeSandbox([SandboxResult("", "docker api error", 1, 0, None, False, infra_error=True)])
    with pytest.raises(runner.JudgeInfrastructureError):
        runner.run_submission("sid", 1, "python", "code", sandbox=sb)
    assert sb.cleaned
    assert all(c["status"] == "RUNNING" for c in callbacks)  # no wrong verdict was sent


def test_test_case_fetch_failure_is_retryable(monkeypatch):
    monkeypatch.setattr(runner, "_post_callback", lambda *a, **k: True)

    def fail(*a, **k):
        raise ConnectionError("backend down")

    monkeypatch.setattr(runner, "_fetch_problem_tests", fail)
    with pytest.raises(runner.JudgeInfrastructureError):
        runner.run_submission("sid", 1, "python", "code", sandbox=FakeSandbox([]))


def test_undeliverable_verdict_raises(monkeypatch):
    monkeypatch.setattr(runner, "_fetch_problem_tests", lambda *a, **k: problem_payload(TESTS[:1]))
    monkeypatch.setattr(runner, "_post_callback", lambda sid, payload, retries=4: payload["status"] == "RUNNING")
    with pytest.raises(runner.JudgeInfrastructureError):
        runner.run_submission("sid", 1, "python", "code", sandbox=FakeSandbox([ok("3")]))


def test_post_callback_does_not_retry_client_errors(monkeypatch):
    class Resp:
        status_code, text = 400, "bad"

    calls = []
    monkeypatch.setattr(runner.requests, "post", lambda *a, **k: calls.append(1) or Resp())
    with pytest.raises(runner.CallbackRejected):
        runner._post_callback("sid", {"status": "RUNNING"}, retries=3)
    assert len(calls) == 1


def test_post_callback_retries_server_errors(monkeypatch):
    responses = [type("R", (), {"status_code": 503, "text": ""})(), type("R", (), {"status_code": 200, "text": ""})()]
    monkeypatch.setattr(runner.requests, "post", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    assert runner._post_callback("sid", {"status": "RUNNING"}, retries=3) is True


# ─── Function-only mode ──────────────────────────────────────────────────────

def function_callbacks(monkeypatch, compare_mode="EXACT"):
    sent = []
    monkeypatch.setattr(runner, "_post_callback", lambda sid, payload, retries=4: sent.append(payload) or True)

    def fetch(pid, sample_only=False):
        return problem_payload(FUNCTION_TESTS, FUNCTION_SIGNATURE, compare_mode)

    monkeypatch.setattr(runner, "_fetch_problem_tests", fetch)
    return sent


def marked(value: str, debug: str = ""):
    return f"{debug}__CQ_RESULT__ {value}\n"


def test_function_mode_wraps_code_and_encodes_input(monkeypatch):
    sent = function_callbacks(monkeypatch)
    sb = FakeSandbox([ok(marked("3")), ok(marked("30"))])
    user_code = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
    assert runner.run_submission("sid", 1, "python", user_code, sandbox=sb) == "ACCEPTED"
    assert "class Solution" in sb.codes[0] and "__CQ_RESULT__" in sb.codes[0]
    assert sb.executed == ["1\n2\n", "10\n20\n"]
    assert sent[-1]["results"][0]["actual_output"] == "3"


def test_function_mode_ignores_debug_stdout(monkeypatch):
    function_callbacks(monkeypatch)
    sb = FakeSandbox([ok(marked("3", "debug line\n")), ok(marked("30"))])
    assert runner.run_submission("sid", 1, "python", "code", sandbox=sb) == "ACCEPTED"
    assert sb.executed == ["1\n2\n", "10\n20\n"]


def test_function_mode_missing_result_is_wrong_answer(monkeypatch):
    function_callbacks(monkeypatch)
    sb = FakeSandbox([ok("debug only\n")])
    assert runner.run_submission("sid", 1, "python", "code", sandbox=sb) == "WRONG_ANSWER"


def test_function_mode_unordered_compare(monkeypatch):
    function_callbacks(monkeypatch, compare_mode="UNORDERED")
    tests = [{"id": 31, "input_data": '{"nums": [2, 7], "target": 9}', "expected_output": "[0,1]", "is_sample": True, "is_hidden": False}]
    signature = {
        "name": "twoSum",
        "params": [{"name": "nums", "type": "int[]"}, {"name": "target", "type": "int"}],
        "return_type": "int[]",
    }
    monkeypatch.setattr(
        runner, "_fetch_problem_tests",
        lambda pid, sample_only=False: problem_payload(tests, signature, "UNORDERED"),
    )
    sb = FakeSandbox([ok(marked("[1,0]"))])
    assert runner.run_submission("sid", 1, "python", "code", sandbox=sb) == "ACCEPTED"


def test_invalid_function_signature_is_retryable(monkeypatch):
    sent = function_callbacks(monkeypatch)
    monkeypatch.setattr(
        runner, "_fetch_problem_tests",
        lambda pid, sample_only=False: problem_payload(FUNCTION_TESTS, {"name": "add", "params": [], "return_type": "void"}),
    )
    with pytest.raises(runner.JudgeInfrastructureError):
        runner.run_submission("sid", 1, "python", "code", sandbox=FakeSandbox([]))
    assert all(c["status"] == "RUNNING" for c in sent)
