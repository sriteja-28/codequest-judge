"""
runner.py — Orchestrates one submission through all test cases.

Flow:
  1. POST callback → backend (status=RUNNING)            [best effort]
  2. Fetch test cases (backend internal API by default, or the DB)
  3. Compile once (C++/Java), run each test case in a fresh container, compare
  4. POST the final verdict with per-test results

Stops at the first non-ACCEPTED test case (standard OJ behaviour); pass
stop_on_fail=False to run all test cases.

Errors:
  JudgeInfrastructureError — retryable (Docker down, backend unreachable,
                              test cases not loadable). The worker retries the
                              job and finally reports INTERNAL_ERROR.
  CallbackRejected         — the backend answered 4xx (contract problem);
                              not retryable.
Verdict callbacks are idempotent on the backend, so re-running a job after a
partial failure is safe.
"""

import logging
import time

import requests
from decouple import config

from codegen import SignatureError, encode_input, split_result, validate_signature, wrap
from comparator import compare_output
from sandbox import Prepared, Sandbox, SandboxResult

logger = logging.getLogger(__name__)

BACKEND_API_URL = config("BACKEND_API_URL", default="http://127.0.0.1:8000").rstrip("/")
JUDGE_INTERNAL_API_KEY = config("JUDGE_INTERNAL_API_KEY", default="judge-secret-key-change-me")
CPU_TIME_LIMIT = config("CPU_TIME_LIMIT", default=10, cast=int)
MEMORY_LIMIT_MB = config("MEMORY_LIMIT_MB", default=256, cast=int)
# "api" (default): GET /api/submissions/judge/problems/<id>/testcases/ with the judge key.
# "db": read the test_cases table directly (needs DATABASE_URL, PostgreSQL only).
TESTCASE_SOURCE = config("TESTCASE_SOURCE", default="api").strip().lower()
CALLBACK_RETRIES = config("CALLBACK_RETRIES", default=4, cast=int)
HTTP_TIMEOUT = config("BACKEND_HTTP_TIMEOUT", default=15, cast=int)
MAX_ERROR_MESSAGE_CHARS = 2_000

HEADERS = {
    "X-Judge-Api-Key": JUDGE_INTERNAL_API_KEY,
    "Content-Type": "application/json",
}


class JudgeInfrastructureError(Exception):
    """Retryable failure that is not the submitted program's fault."""


class CallbackRejected(Exception):
    """The backend rejected a callback with a 4xx response (do not retry)."""


# ─── Backend communication ────────────────────────────────────────────────────

def _post_callback(submission_id: str, payload: dict, retries: int = CALLBACK_RETRIES) -> bool:
    """POST a callback with exponential backoff. Returns True on HTTP 200,
    raises CallbackRejected on a 4xx (except 429), False if never delivered."""
    url = f"{BACKEND_API_URL}/api/submissions/{submission_id}/judge-callback/"
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=HEADERS, timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning("Callback network error for %s (attempt %d/%d): %s", submission_id, attempt, retries, exc)
        else:
            if resp.status_code == 200:
                return True
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise CallbackRejected(f"HTTP {resp.status_code}: {resp.text[:300]}")
            logger.warning("Callback HTTP %d for %s (attempt %d/%d)", resp.status_code, submission_id, attempt, retries)
        if attempt < retries:
            time.sleep(delay)
            delay = min(delay * 2, 10.0)
    logger.error("All %d callback attempts failed for submission %s", retries, submission_id)
    return False


def _fetch_problem_tests(problem_id: int, sample_only: bool = False) -> dict:
    """Return problem judge config + test cases, from the API or the database."""
    if TESTCASE_SOURCE == "db":
        return _fetch_problem_tests_from_db(problem_id, sample_only=sample_only)
    return _fetch_problem_tests_from_api(problem_id, sample_only=sample_only)


def _fetch_problem_tests_from_api(problem_id: int, sample_only: bool = False) -> dict:
    url = f"{BACKEND_API_URL}/api/submissions/judge/problems/{problem_id}/testcases/"
    params = {"sample_only": "1"} if sample_only else None
    resp = requests.get(url, headers=HEADERS, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _fetch_problem_tests_from_db(problem_id: int, sample_only: bool = False) -> dict:
    import dj_database_url
    import psycopg2

    db_url = config("DATABASE_URL", default="postgresql://codequest:codequest@localhost:5432/codequest")
    parsed = dj_database_url.parse(db_url)
    conn = psycopg2.connect(
        host=parsed["HOST"],
        port=parsed.get("PORT") or 5432,
        dbname=parsed["NAME"],
        user=parsed["USER"],
        password=parsed["PASSWORD"],
        connect_timeout=10,
    )
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT function_signature, compare_mode, float_tolerance FROM problems WHERE id = %s",
            (problem_id,),
        )
        problem = cur.fetchone()
        if problem is None:
            raise LookupError(f"problem {problem_id} does not exist")
        query = (
            "SELECT id, input_data, expected_output, is_hidden, is_sample "
            "FROM test_cases WHERE problem_id = %s"
            + (" AND is_sample = TRUE" if sample_only else "")
            + " ORDER BY order_index, id"
        )
        cur.execute(query, (problem_id,))
        test_cases = [
            {"id": r[0], "input_data": r[1], "expected_output": r[2], "is_hidden": r[3], "is_sample": r[4]}
            for r in cur.fetchall()
        ]
        return {
            "problem_id": problem_id,
            "function_signature": problem[0],
            "compare_mode": problem[1],
            "float_tolerance": problem[2],
            "test_cases": test_cases,
        }
    finally:
        conn.close()


# ─── Sandbox (one per worker process) ────────────────────────────────────────

_sandbox = None


def get_sandbox() -> Sandbox:
    global _sandbox
    if _sandbox is None:
        try:
            _sandbox = Sandbox(cpu_time_limit=CPU_TIME_LIMIT, memory_limit_mb=MEMORY_LIMIT_MB)
        except Exception as exc:  # Docker daemon not reachable
            raise JudgeInfrastructureError(f"Docker unavailable: {exc}") from exc
    return _sandbox


def reset_sandbox() -> None:
    global _sandbox
    _sandbox = None


# ─── Verdicts ─────────────────────────────────────────────────────────────────

def classify(
    result: SandboxResult,
    expected_output: str,
    *,
    compare_mode: str = "EXACT",
    float_tolerance: float = 1e-5,
    function_mode: bool = False,
) -> str:
    """Per-test-case status (order matters)."""
    if result.infra_error:
        return "INTERNAL_ERROR"
    if result.compile_error:
        return "COMPILE_ERROR"
    if result.timed_out:
        return "TIME_LIMIT"
    if result.memory_exceeded:
        return "MEMORY_LIMIT"
    if result.output_exceeded or result.exit_code != 0:
        return "RUNTIME_ERROR"

    if function_mode:
        _, answer = split_result(result.stdout)
        if answer is None:
            # The harness never produced a return value: the function returned
            # nothing and the process still exited cleanly.
            return "WRONG_ANSWER"
        actual = answer
    else:
        actual = result.stdout

    return "ACCEPTED" if compare_output(actual, expected_output, compare_mode, float_tolerance) else "WRONG_ANSWER"


def displayed_output(result: SandboxResult, function_mode: bool) -> str:
    """The value shown in 'Actual output' (the return value in function mode)."""
    if function_mode:
        _, answer = split_result(result.stdout)
        if answer is not None:
            return answer
    return result.stdout


_SIGNALS = {134: "SIGABRT", 136: "SIGFPE", 137: "SIGKILL", 139: "SIGSEGV"}


def failure_message(status: str, result: SandboxResult, sandbox: Sandbox) -> str:
    if status == "COMPILE_ERROR":
        return result.compile_stderr[:MAX_ERROR_MESSAGE_CHARS]
    if status == "TIME_LIMIT":
        return f"Time limit exceeded ({sandbox.cpu_time_limit}s)."
    if status == "MEMORY_LIMIT":
        tail = result.stderr.strip().splitlines()[-1:] if result.stderr.strip() else []
        return (f"Memory limit exceeded ({sandbox.memory_limit_mb} MB)." + (f"\n{tail[0]}" if tail else ""))[:MAX_ERROR_MESSAGE_CHARS]
    if status == "RUNTIME_ERROR":
        if result.output_exceeded:
            return "Output limit exceeded (more than 64 KB written to stdout)."
        if result.stderr.strip():
            return result.stderr.strip()[:MAX_ERROR_MESSAGE_CHARS]
        signal = _SIGNALS.get(result.exit_code)
        return f"Process exited with code {result.exit_code}" + (f" ({signal})." if signal else ".")
    return ""


# ─── Main entry point ─────────────────────────────────────────────────────────

def _final(submission_id: str, payload: dict) -> None:
    if not _post_callback(submission_id, payload):
        raise JudgeInfrastructureError("Could not deliver the verdict to the backend.")


def run_submission(
    submission_id: str,
    problem_id: int,
    language: str,
    code: str,
    stop_on_fail: bool = True,
    run_sample: bool = False,
    sandbox: Sandbox | None = None,
) -> str:
    """Judge one submission and POST the verdict. Returns the overall status."""
    logger.info("Starting submission %s | lang=%s | problem=%s | sample=%s",
                submission_id, language, problem_id, run_sample)

    _post_callback(submission_id, {"status": "RUNNING", "results": []}, retries=2)

    try:
        problem = _fetch_problem_tests(problem_id, sample_only=run_sample)
        test_cases = problem["test_cases"]
    except Exception as exc:
        raise JudgeInfrastructureError(f"Could not load test cases: {exc}") from exc

    raw_signature = problem.get("function_signature")
    function_mode = bool(raw_signature)
    signature = None
    if function_mode:
        try:
            signature = validate_signature(raw_signature)
            code = wrap(language, signature, code)
        except (SignatureError, ValueError) as exc:
            raise JudgeInfrastructureError(f"Problem {problem_id} has an invalid function signature: {exc}") from exc

    compare_mode = problem.get("compare_mode") or "EXACT"
    float_tolerance = problem.get("float_tolerance")
    if float_tolerance is None:
        float_tolerance = 1e-5

    if not test_cases:
        logger.error("No test cases found for problem %s (sample_only=%s)", problem_id, run_sample)
        _final(submission_id, {
            "status": "INTERNAL_ERROR",
            "error_message": "No test cases are configured for this problem.",
            "results": [],
        })
        return "INTERNAL_ERROR"

    sandbox = sandbox or get_sandbox()
    prepared = sandbox.prepare(language, code)
    if isinstance(prepared, SandboxResult):  # unsupported language / infrastructure
        if prepared.infra_error:
            raise JudgeInfrastructureError(prepared.stderr)
        _final(submission_id, {"status": "INTERNAL_ERROR", "error_message": prepared.stderr, "results": []})
        return "INTERNAL_ERROR"

    results: list[dict] = []
    overall_status = "ACCEPTED"
    error_message = ""
    max_runtime_ms = 0
    max_memory_kb = 0
    try:
        if not prepared.ok:
            compile_result = prepared.compile_result
            if compile_result.infra_error:
                raise JudgeInfrastructureError(compile_result.stderr)
            overall_status = "COMPILE_ERROR"
            error_message = failure_message("COMPILE_ERROR", compile_result, sandbox)
            results.append({
                "test_case_id": test_cases[0]["id"],
                "status": "COMPILE_ERROR",
                "actual_output": "",
                "error_output": compile_result.compile_stderr,
                "time_ms": None,
                "memory_kb": None,
            })
        else:
            for tc in test_cases:
                try:
                    stdin_data = encode_input(signature, tc["input_data"]) if function_mode else tc["input_data"]
                except ValueError as exc:
                    raise JudgeInfrastructureError(
                        f"Problem {problem_id} test case {tc['id']} has invalid function input: {exc}"
                    ) from exc

                result = sandbox.execute(prepared, stdin_data)
                if result.infra_error:
                    raise JudgeInfrastructureError(result.stderr)
                tc_status = classify(
                    result,
                    tc["expected_output"],
                    compare_mode=compare_mode,
                    float_tolerance=float_tolerance,
                    function_mode=function_mode,
                )
                logger.debug("tc#%s -> %s (%sms, %sKB)", tc["id"], tc_status, result.time_ms, result.memory_kb)

                if result.time_ms:
                    max_runtime_ms = max(max_runtime_ms, result.time_ms)
                if result.memory_kb:
                    max_memory_kb = max(max_memory_kb, result.memory_kb)

                results.append({
                    "test_case_id": tc["id"],
                    "status": tc_status,
                    "actual_output": displayed_output(result, function_mode),
                    "error_output": result.stderr,
                    "time_ms": result.time_ms,
                    "memory_kb": result.memory_kb,
                })

                if tc_status != "ACCEPTED" and overall_status == "ACCEPTED":
                    overall_status = tc_status
                    error_message = failure_message(tc_status, result, sandbox)
                if stop_on_fail and tc_status != "ACCEPTED":
                    break
    finally:
        sandbox.cleanup(prepared)

    logger.info(
        "Submission %s → %s | %dms | %dKB | %d/%d TCs passed",
        submission_id, overall_status, max_runtime_ms, max_memory_kb,
        sum(1 for r in results if r["status"] == "ACCEPTED"), len(test_cases),
    )

    _final(submission_id, {
        "status": overall_status,
        "runtime_ms": max_runtime_ms or None,
        "memory_kb": max_memory_kb or None,
        "error_message": error_message,
        "results": results,
    })
    return overall_status
