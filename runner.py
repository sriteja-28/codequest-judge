"""
runner.py — Orchestrates one submission through all test cases.

Flow:
  1. POST callback → backend (status=RUNNING)
  2. Fetch test cases from DB
  3. For each test case: Sandbox.run() → compare output
  4. POST final callback with all results

Stops at first non-ACCEPTED result (standard OJ behaviour).
Override stop_on_fail=False to run all test cases (useful for debugging).
"""

import logging
import os
import requests
from decouple import config

from sandbox import Sandbox, SandboxResult
from comparator import compare

logger = logging.getLogger(__name__)

BACKEND_API_URL       = config("BACKEND_API_URL",       default="http://localhost:8000")
JUDGE_INTERNAL_API_KEY = config("JUDGE_INTERNAL_API_KEY", default="judge-secret-key-change-me")
CPU_TIME_LIMIT        = config("CPU_TIME_LIMIT",        default=10,  cast=int)
MEMORY_LIMIT_MB       = config("MEMORY_LIMIT_MB",       default=256, cast=int)

HEADERS = {
    "X-Judge-Api-Key": JUDGE_INTERNAL_API_KEY,
    "Content-Type":    "application/json",
}


# ─── Backend communication ────────────────────────────────────────────────────

def _post_callback(submission_id: str, payload: dict, retries: int = 3):
    url = f"{BACKEND_API_URL}/api/submissions/{submission_id}/judge-callback/"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return
            logger.error("Callback HTTP %d (attempt %d): %s", resp.status_code, attempt, resp.text[:200])
        except Exception as exc:
            logger.error("Callback network error (attempt %d): %s", attempt, exc)
    logger.error("All %d callback attempts failed for submission %s", retries, submission_id)


def _fetch_test_cases(problem_id: int, sample_only: bool = False) -> list[dict]:
    """
    Fetch test cases directly from the database.
    Falls back to the backend API if DB creds aren't available.
    """
    return _fetch_test_cases_from_db(problem_id,sample_only=sample_only)


def _fetch_test_cases_from_db(problem_id: int, sample_only: bool = False) -> list[dict]:
    import psycopg2
    import dj_database_url
    from decouple import config as dconfig

    db_url = dconfig(
        "DATABASE_URL",
        default="postgresql://codequest:codequest@localhost:5432/codequest",
    )
    parsed = dj_database_url.parse(db_url)
    conn   = psycopg2.connect(
        host     = parsed["HOST"],
        port     = parsed.get("PORT") or 5432,
        dbname   = parsed["NAME"],
        user     = parsed["USER"],
        password = parsed["PASSWORD"],
    )
    try:
        cur = conn.cursor()

        query = """
            SELECT id, input_data, expected_output, is_hidden, is_sample
            FROM   test_cases
            WHERE  problem_id = %s
            {} ORDER  BY order_index """.format("AND is_sample = TRUE" if sample_only else "") 

        # cur.execute(
        #     """
        #     SELECT id, input_data, expected_output, is_hidden
        #     FROM   test_cases
        #     WHERE  problem_id = %s
        #     ORDER  BY order_index
        #     """,
        #     (problem_id,), )
        cur.execute(query, (problem_id,))
       
        rows = cur.fetchall()
        return [
            {"id": r[0], "input_data": r[1], "expected_output": r[2], "is_hidden": r[3]}
            for r in rows
        ]
    finally:
        conn.close()


# ─── Status mapping ───────────────────────────────────────────────────────────

def _map_status(result: SandboxResult) -> str:
    """
    Convert a SandboxResult to a Django Submission.Status string.
    Order of checks matters — timed_out must precede exit_code check.
    """
    if result.compile_error:
        return "COMPILE_ERROR"
    if result.timed_out:
        return "TIME_LIMIT"
    if result.memory_kb and result.memory_kb * 1024 >= result.memory_kb:
        # Memory limit checked by comparing against the configured cap.
        # Docker kills the container with OOM (exit_code 137) if exceeded.
        pass
    if result.exit_code == 137:
        # SIGKILL — typically OOM kill by Docker
        return "MEMORY_LIMIT"
    if result.exit_code != 0:
        return "RUNTIME_ERROR"
    return "ACCEPTED"  # caller checks output match separately


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_submission(
    submission_id: str,
    problem_id:    int,
    language:      str,
    code:          str,
    stop_on_fail:  bool = True,
    run_sample:    bool = False, 
) -> None:
    """
    Run all test cases for a submission and POST results to the backend.
    """
    logger.info(
        "Starting submission %s | lang=%s | problem=%d",
        submission_id, language, problem_id,
    )

    # Tell backend we've started
    _post_callback(submission_id, {"status": "RUNNING", "results": []})

    # Fetch test cases
    try:
        test_cases = _fetch_test_cases(problem_id, sample_only=run_sample)
    except Exception as exc:
        logger.error("Failed to fetch test cases: %s", exc)
        _post_callback(submission_id, {
            "status":        "INTERNAL_ERROR",
            "error_message": f"Could not load test cases: {exc}",
            "results":       [],
        })
        return

    if not test_cases:
        logger.error("No test cases found for problem %d", problem_id)
        _post_callback(submission_id, {
            "status":        "INTERNAL_ERROR",
            "error_message": "No test cases configured for this problem.",
            "results":       [],
        })
        return

    sandbox = Sandbox(
        cpu_time_limit  = CPU_TIME_LIMIT,
        memory_limit_mb = MEMORY_LIMIT_MB,
    )

    results:         list[dict] = []
    overall_status:  str        = "ACCEPTED"
    max_runtime_ms:  int        = 0
    max_memory_kb:   int        = 0
    error_message:   str        = ""

    for tc in test_cases:
        result: SandboxResult = sandbox.run(language, code, tc["input_data"])


        # adding debug logs here
        logger.debug("----- Test Case Debug -----")
        logger.debug("INPUT: %s", tc["input_data"])
        logger.debug("EXPECTED: %s", repr(tc["expected_output"]))
        logger.debug("GOT: %s", repr(result.stdout))
        logger.debug("---------------------------")

        # Determine per-testcase status
        if result.compile_error:
            tc_status = "COMPILE_ERROR"
        elif result.timed_out:
            tc_status = "TIME_LIMIT"
        elif result.exit_code == 137:
            tc_status = "MEMORY_LIMIT"
        elif result.exit_code != 0:
            tc_status = "RUNTIME_ERROR"
            
        elif compare(result.stdout, tc["expected_output"]):
            tc_status = "ACCEPTED"
        else:
            tc_status = "WRONG_ANSWER"

        # Track peak resource usage across all test cases
        if result.time_ms:
            max_runtime_ms = max(max_runtime_ms, result.time_ms)
        if result.memory_kb:
            max_memory_kb = max(max_memory_kb, result.memory_kb)

        # Build result record
        tc_record = {
            "test_case_id":   tc["id"],
            "status":         tc_status,
            "actual_output":  result.stdout,
            "error_output":   result.stderr if not result.compile_error else result.compile_stderr,
            "time_ms":        result.time_ms,
            "memory_kb":      result.memory_kb,
        }
        results.append(tc_record)

        # Set overall status on first failure
        if tc_status != "ACCEPTED" and overall_status == "ACCEPTED":
            overall_status = tc_status
            if tc_status in ("RUNTIME_ERROR", "COMPILE_ERROR"):
                error_message = (
                    result.compile_stderr if result.compile_error else result.stderr
                )[:2_000]

        # Stop on first failure (standard judge behaviour)
        if stop_on_fail and tc_status != "ACCEPTED":
            logger.debug(
                "Stopping after first failure: %s on tc#%d",
                tc_status, tc["id"],
            )
            break

    logger.info(
        "Submission %s → %s | %dms | %dKB | %d/%d TCs",
        submission_id, overall_status,
        max_runtime_ms, max_memory_kb,
        sum(1 for r in results if r["status"] == "ACCEPTED"),
        len(test_cases),
    )

    _post_callback(submission_id, {
        "status":        overall_status,
        "runtime_ms":    max_runtime_ms or None,
        "memory_kb":     max_memory_kb  or None,
        "error_message": error_message,
        "results":       results,
    })