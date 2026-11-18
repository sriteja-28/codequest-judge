from sandbox import Sandbox
from comparator import compare

def run_code_against_tests(code, language, testcases):
    # initializing sandbox
    sandbox = Sandbox(cpu_time_limit=2, memory_limit_mb=128)

    results = []

    for tc in testcases:
        result = sandbox.run(language, code, tc.input_data)

        # checking status
        if result.compile_error:
            status = "COMPILE_ERROR"
        elif result.timed_out:
            status = "TIME_LIMIT"
        elif result.exit_code != 0:
            status = "RUNTIME_ERROR"
        elif compare(result.stdout, tc.expected_output):
            status = "ACCEPTED"
        else:
            status = "WRONG_ANSWER"

        # building response
        results.append({
            "input": tc.input_data,
            "expected": tc.expected_output,
            "output": result.stdout,
            "status": status,
            "time_ms": result.time_ms,
        })

    return results