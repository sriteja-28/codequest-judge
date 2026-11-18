# CodeQuest Judge Worker

Consumes submission jobs from RabbitMQ and executes them in sandboxed Docker
containers (one compile step, one container per test case, uid 65534, CPU/memory
limits). Verdicts are posted back to the backend through its internal
`X-Judge-Api-Key` API.

## Setup

```bash
cd judge
python -m venv venv
venv\Scripts\activate            # Windows (source venv/bin/activate on POSIX)
pip install -r requirements.txt

cp .env.example .env             # match JUDGE_INTERNAL_API_KEY with backend/.env

# Build the language runner images
docker build -t codequest-python ./languages/python
docker build -t codequest-cpp    ./languages/cpp
docker build -t codequest-java   ./languages/java     # Java 21
docker build -t codequest-js     ./languages/javascript

python worker.py
```

Docker must be running and the four `codequest-*` images must exist before the
worker starts. All knobs (limits, queue names, retries) are documented in
[`.env.example`](.env.example).

## How jobs flow

1. Backend publishes `{submission_id, problem_id, language, code, ...}` to
   `JUDGE_QUEUE_NAME` (default `code_submissions`).
2. Worker fetches test cases via `GET /api/submissions/judge/problems/{id}/testcases/`
   (`TESTCASE_SOURCE=api` — no DB credentials needed).
3. Per test case the sandbox classifies the outcome: ACCEPTED / WRONG_ANSWER /
   RUNTIME_ERROR / TIME_LIMIT / MEMORY_LIMIT, plus COMPILE_ERROR and
   INTERNAL_ERROR for sandbox/infra failures.
4. Worker posts a single callback: `POST /api/submissions/{id}/judge-callback/`
   with `status` (+ `results[]`). The backend applies it idempotently.

## Failure handling

- **Code failures** (wrong answer, TLE, compile error, …) are final — one
  callback, job done.
- **Infrastructure failures** (Docker daemon down, backend unreachable while
  publishing the callback) raise `JudgeInfrastructureError` → the job is
  requeued with a retry header, up to `MAX_JOB_RETRIES`, then dead-lettered to
  `<queue>.dead` for triage.
- **Malformed payloads / unknown enum values** are rejected by the backend
  (400) and the job is dead-lettered — never retried into a corrupt state.

> Deploy backend and judge together: the callback contract (status enums,
> RUNNING + terminal callbacks, idempotency) is versioned with the worker.
