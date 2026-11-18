"""
worker.py — RabbitMQ consumer that processes code submission jobs.

Run one or more workers (each judges one job at a time):
    python worker.py

Each worker:
1. Connects to RabbitMQ (with retries) and declares the durable job queue
   plus a durable dead-letter queue (<queue>.dead).
2. Fetches one job at a time (prefetch=1 for fairness).
3. Validates the payload and calls runner.run_submission() in a helper thread
   while the consumer thread keeps serving RabbitMQ heartbeats — a long Docker
   run can no longer trip AMQPHeartbeatTimeout and kill the connection.
4. ACKs every message exactly once:
     - success                      → ack
     - malformed / invalid payload  → copy to dead-letter queue, ack
     - retryable failure            → republish with x-retry-count+1, ack
       (after MAX_JOB_RETRIES: report INTERNAL_ERROR, dead-letter, ack)
   A message is therefore never requeued in a tight loop (poison message).
5. If the connection dies mid-job (broker restart, heartbeat timeout), the
   worker finishes judging anyway — the backend dedupes verdicts — then
   reconnects; the broker redelivers the unacked message.

Horizontal scaling: launch more worker processes; RabbitMQ round-robins jobs.
Duplicate deliveries are safe: the backend applies the first verdict only.
"""

import json
import logging
import signal
import threading
import time
import uuid

import pika
from decouple import config

from runner import CallbackRejected, JudgeInfrastructureError, _post_callback, reset_sandbox, run_submission
from sandbox import LANGUAGE_CONFIG

logging.basicConfig(
    level=config("LOG_LEVEL", default="INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("judge.worker")

# Default matches the local dev stack (infra/docker-compose.dev.yml).
RABBITMQ_URL = config("RABBITMQ_URL", default="amqp://codequest:codequest_dev@127.0.0.1:5673/")
# Same variable name as the backend (JUDGE_QUEUE_NAME); JUDGE_QUEUE kept for compatibility.
QUEUE_NAME = config("JUDGE_QUEUE_NAME", default=config("JUDGE_QUEUE", default="code_submissions"))
DEAD_LETTER_QUEUE = config("JUDGE_DEAD_LETTER_QUEUE", default=f"{QUEUE_NAME}.dead")
MAX_JOB_RETRIES = config("MAX_JOB_RETRIES", default=3, cast=int)
RETRY_DELAY_SECONDS = config("RETRY_DELAY_SECONDS", default=5, cast=int)
MAX_CODE_BYTES = config("MAX_CODE_BYTES", default=65536, cast=int)
RETRY_HEADER = "x-retry-count"

_running = True


def _shutdown_handler(signum, frame):
    global _running
    logger.info("Received signal %s — finishing current job then shutting down.", signum)
    _running = False


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)


# ─── Job validation ───────────────────────────────────────────────────────────

def validate_job(job) -> str | None:
    """Return an error string for an unusable payload, else None."""
    if not isinstance(job, dict):
        return "payload is not a JSON object"
    try:
        uuid.UUID(str(job.get("submission_id")))
    except ValueError:
        return "submission_id missing or not a UUID"
    if not isinstance(job.get("problem_id"), int) or isinstance(job.get("problem_id"), bool):
        return "problem_id missing or not an integer"
    if job.get("language") not in LANGUAGE_CONFIG:
        return f"unsupported language {job.get('language')!r}"
    code = job.get("code")
    if not isinstance(code, str) or not code.strip():
        return "code missing or empty"
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return f"code exceeds {MAX_CODE_BYTES} bytes"
    if not isinstance(job.get("run_sample", False), bool):
        return "run_sample must be a boolean"
    return None


# ─── Queue helpers ────────────────────────────────────────────────────────────

def _properties(properties, headers: dict) -> pika.BasicProperties:
    return pika.BasicProperties(
        delivery_mode=pika.DeliveryMode.Persistent,
        content_type="application/json",
        message_id=getattr(properties, "message_id", None),
        headers=headers,
    )


def dead_letter(channel, body: bytes, properties, reason: str) -> None:
    headers = dict(getattr(properties, "headers", None) or {})
    headers["x-dead-letter-reason"] = reason[:500]
    headers["x-dead-lettered-at"] = int(time.time())
    channel.basic_publish(exchange="", routing_key=DEAD_LETTER_QUEUE, body=body, properties=_properties(properties, headers))
    logger.error("Dead-lettered job (%s)", reason)


def requeue_with_retry(channel, body: bytes, properties, retry_count: int) -> None:
    headers = dict(getattr(properties, "headers", None) or {})
    headers[RETRY_HEADER] = retry_count
    channel.basic_publish(exchange="", routing_key=QUEUE_NAME, body=body, properties=_properties(properties, headers))


def _report_internal_error(submission_id, message: str) -> None:
    try:
        _post_callback(str(submission_id), {"status": "INTERNAL_ERROR", "error_message": message, "results": []}, retries=2)
    except CallbackRejected as exc:
        logger.error("INTERNAL_ERROR callback rejected for %s: %s", submission_id, exc)


# ─── Consumer ─────────────────────────────────────────────────────────────────

HEARTBEAT_SLICE = 0.5  # seconds between heartbeat-serving checks while judging


def _serve_heartbeats_during(thread: threading.Thread, connection) -> bool:
    """Keep the RabbitMQ connection alive while `thread` judges a job.

    `process_data_events` sends the heartbeat (60 s interval); judging used to
    run inline and starve it, which killed the connection with
    AMQPHeartbeatTimeout. Returns False when the connection dropped mid-job.
    """
    while thread.is_alive():
        if connection is not None:
            try:
                connection.process_data_events(time_limit=HEARTBEAT_SLICE)
            except Exception as exc:
                logger.warning("Connection dropped while judging (%s); letting the job finish anyway.", exc)
                thread.join()
                return False
        thread.join(timeout=HEARTBEAT_SLICE)
    return True


def on_message(channel, method, properties, body):
    """Called for each message from RabbitMQ. Always acks exactly once."""
    try:
        job = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        dead_letter(channel, body, properties, f"invalid JSON: {exc}")
        channel.basic_ack(delivery_tag=method.delivery_tag)
        return

    error = validate_job(job)
    if error:
        if isinstance(job, dict) and job.get("submission_id"):
            _report_internal_error(job["submission_id"], "The judge received an invalid job for this submission.")
        dead_letter(channel, body, properties, f"invalid job: {error}")
        channel.basic_ack(delivery_tag=method.delivery_tag)
        return

    submission_id = job["submission_id"]
    retry_count = int((getattr(properties, "headers", None) or {}).get(RETRY_HEADER, 0))
    logger.info("Received job submission_id=%s lang=%s retry=%d", submission_id, job["language"], retry_count)

    # Judge in a helper thread so this consumer thread can keep answering
    # RabbitMQ heartbeats for the whole run (see _serve_heartbeats_during).
    outcome: dict = {"status": None, "exc": None}

    def _work():
        try:
            outcome["status"] = run_submission(
                submission_id=submission_id,
                problem_id=job["problem_id"],
                language=job["language"],
                code=job["code"],
                run_sample=job.get("run_sample", False),
            )
        except Exception as exc:  # re-handled on the consumer thread below
            outcome["exc"] = exc

    thread = threading.Thread(target=_work, name=f"judge-{submission_id}", daemon=True)
    thread.start()
    connection_ok = _serve_heartbeats_during(thread, getattr(channel, "connection", None))
    if not connection_ok:
        # The verdict was still posted; the unacked message is redelivered by
        # the broker and the backend applies the first verdict only.
        logger.info("Job %s finished after a connection drop; message will be redelivered.", submission_id)
        return

    exc = outcome["exc"]
    if exc is None:
        logger.info("Finished %s -> %s", submission_id, outcome["status"])
    else:  # CallbackRejected, JudgeInfrastructureError or an unexpected bug
        level = logging.WARNING if isinstance(exc, JudgeInfrastructureError) else logging.ERROR
        logger.log(level, "Job %s failed (attempt %d/%d): %s", submission_id, retry_count + 1, MAX_JOB_RETRIES + 1, exc,
                   exc_info=not isinstance(exc, JudgeInfrastructureError))
        if isinstance(exc, CallbackRejected):
            dead_letter(channel, body, properties, f"backend rejected callback: {exc}")
        else:
            reset_sandbox()  # reconnect to Docker on the next attempt
            if retry_count < MAX_JOB_RETRIES:
                time.sleep(min(RETRY_DELAY_SECONDS * (retry_count + 1), 30))
                requeue_with_retry(channel, body, properties, retry_count + 1)
            else:
                _report_internal_error(submission_id, "The judge could not process this submission. Please try again later.")
                dead_letter(channel, body, properties, f"retries exhausted: {exc}")
    channel.basic_ack(delivery_tag=method.delivery_tag)


def connect_with_retry(max_attempts: int = 10, delay: float = 3.0):
    """Retry RabbitMQ connection with linear backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            params = pika.URLParameters(RABBITMQ_URL)
            params.heartbeat = 60
            params.blocked_connection_timeout = 300
            connection = pika.BlockingConnection(params)
            logger.info("Connected to RabbitMQ (attempt %d)", attempt)
            return connection
        except Exception as e:
            logger.warning("RabbitMQ connection attempt %d/%d failed: %s", attempt, max_attempts, e)
            if attempt < max_attempts:
                time.sleep(delay * attempt)
    raise RuntimeError("Could not connect to RabbitMQ after retries")


def main():
    global _running
    logger.info("Judge worker starting — queue: %s (dead letters: %s)", QUEUE_NAME, DEAD_LETTER_QUEUE)
    logger.info("Limits: %ss per test, %s MB", config("CPU_TIME_LIMIT", default=10), config("MEMORY_LIMIT_MB", default=256))

    try:
        from runner import get_sandbox

        sandbox = get_sandbox()
        for lang in LANGUAGE_CONFIG:
            logger.info("Sandbox health %s: %s", lang, "ok" if sandbox.health_check(lang) else "FAILED")
    except Exception as exc:
        logger.error("Sandbox not ready (%s); jobs will be retried until Docker is available.", exc)

    while _running:
        try:
            connection = connect_with_retry()
        except RuntimeError as exc:
            logger.error("%s — retrying in 10 s (stop the worker with Ctrl+C / SIGTERM).", exc)
            time.sleep(10)
            continue
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        channel.queue_declare(queue=DEAD_LETTER_QUEUE, durable=True)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message)
        logger.info("Worker ready — waiting for jobs on '%s'", QUEUE_NAME)
        try:
            while _running:
                connection.process_data_events(time_limit=1)
        except KeyboardInterrupt:
            _running = False
        except Exception as exc:
            # AMQPHeartbeatTimeout, broker restart or a callback error: reconnect
            # instead of exiting. Unacked jobs are redelivered by the broker and
            # the backend applies the first verdict only, so this is safe.
            logger.warning("Connection problem (%s: %s) — reconnecting.", type(exc).__name__, exc)
            time.sleep(1)
        finally:
            try:
                if not connection.is_closed:
                    connection.close()
            except Exception:
                pass
    logger.info("Worker shut down cleanly.")


if __name__ == "__main__":
    main()
