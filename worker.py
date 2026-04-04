"""
worker.py — RabbitMQ consumer that processes code submission jobs.

Run one or more workers:
    python worker.py               # single worker
    python worker.py &             # background; repeat for N workers

Each worker:
1. Connects to RabbitMQ
2. Declares the durable queue (idempotent)
3. Fetches one job at a time (prefetch=1 for fairness)
4. Calls runner.run_submission()
5. ACKs the message on success, NACKs (requeue=False) on permanent failure

Horizontal scaling: just launch more worker processes. RabbitMQ will
round-robin jobs across workers automatically.
"""

import json
import logging
import os
import signal
import sys
import time

import pika
from decouple import config

from runner import run_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("judge.worker")

RABBITMQ_URL = config("RABBITMQ_URL", default="amqp://guest:guest@localhost:5672/")
QUEUE_NAME = config("JUDGE_QUEUE", default="code_submissions")

_running = True


def _sigterm_handler(signum, frame):
    global _running
    logger.info("Received SIGTERM — finishing current job then shutting down.")
    _running = False


signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)


def on_message(channel, method, properties, body):
    """Called for each message from RabbitMQ."""
    try:
        job = json.loads(body)
        logger.info(f"Received job: submission_id={job.get('submission_id')} lang={job.get('language')}")

        run_submission(
            submission_id=job["submission_id"],
            problem_id=job["problem_id"],
            language=job["language"],
            code=job["code"],
            run_sample=job.get("run_sample", False), 
        )

        # ACK — remove from queue
        channel.basic_ack(delivery_tag=method.delivery_tag)
        logger.info(f"ACK'd job {job.get('submission_id')}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in message: {e}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except KeyError as e:
        logger.error(f"Missing field in job payload: {e}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except Exception as e:
        logger.exception(f"Unexpected error processing job: {e}")
        # Requeue once for transient errors (e.g. Docker not ready)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


def connect_with_retry(max_attempts: int = 10, delay: float = 3.0):
    """Retry RabbitMQ connection with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            params = pika.URLParameters(RABBITMQ_URL)
            params.heartbeat = 60
            params.blocked_connection_timeout = 300
            connection = pika.BlockingConnection(params)
            logger.info(f"Connected to RabbitMQ (attempt {attempt})")
            return connection
        except Exception as e:
            logger.warning(f"RabbitMQ connection attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(delay * attempt)
    raise RuntimeError("Could not connect to RabbitMQ after retries")


def main():
    logger.info(f"Judge worker starting — queue: {QUEUE_NAME}")
    logger.info(f"CPU limit: {config('CPU_TIME_LIMIT', 10)}s | Memory: {config('MEMORY_LIMIT_MB', 256)}MB")

    connection = connect_with_retry()
    channel = connection.channel()

    # Declare durable queue (idempotent — safe to call multiple times)
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    # Process one message at a time per worker (fair dispatch)
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message)

    logger.info(f"✅ Worker ready — waiting for jobs on '{QUEUE_NAME}'")

    try:
        while _running:
            connection.process_data_events(time_limit=1)
    except Exception as e:
        logger.error(f"Worker error: {e}")
    finally:
        if not connection.is_closed:
            connection.close()
        logger.info("Worker shut down cleanly.")


if __name__ == "__main__":
    main()