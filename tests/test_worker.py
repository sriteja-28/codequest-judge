import json
import time
import uuid
from types import SimpleNamespace

import pika
import pytest

import worker


class FakeChannel:
    def __init__(self):
        self.acks, self.published = [], []

    def basic_ack(self, delivery_tag):
        self.acks.append(delivery_tag)

    def basic_publish(self, exchange, routing_key, body, properties):
        self.published.append({"queue": routing_key, "body": body, "headers": dict(properties.headers or {})})


def _job(**overrides):
    job = {"submission_id": str(uuid.uuid4()), "problem_id": 1, "language": "python", "code": "print(1)", "run_sample": False}
    job.update(overrides)
    return job


def _deliver(channel, job, headers=None):
    body = job if isinstance(job, bytes) else json.dumps(job).encode()
    worker.on_message(channel, SimpleNamespace(delivery_tag=7), SimpleNamespace(headers=headers, message_id=None), body)


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(worker, "_post_callback", lambda sid, payload, retries=2: calls.append((sid, payload)) or True)
    return calls


@pytest.mark.parametrize("job,reason", [
    ([1, 2], "JSON object"),
    (_job(submission_id="nope"), "UUID"),
    (_job(problem_id="1"), "problem_id"),
    (_job(language="ruby"), "language"),
    (_job(code="  "), "code"),
    (_job(code="x" * 70000), "exceeds"),
    (_job(run_sample="yes"), "run_sample"),
])
def test_validate_job_rejects_bad_payloads(job, reason):
    assert reason in worker.validate_job(job)


def test_valid_job_runs_and_acks(monkeypatch):
    seen = []
    monkeypatch.setattr(worker, "run_submission", lambda **kw: seen.append(kw) or "ACCEPTED")
    ch = FakeChannel()
    _deliver(ch, _job())
    assert ch.acks == [7] and ch.published == [] and seen[0]["language"] == "python"


def test_invalid_json_is_dead_lettered_not_requeued():
    ch = FakeChannel()
    _deliver(ch, b"{not json")
    assert ch.acks == [7]
    assert ch.published[0]["queue"] == worker.DEAD_LETTER_QUEUE


def test_invalid_job_reports_internal_error_and_dead_letters(_fast):
    ch = FakeChannel()
    job = _job(language="ruby")
    _deliver(ch, job)
    assert ch.acks == [7] and ch.published[0]["queue"] == worker.DEAD_LETTER_QUEUE
    assert _fast[0][0] == job["submission_id"] and _fast[0][1]["status"] == "INTERNAL_ERROR"


def test_infrastructure_failure_is_retried_with_counter(monkeypatch):
    def boom(**kw):
        raise worker.JudgeInfrastructureError("docker down")

    monkeypatch.setattr(worker, "run_submission", boom)
    ch = FakeChannel()
    _deliver(ch, _job(), headers={worker.RETRY_HEADER: 1})
    assert ch.acks == [7]
    assert ch.published[0]["queue"] == worker.QUEUE_NAME
    assert ch.published[0]["headers"][worker.RETRY_HEADER] == 2


def test_retries_exhausted_reports_internal_error(monkeypatch, _fast):
    def boom(**kw):
        raise RuntimeError("bug")

    monkeypatch.setattr(worker, "run_submission", boom)
    ch = FakeChannel()
    job = _job()
    _deliver(ch, job, headers={worker.RETRY_HEADER: worker.MAX_JOB_RETRIES})
    assert ch.acks == [7]
    assert [p["queue"] for p in ch.published] == [worker.DEAD_LETTER_QUEUE]
    assert _fast[-1][1]["status"] == "INTERNAL_ERROR"


def test_rejected_callback_is_not_retried(monkeypatch):
    def rejected(**kw):
        raise worker.CallbackRejected("HTTP 400")

    monkeypatch.setattr(worker, "run_submission", rejected)
    ch = FakeChannel()
    _deliver(ch, _job())
    assert ch.acks == [7] and [p["queue"] for p in ch.published] == [worker.DEAD_LETTER_QUEUE]


# ─── heartbeats during judging + reconnect ────────────────────────────────────


class FakeConnection:
    def __init__(self, fail_after=0, on_fail=None):
        self.processed = 0
        self.fail_after = fail_after
        self.on_fail = on_fail
        self.closed = False

    def process_data_events(self, time_limit):
        self.processed += 1
        if self.processed > self.fail_after:
            if self.on_fail:
                self.on_fail()
            raise pika.exceptions.AMQPHeartbeatTimeout("no heartbeats")
        time.sleep(0.01)

    def close(self):
        self.closed = True


class ConnectedChannel(FakeChannel):
    def __init__(self, connection):
        super().__init__()
        self.connection = connection


def test_heartbeat_is_served_while_a_long_job_runs(monkeypatch):
    import threading

    monkeypatch.setattr(worker, "HEARTBEAT_SLICE", 0.05)
    release = threading.Event()

    def slow_run(**kw):
        release.wait(0.5)  # judging takes a while...
        return "ACCEPTED"

    monkeypatch.setattr(worker, "run_submission", slow_run)
    conn = FakeConnection(fail_after=10**9)  # never drops
    ch = ConnectedChannel(conn)
    _deliver(ch, _job())
    release.set()
    assert ch.acks == [7] and ch.published == []
    assert conn.processed >= 2  # ...but the connection was served meanwhile


def test_connection_drop_mid_job_finishes_judging_without_ack(monkeypatch):
    import threading

    monkeypatch.setattr(worker, "HEARTBEAT_SLICE", 0.05)
    release = threading.Event()
    seen = []

    def slow_run(**kw):
        release.wait(2)  # judging is still running when the connection drops
        seen.append(kw)
        return "ACCEPTED"

    monkeypatch.setattr(worker, "run_submission", slow_run)
    conn = FakeConnection(fail_after=1, on_fail=release.set)  # dies mid-job
    ch = ConnectedChannel(conn)
    _deliver(ch, _job())
    assert seen and ch.acks == [] and ch.published == []  # no ack: broker redelivers
    assert conn.closed is False  # the consumer thread did not touch it again


def test_main_reconnects_after_connection_error(monkeypatch):
    import runner

    worker._running = True
    monkeypatch.setattr(worker, "_running", True)
    conns, attempts = [], {"n": 0}

    def fake_connect(max_attempts=10, delay=3.0):
        attempts["n"] += 1
        conn = SimpleNamespace(
            is_closed=False,
            fail=attempts["n"] == 1,  # first connection dies, second is clean
            channel=lambda: SimpleNamespace(
                queue_declare=lambda **kw: None,
                basic_qos=lambda **kw: None,
                basic_consume=lambda **kw: None,
            ),
            close=lambda: None,
        )

        def process_data_events(time_limit):
            if conn.fail:
                conn.fail = False
                raise pika.exceptions.AMQPHeartbeatTimeout("no heartbeats")
            worker._running = False  # second connection: shut down cleanly

        conn.process_data_events = process_data_events
        conns.append(conn)
        return conn

    def no_sandbox():
        raise RuntimeError("no docker in tests")

    monkeypatch.setattr(runner, "get_sandbox", no_sandbox)
    monkeypatch.setattr(worker, "connect_with_retry", fake_connect)
    monkeypatch.setattr(worker.time, "sleep", lambda s: None)
    worker.main()
    assert attempts["n"] == 2  # the worker reconnected instead of exiting
    assert worker._running is False
