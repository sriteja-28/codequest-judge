import os
import sys

import pytest

JUDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if JUDGE_DIR not in sys.path:
    sys.path.insert(0, JUDGE_DIR)


@pytest.fixture(scope="session")
def docker_sandbox():
    """Real sandbox; skips when Docker or the language images are unavailable."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        missing = [img for img in ("codequest-python", "codequest-cpp", "codequest-java", "codequest-js")
                   if not client.images.list(name=img)]
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Docker not available: {exc}")
    if missing:
        pytest.skip(f"Language images not built: {missing}")
    from sandbox import Sandbox

    return Sandbox(cpu_time_limit=3, memory_limit_mb=256)
