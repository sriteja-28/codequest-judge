"""
sandbox.py — Docker-based code execution sandbox for the CodeQuest judge.

Execution model
---------------
1. prepare(language, code)
     Writes the source into a fresh work dir and, for compiled languages
     (C++, Java), compiles ONCE in a short-lived container into /build.
2. execute(prepared, stdin)   (once per test case)
     Starts a fresh container per test case with /code and /build mounted
     read-only, feeds stdin from /code/stdin.txt and measures the program
     *inside* the container (run time excludes container start-up).
3. cleanup(prepared)
     Removes the work dir.
`run(language, code, stdin)` = prepare + execute + cleanup (one-shot API).

Isolation (each container)
--------------------------
  - runs as uid/gid 65534 (nobody) — never root, no `su`
  - no network, read-only root filesystem, all capabilities dropped,
    no-new-privileges, PID limit, memory limit with swap disabled, CPU quota
  - only writable path is a size-limited tmpfs /tmp (mode 1777)
  - stdout/stderr are captured to tmpfs and truncated (64 KB / 8 KB) before
    they reach the Docker log, so output floods cannot exhaust the worker
  - optional seccomp profile (seccomp_default.json next to this file, Linux)

Containers are a mitigation, not a complete security boundary: the worker
talks to the Docker daemon (root-equivalent on the host). Run workers on
dedicated hosts/VMs; consider gVisor (runsc) or Kata for stronger isolation.

Limits/semantics
----------------
  cpu_time_limit   wall-clock seconds per test case, enforced in-container by
                   coreutils `timeout` (single CPU by default, so wall ≈ CPU)
  memory_limit_mb  container memory cap for the run phase (MLE on OOM kill or
                   language OOM errors such as java.lang.OutOfMemoryError)
  time_ms          program wall time measured inside the container
  memory_kb        container cgroup peak (memory.peak / max_usage_in_bytes);
                   None when the kernel does not expose it
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import docker
from docker.errors import APIError, DockerException, ImageNotFound
from docker.types import LogConfig

logger = logging.getLogger(__name__)

# ─── Platform detection ───────────────────────────────────────────────────────

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# ─── Optional seccomp profile (Linux production only) ────────────────────────
# Drop moby's default profile (or a stricter one) next to this file:
#   https://github.com/moby/moby/blob/master/profiles/seccomp/default.json
# Docker already applies its default profile when none is given.
_HERE = Path(__file__).parent
SECCOMP_PROFILE = _HERE / "seccomp_default.json"

SANDBOX_USER = "65534:65534"  # nobody:nogroup in the Debian/Ubuntu base images
STATS_MARKER = "__CQ_STATS__"
MAX_STDOUT_BYTES = 65_536
MAX_STDERR_BYTES = 8_192
MAX_COMPILE_OUTPUT_BYTES = 16_384
HOST_GRACE_SECONDS = 15  # host-side backstop on top of the in-container limit
SANDBOX_LABEL = {"codequest.sandbox": "1"}

# Base dir for each run's payload (source, stdin, wrapper scripts). The worker
# bind-mounts it into the language containers, and the Docker daemon resolves
# bind sources on the HOST — so when the worker itself runs in a container
# (infra/docker-compose.prod.yml), this must be a directory the host can see
# at the same path; compose mounts it accordingly (SANDBOX_DIR /
# SANDBOX_HOST_DIR in infra/.env.production.example). Unset → system temp dir,
# which is correct when the worker runs directly on the host (dev).
SANDBOX_DIR = os.environ.get("SANDBOX_DIR") or None
if SANDBOX_DIR:
    os.makedirs(SANDBOX_DIR, exist_ok=True)


# ─── Language configuration ───────────────────────────────────────────────────

@dataclass(frozen=True)
class LanguageConfig:
    image: str                      # Docker image tag (judge/languages/<dir>)
    filename: str                   # source file name inside /code
    compile_cmd: Optional[str]      # writes artefacts to /build (None = interpreted)
    run_cmd: str                    # program command; stdin is redirected by the wrapper
    pre_run_cmd: Optional[str] = None   # untimed setup before the program starts
    version_cmd: str = "echo unknown"
    oom_markers: tuple = ()         # stderr fragments meaning "out of memory"
    build_dir: str = ""             # judge/languages/<dir> (for error messages)


LANGUAGE_CONFIG: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        image="codequest-python",
        filename="solution.py",
        compile_cmd=None,
        run_cmd="python3 -u /code/solution.py",
        version_cmd="python3 --version",
        oom_markers=("MemoryError",),
        build_dir="python",
    ),
    "cpp": LanguageConfig(
        image="codequest-cpp",
        filename="solution.cpp",
        compile_cmd="g++ -O2 -std=c++17 -Wall -o /build/sol /code/solution.cpp",
        # Copy to the exec-enabled tmpfs: host temp dirs may be mounted noexec.
        pre_run_cmd="cp /build/sol /tmp/sol",
        run_cmd="/tmp/sol",
        version_cmd="g++ --version",
        oom_markers=("std::bad_alloc",),
        build_dir="cpp",
    ),
    "java": LanguageConfig(
        image="codequest-java",
        filename="Solution.java",
        compile_cmd="javac -encoding UTF-8 -d /build /code/Solution.java",
        run_cmd="java -Xmx200m -XX:+UseSerialGC -XX:-UsePerfData -cp /build Solution",
        version_cmd="java --version",
        oom_markers=("java.lang.OutOfMemoryError",),
        build_dir="java",
    ),
    "javascript": LanguageConfig(
        image="codequest-js",
        filename="solution.js",
        compile_cmd=None,
        run_cmd="node --max-old-space-size=200 /code/solution.js",
        version_cmd="node --version",
        oom_markers=("JavaScript heap out of memory",),
        build_dir="javascript",
    ),
}


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    time_ms: int
    memory_kb: Optional[int]
    timed_out: bool

    # Compile-phase fields (populated for compiled languages)
    compile_error: bool = False
    compile_stderr: str = ""

    memory_exceeded: bool = False   # OOM kill or language-level OOM error
    output_exceeded: bool = False   # stdout larger than MAX_STDOUT_BYTES
    infra_error: bool = False       # Docker/image problem — not the user's fault

    @property
    def success(self) -> bool:
        return (
            self.exit_code == 0 and not self.timed_out and not self.compile_error
            and not self.memory_exceeded and not self.output_exceeded and not self.infra_error
        )

    def __repr__(self) -> str:
        return (
            f"<SandboxResult success={self.success} exit={self.exit_code} time={self.time_ms}ms "
            f"mem={self.memory_kb}KB timed_out={self.timed_out} compile_error={self.compile_error} "
            f"mle={self.memory_exceeded} ole={self.output_exceeded} infra={self.infra_error}>"
        )


@dataclass
class Prepared:
    language: str
    config: LanguageConfig
    workdir: str
    code_dir: str
    build_dir: str
    compile_result: Optional[SandboxResult] = None  # set when compilation failed

    @property
    def ok(self) -> bool:
        return self.compile_result is None


def _infra_result(message: str) -> SandboxResult:
    return SandboxResult("", message, 1, 0, None, False, infra_error=True)


# ─── Wrapper scripts (POSIX sh, executed by /bin/sh inside the container) ────

def _run_script(config: LanguageConfig, limit_s: int) -> str:
    pre = f"{config.pre_run_cmd} || exit 98\n" if config.pre_run_cmd else ""
    return (
        "#!/bin/sh\n"
        f"{pre}"
        "s=$(date +%s%N)\n"
        f"timeout -k 1 {limit_s}s {config.run_cmd} < /code/stdin.txt > /tmp/out.txt 2> /tmp/err.txt\n"
        "rc=$?\n"
        "e=$(date +%s%N)\n"
        "m=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null || echo -)\n"
        "ob=$(wc -c < /tmp/out.txt)\n"
        f"head -c {MAX_STDOUT_BYTES} /tmp/out.txt\n"
        f"head -c {MAX_STDERR_BYTES} /tmp/err.txt >&2\n"
        f"printf '\\n{STATS_MARKER} rc=%s ns=%s mem=%s out=%s\\n' \"$rc\" \"$((e - s))\" \"$m\" \"$ob\" >&2\n"
        "exit $rc\n"
    )


def _compile_script(config: LanguageConfig, limit_s: int) -> str:
    return (
        "#!/bin/sh\n"
        "umask 000\n"  # artefacts must stay deletable by the (non-root) worker
        f"timeout -k 1 {limit_s}s {config.compile_cmd} > /tmp/compile.txt 2>&1\n"
        "rc=$?\n"
        f"head -c {MAX_COMPILE_OUTPUT_BYTES} /tmp/compile.txt\n"
        "exit $rc\n"
    )


_STATS_RE = re.compile(
    rf"^{STATS_MARKER} rc=(?P<rc>-?\d+) ns=(?P<ns>-?\d+) mem=(?P<mem>\S*) out=\s*(?P<out>\d+)\s*$"
)


def _split_stats(stderr: str):
    """Remove the wrapper's stats line (the last one; a program could print a
    fake marker earlier) from stderr. Returns (stderr, stats | None)."""
    idx = stderr.rfind(f"{STATS_MARKER} rc=")
    if idx < 0:
        return stderr, None
    match = _STATS_RE.match(stderr[idx:])
    if not match:
        return stderr, None
    clean = stderr[:idx]
    if clean.endswith("\n"):
        clean = clean[:-1]
    stats = {
        "rc": int(match.group("rc")),
        "ns": int(match.group("ns")),
        "mem": int(match.group("mem")) if match.group("mem").isdigit() else None,
        "out": int(match.group("out")),
    }
    return clean, stats


# ─── Sandbox ──────────────────────────────────────────────────────────────────

class Sandbox:
    """
    One instance per worker process is fine; every execution uses a fresh
    container. Constructor arguments keep the original signature.
    """

    def __init__(
        self,
        cpu_time_limit: int = 10,       # seconds per test case
        memory_limit_mb: int = 256,
        compile_time_limit: Optional[int] = None,
        compile_memory_limit_mb: Optional[int] = None,
        cpus: Optional[float] = None,
    ):
        self.cpu_time_limit = int(cpu_time_limit)
        self.memory_limit_mb = int(memory_limit_mb)
        self.compile_time_limit = int(compile_time_limit or os.environ.get("COMPILE_TIME_LIMIT", 30))
        self.compile_memory_limit_mb = int(compile_memory_limit_mb or os.environ.get("COMPILE_MEMORY_LIMIT_MB", 512))
        self.cpus = float(cpus or os.environ.get("JUDGE_CPUS", 1.0))
        self.client = docker.from_env()
        self._log_env()

    def _log_env(self):
        info = self.client.info()
        logger.info(
            "Docker connected: %s | OS: %s | Arch: %s | cgroup v%s",
            info.get("ServerVersion", "?"), info.get("OSType", "?"),
            info.get("Architecture", "?"), info.get("CgroupVersion", "?"),
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, language: str, code: str, stdin_data: str) -> SandboxResult:
        """One-shot execution (compile + single run). Never raises."""
        prepared = self.prepare(language, code)
        try:
            if isinstance(prepared, SandboxResult):
                return prepared
            if not prepared.ok:
                return prepared.compile_result
            return self.execute(prepared, stdin_data)
        finally:
            if not isinstance(prepared, SandboxResult):
                self.cleanup(prepared)

    def prepare(self, language: str, code: str):
        """Write the source and compile it once. Returns a Prepared handle, or a
        SandboxResult describing an unsupported language / infrastructure error.
        A compile failure is reported via Prepared.compile_result."""
        config = LANGUAGE_CONFIG.get(language)
        if config is None:
            supported = ", ".join(LANGUAGE_CONFIG)
            return SandboxResult("", f"Unsupported language '{language}'. Supported: {supported}", 1, 0, None, False)

        workdir = tempfile.mkdtemp(prefix="cq-judge-", dir=SANDBOX_DIR)
        code_dir = os.path.join(workdir, "code")
        build_dir = os.path.join(workdir, "build")
        os.makedirs(code_dir)
        os.makedirs(build_dir)
        Path(code_dir, config.filename).write_text(code, encoding="utf-8")
        Path(code_dir, "stdin.txt").write_text("", encoding="utf-8")
        Path(code_dir, ".cq_run.sh").write_text(_run_script(config, self.cpu_time_limit), encoding="utf-8", newline="\n")
        if config.compile_cmd:
            Path(code_dir, ".cq_compile.sh").write_text(
                _compile_script(config, self.compile_time_limit), encoding="utf-8", newline="\n"
            )
        # tempfile creates 0700 dirs; the sandbox user (65534) must be able to
        # read /code and write /build on Linux hosts.
        os.chmod(workdir, 0o755)
        os.chmod(code_dir, 0o755)
        os.chmod(build_dir, 0o777)
        for name in os.listdir(code_dir):
            os.chmod(os.path.join(code_dir, name), 0o644)

        prepared = Prepared(language, config, workdir, code_dir, build_dir)
        if config.compile_cmd:
            result = self._compile(prepared)
            if result is not None:
                prepared.compile_result = result
        return prepared

    def execute(self, prepared: Prepared, stdin_data: str) -> SandboxResult:
        """Run the prepared program against one input. Never raises."""
        if not prepared.ok:
            return prepared.compile_result
        Path(prepared.code_dir, "stdin.txt").write_text(stdin_data or "", encoding="utf-8")
        return self._run(prepared)

    def cleanup(self, prepared: Prepared) -> None:
        shutil.rmtree(prepared.workdir, ignore_errors=True)
        if os.path.exists(prepared.workdir):
            logger.warning("Could not fully remove sandbox work dir %s", prepared.workdir)

    def health_check(self, language: str = "python") -> bool:
        programs = {
            "python": "print('ok')\n",
            "javascript": "console.log('ok')\n",
            "cpp": "#include <cstdio>\nint main(){std::puts(\"ok\");}\n",
            "java": "public class Solution{public static void main(String[] a){System.out.println(\"ok\");}}\n",
        }
        result = self.run(language, programs[language], "")
        ok = result.success and result.stdout.strip() == "ok"
        if not ok:
            logger.error("Sandbox health check FAILED for %s: %r %s", language, result, result.stderr[:300])
        return ok

    # ── Container plumbing ──────────────────────────────────────────────────

    def _container_kwargs(self, prepared: Prepared, *, build_mode: str, memory_mb: int,
                          tmpfs_opts: str, pids: int, script: str) -> dict:
        kwargs: dict = {
            "image": prepared.config.image,
            "command": ["/bin/sh", f"/code/{script}"],
            "user": SANDBOX_USER,
            "working_dir": "/tmp",
            "environment": {"HOME": "/tmp", "TMPDIR": "/tmp"},
            "volumes": {
                os.path.abspath(prepared.code_dir): {"bind": "/code", "mode": "ro"},
                os.path.abspath(prepared.build_dir): {"bind": "/build", "mode": build_mode},
            },
            "tmpfs": {"/tmp": tmpfs_opts},
            "read_only": True,
            "network_disabled": True,
            "mem_limit": f"{memory_mb}m",
            "memswap_limit": f"{memory_mb}m",  # no swap
            "nano_cpus": int(self.cpus * 1_000_000_000),
            "pids_limit": pids,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "stdin_open": False,
            "detach": True,
            "labels": SANDBOX_LABEL,
            "log_config": LogConfig(type=LogConfig.types.JSON, config={"max-size": "1m", "max-file": "1"}),
        }
        if IS_LINUX and SECCOMP_PROFILE.exists():
            kwargs["security_opt"].append(f"seccomp={SECCOMP_PROFILE.read_text()}")
        return kwargs

    def _start_and_wait(self, kwargs: dict, timeout_s: int):
        """Returns (container, exit_code, host_timed_out, elapsed_ms)."""
        start = time.monotonic()
        container = self.client.containers.run(**kwargs)
        host_timed_out = False
        exit_code = 1
        try:
            exit_code = container.wait(timeout=timeout_s).get("StatusCode", 1)
        except Exception:
            host_timed_out = True
            try:
                container.kill()
            except Exception:
                pass
        return container, exit_code, host_timed_out, int((time.monotonic() - start) * 1000)

    @staticmethod
    def _remove(container):
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass

    def _image_missing(self, config: LanguageConfig) -> SandboxResult:
        msg = (
            f"Judge image '{config.image}' is not built. "
            f"Run: docker build -t {config.image} judge/languages/{config.build_dir}"
        )
        logger.error(msg)
        return _infra_result(msg)

    def _compile(self, prepared: Prepared) -> Optional[SandboxResult]:
        """Compile into /build. Returns None on success, otherwise a result."""
        container = None
        try:
            kwargs = self._container_kwargs(
                prepared, build_mode="rw", memory_mb=self.compile_memory_limit_mb,
                tmpfs_opts="size=256m,mode=1777", pids=256, script=".cq_compile.sh",
            )
            container, rc, host_timed_out, elapsed_ms = self._start_and_wait(
                kwargs, self.compile_time_limit + HOST_GRACE_SECONDS
            )
            output = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            timed_out = host_timed_out or rc == 124
            if rc == 0 and not timed_out:
                return None
            if timed_out:
                output = (output + f"\nCompilation exceeded {self.compile_time_limit}s.").strip()
            return SandboxResult(
                stdout="", stderr=output, exit_code=rc, time_ms=elapsed_ms, memory_kb=None,
                timed_out=False, compile_error=True, compile_stderr=output,
            )
        except ImageNotFound:
            return self._image_missing(prepared.config)
        except (APIError, DockerException) as exc:
            logger.error("Docker error during compile: %s", exc)
            return _infra_result(f"Internal judge error during compilation: {exc}")
        finally:
            self._remove(container)

    def _run(self, prepared: Prepared) -> SandboxResult:
        config = prepared.config
        container = None
        try:
            kwargs = self._container_kwargs(
                prepared, build_mode="ro", memory_mb=self.memory_limit_mb,
                tmpfs_opts="size=64m,mode=1777,exec", pids=64, script=".cq_run.sh",
            )
            container, rc, host_timed_out, elapsed_ms = self._start_and_wait(
                kwargs, self.cpu_time_limit + HOST_GRACE_SECONDS
            )
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            raw_stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            stderr, stats = _split_stats(raw_stderr)
            try:
                container.reload()
                oom_killed = bool(container.attrs.get("State", {}).get("OOMKilled"))
            except Exception:
                oom_killed = False

            if stats is not None:
                exit_code = stats["rc"]
                time_ms = max(0, stats["ns"] // 1_000_000)
                memory_kb = stats["mem"] // 1024 if stats["mem"] else None
                output_exceeded = stats["out"] > MAX_STDOUT_BYTES
            else:  # wrapper itself was killed (host timeout / OOM of the shell)
                exit_code, time_ms, memory_kb, output_exceeded = rc, elapsed_ms, None, False

            if rc == 98 and stats is None and config.pre_run_cmd:
                return _infra_result(f"Sandbox setup failed: {stderr.strip()[:500]}")

            limit_ms = self.cpu_time_limit * 1000
            timed_out = (
                host_timed_out
                or exit_code == 124
                or (exit_code == 137 and time_ms >= limit_ms - 50)
                or time_ms > limit_ms
            )
            memory_exceeded = not timed_out and (
                oom_killed
                or exit_code == 137
                or any(marker in stderr for marker in config.oom_markers)
            )
            return SandboxResult(
                stdout=stdout,
                stderr=stderr[:MAX_STDERR_BYTES],
                exit_code=exit_code,
                time_ms=min(time_ms, limit_ms) if timed_out else time_ms,
                memory_kb=memory_kb,
                timed_out=timed_out,
                memory_exceeded=memory_exceeded,
                output_exceeded=output_exceeded and not timed_out,
            )
        except ImageNotFound:
            return self._image_missing(config)
        except (APIError, DockerException) as exc:
            logger.error("Docker error during run: %s", exc)
            return _infra_result(f"Internal judge error: {exc}")
        finally:
            self._remove(container)


# ─── Self-test ────────────────────────────────────────────────────────────────
# Run:   python sandbox.py        (requires Docker + the four codequest-* images)

def _assert(condition: bool, label: str, result: SandboxResult):
    icon = "PASS" if condition else "FAIL"
    print(f"  [{icon}] {label}")
    if not condition:
        print(f"     {result!r}")
        print(f"     stdout: {result.stdout[:300]!r}")
        print(f"     stderr: {(result.compile_stderr or result.stderr)[:300]!r}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(f"\nSandbox self-test on {platform.system()} {platform.release()} / Python {sys.version.split()[0]}\n")

    sb = Sandbox(cpu_time_limit=5)
    add = {
        "python": "a, b = map(int, open(0).read().split())\nprint(a + b)\n",
        "javascript": "const [a, b] = require('fs').readFileSync(0, 'utf8').trim().split(/\\s+/).map(Number);\nconsole.log(a + b);\n",
        "cpp": "#include <iostream>\nint main(){long long a,b;std::cin>>a>>b;std::cout<<a+b<<std::endl;}\n",
        "java": "import java.util.*;\npublic class Solution{public static void main(String[] x){Scanner s=new Scanner(System.in);System.out.println(s.nextLong()+s.nextLong());}}\n",
    }
    for lang, src in add.items():
        r = sb.run(lang, src, "4 6\n")
        _assert(r.success and r.stdout.strip() == "10", f"{lang}: stdin -> stdout (time={r.time_ms}ms mem={r.memory_kb}KB)", r)

    r = sb.run("python", "raise ValueError('boom')\n", "")
    _assert(not r.success and r.exit_code != 0 and "boom" in r.stderr and not r.timed_out, "python: runtime error", r)
    r = sb.run("cpp", "int main() { this is not c++; }\n", "")
    _assert(r.compile_error and "error" in r.compile_stderr, "cpp: compile error", r)
    r = sb.run("java", "public class Solution { void x( }\n", "")
    _assert(r.compile_error, "java: compile error", r)
    r = Sandbox(cpu_time_limit=2).run("python", "while True: pass\n", "")
    _assert(r.timed_out, f"python: time limit (time={r.time_ms}ms)", r)
    r = sb.run("python", "x = bytearray(1024 * 1024 * 1024)\n", "")
    _assert(r.memory_exceeded, f"python: memory limit (exit={r.exit_code})", r)
    r = sb.run("python", "import sys\nsys.stdout.write('x' * 200000)\n", "")
    _assert(r.output_exceeded, "python: output limit", r)
    r = sb.run("python", "import os\nprint(os.getuid())\n", "")
    _assert(r.success and r.stdout.strip() == "65534", "python: runs as nobody", r)
    r = sb.run("python", "import urllib.request\nurllib.request.urlopen('http://1.1.1.1', timeout=3)\n", "")
    _assert(not r.success, "python: no network", r)
    print("\nAll sandbox self-tests passed.\n")
