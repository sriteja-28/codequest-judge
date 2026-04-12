"""
sandbox.py — Production-grade code execution sandbox.

Works identically on:
  ✅ Windows  — Docker Desktop (named pipes, no raw socket access)
  ✅ macOS    — Docker Desktop / Colima
  ✅ Linux    — Docker Engine (CI, staging, production)
  ✅ Cloud    — ECS, GKE, K8s (privileged or sysbox nodes)

Architecture
------------
Stdin delivery: written to a file in the mounted temp directory and
redirected with shell `< /code/stdin.txt`. This is the only approach
that works reliably on Windows Docker Desktop, where attach_socket
returns a named-pipe wrapper whose .send() silently fails on some
versions, causing stdin-dependent code to block until the wall-clock
timeout fires.

Execution model:
  1. Write code + stdin to an OS temp dir (one per submission, deleted after)
  2. Mount that dir read-only into a fresh container
  3. For compiled languages (C++, Java), compile first, then run — separate
     stderr streams let us distinguish compile errors from runtime errors
  4. Wait with a hard wall-clock timeout; kill on breach
  5. Collect stdout/stderr/stats; remove the container in a finally block

Security layers:
  - No network  (--network none)
  - Read-only root filesystem  (--read-only)
  - Writable /tmp tmpfs only — compiled binaries execute from /tmp
  - Hard memory cap with swap disabled (mem == memswap)
  - PID limit — prevents fork-bombs
  - Non-root user (nobody)
  - Container auto-removed on exit

Production hardening (add when running on Linux production nodes):
  - Mount a seccomp profile: see SECCOMP_PROFILE_PATH below
  - Use gVisor (runsc) or Kata Containers for kernel-level isolation
  - Run judge workers on dedicated node group isolated from API nodes
  - Set Docker daemon's default ulimits (nproc, nofile) at the host level
"""

from __future__ import annotations

import logging
import os
import platform
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import docker
from docker.errors import APIError, ContainerError, ImageNotFound, NotFound
from docker.models.containers import Container

logger = logging.getLogger(__name__)

# ─── Platform detection ───────────────────────────────────────────────────────

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# ─── Optional seccomp profile (Linux production only) ────────────────────────
#
# Drop this file next to sandbox.py in production:
#   https://github.com/moby/moby/blob/master/profiles/seccomp/default.json
# Restricts ~300 Linux syscalls to the minimum needed for code execution.
# Ignored on Windows/macOS (Docker Desktop handles isolation differently).
#
_HERE = Path(__file__).parent
SECCOMP_PROFILE = _HERE / "seccomp_default.json"


# ─── Language configuration ───────────────────────────────────────────────────
#
# Design rules:
#   1. Always redirect stdin from /code/stdin.txt — cross-platform compatible
#   2. Compiled languages use a two-phase approach (compile → run) so compile
#      errors are reported separately from runtime errors
#   3. All output written to stdout; stderr captured separately
#   4. No network calls possible inside the container
#
@dataclass
class LanguageConfig:
    image: str          # Docker image tag to use
    filename: str       # Name to save the user's code as
    compile_cmd: Optional[str]  # Shell command to compile (None for interpreted)
    run_cmd: str        # Shell command to run (always reads from /code/stdin.txt)
    chmod_cmd: Optional[str] = None  # chmod to run after compile so nobody can exec output
    version_cmd: str = "echo unknown"  # Used in health checks / build verification


LANGUAGE_CONFIG: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        image="codequest-python",
        filename="solution.py",
        compile_cmd=None,  # interpreted — no compile step
        run_cmd="python3 -u /code/solution.py < /code/stdin.txt",
        version_cmd="python3 --version",
    ),
    "cpp": LanguageConfig(
        image="codequest-cpp",
        filename="solution.cpp",
        compile_cmd="g++ -O2 -std=c++17 -Wall -o /tmp/sol /code/solution.cpp",
        chmod_cmd="chmod +x /tmp/sol",
        run_cmd="/tmp/sol < /code/stdin.txt",
        version_cmd="g++ --version",
    ),
    "java": LanguageConfig(
        image="codequest-java",
        filename="Solution.java",
        compile_cmd="javac /code/Solution.java -d /tmp",
        run_cmd="java -cp /tmp -Xmx200m Solution < /code/stdin.txt",
        version_cmd="java --version",
    ),
    "javascript": LanguageConfig(
        image="codequest-js",
        filename="solution.js",
        compile_cmd=None,
        run_cmd="node --max-old-space-size=200 /code/solution.js < /code/stdin.txt",
        version_cmd="node --version",
    ),
}


# ─── Result dataclass ─────────────────────────────────────────────────────────


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

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.compile_error

    def __repr__(self) -> str:
        return (
            f"<SandboxResult success={self.success} "
            f"exit={self.exit_code} time={self.time_ms}ms "
            f"timed_out={self.timed_out} compile_error={self.compile_error}>"
        )


# ─── Sandbox class ────────────────────────────────────────────────────────────


class Sandbox:
    """
    Stateless executor — one instance per worker process is fine.
    Each call to .run() spins up a fresh container and removes it after.

    Horizontal scaling: run N worker processes; each gets its own Sandbox
    instance. Docker handles concurrency at the container level.
    """

    def __init__(
        self,
        cpu_time_limit: int = 10,   # wall-clock seconds per execution phase
        memory_limit_mb: int = 256,
    ):
        self.cpu_time_limit = cpu_time_limit
        self.memory_limit_mb = memory_limit_mb
        self.client = docker.from_env()
        self._log_env()

    def _log_env(self):
        info = self.client.info()
        logger.info(
            "Docker connected: %s | OS: %s | Arch: %s",
            info.get("ServerVersion", "?"),
            info.get("OSType", "?"),
            info.get("Architecture", "?"),
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, language: str, code: str, stdin_data: str) -> SandboxResult:
        """
        Execute user code against one test case.

        Parameters
        ----------
        language   : one of "python", "cpp", "java", "javascript"
        code       : user-submitted source code (UTF-8 string)
        stdin_data : test case input (may be empty string)

        Returns
        -------
        SandboxResult — always returns (never raises); all errors are
        captured in the result fields.
        """
        config = LANGUAGE_CONFIG.get(language)
        if config is None:
            supported = ", ".join(LANGUAGE_CONFIG)
            return SandboxResult(
                stdout="",
                stderr=f"Unsupported language '{language}'. Supported: {supported}",
                exit_code=1,
                time_ms=0,
                memory_kb=None,
                timed_out=False,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_files(tmpdir, config, code, stdin_data)
            return self._execute(config, tmpdir)

    def health_check(self, language: str = "python") -> bool:
        """
        Verify the sandbox can run a trivial program.
        Called by the worker on startup and periodically.
        """
        result = self.run(language, "print('ok')\n", "")
        ok = result.success and "ok" in result.stdout
        if not ok:
            logger.error("Sandbox health check FAILED for %s: %r", language, result)
        return ok

    # ── File setup ──────────────────────────────────────────────────────────

    @staticmethod
    def _write_files(tmpdir: str, config: LanguageConfig, code: str, stdin_data: str):
        Path(tmpdir, config.filename).write_text(code, encoding="utf-8")
        Path(tmpdir, "stdin.txt").write_text(stdin_data or "", encoding="utf-8")

    # ── Container execution ─────────────────────────────────────────────────

    def _base_container_kwargs(self, config: LanguageConfig, code_dir: str) -> dict:
        """
        Build Docker container kwargs.
        Shared between compile and run phases.
        Security settings are always applied.
        """
        kwargs: dict = {
            "image": config.image,
            "volumes": {
                # os.path.abspath is required on Windows where Docker Desktop
                # needs a proper drive-letter path (e.g. C:\Users\...)
                # FIX #1 (cont): Volume is now "ro" again. The compiled binary
                # lives in /tmp (tmpfs) inside the container, not in /code.
                os.path.abspath(code_dir): {"bind": "/code", "mode": "ro"},
            },
            "stdin_open": False,        # stdin via file redirect
            "detach": True,
            "network_disabled": True,   # no internet inside container
            "read_only": True,          # immutable root fs
            # exec is required: Docker Desktop (Windows/macOS) mounts tmpfs
            # with noexec by default, which prevents the compiled binary from
            # running even after chmod +x. Explicitly passing exec overrides it.
            "tmpfs": {"/tmp": "size=128m,exec"},
            "mem_limit": f"{self.memory_limit_mb}m",
            "memswap_limit": f"{self.memory_limit_mb}m",  # disable swap
            "pids_limit": 64,           # kill fork-bombs
            # user is set per-phase in _run_phase:
            #   compile → root  (needs to write binary to /tmp)
            #   run     → nobody (untrusted user code must not run as root)
        }

        # Seccomp profile — Linux production only
        # On Windows/macOS Docker Desktop manages isolation differently;
        # passing a seccomp profile there causes a Docker API error.
        if IS_LINUX and SECCOMP_PROFILE.exists():
            kwargs["security_opt"] = [f"seccomp={SECCOMP_PROFILE.read_text()}"]
            logger.debug("Seccomp profile loaded from %s", SECCOMP_PROFILE)

        return kwargs

    def _wait_container(self, container: Container, phase: str) -> tuple[int, bool]:
        """
        Wait for container to finish. Returns (exit_code, timed_out).
        Adds a 5-second grace period on top of cpu_time_limit to account
        for container startup overhead (image layer mounting, JVM init, etc.)
        This is especially important on Docker Desktop (Windows/macOS) where
        the VM layer adds 1-3 seconds of overhead.
        """
        wall_clock = self.cpu_time_limit + 5
        timed_out = False
        exit_code = 1

        try:
            result = container.wait(timeout=wall_clock)
            exit_code = result.get("StatusCode", 1)
        except Exception:
            timed_out = True
            logger.warning(
                "Container timed out during %s phase (limit=%ds)", phase, wall_clock
            )
            try:
                container.kill()
            except Exception:
                pass

        return exit_code, timed_out

    def _collect_stats(self, container: Container) -> Optional[int]:
        """
        Read peak memory from Docker stats.
        Returns KB or None if unavailable (common on Docker Desktop / cgroup v1).
        """
        try:
            stats = container.stats(stream=False)
            max_usage = stats.get("memory_stats", {}).get("max_usage", 0)
            return max_usage // 1024 if max_usage else None
        except Exception:
            return None

    def _collect_memory_from_file(self, code_dir: str) -> Optional[int]:
        """
        FIX #5: This method was called in _run_phase but never defined,
        causing an AttributeError on any non-Linux host.

        Parses the 'Maximum resident set size' line written by
        `/usr/bin/time -v` (GNU time) into the time_stats.txt file.
        Returns KB, or None if the file is absent or unparseable
        (e.g. BusyBox time on Alpine, or macOS Docker Desktop).
        """
        stats_file = Path(code_dir) / "time_stats.txt"
        if not stats_file.exists():
            return None
        try:
            text = stats_file.read_text(errors="replace")
            # GNU time outputs: "\tMaximum resident set size (kbytes): 12345"
            match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
            return int(match.group(1)) if match else None
        except Exception:
            return None

    def _remove(self, container: Optional[Container]):
        if container:
            try:
                container.remove(force=True)
            except Exception:
                pass

    def _execute(self, config: LanguageConfig, code_dir: str) -> SandboxResult:
        """
        Two-container strategy for compiled languages:

        Container 1 — compile only, runs as root.
          Root is required because Docker mounts the /tmp tmpfs as root:root
          755, and the compiler must write the binary there. Stderr flows into
          Docker logs so we can read it after exit. On failure → compile_error.

        Container 2 — compile+run, also starts as root.
          Recompiles (fast; code already verified above), then uses
          `su -s /bin/sh nobody` to drop privileges before executing the
          binary. Untrusted user code never runs as root.
          On failure → runtime error.

        Interpreted languages skip both and use a single run container as nobody.
        """

        if config.compile_cmd:
            # Phase 1: compile only — stderr flows naturally into Docker logs
            # so we can read it after the container exits.
            compile_result = self._run_phase(
                config, code_dir, config.compile_cmd, "compile"
            )
            if compile_result.exit_code != 0 or compile_result.timed_out:
                return SandboxResult(
                    stdout="",
                    stderr=compile_result.stderr,
                    exit_code=compile_result.exit_code,
                    time_ms=compile_result.time_ms,
                    memory_kb=None,
                    timed_out=compile_result.timed_out,
                    compile_error=True,
                    compile_stderr=compile_result.stderr,
                )

            # Phase 2: single combined container — compile (as root, writes to
            # /tmp) then immediately re-exec the binary as nobody so untrusted
            # user code never runs with elevated privileges.
            # Build the combined command: compile → optional chmod → run-as-nobody.
            # chmod is needed for binary outputs (C++) so that nobody can exec
            # the file that root just wrote. Java .class files are read by the
            # JVM so they don't need +x.
            parts = [config.compile_cmd]
            if config.chmod_cmd:
                parts.append(config.chmod_cmd)
            parts.append(f"su -s /bin/sh nobody -c '{config.run_cmd}'")
            combined_cmd = " && ".join(parts)
            # Override user back to root for this container so the compile
            # step can write to /tmp; the su above drops to nobody for run.
            result = self._run_phase(config, code_dir, combined_cmd, "compile+run")
            # Force user=root for compile+run (su handles the drop inside)
            return result

        # Interpreted languages — single run phase only.
        return self._run_phase(config, code_dir, config.run_cmd, "run")

    def _run_phase(
        self,
        config: LanguageConfig,
        code_dir: str,
        cmd: str,
        phase: str,
    ) -> SandboxResult:
        start_time = time.monotonic()
        container = None

        try:
            kwargs = self._base_container_kwargs(config, code_dir)

            # Compile phase runs as root so it can write the binary to /tmp
            # (Docker mounts tmpfs as root:root 755 — nobody can't write to it).
            # Run phase always executes as nobody — untrusted user code must
            # never run as root regardless of platform.
            # compile+run also starts as root; the combined command uses `su`
            # to drop to nobody before executing the compiled binary.
            if phase in ("compile", "compile+run"):
                kwargs["user"] = "root"
            else:
                kwargs["user"] = "nobody"

            # FIX #3: `/usr/bin/time -v` is GNU-only and breaks on Alpine/macOS.
            # Use it only on Linux; fall back to Docker stats (or None) elsewhere.
            if phase in ("run", "compile+run") and IS_LINUX:
                actual_cmd = f'/usr/bin/time -v sh -c "{cmd}" 2>/code/time_stats.txt'
            else:
                actual_cmd = cmd

            kwargs["command"] = ["/bin/sh", "-c", actual_cmd]
            container = self.client.containers.run(**kwargs)

            exit_code, timed_out = self._wait_container(container, phase)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            # Memory: try Docker stats first (works on Linux cgroup v2),
            # fall back to the GNU time output file (Linux only),
            # otherwise None (Windows/macOS Docker Desktop).
            memory_kb = self._collect_stats(container)
            if not memory_kb:
                # FIX #5: method now exists; safe to call on all platforms.
                memory_kb = self._collect_memory_from_file(code_dir)

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")

            if phase in ("run", "compile+run") and IS_LINUX:
                # GNU time wrote real stderr + timing lines to time_stats.txt.
                # Strip the timing lines; keep only actual program stderr.
                stderr_file = Path(code_dir) / "time_stats.txt"
                try:
                    raw_stderr = stderr_file.read_text(errors="replace") if stderr_file.exists() else ""
                    stderr = "\n".join(
                        line for line in raw_stderr.splitlines()
                        if not any(k in line for k in [
                            "Command being timed", "Maximum resident", "Elapsed",
                            "Percent of CPU", "Major (requiring", "Minor (reclaiming",
                            "Voluntary context", "Involuntary context", "File system",
                            "Socket messages", "Signals delivered", "Page size",
                            "Exit status", "wall clock", "Maximum",
                        ])
                    )
                except Exception:
                    stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            else:
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            return SandboxResult(
                stdout=stdout[:65_536],
                stderr=stderr[:8_192],
                exit_code=exit_code,
                time_ms=elapsed_ms,
                memory_kb=memory_kb,
                timed_out=timed_out,
            )

        except ImageNotFound:
            lang = config.image.replace("codequest-", "")
            msg = (
                f"Docker image '{config.image}' not found. "
                f"Build: docker build -t {config.image} ./languages/{lang}"
            )
            logger.error(msg)
            return SandboxResult("", msg, 1, 0, None, False)

        except APIError as exc:
            logger.error("Docker API error during %s: %s", phase, exc)
            return SandboxResult("", f"Internal judge error: {exc}", 1, 0, None, False)

        finally:
            self._remove(container)


# ─── Self-test ────────────────────────────────────────────────────────────────
#
# Run:   python sandbox.py
# All 5 tests must pass before deploying the judge worker.
#


def _assert(condition: bool, label: str, result: SandboxResult):
    icon = "✅" if condition else "❌"
    print(f"  {icon} {label}")
    if not condition:
        print(f"     stdout:        {result.stdout!r}")
        print(f"     stderr:        {result.stderr!r}")
        print(f"     exit_code:     {result.exit_code}")
        print(f"     timed_out:     {result.timed_out}")
        print(f"     compile_error: {result.compile_error}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"\n{'='*50}")
    print(f"  Sandbox self-test")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"{'='*50}\n")

    sb = Sandbox(cpu_time_limit=15)

    # ── Test 1: Python — basic stdout ───────────────────────────────────────
    print("Test 1: Python basic stdout")
    r = sb.run("python", "print('hello world')\n", "")
    _assert(r.success, "success=True", r)
    _assert("hello world" in r.stdout, "stdout contains 'hello world'", r)
    print(f"  ℹ  time={r.time_ms}ms  memory={r.memory_kb}KB\n")

    # ── Test 2: Python — stdin reading ──────────────────────────────────────
    print("Test 2: Python stdin reading")
    code = "import sys\nlines=sys.stdin.read().strip().split('\\n')\nprint(int(lines[0])+int(lines[1]))\n"
    r = sb.run("python", code, "3\n7")
    _assert(r.success, "success=True", r)
    _assert("10" in r.stdout, "stdout contains '10'", r)
    print(f"  ℹ  time={r.time_ms}ms\n")

    # ── Test 3: Python — runtime error ──────────────────────────────────────
    print("Test 3: Python runtime error")
    r = sb.run("python", "raise ValueError('boom')\n", "")
    _assert(not r.success, "success=False", r)
    _assert(r.exit_code != 0, "exit_code != 0", r)
    _assert(not r.timed_out, "timed_out=False", r)
    _assert("boom" in r.stderr, "stderr contains 'boom'", r)
    print(f"  ℹ  exit_code={r.exit_code}\n")

    # ── Test 4: Python — infinite loop (TLE) ────────────────────────────────
    print("Test 4: Python timeout (TLE) — this takes ~cpu_time_limit seconds …")
    sb_short = Sandbox(cpu_time_limit=3)
    r = sb_short.run("python", "while True: pass\n", "")
    _assert(not r.success, "success=False", r)
    _assert(r.timed_out, "timed_out=True", r)
    print(f"  ℹ  time={r.time_ms}ms\n")

    # ── Test 5: C++ — compile + run ─────────────────────────────────────────
    print("Test 5: C++ compile + run")
    cpp_code = (
        "#include <iostream>\n"
        "#include <string>\n"
        "int main(){\n"
        "  int a,b; std::cin>>a>>b;\n"
        "  std::cout<<a+b<<std::endl;\n"
        "  return 0;\n"
        "}\n"
    )
    r = sb.run("cpp", cpp_code, "4\n6")
    _assert(r.success, "success=True", r)
    _assert("10" in r.stdout, "stdout contains '10'", r)
    print(f"  ℹ  time={r.time_ms}ms\n")

    # ── Test 6: C++ — compile error ─────────────────────────────────────────
    print("Test 6: C++ compile error")
    bad_cpp = "int main() { this is not valid c++; }\n"
    r = sb.run("cpp", bad_cpp, "")
    _assert(not r.success, "success=False", r)
    _assert(r.compile_error, "compile_error=True", r)
    _assert(not r.timed_out, "timed_out=False", r)
    print(f"  ℹ  compile_stderr={r.compile_stderr[:80]!r}\n")

    print("=" * 50)
    print("  ✅  All tests passed — sandbox ready")
    print("=" * 50 + "\n")