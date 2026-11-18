import pytest

from sandbox import LANGUAGE_CONFIG, STATS_MARKER, _compile_script, _run_script, _split_stats


def test_split_stats_strips_marker_and_parses():
    raw = f"Traceback\nValueError: x\n{STATS_MARKER} rc=1 ns=2500000 mem=10485760 out=12\n"
    stderr, stats = _split_stats(raw)
    assert stderr == "Traceback\nValueError: x"
    assert stats == {"rc": 1, "ns": 2500000, "mem": 10485760, "out": 12}


def test_split_stats_uses_last_marker_and_handles_missing_memory():
    raw = f"{STATS_MARKER} rc=0 ns=1 mem=1 out=1\nprogram noise\n{STATS_MARKER} rc=0 ns=5 mem=- out=0\n"
    stderr, stats = _split_stats(raw)
    assert stats["ns"] == 5 and stats["mem"] is None
    assert "program noise" in stderr


def test_split_stats_absent():
    assert _split_stats("killed") == ("killed", None)


def test_scripts_never_use_root_or_su():
    for config in LANGUAGE_CONFIG.values():
        script = _run_script(config, 5)
        assert "su " not in script and "timeout -k 1 5s" in script and "< /code/stdin.txt" in script
        if config.compile_cmd:
            assert "/build" in _compile_script(config, 30)


# ─── Docker integration (skipped without Docker / images) ────────────────────

pytestmark_docker = pytest.mark.docker

PROGRAMS = {
    "python": "a, b = map(int, input().split())\nprint(a + b)\n",
    "javascript": "const [a,b]=require('fs').readFileSync(0,'utf8').trim().split(/\\s+/).map(Number);console.log(a+b);\n",
    "cpp": "#include <iostream>\nint main(){int a,b;std::cin>>a>>b;std::cout<<a+b<<\"\\n\";}\n",
    "java": "import java.util.*;public class Solution{public static void main(String[] x){Scanner s=new Scanner(System.in);System.out.println(s.nextInt()+s.nextInt());}}\n",
}


@pytest.mark.docker
@pytest.mark.parametrize("language", sorted(PROGRAMS))
def test_compile_once_run_many(docker_sandbox, language):
    prepared = docker_sandbox.prepare(language, PROGRAMS[language])
    try:
        assert prepared.ok, prepared.compile_result
        outputs = [docker_sandbox.execute(prepared, stdin).stdout.strip() for stdin in ("1 2", "40 2")]
        assert outputs == ["3", "42"]
    finally:
        docker_sandbox.cleanup(prepared)


@pytest.mark.docker
def test_java_runtime_exception_is_runtime_error(docker_sandbox):
    src = "public class Solution{public static void main(String[] a){int[] x=new int[1];System.out.println(x[3]);}}"
    r = docker_sandbox.run("java", src, "")
    assert not r.success and not r.compile_error and "ArrayIndexOutOfBoundsException" in r.stderr


@pytest.mark.docker
def test_java_heap_exhaustion_is_memory_limit(docker_sandbox):
    src = ("import java.util.*;public class Solution{public static void main(String[] a){"
           "List<long[]> l=new ArrayList<>();while(true)l.add(new long[1<<20]);}}")
    r = docker_sandbox.run("java", src, "")
    assert r.memory_exceeded and not r.timed_out


@pytest.mark.docker
def test_cpp_infinite_loop_times_out(docker_sandbox):
    r = docker_sandbox.run("cpp", "int main(){volatile long x=0;while(true)x++;}", "")
    assert r.timed_out and r.time_ms <= docker_sandbox.cpu_time_limit * 1000


@pytest.mark.docker
def test_stderr_flood_is_truncated(docker_sandbox):
    r = docker_sandbox.run("python", "import sys\nfor _ in range(200000): sys.stderr.write('e' * 50)\n", "")
    assert len(r.stderr) <= 8192 and r.exit_code == 0


@pytest.mark.docker
def test_no_writes_outside_tmp(docker_sandbox):
    r = docker_sandbox.run("python", "open('/code/hack.txt','w').write('x')\n", "")
    assert not r.success and ("Read-only file system" in r.stderr or "Permission denied" in r.stderr)
