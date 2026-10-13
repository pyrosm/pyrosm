"""Run a command in its own process and measure its wall-clock time and peak memory.

The memory of the process and of its child processes is sampled every 50 ms and summed: on
macOS their physical footprint (what Activity Monitor shows; resident memory leaves out the
compressed and swapped pages of a large read), elsewhere their resident memory. The command
runs in its own process group, which is stopped when the run exceeds its timeout, when the
caller is interrupted, and after the run, so no process of one run outlives it.

Each run is logged to benchmark.log in the working directory (its start, every 30 seconds its
time and memory so far, and its end), so a long run can be followed with
`tail -f benchmark.log`. cool_down() pauses after a run, so that the heat of one run does not
slow down the next.

benchmarks_scaling.ipynb imports run_monitored() and cool_down(); osmextract_benchmark_scaling.ipynb
calls this file, which pauses after the run with --cool-down and prints the result as JSON:

    python run_monitored.py [--label <label>] [--cool-down] <timeout seconds> <command> [<argument> ...]
"""

import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import psutil

SAMPLE_SECONDS = 0.05
LOG_FILE = "benchmark.log"
LOG_EVERY_SECONDS = 30
# A failed command's last error line is read in chunks of this size, and a longer line keeps
# this much of its start and of its end.
ERROR_CHUNK_BYTES = 4096
# The pause after a run: half as long as the run took, at least 2 s and at most 5 minutes.
COOLDOWN_RATIO = 0.5
COOLDOWN_MIN = 2
COOLDOWN_MAX = 300

if sys.platform == "darwin":
    _LIBC = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

    class _RusageInfoV4(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
            ("ri_" + name, ctypes.c_uint64)
            for name in (
                "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
                "resident_size phys_footprint proc_start_abstime proc_exit_abstime "
                "child_user_time child_system_time child_pkg_idle_wkups "
                "child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread "
                "diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance "
                "cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy "
                "cpu_time_qos_user_initiated cpu_time_qos_user_interactive "
                "billed_system_time serviced_system_time logical_writes "
                "lifetime_max_phys_footprint instructions cycles billed_energy "
                "serviced_energy interval_max_phys_footprint runnable_time"
            ).split()
        ]

    def process_memory(pid):
        """Physical footprint of process ``pid`` in bytes, 0 once it has exited."""
        info = _RusageInfoV4()
        if _LIBC.proc_pid_rusage(pid, 4, ctypes.byref(info)) != 0:
            return 0
        return info.ri_phys_footprint

else:

    def process_memory(pid):
        """Resident memory of process ``pid`` in bytes, 0 once it has exited."""
        try:
            return psutil.Process(pid).memory_info().rss
        except psutil.Error:
            return 0


def log(message):
    """Append a time-stamped line to the log file."""
    with open(LOG_FILE, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), message))


def cool_down(run_seconds):
    """Pause after a run that took ``run_seconds``."""
    pause = min(COOLDOWN_MAX, max(COOLDOWN_MIN, COOLDOWN_RATIO * run_seconds))
    log("pause    %.0f s" % pause)
    time.sleep(pause)


def _tree_memory(process):
    try:
        children = process.children(recursive=True)
    except psutil.Error:
        children = []
    return process_memory(process.pid) + sum(process_memory(c.pid) for c in children)


def _stop(proc, process):
    """Stop the command's process group (on Windows, its process tree)."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return
    try:
        children = process.children(recursive=True)
    except psutil.Error:
        children = []
    for child in children + [process]:
        try:
            child.kill()
        except psutil.Error:
            pass


def _last_line(stream):
    """The last non-empty line of the binary file ``stream``, found by reading backwards from its
    end; of a line longer than ERROR_CHUNK_BYTES only its start and end are read."""
    stream.seek(0, os.SEEK_END)
    end = stream.tell()
    while end > 0:
        stream.seek(end - 1)
        if stream.read(1) not in b" \t\r\n":
            break
        end -= 1
    start = end
    while start > 0:
        size = min(ERROR_CHUNK_BYTES, start)
        stream.seek(start - size)
        newline = stream.read(size).rfind(b"\n")
        if newline >= 0:
            start = start - size + newline + 1
            break
        start -= size
    stream.seek(start)
    if end - start <= 2 * ERROR_CHUNK_BYTES:
        return stream.read(end - start).decode("utf-8", "replace").strip()
    head = stream.read(ERROR_CHUNK_BYTES).decode("utf-8", "replace")
    stream.seek(end - ERROR_CHUNK_BYTES)
    return head + " ... " + stream.read(ERROR_CHUNK_BYTES).decode("utf-8", "replace")


def run_monitored(command, timeout, env=None, label=None):
    """Run ``command`` (a list) for at most ``timeout`` seconds, with the environment ``env``;
    ``label`` names the run in the log.

    Returns a dict with the ``status`` ("ok", "timeout", "OOM/crash (exit -N)" when a signal
    such as an out-of-memory kill stopped it, or "error (exit N): <the last line of its error
    output>" when it exited with an error; a long line keeps its start and end), the ``wall_seconds`` the run took, the ``peak_mb`` of the process and its child processes,
    and the ``stdout`` of the command.
    """
    label = label or " ".join(str(part) for part in command[-2:])
    log("start    %s" % label)
    started = time.perf_counter()
    next_log = LOG_EVERY_SECONDS
    # The output goes to a file, so a command that prints a lot cannot fill a pipe and stall.
    with tempfile.TemporaryFile("w+") as output, tempfile.TemporaryFile("w+b") as errors:
        proc = subprocess.Popen(
            command, stdout=output, stderr=errors, env=env,
            start_new_session=os.name == "posix",
        )
        process = psutil.Process(proc.pid)
        peak = 0
        timed_out = False
        try:
            while proc.poll() is None:
                elapsed = time.perf_counter() - started
                if elapsed > timeout:
                    timed_out = True
                    break
                memory = _tree_memory(process)
                peak = max(peak, memory)
                if elapsed >= next_log:
                    log("running  %s  %.0f s  %.1f GB now, %.1f GB peak"
                        % (label, elapsed, memory / 1e9, peak / 1e9))
                    next_log += LOG_EVERY_SECONDS
                time.sleep(SAMPLE_SECONDS)
        except BaseException:
            log("end      %s  interrupted, %.0f s" % (label, time.perf_counter() - started))
            raise
        finally:
            _stop(proc, process)
            proc.wait()
        output.seek(0)
        stdout = output.read()
        last_error = _last_line(errors)
    if timed_out:
        status = "timeout"
    elif proc.returncode == 0:
        status = "ok"
    elif proc.returncode < 0:
        status = "OOM/crash (exit %d)" % proc.returncode
    else:
        status = "error (exit %d)" % proc.returncode
        if last_error:
            status += ": " + (last_error if len(last_error) <= 240
                              else last_error[:80] + " ... " + last_error[-155:])
    wall_seconds = round(time.perf_counter() - started, 2)
    log("end      %s  %s, %.0f s, %.1f GB peak" % (label, status, wall_seconds, peak / 1e9))
    return {
        "status": status,
        "wall_seconds": wall_seconds,
        "peak_mb": round(peak / 1e6, 1),
        "stdout": stdout,
    }


if __name__ == "__main__":
    arguments = sys.argv[1:]
    label, pause = None, False
    while arguments[0] in ("--label", "--cool-down"):
        if arguments[0] == "--label":
            label, arguments = arguments[1], arguments[2:]
        else:
            pause, arguments = True, arguments[1:]
    result = run_monitored(arguments[1:], float(arguments[0]), label=label)
    if pause:
        cool_down(result["wall_seconds"])
    print(json.dumps(result))
