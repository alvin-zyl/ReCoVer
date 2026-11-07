"""Make a rank that dies of an uncaught Python exception count as failed.

Such a rank would otherwise exit normally: the extension's atexit handler
calls MPI_Finalize, and ULFM does not report a finalized process as failed,
so the other ranks never recover from its loss (in our tests the job hung
until its time limit). A rank that leaves with _exit() is not reported
either. A rank killed by a signal is, so print the traceback and SIGKILL the
process, as the failure simulator does.
"""

import os
import signal
import sys
import traceback


def _excepthook(exc_type, exc, tb):
    # Write to the process's own stderr: torch.distributed's init_process_group
    # wraps this hook and points sys.stderr at a buffer that it prints only
    # after the hook returns, which this one never does.
    out = sys.__stderr__ or sys.stderr
    try:
        rank = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "?"))
        lines = "".join(traceback.format_exception(exc_type, exc, tb)).rstrip("\n").split("\n")
        lines.append(
            f"[fail-stop] uncaught {exc_type.__name__}: killing this rank with "
            f"SIGKILL so that the other ranks recover from its loss"
        )
        out.write("".join(f"[rank{rank}]: {line}\n" for line in lines))
        out.flush()
    finally:
        os.kill(os.getpid(), signal.SIGKILL)


def kill_self(message: str) -> None:
    """Leave as a failed process (SIGKILL), after writing ``message``."""
    out = sys.__stderr__ or sys.stderr
    try:
        rank = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "?"))
        out.write(f"[rank{rank}]: [fail-stop] {message}\n")
        out.flush()
    finally:
        os.kill(os.getpid(), signal.SIGKILL)


def install_fail_stop() -> None:
    """Turn uncaught exceptions on the main thread into a SIGKILL."""
    sys.excepthook = _excepthook
