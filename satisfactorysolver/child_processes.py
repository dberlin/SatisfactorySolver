"""
Tie child processes' lives to their parent's, so a parent killed without the
chance to stop them (Task Manager, kill -9) does not leave them running.
"""

import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

# Held until this process exits: closing the last handle kills the job.
_job = None


def kill_children_on_exit() -> None:
    """Have the OS kill every process this one starts once it exits, however it
    exits. Only Windows can; elsewhere children call exit_with_parent."""
    global _job
    if sys.platform != "win32" or _job is not None:
        return
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (f"{kind}{unit}", ctypes.c_uint64)
            for unit in ("OperationCount", "TransferCount")
            for kind in ("Read", "Write", "Other")
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job_object_extended_limit_information = 9
    job_object_limit_kill_on_job_close = 0x2000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)

    # The handle is not inheritable, so only this process holds it open.
    job = kernel32.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
    if not (
        job
        and kernel32.SetInformationJobObject(
            job,
            job_object_extended_limit_information,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
        and kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
    ):
        logger.warning(
            "Cannot tie child processes to this one: %s",
            ctypes.FormatError(ctypes.get_last_error()),
        )
        return
    _job = job


def exit_with_parent(parent_pid: int, interval: float = 1) -> None:
    """Exit this process soon after the given parent exits, however it exits.

    Windows needs nothing here: kill_children_on_exit has the OS do it.
    """
    if sys.platform == "win32":
        return

    def watch():
        # An orphan is adopted by init or a subreaper, changing its parent.
        while os.getppid() == parent_pid:
            time.sleep(interval)
        os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()
