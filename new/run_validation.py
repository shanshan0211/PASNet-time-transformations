"""Run all repository tests with explicit Windows peak-RSS compatibility.

The adapter reports native PeakWorkingSetSize in KiB for importable legacy
runners. It does not rewrite their sealed Linux benchmark measurements.
Failures, including inaccessible historical provenance, stay failures.
"""
import ctypes
import argparse
from ctypes import wintypes
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def install_windows_resource():
    if os.name != "nt":
        return
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def usage(who):
        if who != 0:
            raise NotImplementedError("only current-process RSS is supported")
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return types.SimpleNamespace(ru_maxrss=counters.PeakWorkingSetSize / 1024)
    module = types.ModuleType("resource")
    module.RUSAGE_SELF = 0
    module.getrusage = usage
    sys.modules["resource"] = module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("workspace", "archive"), default="workspace")
    args = parser.parse_args()
    install_windows_resource()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tests"))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    excluded = []
    if args.scope == "archive":
        external_id = "test_monitor_generalisation_artifacts.MonitorGeneralisationArtifactTests.test_recovered_multiseed_excludes_invalid_known_h"
        def flatten(items):
            for item in items:
                if isinstance(item, unittest.TestSuite):
                    yield from flatten(item)
                else:
                    yield item
        all_tests = list(flatten(suite))
        excluded = [t.id() for t in all_tests if t.id() == external_id]
        if excluded != [external_id]:
            raise RuntimeError("archive scope must explicitly exclude exactly the one external historical provenance check")
        suite = unittest.TestSuite(t for t in all_tests if t.id() != external_id)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    stem = "validation_tests_current" if args.scope == "workspace" else "validation_tests_archive"
    (ROOT / "new" / (stem + ".log")).write_text(stream.getvalue(), encoding="utf-8")
    report = {"tests_run": result.testsRun, "failures": [{"test": str(t), "traceback": tb} for t, tb in result.failures],
              "errors": [{"test": str(t), "traceback": tb} for t, tb in result.errors],
              "skipped": result.skipped, "passed": result.wasSuccessful(),
              "runtime": sys.version, "scope": args.scope, "excluded_tests": excluded,
              "windows_adapter": "native PeakWorkingSetSize/KiB; no legacy results changed"}
    (ROOT / "new" / (stem + ".json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(stream.getvalue())
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
