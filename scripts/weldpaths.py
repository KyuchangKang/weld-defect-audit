# -*- coding: utf-8 -*-
"""Where the scripts look for the project folder and the training-run folder.

The scripts were written on one workstation and keep its paths as defaults, so they behave exactly as they did
for the paper when nothing is set. To run them elsewhere, give either location on the command line or in the
environment (the command line wins):

  --root PATH   or  WELD_AUDIT_ROOT   project folder: holds 04_experiments/data/, 05_results/, ...
  --runs PATH   or  WELD_AUDIT_RUNS   training-run folder: holds E2_v5n/, upgrade/, review4/, ...
  --out PATH    or  WELD_AUDIT_OUT    output folder of render_samples.py only

The options are removed from sys.argv when this module is imported, so a script's own argument parsing is
unaffected. Each script passes its original absolute path as the default and joins its sub-folders onto the
result, e.g. project_dir(r"D:\\...\\paper_wedling2") / "04_experiments" / "data".
"""
import os
import sys
from pathlib import Path

_FLAGS = {"--root": "WELD_AUDIT_ROOT", "--runs": "WELD_AUDIT_RUNS", "--out": "WELD_AUDIT_OUT"}


def _take_flags(argv):
    found, rest, i = {}, [argv[0]] if argv else [], 1
    while i < len(argv):
        a = argv[i]
        name, eq, val = a.partition("=")
        if name in _FLAGS:
            if not eq:
                if i + 1 >= len(argv):
                    sys.exit(f"{name} needs a path")
                val = argv[i + 1]
                i += 1
            found[name] = val
        else:
            rest.append(a)
        i += 1
    return found, rest


_GIVEN, sys.argv[:] = _take_flags(sys.argv)


def _resolve(flag, default):
    value = _GIVEN.get(flag) or os.environ.get(_FLAGS[flag])
    return Path(value) if value else Path(default)


def project_dir(default):
    return _resolve("--root", default)


def runs_dir(default):
    return _resolve("--runs", default)


def out_dir(default):
    return _resolve("--out", default)
