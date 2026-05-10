"""
inline_script_libyt.py — libyt in-situ analysis for Dyablo blast test.

Uses yt.SlicePlot via the yt_libyt_oct frontend, which exposes libyt's
in-memory oct data as a proper yt Dataset — exactly as a post-processing
script would, with just two lines of change.
"""

import sys
import site
import os

# addsitedir processes .pth files (needed for editable installs).
site.addsitedir("/home/cphyc/Documents/prog/yt/venv/lib/python3.14/site-packages")
# yt is installed in editable mode; its source tree must also be on sys.path.
_YT_SRC = "/home/cphyc/Documents/prog/yt"
if _YT_SRC not in sys.path:
    sys.path.insert(0, _YT_SRC)

import matplotlib
matplotlib.use("Agg")

import yt
from yt_libyt_oct import LibytOctDataset

yt.set_log_level("warning")  # suppress verbose yt output during in-situ runs

_step = 0


def yt_inline():
    """Called by libyt at each output snapshot."""
    global _step

    import libyt
    grids    = libyt.oct_grids
    n_leaves = grids["num_leaves_local"]
    levels   = grids["level"]

    print(f"[libyt] step={_step}  n_leaves={n_leaves}  "
          f"max_level={int(levels.max()) if n_leaves > 0 else 0}")

    # ── Build a yt Dataset from libyt's in-memory oct data ───────────────────
    ds = LibytOctDataset()

    # ── Slice plot for every field ────────────────────────────────────────────
    for field_name in libyt.oct_field_list:
        field = ("libyt", field_name)
        p = yt.SlicePlot(ds, "z", field)
        p.annotate_grids()      # overlay AMR block boundaries
        fname = f"slice_{field_name}_step{_step:04d}.png"
        p.save(fname)
        print(f"[libyt]   → {os.path.abspath(fname)}")

    _step += 1
    print("[libyt] yt_inline() complete\n")
