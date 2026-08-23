"""Content-addressed guard for ``linear_mlp_scores_*.npz``.

Why this exists (2026-08-05 trap)
---------------------------------
When the merged panel grew from 880 to 889 rows, the stale ``.npz`` built on
the *old* panel was still loaded and ``pd.DataFrame(d["linear"], index=...,
columns=...)`` raised::

    ValueError: Shape of passed values is (880,5910), indices imply (889,5910)

The ``.npz`` was keyed only by ``universe``, never by the panel it was built
against, so a panel change silently produced a shape mismatch at load time.

Fix (ported in spirit from model4's ``pipeline_cache.py``, which content-
addresses by ``base_panel + benchmark + daily files + code/config hashes``)
------------------------------------------------------------------------------
We store a *panel fingerprint* **inside** the ``.npz`` (a ``meta`` array) and
re-check it on every load. If the current panel's fingerprint differs, the
cache is treated as stale -> deleted and rebuilt from the current panel.

The fingerprint captures exactly the dimensions that caused the crash
(``n_dates``, ``n_symbols``, first/last date) plus the *source file* size and
mtime so that a re-merge of the panel forces a rebuild too. This is a
right-sized subset of model4's manifest system -- sufficient to kill the
shape-error class without dragging in the full cache orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def panel_fingerprint(idx, symbols, panel_csv=None) -> dict:
    """Compute a cheap, dimension-strong fingerprint of the panel in use."""
    fp = {
        "n_dates": int(len(idx)),
        "n_symbols": int(len(symbols)),
        "first_date": str(idx[0].date()) if len(idx) else "",
        "last_date": str(idx[-1].date()) if len(idx) else "",
    }
    if panel_csv is not None:
        p = Path(panel_csv)
        if p.exists():
            st = p.stat()
            fp["panel_path"] = str(p.resolve())
            fp["panel_size"] = int(st.st_size)
            fp["panel_mtime"] = float(st.st_mtime)
    return fp


def _fp_str(fp: dict) -> str:
    return json.dumps(fp, ensure_ascii=False, sort_keys=True)


def load_or_build(scores_npz, idx, symbols, build_fn, panel_csv=None, use_cache=True):
    """Return ``(linear_score, mlp_score)`` DataFrames built from the *current* panel.

    Parameters
    ----------
    scores_npz : path to the cache file (already namespaced by universe if needed)
    idx        : the panel's label/date index (used both for fingerprint and to
                 reconstruct the returned DataFrames)
    symbols    : the panel's symbol columns
    build_fn   : callable returning ``(linear_score, mlp_score)`` frames, built
                 against the current panel. Only invoked on a cache miss/corruption.
    panel_csv  : optional path to the source panel file; its size/mtime deepen
                 the fingerprint (a re-merge invalidates the cache).
    use_cache  : when False, always rebuild.

    On a fingerprint mismatch, missing ``meta`` (legacy cache), missing arrays,
    or any load corruption, the cached file is removed and ``build_fn`` runs
    fresh -- so a panel change can never again raise a shape error.
    """
    scores_npz = Path(scores_npz)
    cur_fp = _fp_str(panel_fingerprint(idx, symbols, panel_csv))

    if use_cache and scores_npz.exists():
        try:
            d = np.load(scores_npz, allow_pickle=True)
            try:
                cached = d["meta"]
            except KeyError:
                cached = None
            cached_fp = "" if cached is None else str(np.asarray(cached).item())
            if cached_fp == cur_fp and "linear" in d.files and "mlp" in d.files:
                linear = pd.DataFrame(d["linear"], index=idx, columns=symbols)
                mlp = pd.DataFrame(d["mlp"], index=idx, columns=symbols)
                return linear, mlp
            # Legacy cache (no meta) or fingerprint mismatch -> stale.
            scores_npz.unlink(missing_ok=True)
        except Exception:
            # Corrupt cache -> rebuild.
            try:
                scores_npz.unlink(missing_ok=True)
            except OSError:
                pass

    linear, mlp = build_fn()
    scores_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        scores_npz,
        linear=linear.values,
        mlp=mlp.values,
        meta=np.array(cur_fp),
    )
    return linear, mlp
