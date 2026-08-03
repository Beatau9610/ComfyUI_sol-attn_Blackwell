"""Persistent fix for the torch Inductor duplicate-template crash.

In this environment the installed torch 2.11 (cu130) has a stale install
layout: the ``torch/_inductor/kernel/`` directory contains BOTH the old
top-level files ``flex_attention.py`` / ``flex_decoding.py`` /
``mm_scaled_grouped.py`` (dated Aug 2025) AND the new
``torch/_inductor/kernel/flex/`` subpackage (dated Mar 2026) that replaced
them.  When Inductor loads, ``lowering.py`` iterates ``kernel/*`` and
imports every module, so the old top-level ``flex_attention.py`` registers a
``TritonTemplate(name="flex_attention")`` and then the new
``flex/flex_attention.py`` registers the same name again.  TritonTemplate
asserts that template names are unique, so ``torch.compile`` of any
flex_attention call raises ``AssertionError: duplicate template name``.

The old files are functionally obsolete (the flex/ subpackage is the current
implementation).  This module removes them from the package directory so
Inductor only sees the new subpackage.  If the files are already gone (or
were never present) this is a no-op.

Call this at the very start of the plugin's ``__init__`` (before any
``torch.compile``) so both the SolAttn-Flex backend and ComfyUI's own
``torch.compile`` usage work.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Offending stale top-level files (obsolete duplicates of the flex/ subpackage).
_STALE_NAMES = (
    "flex_attention.py",
    "flex_decoding.py",
    "mm_scaled_grouped.py",
)

# Where we park them instead of deleting (so the removal is reversible).
_BACKUP_DIR = Path("/tmp/stale_inductor_kernels")


def _inductor_kernel_dir() -> Path:
    try:
        import torch._inductor.kernel as k
    except Exception:
        return Path()
    return Path(k.__file__).parent


def apply_inductor_fix(force: bool = False) -> bool:
    """Move stale duplicate Inductor kernel files out of the package.

    Returns True if any file was moved (i.e. the fix was needed), False if
    the environment was already clean.
    """
    kernel_dir = _inductor_kernel_dir()
    if kernel_dir == Path() or not kernel_dir.is_dir():
        logger.warning("[InductorFix] could not locate torch._inductor.kernel dir")
        return False

    moved_any = False
    try:
        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        for name in _STALE_NAMES:
            src = kernel_dir / name
            if src.is_file():
                dst = _BACKUP_DIR / name
                shutil.move(str(src), str(dst))
                logger.info(f"[InductorFix] moved obsolete {src.name} -> {dst}")
                moved_any = True
        if moved_any:
            # Clear any partially-imported Inductor module state so the next
            # import rebuilds the template registry cleanly.
            _purge_inductor_cache()
    except Exception as exc:
        logger.warning(f"[InductorFix] failed to apply fix: {exc}")
        return False
    return moved_any


def _purge_inductor_cache():
    """Drop cached Inductor/flex modules so re-import rebuilds cleanly."""
    import sys

    for mod in list(sys.modules):
        if mod.startswith("torch._inductor"):
            del sys.modules[mod]


def apply_inductor_fix_if_needed() -> bool:
    """Idempotent wrapper: only act if the duplicate-template problem would
    otherwise occur.  Cheap check: presence of any stale top-level file."""
    import torch

    # If the new flex/ subpackage is present, the stale files, if any, are a
    # problem.  If the subpackage is absent, the old files are the only impl
    # and we must leave them alone.
    kernel_dir = _inductor_kernel_dir()
    has_flex_subpkg = (kernel_dir / "flex").is_dir()
    stale = [n for n in _STALE_NAMES if (kernel_dir / n).is_file()]
    if not has_flex_subpkg or not stale:
        return False
    return apply_inductor_fix()