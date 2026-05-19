"""Register local ``src/diffusers`` as ``fd_diffusers`` without shadowing Hugging Face diffusers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

_REGISTERED = False


def register_fd_diffusers() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    root = Path(__file__).resolve().parents[1] / "src" / "diffusers"
    if not root.is_dir():
        raise ImportError(f"FD diffusers extensions not found at {root}")

    if "fd_diffusers" not in sys.modules:
        import importlib.util

        init_path = root / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "fd_diffusers",
            init_path,
            submodule_search_locations=[str(root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load fd_diffusers package from {init_path}")
        pkg = importlib.util.module_from_spec(spec)
        pkg.__path__ = [str(root)]
        pkg.__package__ = "fd_diffusers"
        sys.modules["fd_diffusers"] = pkg
        spec.loader.exec_module(pkg)

    _REGISTERED = True
