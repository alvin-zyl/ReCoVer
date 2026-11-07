"""
conftest.py for pure unit tests.

ulfm_collectives/__init__.py imports the compiled C extension (_C), which
requires torch + MPI at runtime. Pure helper modules (e.g. hsdp_groups.py)
have no such dependency. This conftest inserts a lightweight package stub
into sys.modules before any test import runs, so that
``from ulfm_collectives.hsdp_groups import ...`` works without triggering
the full __init__.py.
"""
import sys
import types
import importlib
import pathlib

# Only inject the stub if the real package can't be imported cleanly.
if "ulfm_collectives" not in sys.modules:
    pkg_dir = pathlib.Path(__file__).parent.parent / "ulfm_collectives"
    stub = types.ModuleType("ulfm_collectives")
    stub.__path__ = [str(pkg_dir)]
    stub.__package__ = "ulfm_collectives"
    stub.__spec__ = importlib.util.spec_from_file_location(
        "ulfm_collectives",
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    sys.modules["ulfm_collectives"] = stub
