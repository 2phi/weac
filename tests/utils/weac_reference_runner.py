"""
Utility to run code against a reference (pinned) PyPI weac version in isolation.

Creates and caches a dedicated virtual environment per version under
`.weac-reference/<version>` (overridable via WEAC_REFERENCE_HOME), installs the
requested version, executes a small helper script inside that environment, and
returns computed results to the tests via JSON.

This avoids import-name conflicts with the local in-repo `weac` package.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# For type hints without importing numpy at module import time
try:
    import numpy as _np
except ImportError:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        import numpy as _np
    else:
        _np = Any  # type: ignore[assignment, misc]


DEFAULT_REFERENCE_VERSION = os.environ.get("WEAC_REFERENCE_VERSION", "2.6.4")
REFERENCE_HOME = os.environ.get("WEAC_REFERENCE_HOME", None)


@dataclass
class ReferenceEnv:
    python_exe: str
    venv_dir: str
    version: str


# New: ensure subprocesses don't see local project on sys.path or user site
def _clean_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _project_root() -> str:
    # tests/utils/weac_reference_runner.py -> tests -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _venv_dir(version: str) -> str:
    # Place under project root to cache between test runs
    root = _project_root()
    base = REFERENCE_HOME or os.path.join(root, ".weac-reference")
    return os.path.join(base, version)


def _venv_python(venv_dir: str) -> str:
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def ensure_weac_reference_env(
    version: str = DEFAULT_REFERENCE_VERSION,
) -> Optional[ReferenceEnv]:
    """Create a dedicated venv with weac==version installed if missing.

    Returns ReferenceEnv on success, or None on failure (e.g., no network).
    """
    venv_dir = _venv_dir(version)
    py_exe = _venv_python(venv_dir)

    try:
        if not os.path.exists(py_exe):
            os.makedirs(venv_dir, exist_ok=True)
            # Create venv
            subprocess.run(
                [sys.executable, "-m", "venv", venv_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

        # Ensure pip is up to date
        subprocess.run(
            [py_exe, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Ensure numpy is available for the runner script regardless of weac deps
        subprocess.run(
            [py_exe, "-m", "pip", "install", "--upgrade", "numpy"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Install exact version if not present or mismatched
        code = (
            "import sys\n"
            "try:\n"
            "    from importlib.metadata import version, PackageNotFoundError\n"
            "except Exception:\n"
            "    from importlib_metadata import version, PackageNotFoundError\n"
            "try:\n"
            f"    v = version('weac'); sys.exit(0 if v == '{version}' else 1)\n"
            "except PackageNotFoundError:\n"
            "    sys.exit(2)\n"
        )
        check_proc = subprocess.run(
            [py_exe, "-c", code], cwd=venv_dir, env=_clean_env()
        )
        if check_proc.returncode != 0:
            # Install pinned reference version and its deps
            subprocess.run(
                [
                    py_exe,
                    "-m",
                    "pip",
                    "install",
                    f"weac=={version}",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=_clean_env(),
            )

        return ReferenceEnv(python_exe=py_exe, venv_dir=venv_dir, version=version)
    except subprocess.CalledProcessError:
        return None


def _write_runner_script(script_path: str) -> None:
    """Write the Python script executed inside the reference venv.

    The script reads a JSON config path from argv[1], executes the reference API,
    and prints JSON to stdout.
    """
    script = r"""
import json
import sys
import numpy as np
from json_helpers import json_default


def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    import weac as ref_weac

    # Build model
    system = cfg.get('system', 'skier')
    layers_profile = cfg['layers_profile']
    touchdown = bool(cfg.get('touchdown', False))
    model = ref_weac.Layered(system=system, layers=layers_profile, touchdown=touchdown)

    set_foundation = cfg.get('set_foundation')
    if set_foundation:
        # e.g. {"t": 20, "E": 0.35, "nu": 0.1}
        model.set_foundation_properties(update=True, **set_foundation)

    L = float(cfg['L'])
    a = float(cfg['a'])
    m = float(cfg['m'])
    phi = float(cfg['phi'])

    segs = model.calc_segments(L=L, a=a, m=m, li=None, mi=None, ki=None, phi=phi)["crack"]
    constants = model.assemble_and_solve(phi=phi, **segs)

    z_parts = []
    num_segments = constants.shape[1]
    # The 'pst-' system returns segments in lists under 'li', 'mi', 'ki'
    seg_lengths = segs.get('li', [])
    seg_foundations = segs.get('ki', [])

    for i in range(num_segments):
        seg_len = seg_lengths[i]
        is_bed = seg_foundations[i]
        x_coords = [0, seg_len/2, seg_len]
        C_seg = constants[:, [i]]

        z_segment = model.z(
            x=x_coords,
            C=C_seg,
            l=seg_len,
            phi=phi,
            bed=is_bed
        )
        z_parts.append(np.asarray(z_segment))

    if z_parts:
        z_combined = np.hstack(z_parts)
        z_list = z_combined.tolist()
    else:
        z_list = []


    # Extract state needed by tests
    state = {
        "weak": {
            "nu": model.weak.get("nu"),
            "E": model.weak.get("E"),
        },
        "t": getattr(model, 't', None),
        "kn": getattr(model, 'kn', None),
        "kt": getattr(model, 'kt', None),
        "slab": model.slab.tolist() if hasattr(model, 'slab') else None,
        "h": getattr(model, 'h', None),
        "zs": getattr(model, 'zs', None),
        "a": getattr(model, 'a', None),
        "touchdown": {
            "tc": getattr(model, 'tc', None),
            "a1": getattr(model, 'a1', None),
            "a2": getattr(model, 'a2', None),
            "td": getattr(model, 'td', None),
        },
        "segs": segs,
    }

    out = {"constants": np.asarray(constants).tolist(), "state": state, "z": z_list}
    print(json.dumps(out, default=json_default))

if __name__ == '__main__':
    main()
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)


def compute_reference_model_results(
    *,
    system: str,
    layers_profile: Any,
    touchdown: bool,
    L: float,
    a: float,
    m: float,
    phi: float,
    set_foundation: Optional[Dict[str, Any]] = None,
    version: str = DEFAULT_REFERENCE_VERSION,
) -> Tuple["_np.ndarray", Dict[str, Any], "_np.ndarray"]:
    """Run the reference published weac implementation and return (constants, state, z).

    The return constants is a numpy array; state is a JSON-serializable dict
    with selected model attributes used in tests.
    """
    env = ensure_weac_reference_env(version=version)
    if env is None:
        raise RuntimeError(
            f"Unable to provision reference weac environment (weac=={version})."
        )

    tmp_dir = tempfile.mkdtemp(prefix="weac_reference_run_")
    try:
        # Copy helper to be available to the runner script
        json_helpers_src = os.path.join(os.path.dirname(__file__), "json_helpers.py")
        shutil.copy(json_helpers_src, tmp_dir)

        cfg = {
            "system": system,
            "layers_profile": layers_profile,
            "touchdown": touchdown,
            "L": L,
            "a": a,
            "m": m,
            "phi": phi,
            "set_foundation": set_foundation,
        }

        cfg_path = os.path.join(tmp_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        runner_path = os.path.join(tmp_dir, "reference_runner.py")
        _write_runner_script(runner_path)

        proc = subprocess.run(
            [env.python_exe, runner_path, cfg_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_dir,
            env=_clean_env(),
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Reference runner failed with code {proc.returncode}: {proc.stderr.strip()}"
            )

        data = json.loads(proc.stdout)

        # Lazy import numpy only in the main environment
        import numpy as np  # type: ignore

        constants = np.asarray(data["constants"])
        state = data["state"]
        z = np.asarray(data["z"])
        return constants, state, z
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
