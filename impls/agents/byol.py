"""
Adapter for BYOL-<y.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import utils as ogbench_utils


REFERENCE_COMMIT = "a74aad4bbc29798e216bc042f78e53ed0496983e"
REFERENCE_SHA256 = {
    "agents/byol.py": "90eafb48dd508844c7863dbadb6c3990d7127471c16cb1a1ba9058b150e445eb",
    "utils/networks.py": "ccacd24f60f89d9724fe035e400b6cbc6a8593d4528644e14ebb39ea85331d65",
    "utils/encoders.py": "6116b0c33b23a45f064314befe58ef1992f3f29c6b228ad52919b0bf3cdcb023",
    "utils/flax_utils.py": "febe623b0c1bb28f6fb54e9e7211784209f0304ef81f655729eb880cefabc5f1",
}

_REFERENCE_ROOT = Path(__file__).resolve().parents[3] / "self-pred-bc"
_PRIVATE_MODULE_NAMES = {
    "networks": "_thesis_self_pred_bc_networks",
    "encoders": "_thesis_self_pred_bc_encoders",
    "flax_utils": "_thesis_self_pred_bc_flax_utils",
    "byol": "_thesis_self_pred_bc_byol",
}
_ADAPTER_PATH = Path(__file__).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REFERENCE_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not resolve the pinned BYOL repository commit at {_REFERENCE_ROOT}."
        ) from exc


def _verify_reference_sources() -> None:
    actual_commit = _reference_git_commit()
    if actual_commit != REFERENCE_COMMIT:
        raise RuntimeError(
            "Pinned BYOL repository commit changed: "
            f"{actual_commit}, expected {REFERENCE_COMMIT}."
        )
    for relative_path, expected_sha256 in REFERENCE_SHA256.items():
        path = _REFERENCE_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"BYOL reference source not found: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Pinned BYOL reference source changed: "
                f"{path} has SHA-256 {actual_sha256}, expected {expected_sha256}."
            )


def _load_private_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@contextmanager
def _temporary_utils_aliases(**modules):
    """Temporarily expose private reference modules as ``utils.<name>``."""

    missing = object()
    previous_modules = {}
    previous_attributes = {}
    try:
        for short_name, module in modules.items():
            full_name = f"utils.{short_name}"
            previous_modules[full_name] = sys.modules.get(full_name, missing)
            previous_attributes[short_name] = getattr(ogbench_utils, short_name, missing)
            sys.modules[full_name] = module
            setattr(ogbench_utils, short_name, module)
        yield
    finally:
        for short_name in reversed(tuple(modules)):
            full_name = f"utils.{short_name}"
            previous_module = previous_modules[full_name]
            if previous_module is missing:
                sys.modules.pop(full_name, None)
            else:
                sys.modules[full_name] = previous_module

            previous_attribute = previous_attributes[short_name]
            if previous_attribute is missing:
                try:
                    delattr(ogbench_utils, short_name)
                except AttributeError:
                    pass
            else:
                setattr(ogbench_utils, short_name, previous_attribute)


def _load_reference_modules():
    _verify_reference_sources()

    networks = _load_private_module(
        _PRIVATE_MODULE_NAMES["networks"],
        _REFERENCE_ROOT / "utils" / "networks.py",
    )
    flax_utils = _load_private_module(
        _PRIVATE_MODULE_NAMES["flax_utils"],
        _REFERENCE_ROOT / "utils" / "flax_utils.py",
    )
    with _temporary_utils_aliases(networks=networks):
        encoders = _load_private_module(
            _PRIVATE_MODULE_NAMES["encoders"],
            _REFERENCE_ROOT / "utils" / "encoders.py",
        )
    with _temporary_utils_aliases(
        networks=networks,
        encoders=encoders,
        flax_utils=flax_utils,
    ):
        byol = _load_private_module(
            _PRIVATE_MODULE_NAMES["byol"],
            _REFERENCE_ROOT / "agents" / "byol.py",
        )
    return byol


_REFERENCE_BYOL = _load_reference_modules()
BYOLAgent = _REFERENCE_BYOL.BYOLAgent


def get_reference_provenance():
    """Return immutable source identity for run metadata and tests."""

    return {
        "repository": "external/self-pred-bc",
        "commit": _reference_git_commit(),
        "expected_commit": REFERENCE_COMMIT,
        "root": str(_REFERENCE_ROOT),
        "sha256": dict(REFERENCE_SHA256),
        "adapter_path": str(_ADAPTER_PATH),
        "adapter_sha256": _sha256(_ADAPTER_PATH),
    }


def get_config():
    """Return the reference config with only integration metadata added."""

    config = _REFERENCE_BYOL.get_config()
    config.agent_name = "byol_gamma"
    config.reference_repository = "external/self-pred-bc"
    config.reference_commit = REFERENCE_COMMIT
    config.reference_agent_sha256 = REFERENCE_SHA256["agents/byol.py"]
    config.atomic_train_manifest_path = ""
    config.atomic_val_manifest_path = ""
    config.atomic_goal_stack_mode = "repeat_endpoint"
    config.atomic_require_source_fingerprint = False
    return config
