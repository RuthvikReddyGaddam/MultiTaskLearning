# utils/config.py

import os
from pathlib import Path
import yaml


def _get_project_root() -> Path:
    """Find the root directory of newimp."""
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # utils/ is 1 level below newimp/
    return Path(__file__).resolve().parent.parent


def resolve_path(path_str: str, project_root: Path = None) -> str:
    """
    Resolve a path string to a valid, portable path.
    - Resolves environment variables (e.g. $PROJECT_ROOT, $DATA_ROOT)
    - If path starts with legacy '/home/rgnm9/...', redirects to current workspace if legacy does not exist
    - If path is relative, resolves relative to project_root
    """
    if not isinstance(path_str, str) or not path_str.strip():
        return path_str

    if project_root is None:
        project_root = _get_project_root()

    # Expand environment variables and user home
    expanded = os.path.expandvars(os.path.expanduser(path_str))

    # Redirect legacy paths to current project root without querying dead /home/rgnm9 mounts
    legacy_prefix = "/home/rgnm9/newimp/"
    if expanded.startswith(legacy_prefix):
        rel_part = expanded[len(legacy_prefix):]
        return str(project_root / rel_part)

    legacy_sdenet = "/home/rgnm9/SDENet/"
    if expanded.startswith(legacy_sdenet):
        rel_part = expanded[len(legacy_sdenet):]
        alt_path = project_root.parent / "SDENet" / rel_part
        return str(alt_path)

    legacy_sdenetfusion = "/home/rgnm9/SDENetFusion/"
    if expanded.startswith(legacy_sdenetfusion):
        rel_part = expanded[len(legacy_sdenetfusion):]
        alt_path = project_root.parent / "SDENetFusion" / rel_part
        return str(alt_path)

    # Relative path resolution
    p = Path(expanded)
    if not p.is_absolute():
        p = (project_root / p).resolve()
        return str(p)

    return expanded


def resolve_config_paths(data, project_root: Path = None, key: str = None):
    """Recursively resolve path strings inside config dictionary."""
    if project_root is None:
        project_root = _get_project_root()

    if isinstance(data, dict):
        return {k: resolve_config_paths(v, project_root, key=k) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_config_paths(x, project_root, key=key) for x in data]
    elif isinstance(data, str) and ("/" in data or "\\" in data or "$" in data):
        NON_PATH_KEYS = {
            "model_name",
            "appearance_model_name",
            "name",
            "type",
            "arch",
            "backbone",
            "pretrained_model_name_or_path",
            "loss_type",
            "mode",
        }
        if key in NON_PATH_KEYS or (key and (key.endswith("_name") or key.endswith("_type"))):
            expanded = os.path.expandvars(os.path.expanduser(data))
            if os.path.exists(expanded) or (project_root / expanded).exists():
                return resolve_path(data, project_root)
            return data

        expanded = os.path.expandvars(os.path.expanduser(data))
        parts = data.split("/")
        if len(parts) == 2 and not (project_root / expanded).exists() and not os.path.exists(expanded):
            import re
            if re.match(r"^[a-zA-Z0-9_\.-]+/[a-zA-Z0-9_\.-]+$", data) and not data.startswith((".", "/")):
                return data

        return resolve_path(data, project_root)
    return data


def load_config(path: str, resolve_paths: bool = True):
    """
    Load YAML configuration file with automatic path normalization.
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = (_get_project_root() / path_obj).resolve()

    with open(path_obj, "r") as file:
        config = yaml.safe_load(file)

    if resolve_paths and isinstance(config, dict):
        config = resolve_config_paths(config, _get_project_root())

    return config