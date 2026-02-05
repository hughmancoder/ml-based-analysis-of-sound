import os
from pathlib import Path


def guard_path(path: Path, repo_root: Path, label: str) -> Path:
    """
    Prevent writes outside repo unless ALLOW_UNSAFE_PATHS=1 is set.
    """
    allow_unsafe = os.environ.get("ALLOW_UNSAFE_PATHS") == "1"
    p = Path(path).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()

    if allow_unsafe:
        return p
    if p.parent == p:
        raise SystemExit(f"[safe-guard] Refusing {label} at drive root: {p}")
    if repo not in p.parents and p != repo:
        raise SystemExit(f"[safe-guard] Refusing {label} outside repo: {p}")
    return p
