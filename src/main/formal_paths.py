from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_NETWORK_TYPES = ("mlp", "kan")
DEFAULT_FORMAL_ROUND = 2


def _normalized_network_type(network_type: str) -> str:
    backend = str(network_type).strip().lower()
    if backend not in SUPPORTED_NETWORK_TYPES:
        raise ValueError(
            "formal DQN-MPC network_type must be 'mlp' or 'kan'"
        )
    return backend


def formal_output_dir(
    network_type: str,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> Path:
    backend = _normalized_network_type(network_type)
    return (
        Path(repo_root)
        / "outputs"
        / f"dqn_mpc_{backend}_causal_deficit_a2_formal_rounds"
    )


def formal_checkpoint_path(
    network_type: str,
    *,
    round_id: int = DEFAULT_FORMAL_ROUND,
    repo_root: str | Path = REPO_ROOT,
) -> Path:
    backend = _normalized_network_type(network_type)
    resolved_round = int(round_id)
    if resolved_round < 1:
        raise ValueError("formal round_id must be positive")
    return (
        formal_output_dir(backend, repo_root=repo_root)
        / f"round_{resolved_round}"
        / f"model_round{resolved_round}.pt"
    )


def require_formal_checkpoint(
    network_type: str,
    *,
    checkpoint_path: str | Path | None = None,
    round_id: int = DEFAULT_FORMAL_ROUND,
    repo_root: str | Path = REPO_ROOT,
) -> Path:
    backend = _normalized_network_type(network_type)
    expected_root = formal_output_dir(
        backend,
        repo_root=repo_root,
    ).resolve()
    candidate = (
        formal_checkpoint_path(
            backend,
            round_id=round_id,
            repo_root=repo_root,
        )
        if checkpoint_path is None
        else Path(checkpoint_path)
    ).resolve()

    if not candidate.is_relative_to(expected_root):
        raise ValueError(
            f"{backend.upper()} checkpoint must be inside {expected_root}: "
            f"{candidate}"
        )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"required {backend.upper()} checkpoint does not exist: "
            f"{candidate}"
        )
    return candidate
