"""Backward feature elimination for an MPS Born machine, in one file.

This module keeps only the functionality needed for greedy backward elimination:

- exact marginalisation of dropped features by partial trace of the MPS density;
- reduced anomaly score ``-log P(v_S)``;
- global and per-family AUC-ROC / AUC-PR;
- leakage-safe selection/evaluation split;
- greedy backward elimination and feature ranking;
- CSV/JSON export and a minimal CLI.

It intentionally removes the forward/sweep feature-selection strategies.
The only project-specific runtime dependency is ``mps.MPS`` when using the CLI
or ``--selftest``. Programmatic use only needs an already-loaded MPS object.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)
Objective = Union[str, Callable[[Dict], float]]


# ---------------------------------------------------------------------------
# Generic row/meta helpers
# ---------------------------------------------------------------------------


@dataclass
class EvaluationSplit:
    """Leakage-safe split used by backward elimination."""

    X_select: torch.Tensor
    meta_select: Dict
    X_eval: torch.Tensor
    meta_eval: Dict


def _as_numpy(value) -> np.ndarray:
    """Convert tensors/lists/arrays to a CPU numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _take_rows(X, indices: np.ndarray):
    """Row-select ``X`` preserving torch tensors when possible."""
    indices = np.asarray(indices, dtype=np.int64)
    if torch.is_tensor(X):
        idx = torch.as_tensor(indices, dtype=torch.long, device=X.device)
        return X.index_select(0, idx)
    return np.asarray(X)[indices]


def _slice_meta(meta: Mapping, indices: np.ndarray, n_rows: int) -> Dict:
    """Slice per-row metadata while preserving global entries such as family_names."""
    indices = np.asarray(indices, dtype=np.int64)
    out: Dict = {}
    for key, value in meta.items():
        if key == "family_names":
            out[key] = value
            continue

        if torch.is_tensor(value):
            if value.ndim > 0 and len(value) == n_rows:
                idx = torch.as_tensor(indices, dtype=torch.long, device=value.device)
                out[key] = value.index_select(0, idx)
            else:
                out[key] = value
            continue

        if isinstance(value, np.ndarray):
            out[key] = value[indices] if value.ndim > 0 and len(value) == n_rows else value
            continue

        if isinstance(value, (list, tuple)):
            if len(value) == n_rows:
                out[key] = [value[int(i)] for i in indices]
            else:
                out[key] = value
            continue

        out[key] = value
    return out


def split_dataset(
    X,
    meta: Mapping,
    frac_select: float = 0.5,
    seed: int = 0,
    stratify_key: str = "family_code",
) -> EvaluationSplit:
    """Split rows into selection/evaluation halves.

    The greedy elimination decision uses the selection half; the reported curve
    uses the held-out evaluation half. If ``meta[stratify_key]`` exists, the
    split is stratified by that metadata field.
    """
    if not 0.0 < frac_select < 1.0:
        raise ValueError("frac_select must be between 0 and 1")

    n_rows = len(X)
    if n_rows < 2:
        raise ValueError("at least two rows are needed to create select/eval split")

    rng = np.random.default_rng(seed)
    select_idx: List[int] = []
    eval_idx: List[int] = []

    if stratify_key in meta:
        labels = _as_numpy(meta[stratify_key])
        if len(labels) != n_rows:
            raise ValueError(f"meta[{stratify_key!r}] length does not match X")
        for label in np.unique(labels):
            group = np.flatnonzero(labels == label)
            rng.shuffle(group)
            if len(group) == 1:
                # Singleton classes cannot be split; keep them in selection so
                # they can still influence the greedy choice if present.
                n_select = 1
            else:
                n_select = int(round(len(group) * frac_select))
                n_select = min(max(1, n_select), len(group) - 1)
            select_idx.extend(group[:n_select].tolist())
            eval_idx.extend(group[n_select:].tolist())
    else:
        perm = rng.permutation(n_rows)
        n_select = int(round(n_rows * frac_select))
        n_select = min(max(1, n_select), n_rows - 1)
        select_idx = perm[:n_select].tolist()
        eval_idx = perm[n_select:].tolist()

    select_idx_arr = np.asarray(select_idx, dtype=np.int64)
    eval_idx_arr = np.asarray(eval_idx, dtype=np.int64)
    rng.shuffle(select_idx_arr)
    rng.shuffle(eval_idx_arr)

    return EvaluationSplit(
        X_select=_take_rows(X, select_idx_arr),
        meta_select=_slice_meta(meta, select_idx_arr, n_rows),
        X_eval=_take_rows(X, eval_idx_arr),
        meta_eval=_slice_meta(meta, eval_idx_arr, n_rows),
    )


# ---------------------------------------------------------------------------
# Exact marginalisation over dropped sites
# ---------------------------------------------------------------------------


@torch.no_grad()
def marginal_log_prob(
    mps,
    X: torch.Tensor,
    kept_sites: Sequence[int],
    row_batch: int = 2048,
) -> torch.Tensor:
    """Compute ``log P(v_S)`` keeping ``kept_sites`` and marginalising the rest.

    For kept sites, bra and ket are projected to the observed value. For dropped
    sites, the physical index is summed exactly, as in the MPS norm contraction.
    """
    if row_batch <= 0:
        raise ValueError("row_batch must be positive")

    kept = {int(site) for site in kept_sites}
    invalid = [site for site in kept if site < 0 or site >= mps.num_sites]
    if invalid:
        raise ValueError(f"invalid kept site indices: {invalid}")

    if not torch.is_tensor(X):
        X = torch.as_tensor(np.asarray(X))
    if X.ndim != 2 or X.shape[1] < mps.num_sites:
        raise ValueError(f"X must have shape (n_rows, >= {mps.num_sites})")

    device = mps.site_tensors[0].device
    dtype = mps.dtype
    floor = mps._numerical_floor
    X = X.to(device=device, dtype=torch.long)
    log_z = mps.log_norm().double()

    out = torch.empty(len(X), dtype=torch.float64, device=device)
    for start in range(0, len(X), row_batch):
        xb = X[start:start + row_batch]
        batch_size = len(xb)
        env = torch.ones(batch_size, 1, 1, dtype=dtype, device=device)
        log_scale = torch.zeros(batch_size, dtype=torch.float64, device=device)

        for site in range(mps.num_sites):
            A = mps._as_matrices(mps.site_tensors[site])  # (d, chiL, chiR)
            if site in kept:
                selected = A[xb[:, site]]
                env = torch.einsum("Baj,Bab,Bbk->Bjk", selected.conj(), env, selected)
            else:
                env = torch.einsum("saj,Bab,sbk->Bjk", A.conj(), env, A)

            scale = env.abs().amax(dim=(-2, -1)).clamp_min(floor)
            env = env / scale[:, None, None]
            log_scale = log_scale + scale.double().log()

        marginal = env.reshape(batch_size, -1)[:, 0].real.double().clamp_min(floor)
        out[start:start + row_batch] = marginal.log() + log_scale

    return (out - log_z).to(device)


@torch.no_grad()
def marginal_anomaly_score(
    mps,
    X,
    kept_sites: Sequence[int],
    row_batch: int = 2048,
) -> np.ndarray:
    """Reduced-feature anomaly score: ``-log P(v_S)`` as a numpy array."""
    scores = -marginal_log_prob(mps, X, kept_sites, row_batch=row_batch)
    return scores.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# AUC metrics and objective functions
# ---------------------------------------------------------------------------


def auc_global_and_per_family(
    scores: np.ndarray,
    meta: Mapping,
    normal_label: str = "normal",
) -> Dict:
    """Compute global and family-vs-normal AUC-ROC/AUC-PR from anomaly scores."""
    family_code = _as_numpy(meta["family_code"])
    is_attack = _as_numpy(meta["is_attack"])
    family_names = list(meta["family_names"])

    if normal_label not in family_names:
        raise ValueError(f"normal_label {normal_label!r} is not in family_names")
    normal_idx = family_names.index(normal_label)

    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    s = scores[finite]
    fam = family_code[finite]
    atk = is_attack[finite]

    if len(s) == 0:
        raise ValueError("no finite scores available for AUC")
    if len(np.unique(atk)) < 2:
        raise ValueError("global AUC needs both normal and attack rows")

    result: Dict = {
        "global": {
            "auc_roc": float(roc_auc_score(atk, s)),
            "auc_pr": float(average_precision_score(atk, s)),
            "n": int(len(s)),
        }
    }

    normal_mask = fam == normal_idx
    for family_idx, family_name in enumerate(family_names):
        if family_idx == normal_idx:
            continue
        mask = normal_mask | (fam == family_idx)
        y = (fam[mask] == family_idx).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        result[family_name] = {
            "auc_roc": float(roc_auc_score(y, s[mask])),
            "auc_pr": float(average_precision_score(y, s[mask])),
            "n": int(y.sum()),
        }
    return result


def objective_value(aucs: Dict, objective: Objective) -> float:
    """Scalar objective to maximise from a global/per-family AUC dictionary."""
    if callable(objective):
        return float(objective(aucs))
    if objective == "auc_roc":
        return float(aucs["global"]["auc_roc"])
    if objective == "auc_pr":
        return float(aucs["global"]["auc_pr"])

    family_roc = [v["auc_roc"] for k, v in aucs.items() if k != "global"]
    family_pr = [v["auc_pr"] for k, v in aucs.items() if k != "global"]
    if objective == "mean_family_roc":
        if not family_roc:
            raise ValueError("mean_family_roc needs at least one attack family")
        return float(np.mean(family_roc))
    if objective == "min_family_roc":
        if not family_roc:
            raise ValueError("min_family_roc needs at least one attack family")
        return float(np.min(family_roc))
    if objective == "mean_family_pr":
        if not family_pr:
            raise ValueError("mean_family_pr needs at least one attack family")
        return float(np.mean(family_pr))

    raise ValueError(f"unknown objective {objective!r}")


# ---------------------------------------------------------------------------
# Backward elimination
# ---------------------------------------------------------------------------


def backward_eliminate(
    mps,
    X_select: torch.Tensor,
    meta_select: Mapping,
    X_eval: torch.Tensor,
    meta_eval: Mapping,
    feature_names: Sequence[str],
    objective: Objective = "auc_roc",
    min_features: int = 1,
    row_batch: int = 2048,
    normal_label: str = "normal",
) -> List[Dict]:
    """Run greedy backward elimination and return one record per subset size.

    The greedy drop decision is made on ``X_select/meta_select``. The recorded
    AUC values are measured on ``X_eval/meta_eval``. Records run from all
    features down to ``min_features`` and include ``removed_next``: the feature
    removed to reach the next smaller subset.
    """
    n_sites = len(feature_names)
    if hasattr(mps, "num_sites") and n_sites != int(mps.num_sites):
        raise ValueError(f"feature_names has {n_sites} entries but mps.num_sites={mps.num_sites}")
    if not 1 <= min_features <= n_sites:
        raise ValueError(f"min_features must be in [1, {n_sites}]")

    if not torch.is_tensor(X_select):
        X_select = torch.as_tensor(np.asarray(X_select))
    if not torch.is_tensor(X_eval):
        X_eval = torch.as_tensor(np.asarray(X_eval))

    kept = list(range(n_sites))
    trajectory: List[Dict] = []

    while True:
        eval_scores = marginal_anomaly_score(mps, X_eval, kept, row_batch=row_batch)
        eval_aucs = auc_global_and_per_family(eval_scores, meta_eval, normal_label=normal_label)
        record: Dict = {
            "n_features": len(kept),
            "kept_sites": list(kept),
            "kept_features": [feature_names[i] for i in kept],
            "eval_auc_roc": eval_aucs["global"]["auc_roc"],
            "eval_auc_pr": eval_aucs["global"]["auc_pr"],
            "removed_next": None,
            "removed_next_name": None,
            "select_objective_next": None,
        }
        for family_name, data in eval_aucs.items():
            if family_name == "global":
                continue
            record[f"eval_auc_roc_{family_name}"] = data["auc_roc"]
            record[f"eval_auc_pr_{family_name}"] = data["auc_pr"]
            record[f"n_{family_name}"] = data["n"]

        if len(kept) <= min_features:
            trajectory.append(record)
            logger.info("  n=%2d  eval AUC-ROC=%.4f  (final)", len(kept), record["eval_auc_roc"])
            break

        best_site: Optional[int] = None
        best_score = -np.inf
        for site in kept:
            candidate = [s for s in kept if s != site]
            select_scores = marginal_anomaly_score(mps, X_select, candidate, row_batch=row_batch)
            select_aucs = auc_global_and_per_family(select_scores, meta_select, normal_label=normal_label)
            value = objective_value(select_aucs, objective)
            if value > best_score:
                best_score = value
                best_site = site

        if best_site is None:
            raise RuntimeError("no removable feature candidate was evaluated")

        record["removed_next"] = int(best_site)
        record["removed_next_name"] = feature_names[best_site]
        record["select_objective_next"] = float(best_score)
        trajectory.append(record)

        obj_name = objective if isinstance(objective, str) else "custom"
        logger.info(
            "  n=%2d  eval AUC-ROC=%.4f  -> drop %-24s (select %s=%.4f)",
            len(kept), record["eval_auc_roc"], feature_names[best_site], obj_name, best_score,
        )
        kept.remove(best_site)

    return trajectory


def from_selector(mps, selector, **kwargs) -> List[Dict]:
    """Compatibility helper for an existing selector carrying ``eval_split``."""
    if getattr(selector, "eval_split", None) is None:
        raise ValueError("selector has no eval_split")
    parts = selector.eval_split
    return backward_eliminate(
        mps,
        parts.X_select,
        parts.meta_select,
        parts.X_eval,
        parts.meta_eval,
        selector.feature_names,
        **kwargs,
    )


def elimination_ranking(trajectory: Sequence[Mapping]) -> List[int]:
    """Return site indices ordered most useful first.

    The final survivors are most useful; then come removed sites in reverse
    removal order.
    """
    if not trajectory:
        return []
    removed = [rec["removed_next"] for rec in trajectory if rec.get("removed_next") is not None]
    survivors = list(trajectory[-1]["kept_sites"])
    return list(survivors) + list(reversed(removed))


def elimination_ranking_names(trajectory: Sequence[Mapping], feature_names: Sequence[str]) -> List[str]:
    """Return feature names ordered most useful first."""
    return [feature_names[i] for i in elimination_ranking(trajectory)]


# ---------------------------------------------------------------------------
# I/O and CLI helpers
# ---------------------------------------------------------------------------


def _csv_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return " | ".join(map(str, value))
    return value


def save_trajectory(trajectory: Sequence[Mapping], output_csv: Union[str, Path]) -> Tuple[Path, Path]:
    """Save trajectory to CSV and JSON. Returns ``(csv_path, json_path)``."""
    if not trajectory:
        raise ValueError("empty trajectory")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json = output_csv.with_suffix(".json")

    first_cols = [
        "n_features", "eval_auc_roc", "eval_auc_pr", "removed_next",
        "removed_next_name", "select_objective_next", "kept_sites", "kept_features",
    ]
    all_keys = set().union(*(record.keys() for record in trajectory))
    fieldnames = first_cols + sorted(k for k in all_keys if k not in first_cols)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in trajectory:
            writer.writerow({k: _csv_value(record.get(k, "")) for k in fieldnames})

    output_json.write_text(json.dumps(list(trajectory), indent=2), encoding="utf-8")
    return output_csv, output_json


def load_feature_names(schema_path: Union[str, Path]) -> List[str]:
    """Load feature names from ``encoding_schema.json``."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return [feature["name"] for feature in schema["features"]]

def load_model_and_test(data_dir: Union[str, Path], checkpoint: str = "mps_trained.pt"):
    """Load trained MPS, test_X, test_meta and feature_names from a data folder."""
    from mps import MPS

    data_dir = Path(data_dir)
    model_path = data_dir / checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"missing {model_path}")

    mps = MPS.load(str(model_path))
    if hasattr(mps, "eval"):
        mps.eval()
    test_X = torch.load(data_dir / "test_X.pt", weights_only=True).long()
    test_meta = torch.load(data_dir / "test_meta.pt", weights_only=True)
    feature_names = load_feature_names(data_dir / "encoding_schema.json")
    logger.info("loaded MPS: %d sites", mps.num_sites)
    return mps, test_X, test_meta, feature_names

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy backward feature elimination for a trained MPS.")
    parser.add_argument(
        "data_dir", nargs="?", default="./nsl_kdd_binary",
        help="folder with mps_trained.pt, test_X.pt, test_meta.pt and encoding_schema.json",
    )
    parser.add_argument("--checkpoint", default="mps_trained.pt")
    parser.add_argument(
        "--objective", default="auc_roc",
        choices=["auc_roc", "auc_pr", "mean_family_roc", "min_family_roc", "mean_family_pr"],
    )
    parser.add_argument("--min-features", type=int, default=1)
    parser.add_argument("--frac-select", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-batch", type=int, default=2048)
    parser.add_argument("--normal-label", default="normal")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=None, help="output CSV; JSON is written beside it")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    data_dir = Path(args.data_dir)
    mps, test_X, test_meta, feature_names = load_model_and_test(data_dir, checkpoint=args.checkpoint)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    mps = mps.to(device)

    split = split_dataset(test_X, test_meta, frac_select=args.frac_select, seed=args.seed)
    if device == "cuda":
        split = EvaluationSplit(
            X_select=split.X_select.to(device),
            meta_select=split.meta_select,
            X_eval=split.X_eval.to(device),
            meta_eval=split.meta_eval,
        )

    logger.info(
        "running backward elimination: objective=%s, min_features=%d, device=%s",
        args.objective, args.min_features, device,
    )
    trajectory = backward_eliminate(
        mps,
        split.X_select,
        split.meta_select,
        split.X_eval,
        split.meta_eval,
        feature_names,
        objective=args.objective,
        min_features=args.min_features,
        row_batch=args.row_batch,
        normal_label=args.normal_label,
    )

    out_csv = Path(args.out) if args.out else data_dir / f"backward_elimination_{args.objective}.csv"
    csv_path, json_path = save_trajectory(trajectory, out_csv)
    ranking = elimination_ranking_names(trajectory, feature_names)

    logger.info("wrote %s and %s", csv_path, json_path)
    logger.info("feature ranking, most useful first: %s", ranking)


if __name__ == "__main__":
    main()
