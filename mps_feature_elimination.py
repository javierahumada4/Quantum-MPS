"""Selection-only backward elimination + retraining for an MPS Born machine.

Flow implemented by this module
-------------------------------
1. Run greedy backward elimination on *one* labelled selection set.
2. Record the whole AUC/objective trajectory on that same selection set.
3. Choose ``k`` from those selection-set results, or use an explicit ``--k``.
4. Build a reduced training directory with only the selected columns.
5. Optionally delegate training to ``train_mps_nsl_kdd.main``.

No evaluation/test split is created or consumed by the backward-elimination
logic. Metrics are intentionally named ``select_*`` to make this explicit.

Typical usage
-------------
    python mps_backward_select_and_train.py /path/to/nsl_kdd \
        --objective auc_roc

    python mps_backward_select_and_train.py /path/to/nsl_kdd \
        --k 9

The selection set is always loaded from ``<data_dir>/fs_X.pt`` and
``<data_dir>/fs_meta.pt`` -- the feature-selection slice the encoder carves out
of KDDTrain+ (with attacks) before the train/validation split, so it never
overlaps the rows the MPS is trained or calibrated on. KDDTest+ is not accepted
as a feature-selection source; it is reserved for final evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("mps_be_select_train")
Objective = Union[str, Callable[[Dict], float]]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _torch_load(path: Union[str, Path]):
    """Load a torch artifact, supporting both newer and older torch versions."""
    try:
        return torch.load(path, weights_only=True)
    except TypeError:  # torch<2.0 compatibility
        return torch.load(path)


def _as_numpy(value) -> np.ndarray:
    """Convert tensors/lists/arrays to a CPU numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _csv_value(value):
    if isinstance(value, (list, tuple)):
        return " | ".join(map(str, value))
    return value


# ---------------------------------------------------------------------------
# Exact marginalisation over dropped MPS sites
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
# Selection-only backward elimination
# ---------------------------------------------------------------------------


def _selection_record(
    *,
    kept: Sequence[int],
    feature_names: Sequence[str],
    aucs: Dict,
    objective: float,
) -> Dict:
    record: Dict = {
        "n_features": len(kept),
        "kept_sites": list(kept),
        "kept_features": [feature_names[i] for i in kept],
        "select_auc_roc": aucs["global"]["auc_roc"],
        "select_auc_pr": aucs["global"]["auc_pr"],
        "select_objective": float(objective),
        "removed_next": None,
        "removed_next_name": None,
        "removed_next_select_objective": None,
    }
    for family_name, data in aucs.items():
        if family_name == "global":
            continue
        record[f"select_auc_roc_{family_name}"] = data["auc_roc"]
        record[f"select_auc_pr_{family_name}"] = data["auc_pr"]
        record[f"n_{family_name}"] = data["n"]
    return record


def backward_eliminate_on_selection(
    mps,
    X_select: torch.Tensor,
    meta_select: Mapping,
    feature_names: Sequence[str],
    objective: Objective = "auc_roc",
    min_features: int = 1,
    row_batch: int = 2048,
    normal_label: str = "normal",
) -> List[Dict]:
    """Run greedy backward elimination using only the selection set.

    The greedy drop decision, the recorded AUC curve, and the final ``k`` choice
    are all based on ``X_select/meta_select``. No evaluation/test rows are
    passed to this function.
    """
    n_sites = len(feature_names)
    if hasattr(mps, "num_sites") and n_sites != int(mps.num_sites):
        raise ValueError(f"feature_names has {n_sites} entries but mps.num_sites={mps.num_sites}")
    if not 1 <= min_features <= n_sites:
        raise ValueError(f"min_features must be in [1, {n_sites}]")

    if not torch.is_tensor(X_select):
        X_select = torch.as_tensor(np.asarray(X_select))

    kept = list(range(n_sites))
    trajectory: List[Dict] = []
    cache: Dict[Tuple[int, ...], Dict] = {}

    def evaluate_subset(sites: Sequence[int]) -> Dict:
        key = tuple(sites)
        if key not in cache:
            scores = marginal_anomaly_score(mps, X_select, sites, row_batch=row_batch)
            aucs = auc_global_and_per_family(scores, meta_select, normal_label=normal_label)
            cache[key] = {
                "aucs": aucs,
                "objective": objective_value(aucs, objective),
            }
        return cache[key]

    while True:
        current = evaluate_subset(kept)
        record = _selection_record(
            kept=kept,
            feature_names=feature_names,
            aucs=current["aucs"],
            objective=current["objective"],
        )

        if len(kept) <= min_features:
            trajectory.append(record)
            logger.info(
                "  n=%2d  select AUC-ROC=%.4f  select objective=%.4f  (final)",
                len(kept), record["select_auc_roc"], record["select_objective"],
            )
            break

        best_site: Optional[int] = None
        best_candidate: Optional[Dict] = None
        best_value = -np.inf
        for site in kept:
            candidate_sites = [s for s in kept if s != site]
            candidate = evaluate_subset(candidate_sites)
            value = candidate["objective"]
            if value > best_value:
                best_value = value
                best_site = site
                best_candidate = candidate

        if best_site is None or best_candidate is None:
            raise RuntimeError("no removable feature candidate was evaluated")

        record["removed_next"] = int(best_site)
        record["removed_next_name"] = feature_names[best_site]
        record["removed_next_select_objective"] = float(best_value)
        record["removed_next_select_auc_roc"] = best_candidate["aucs"]["global"]["auc_roc"]
        record["removed_next_select_auc_pr"] = best_candidate["aucs"]["global"]["auc_pr"]
        trajectory.append(record)

        obj_name = objective if isinstance(objective, str) else "custom"
        logger.info(
            "  n=%2d  select AUC-ROC=%.4f  -> drop %-24s (next select %s=%.4f)",
            len(kept), record["select_auc_roc"], feature_names[best_site], obj_name, best_value,
        )
        kept.remove(best_site)

    return trajectory


# ---------------------------------------------------------------------------
# Choosing k from selection results
# ---------------------------------------------------------------------------


def row_for_k(trajectory: Sequence[Dict], k: int) -> Dict:
    """Return the trajectory row with ``n_features == k``."""
    for row in trajectory:
        if row.get("n_features") == k:
            return row
    available = sorted({r.get("n_features") for r in trajectory if "n_features" in r})
    raise ValueError(f"No trajectory row with k={k}. Available: {available}")


def choose_k_from_selection(
    trajectory: Sequence[Dict],
    *,
    k: Optional[int] = None,
    metric: str = "select_objective",
    tie_break: str = "fewer",
) -> Dict:
    """Choose the row used for retraining from selection-only results.

    If ``k`` is given, the row with that feature count is returned. Otherwise,
    the row with the largest ``metric`` is selected. Ties are resolved by
    choosing either the smaller or larger model according to ``tie_break``.
    """
    if not trajectory:
        raise ValueError("empty trajectory")
    if k is not None:
        return row_for_k(trajectory, k)
    if tie_break not in {"fewer", "more"}:
        raise ValueError("tie_break must be 'fewer' or 'more'")

    def key(row: Dict):
        tie = -row["n_features"] if tie_break == "fewer" else row["n_features"]
        return (float(row.get(metric, float("-inf"))), tie)

    return max(trajectory, key=key)


# ---------------------------------------------------------------------------
# I/O, schemas and reduced-directory construction
# ---------------------------------------------------------------------------


def save_trajectory(trajectory: Sequence[Mapping], output_csv: Union[str, Path]) -> Tuple[Path, Path]:
    """Save trajectory to CSV and JSON. Returns ``(csv_path, json_path)``."""
    if not trajectory:
        raise ValueError("empty trajectory")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json = output_csv.with_suffix(".json")

    first_cols = [
        "n_features", "select_objective", "select_auc_roc", "select_auc_pr",
        "removed_next", "removed_next_name", "removed_next_select_objective",
        "removed_next_select_auc_roc", "removed_next_select_auc_pr",
        "kept_sites", "kept_features",
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


def load_schema(data_dir: Union[str, Path]) -> Dict:
    schema_path = Path(data_dir) / "encoding_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema["n_features"] != len(schema["physical_dims"]):
        raise ValueError("Corrupt schema: n_features != len(physical_dims).")
    if schema["n_features"] != len(schema["features"]):
        raise ValueError("Corrupt schema: n_features != len(features).")
    return schema


def load_feature_names(data_dir: Union[str, Path]) -> List[str]:
    """Load feature names from ``encoding_schema.json``."""
    schema = load_schema(data_dir)
    return [feature["name"] for feature in schema["features"]]


def load_partition(data_dir: Union[str, Path], split: str) -> Tuple[torch.Tensor, Dict]:
    """Load ``<split>_X.pt`` and ``<split>_meta.pt`` from ``data_dir``."""
    data_dir = Path(data_dir)
    x_path = data_dir / f"{split}_X.pt"
    meta_path = data_dir / f"{split}_meta.pt"
    if not x_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {x_path.name} or {meta_path.name} in {data_dir}. "
            "Run the encoder first so fs_X.pt/fs_meta.pt exist."
        )
    return _torch_load(x_path).long(), _torch_load(meta_path)


def load_partition_from_paths(x_path: Union[str, Path], meta_path: Union[str, Path]) -> Tuple[torch.Tensor, Dict]:
    """Load a custom ``X``/``meta`` pair for the selection set."""
    return _torch_load(x_path).long(), _torch_load(meta_path)


def load_model(data_dir: Union[str, Path], checkpoint: str = "mps_trained.pt"):
    """Load the full trained MPS used for feature ranking."""
    from mps import MPS

    data_dir = Path(data_dir)
    model_path = data_dir / checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"missing {model_path}")
    mps = MPS.load(str(model_path))
    if hasattr(mps, "eval"):
        mps.eval()
    logger.info("loaded MPS: %d sites", mps.num_sites)
    return mps


def subset_schema(schema: Dict, sites: Sequence[int]) -> Dict:
    """Build a reduced encoding schema keeping only ``sites``.

    Each surviving feature keeps its encoding and records ``original_site`` for
    provenance; ``site`` is renumbered so the file is drop-in for training.
    """
    features = schema["features"]
    out_features: List[Dict] = []
    for new_site, original_site in enumerate(sites):
        entry = dict(features[original_site])
        entry["original_site"] = original_site
        entry["site"] = new_site
        out_features.append(entry)
    return {
        "n_features": len(sites),
        "physical_dims": [schema["physical_dims"][s] for s in sites],
        "features": out_features,
    }


def _subset_split_columns(
    data_dir: Path,
    in_split: str,
    sites: Sequence[int],
    n_features: int,
    out_dir: Path,
    out_prefix: str,
    *,
    required: bool,
) -> Optional[Tuple[int, int]]:
    """Column-subset one ``<in_split>_X.pt``/meta and save it as ``<out_prefix>_*``.

    Rows are untouched (so seeded splits reconstructed downstream stay identical);
    only the kept feature columns survive. Returns ``(n_rows, k)`` or ``None`` when
    the split is absent and ``required`` is False.
    """
    x_path = data_dir / f"{in_split}_X.pt"
    meta_path = data_dir / f"{in_split}_meta.pt"
    if not x_path.exists() or not meta_path.exists():
        if required:
            raise FileNotFoundError(f"Missing {x_path.name} or {meta_path.name} in {data_dir}.")
        logger.warning("split %r not found in %s; skipping", in_split, data_dir)
        return None

    X = _torch_load(x_path).long()
    meta = _torch_load(meta_path)
    if X.dim() != 2 or X.shape[1] != n_features:
        raise ValueError(
            f"{in_split}_X has shape {tuple(X.shape)}, incompatible with a {n_features}-site schema."
        )
    site_index = torch.as_tensor(list(sites), dtype=torch.long, device=X.device)
    X_reduced = X.index_select(1, site_index).contiguous()
    torch.save(X_reduced.cpu(), out_dir / f"{out_prefix}_X.pt")
    torch.save(meta, out_dir / f"{out_prefix}_meta.pt")
    logger.info("%s -> %s: %s -> %s", in_split, out_prefix, tuple(X.shape), tuple(X_reduced.shape))
    return int(X.shape[0]), int(X_reduced.shape[1])


def build_reduced_training_dir(
    data_dir: Union[str, Path],
    sites: Sequence[int],
    out_dir: Union[str, Path],
    *,
    train_split: str = "train",
    train_normal_split: Optional[str] = "train_normal",
    val_normal_split: Optional[str] = "val_normal",
    fs_split: Optional[str] = "fs",
    selection_split: Optional[str] = None,
    evaluation_split: Optional[str] = "evaluation",
    write_test_alias: bool = False,
    selected_row: Optional[Mapping] = None,
    trajectory_path: Optional[Path] = None,
) -> Path:
    """Write a reduced directory with only the selected feature columns.

    Always writes the reduced training partitions: the full ``train`` (kept for
    the explainer baseline) and the normal-only ``train_normal`` / ``val_normal``
    pair the reduced trainer actually consumes. The normal split is produced by
    the encoder, so here it is only column-subset (rows untouched), keeping it
    identical to the full-feature run. When present, the feature-selection set
    ``fs`` and the full KDDTest+ ``evaluation`` split are written too. No
    KDDTest+ selection partition is copied or created; KDDTest+ is reserved for
    final evaluation only. The legacy ``test_*`` alias is disabled by default.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = load_schema(data_dir)
    sites = sorted(int(s) for s in sites)
    if len(set(sites)) != len(sites):
        raise ValueError(f"Duplicate sites in selection: {sites}")
    for s in sites:
        if not 0 <= s < schema["n_features"]:
            raise ValueError(f"Site {s} out of range [0, {schema['n_features']}).")
    n_features = schema["n_features"]

    written: Dict[str, str] = {}
    _subset_split_columns(data_dir, train_split, sites, n_features, out_dir, "train", required=True)
    written["train"] = train_split

    # Normal-only train/val split (what the reduced trainer reads).
    if train_normal_split is not None:
        _subset_split_columns(
            data_dir, train_normal_split, sites, n_features, out_dir, "train_normal", required=True
        )
        written["train_normal"] = train_normal_split
    if val_normal_split is not None:
        if _subset_split_columns(
            data_dir, val_normal_split, sites, n_features, out_dir, "val_normal", required=False
        ):
            written["val_normal"] = val_normal_split

    # Feature-selection set (KDDTrain+ slice with attacks), carried through so
    # the reduced directory is self-contained.
    if fs_split is not None:
        if _subset_split_columns(data_dir, fs_split, sites, n_features, out_dir, "fs", required=False):
            written["fs"] = fs_split

    if selection_split is not None:
        if _subset_split_columns(data_dir, selection_split, sites, n_features, out_dir, "selection", required=False):
            written["selection"] = selection_split
    if evaluation_split is not None:
        got_eval = _subset_split_columns(data_dir, evaluation_split, sites, n_features, out_dir, "evaluation", required=False)
        if got_eval:
            written["evaluation"] = evaluation_split
            if write_test_alias:
                _subset_split_columns(data_dir, evaluation_split, sites, n_features, out_dir, "test", required=False)
                written["test"] = f"{evaluation_split} (legacy alias)"

    reduced_schema = subset_schema(schema, sites)
    (out_dir / "encoding_schema.json").write_text(json.dumps(reduced_schema, indent=2), encoding="utf-8")

    provenance: Dict = {
        "k": len(sites),
        "kept_sites": sites,
        "kept_features": [f["name"] for f in reduced_schema["features"]],
        "source_data_dir": str(data_dir),
        "source_train_split": train_split,
        "source_schema_n_features": schema["n_features"],
        "selection_only": True,
        "written_splits": written,
    }
    if trajectory_path is not None:
        provenance["selection_trajectory"] = str(trajectory_path)
    if selected_row is not None:
        provenance["selection_reference"] = {
            key: selected_row[key]
            for key in (
                "n_features", "select_objective", "select_auc_roc", "select_auc_pr",
                "kept_sites", "kept_features",
            )
            if key in selected_row
        }
    (out_dir / "feature_selection.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info("reduced directory ready: %s  (splits: %s)", out_dir, ", ".join(sorted(written)))
    return out_dir


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def train_reduced_model(out_dir: Union[str, Path]) -> None:
    """Delegate model training to the existing project training entry point."""
    from train_mps_nsl_kdd import main as train_main

    train_main(Path(out_dir))


def run(
    data_dir: Union[str, Path],
    *,
    checkpoint: str = "mps_trained.pt",
    selection_split: str = "fs",
    selection_x: Optional[Union[str, Path]] = None,
    selection_meta: Optional[Union[str, Path]] = None,
    train_split: str = "train",
    train_normal_split: str = "train_normal",
    val_normal_split: str = "val_normal",
    evaluation_split: str = "evaluation",
    write_test_alias: bool = False,
    objective: Objective = "auc_roc",
    min_features: int = 1,
    row_batch: int = 2048,
    normal_label: str = "normal",
    k: Optional[int] = None,
    tie_break: str = "fewer",
    device: str = "auto",
    trajectory_csv: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    train: bool = True,
) -> Tuple[List[Dict], Dict, Path]:
    """Run selection-only BE, choose k, build reduced data and optionally train."""
    data_dir = Path(data_dir)
    feature_names = load_feature_names(data_dir)
    mps = load_model(data_dir, checkpoint=checkpoint)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    mps = mps.to(device)

    if selection_split != "fs":
        raise ValueError(
            "Feature selection must use fs_X.pt/fs_meta.pt from KDDTrain+. "
            f"Refusing selection_split={selection_split!r}."
        )
    if selection_x is not None or selection_meta is not None:
        raise ValueError(
            "Explicit --selection-x/--selection-meta is disabled: feature selection "
            "must use fs_X.pt/fs_meta.pt from KDDTrain+."
        )

    X_select, meta_select = load_partition(data_dir, "fs")
    selection_source = "fs_X.pt / fs_meta.pt"

    X_select = X_select.to(device) if device == "cuda" else X_select
    logger.info(
        "running selection-only backward elimination: selection=%s, objective=%s, min_features=%d, device=%s",
        selection_source, objective if isinstance(objective, str) else "custom", min_features, device,
    )

    trajectory = backward_eliminate_on_selection(
        mps,
        X_select,
        meta_select,
        feature_names,
        objective=objective,
        min_features=min_features,
        row_batch=row_batch,
        normal_label=normal_label,
    )

    if trajectory_csv is None:
        objective_name = objective if isinstance(objective, str) else "custom"
        trajectory_csv = data_dir / f"backward_elimination_selection_{objective_name}.csv"
    csv_path, json_path = save_trajectory(trajectory, trajectory_csv)
    logger.info("wrote %s and %s", csv_path, json_path)

    selected_row = choose_k_from_selection(trajectory, k=k, tie_break=tie_break)
    selected_sites = list(selected_row["kept_sites"])
    k_eff = selected_row["n_features"]
    if out_dir is None:
        out_dir = f"k_selection_only"

    logger.info(
        "chosen k=%d from selection results: select objective=%.4f, select AUC-ROC=%.4f",
        k_eff, selected_row["select_objective"], selected_row["select_auc_roc"],
    )
    logger.info("selected features: %s", selected_row["kept_features"])

    out_dir = build_reduced_training_dir(
        data_dir,
        selected_sites,
        out_dir,
        train_split=train_split,
        train_normal_split=train_normal_split,
        val_normal_split=val_normal_split,
        fs_split="fs",
        selection_split=None,
        evaluation_split=evaluation_split,
        write_test_alias=write_test_alias,
        selected_row=selected_row,
        trajectory_path=json_path,
    )

    if train:
        logger.info("training reduced model in %s", out_dir)
        train_reduced_model(out_dir)
    else:
        logger.info("--no-train: directory built, skipping training")

    return trajectory, selected_row, Path(out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selection-only backward elimination followed by reduced-feature MPS training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_dir", type=Path, help="directory with full encoder artifacts and full MPS checkpoint")
    parser.add_argument("--checkpoint", default="mps_trained.pt", help="full MPS checkpoint used for feature ranking")

    selection = parser.add_argument_group("selection set used by backward elimination")
    selection.add_argument(
        "--selection-split", default="fs",
        help="must remain 'fs': backward elimination always uses fs_X.pt/fs_meta.pt from KDDTrain+",
    )

    parser.add_argument(
        "--objective", default="auc_roc",
        choices=["auc_roc", "auc_pr", "mean_family_roc", "min_family_roc", "mean_family_pr"],
        help="selection-set objective maximised by the greedy elimination and automatic k choice",
    )
    parser.add_argument("--min-features", type=int, default=1, help="smallest subset considered during BE")
    parser.add_argument("--row-batch", type=int, default=2048, help="row batch size for marginalisation")
    parser.add_argument("--normal-label", default="normal", help="normal class label in meta['family_names']")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="device for BE contractions")

    choose = parser.add_argument_group("k choice")
    choose.add_argument("--k", type=int, default=None, help="explicit k; otherwise best select_objective is chosen")
    choose.add_argument(
        "--tie-break", choices=["fewer", "more"], default="fewer",
        help="automatic-k tie break when several rows have the same selection metric",
    )

    train = parser.add_argument_group("reduced training directory")
    train.add_argument("--train-split", default="train", help="load <train-split>_X.pt/meta and write it as train_X.pt/meta")
    train.add_argument("--train-normal-split", default="train_normal",
                       help="normal-only training split to column-subset into the reduced directory")
    train.add_argument("--val-normal-split", default="val_normal",
                       help="normal-only validation split to column-subset into the reduced directory")
    train.add_argument("--evaluation-split", default="evaluation",
                       help="evaluation split to column-subset and copy into the reduced directory")
    train.add_argument("--write-test-alias", action="store_true",
                       help="also write evaluation as legacy test_* alias (off by default)")
    train.add_argument("--out-dir", type=Path, default=None, help="output reduced directory")
    train.add_argument("--trajectory-csv", type=Path, default=None, help="where to save the selection-only BE CSV")
    train.add_argument("--no-train", action="store_true", help="build reduced directory only; do not train")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    run(
        args.data_dir,
        checkpoint=args.checkpoint,
        selection_split=args.selection_split,
        train_split=args.train_split,
        train_normal_split=args.train_normal_split,
        val_normal_split=args.val_normal_split,
        evaluation_split=args.evaluation_split,
        write_test_alias=args.write_test_alias,
        objective=args.objective,
        min_features=args.min_features,
        row_batch=args.row_batch,
        normal_label=args.normal_label,
        k=args.k,
        tie_break=args.tie_break,
        device=args.device,
        trajectory_csv=args.trajectory_csv,
        out_dir=args.out_dir,
        train=not args.no_train,
    )


if __name__ == "__main__":
    main()