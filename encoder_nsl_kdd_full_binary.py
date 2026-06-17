"""Full-binary NSL-KDD encoder, self-contained.

Maps every NSL-KDD feature onto a single binary MPS site (``d = 2``), so the
chain is a uniform string of qubits ready for a state-preparation circuit. This
is the consolidation of what used to be three files (the base encoder, the
numeric-only binary encoder, and the full-binary encoder): only the machinery
needed for the full-binary encoding is kept.

The chain is 40 sites (41 NSL-KDD features minus the zero-variance
``num_outbound_cmds``), all ``d = 2``:

* numeric features -> one binary site, decided on *normal* traffic:
    - constant / quasi-constant in normal -> ``constant_normal`` site, encoded as
      "equals the normal value (0) vs differs (1)";
    - otherwise -> ``numeric`` site split at the normal median ("<= median (0)
      vs > median (1)"), falling back to same-vs-different if that split is
      degenerate on normal traffic.
* the three string categoricals (``protocol_type``, ``service``, ``flag``) ->
  one binary ``categorical_binary`` site each. A site is 0 when the value is in a
  set ``S`` derived from normal traffic, 1 otherwise. ``S`` is built one of two
  ways, selected by ``categorical_strategy``:
    - ``"unknown"``  : ``S`` = every value seen in normal -> a *novelty* flag,
      constant (0) on the all-normal training data, firing only on unseen values.
    - ``"frequency"``: ``S`` = values at least ``frequency_threshold`` frequent in
      normal -> a *rarity* flag that also carries within-normal variance.

The fitted state is one :class:`FeatureSpec` per site; ``physical_dims`` is just
their ``d`` values (all 2s) and is what the MPS constructor needs. Run as a
script it reads ``KDDTrain+.txt`` / ``KDDTest+.txt`` and writes the encoded
tensors, per-split metadata and ``encoding_schema.json``:

    python encoder_nsl_kdd_full_binary.py ./nsl_kdd ./nsl_kdd_fullbin_freq frequency
    python encoder_nsl_kdd_full_binary.py ./nsl_kdd ./nsl_kdd_fullbin_unk  unknown
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------

COLUMNS: List[str] = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CATEGORICAL_COLS = {"protocol_type", "service", "flag"}
DROP_COLS = {"num_outbound_cmds"}                  # zero-variance feature
META_COLS = {"label", "difficulty"}                # metadata, not model inputs

ATTACK_FAMILY: Dict[str, str] = {
    "normal": "normal",
    # DoS (10)
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    # Probe (6)
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L (15)
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "sendmail": "r2l", "named": "r2l", "worm": "r2l",
    # U2R (8)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
    "httptunnel": "u2r",
}

# Site kinds produced by this encoder.
KIND_CONSTANT = "constant_normal"        # same-vs-different on a reference value
KIND_NUMERIC = "numeric"                 # threshold at the normal median
KIND_CATEGORICAL = "categorical_binary"  # in-normal-set vs not


# ----------------------------------------------------------------------
# Feature specification
# ----------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """How a single feature is encoded into one binary site (``d == 2``)."""
    name: str
    kind: str                                # KIND_CONSTANT | KIND_NUMERIC | KIND_CATEGORICAL
    d: int                                   # always 2 here
    vocab: Optional[List] = None             # set S of normal values (categorical_binary)
    edges: Optional[List[float]] = None      # [-inf, median, inf] (numeric)
    normal_value: Optional[float] = None     # reference value (constant_normal)


class EncodingError(ValueError):
    """Encoded data violates the schema (e.g. a column outside [0, d))."""


# ----------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------

class NSLKDDFullBinaryEncoder:
    """Encode every NSL-KDD feature as one binary site (``d = 2``).

    Parameters
    ----------
    categorical_strategy:
        ``"unknown"``  -> categorical site is 0 for any value seen in normal,
        1 otherwise (novelty flag, constant on the all-normal training data).
        ``"frequency"`` -> site is 0 for values at least ``frequency_threshold``
        frequent in normal, 1 otherwise (rarity flag, carries within-normal
        variance).
    frequency_threshold:
        Minimum normalised frequency in normal traffic for a categorical value to
        count as "common" (``"frequency"`` strategy only). E.g. ``0.01`` -> a
        category must be >= 1% of normal rows to map to 0.
    quasi_constant_threshold:
        Mode share in normal above which a numeric feature is treated as
        constant-in-normal and collapsed to a same-vs-different site.
    degenerate_eps:
        Minimum share of normal rows that must fall on each side of a numeric
        median split for it to be accepted; below it the feature falls back to
        same-vs-different.
    """

    def __init__(
        self,
        categorical_strategy: str = "frequency",
        frequency_threshold: float = 0.01,
        quasi_constant_threshold: float = 0.95,
        degenerate_eps: float = 0.01,
    ) -> None:
        if categorical_strategy not in ("unknown", "frequency"):
            raise ValueError(
                "categorical_strategy must be 'unknown' or 'frequency', "
                f"got {categorical_strategy!r}"
            )
        if not (0.0 < frequency_threshold < 1.0):
            raise ValueError(f"frequency_threshold must be in (0, 1), got {frequency_threshold}")
        if not (0.5 < quasi_constant_threshold < 1.0):
            raise ValueError(f"quasi_constant_threshold must be in (0.5, 1.0), got {quasi_constant_threshold}")
        if not (0.0 <= degenerate_eps < 0.5):
            raise ValueError(f"degenerate_eps must be in [0, 0.5), got {degenerate_eps}")

        self.categorical_strategy = categorical_strategy
        self.frequency_threshold = frequency_threshold
        self.quasi_constant_threshold = quasi_constant_threshold
        self.degenerate_eps = degenerate_eps
        self.specs: List[FeatureSpec] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> "NSLKDDFullBinaryEncoder":
        """Build one binary FeatureSpec per feature from normal-traffic values.

        Expects ``df_train`` with all NSL-KDD columns including ``label``; the
        ``label == 'normal'`` subset is what every fitting decision looks at.
        """
        normal_mask = (df_train["label"].str.rstrip(".") == "normal").to_numpy()
        df_normal = df_train.loc[normal_mask]

        specs: List[FeatureSpec] = []
        for col in COLUMNS:
            if col in DROP_COLS or col in META_COLS:
                continue
            if col in CATEGORICAL_COLS:
                specs.append(self._fit_categorical_binary(col, df_normal[col]))
            else:
                specs.append(self._fit_numeric_binary(col, df_normal[col]))

        self.specs = specs
        return self

    def _fit_numeric_binary(self, col: str, x_normal: pd.Series) -> FeatureSpec:
        """Decide the binary site for one numeric column.

        3-step policy on normal traffic: constant -> quasi-constant -> median
        threshold (with a degeneracy fallback to same-vs-different).
        """
        x_normal = x_normal.astype(float)

        # (1) constant in normal -> same-vs-different
        if int(x_normal.nunique()) == 1:
            return FeatureSpec(name=col, kind=KIND_CONSTANT, d=2,
                               normal_value=float(x_normal.iloc[0]))

        value_counts = x_normal.value_counts()
        mode_value = float(value_counts.index[0])
        mode_share = float(value_counts.iloc[0]) / float(len(x_normal))

        # (2) quasi-constant in normal -> same-vs-different
        if mode_share >= self.quasi_constant_threshold:
            return FeatureSpec(name=col, kind=KIND_CONSTANT, d=2, normal_value=mode_value)

        # (3) otherwise -> median threshold, unless degenerate on normal
        arr = x_normal.to_numpy()
        median = float(np.median(arr))
        edges = [-np.inf, median, np.inf]
        codes = pd.cut(arr, bins=edges, labels=False, include_lowest=True)
        share_high = float(np.mean(codes == 1))

        if share_high < self.degenerate_eps or share_high > 1.0 - self.degenerate_eps:
            logger.info(
                "  %s: median split degenerate (share_high=%.4f); "
                "using same-vs-different on mode=%.4g", col, share_high, mode_value,
            )
            return FeatureSpec(name=col, kind=KIND_CONSTANT, d=2, normal_value=mode_value)

        return FeatureSpec(name=col, kind=KIND_NUMERIC, d=2, edges=edges)

    def _fit_categorical_binary(self, col: str, x_normal: pd.Series) -> FeatureSpec:
        """Decide the binary site for one categorical column.

        ``vocab`` stores the set ``S`` of normal values that map to 0; everything
        else (rare and/or unseen) maps to 1 at transform time.
        """
        values = x_normal.astype(str)

        if self.categorical_strategy == "unknown":
            normal_set = sorted(values.unique().tolist())
        else:  # "frequency"
            freq = values.value_counts(normalize=True)
            common = freq.index[freq >= self.frequency_threshold].tolist()
            if not common:
                common = [freq.index[0]]
                logger.info(
                    "  %s: frequency_threshold=%.3g excludes every category; "
                    "keeping mode %r as the only common value",
                    col, self.frequency_threshold, common[0],
                )
            normal_set = sorted(common)

        return FeatureSpec(name=col, kind=KIND_CATEGORICAL, d=2, vocab=normal_set)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Map a DataFrame onto a (n_rows, n_features) LongTensor of 0/1 codes."""
        if not self.specs:
            raise RuntimeError("Encoder not fitted yet; call fit() first.")
        cols_out = [self._transform_one(spec, df[spec.name]) for spec in self.specs]
        arr = np.stack(cols_out, axis=1).astype(np.int64)
        return torch.from_numpy(arr)

    def _transform_one(self, spec: FeatureSpec, x: pd.Series) -> np.ndarray:
        """Apply one fitted ``FeatureSpec`` to a column, returning 0/1 codes."""
        if spec.kind == KIND_CONSTANT:
            arr = x.astype(float).to_numpy()
            return (~np.isclose(arr, spec.normal_value)).astype(np.int64)

        if spec.kind == KIND_NUMERIC:
            arr = x.astype(float).to_numpy()
            out = pd.cut(arr, bins=spec.edges, labels=False, include_lowest=True).astype(np.int64)
            if np.isnan(out).any():
                raise EncodingError(f"NaN bins for feature {spec.name!r}; edges={spec.edges}")
            return out

        if spec.kind == KIND_CATEGORICAL:
            arr = x.astype(str).to_numpy()
            normal = np.asarray(spec.vocab, dtype=object)
            return (~np.isin(arr, normal)).astype(np.int64)

        raise ValueError(f"unknown FeatureSpec.kind: {spec.kind!r}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def physical_dims(self) -> List[int]:
        """Per-site physical dimensions (all 2s) — the MPS shape."""
        return [s.d for s in self.specs]

    @property
    def feature_names(self) -> List[str]:
        """Feature name at each site, in MPS order."""
        return [s.name for s in self.specs]

    def schema_dict(self) -> Dict:
        """Serialisable description of the fitted encoding.

        One entry per site (name, kind, dimension, and whichever of vocab / edges
        / normal_value applies) plus the top-level ``physical_dims``. Same format
        the training / evaluation / explainability steps read back, so it stays
        drop-in. Infinite bin edges are surfaced as ``null``.
        """
        out: List[Dict] = []
        for i, s in enumerate(self.specs):
            entry: Dict = {"site": i, "name": s.name, "kind": s.kind, "d": s.d}
            if s.vocab is not None:
                entry["vocab"] = [
                    v.item() if isinstance(v, np.generic) else v for v in s.vocab
                ]
            if s.edges is not None:
                entry["edges"] = [None if not np.isfinite(e) else float(e) for e in s.edges]
            if s.normal_value is not None:
                entry["normal_value"] = float(s.normal_value)
            out.append(entry)
        return {
            "n_features": len(self.specs),
            "physical_dims": self.physical_dims,
            "features": out,
        }


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def load_split(path: Path) -> pd.DataFrame:
    """Read one NSL-KDD split and attach the derived ``family`` / ``is_attack``."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: esperadas {len(COLUMNS)} columnas, halladas {df.shape[1]}")
    df.columns = COLUMNS
    df["label"] = df["label"].str.rstrip(".")
    df["family"] = df["label"].map(ATTACK_FAMILY)
    if df["family"].isna().any():
        unknown = sorted(df.loc[df["family"].isna(), "label"].unique())
        raise ValueError(f"Unknown labels in {path.name}: {unknown}")
    df["is_attack"] = (df["family"] != "normal").astype(int)
    return df


def build_meta(df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """Pack the labels/metadata side-by-side with the encoded X."""
    families_sorted = sorted(set(ATTACK_FAMILY.values()))
    family_to_code = {f: i for i, f in enumerate(families_sorted)}
    return {
        "is_attack": torch.tensor(df["is_attack"].to_numpy(), dtype=torch.long),
        "family_code": torch.tensor(df["family"].map(family_to_code).to_numpy(), dtype=torch.long),
        "family_names": families_sorted,
        "difficulty": torch.tensor(df["difficulty"].to_numpy(), dtype=torch.long),
        "label": df["label"].tolist(),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(data_dir: Path, out_dir: Path, strategy: str) -> None:
    """Fit on the train split with ``strategy`` and write the artefacts to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(data_dir / "KDDTrain+.txt")
    test = load_split(data_dir / "KDDTest+.txt")
    logger.info("loaded: %d train rows, %d test rows", len(train), len(test))

    encoder = NSLKDDFullBinaryEncoder(categorical_strategy=strategy)
    encoder.fit(train)

    train_X = encoder.transform(train)
    test_X = encoder.transform(test)

    physical_dims = encoder.physical_dims
    if any(d != 2 for d in physical_dims):
        raise EncodingError(
            "full-binary encoder produced non-binary sites: "
            f"{[(n, d) for n, d in zip(encoder.feature_names, physical_dims) if d != 2]}"
        )

    for split_name, X in (("train", train_X), ("test", test_X)):
        col_max = X.max(dim=0).values
        col_min = X.min(dim=0).values
        for k, (lo, hi, d) in enumerate(zip(col_min.tolist(), col_max.tolist(), physical_dims)):
            if lo < 0 or hi >= d:
                raise EncodingError(
                    f"{split_name}: site {k} ({encoder.feature_names[k]}) "
                    f"has range [{lo}, {hi}] outside [0, {d})"
                )

    # categorical-site diagnostics on normal training rows
    normal_mask = (train["label"].str.rstrip(".") == "normal").to_numpy()
    for k, spec in enumerate(encoder.specs):
        if spec.kind == KIND_CATEGORICAL:
            share_one = float(train_X[normal_mask, k].float().mean().item())
            logger.info(
                "  categorical %-13s (strategy=%s): |S|=%d kept, %.2f%% of normal rows map to 1",
                spec.name, strategy, len(spec.vocab), 100.0 * share_one,
            )

    logger.info("schema: %d sites, all d=2  ->  %d physical qubits for the chain",
                len(encoder.specs), len(encoder.specs))
    logger.info("  by kind: %s", dict(Counter(s.kind for s in encoder.specs)))

    torch.save(train_X, out_dir / "train_X.pt")
    torch.save(test_X, out_dir / "test_X.pt")
    torch.save(build_meta(train), out_dir / "train_meta.pt")
    torch.save(build_meta(test), out_dir / "test_meta.pt")
    (out_dir / "encoding_schema.json").write_text(json.dumps(encoder.schema_dict(), indent=2))

    logger.info("wrote artefacts to %s/ (strategy=%s)", out_dir, strategy)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./nsl_kdd")
    strategy = sys.argv[3] if len(sys.argv) > 3 else "frequency"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / f"fullbin_{strategy}"
    main(data_dir, out_dir, strategy)
