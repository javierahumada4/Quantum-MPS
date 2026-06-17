"""Binary NSL-KDD encoder: every site is a qubit (d = 2).

A variant of :class:`encoder_nsl_kdd.NSLKDDEncoder` built for deploying the MPS
Born machine on a quantum computer, where each site has to map onto a qubit.
Two changes versus the base encoder:

* the three string-categorical features (``protocol_type``, ``service``,
  ``flag``) are dropped entirely.  They are the only features whose natural
  encoding needs more than one qubit (``service`` alone is ~7 levels), so
  removing them is what keeps the chain uniform and the state-preparation
  circuit clean.
* every remaining feature is forced to physical dimension ``d = 2``, instead of
  the multi-level quantile bins / low-cardinality vocabularies the base encoder
  produces.

The result is a chain of 37 qubit-sized sites (41 NSL-KDD features, minus the
zero-variance ``num_outbound_cmds`` and the 3 categoricals).  Every other piece
of machinery -- ``transform``, ``schema_dict``, ``physical_dims``, the I/O
helpers -- is inherited unchanged, so the artefacts written here are drop-in
compatible with the training / evaluation / explainability steps; only
``physical_dims`` differs (it is now all 2s).

Binarisation policy for a non-categorical feature, decided on *normal* traffic:

* constant or quasi-constant in normal  ->  ``constant_normal`` site, encoded as
  "equals the normal value (0) vs differs (1)";
* otherwise  ->  ``numeric`` site split at the normal median, i.e.
  "<= median (0) vs > median (1)".  If that split turns out degenerate on normal
  traffic (essentially everything on one side, e.g. the median coincides with
  the maximum) it falls back to the same-vs-different rule above, which is never
  degenerate once quasi-constant features have been filtered out.

Run as a script it mirrors ``encoder_nsl_kdd.main`` but writes its artefacts to
a *separate* output directory, leaving the full-resolution encoding intact so
the two can be compared side by side:

    python encoder_nsl_kdd_binary.py ./nsl_kdd ./nsl_kdd_binary
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from encoder_nsl_kdd import (
    CATEGORICAL_COLS,
    COLUMNS,
    DROP_COLS,
    META_COLS,
    EncodingError,
    FeatureSpec,
    NSLKDDEncoder,
    build_meta,
    load_split,
)

logger = logging.getLogger(__name__)


class NSLKDDBinaryEncoder(NSLKDDEncoder):
    """NSL-KDD encoder where every site is binary (``d = 2``).

    Same fitted-state contract as the base class (one :class:`FeatureSpec` per
    site, ``physical_dims``, ``feature_names``, ``schema_dict``), but:

    * categorical string features are skipped during :meth:`fit`;
    * :meth:`_fit_one` only ever returns ``d = 2`` specs.

    ``degenerate_eps`` is the minimum share of normal rows that must fall on each
    side of a median split for it to be accepted; below it, the feature falls
    back to a same-vs-different binary site.
    """

    def __init__(
        self,
        quasi_constant_threshold: float = 0.95,
        degenerate_eps: float = 0.01,
    ) -> None:
        # target_d_numeric / max_implicit_categorical are irrelevant in binary
        # mode (everything is forced to d = 2); pass valid placeholders so the
        # base validation is happy.
        super().__init__(
            target_d_numeric=2,
            max_implicit_categorical=2,
            quasi_constant_threshold=quasi_constant_threshold,
        )
        if not (0.0 <= degenerate_eps < 0.5):
            raise ValueError(
                f"degenerate_eps must be in [0, 0.5), got {degenerate_eps}"
            )
        self.degenerate_eps = degenerate_eps

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> "NSLKDDBinaryEncoder":
        """Build one binary FeatureSpec per non-categorical feature.

        Identical to the base ``fit`` except that the three categorical string
        columns are skipped, so they never reach :meth:`_fit_one`.
        """
        normal_mask = (df_train["label"].str.rstrip(".") == "normal").to_numpy()
        df_normal = df_train.loc[normal_mask]

        specs = []
        for col in COLUMNS:
            if col in DROP_COLS or col in META_COLS:
                continue
            if col in CATEGORICAL_COLS:          # <-- drop categoricals
                continue
            specs.append(self._fit_one(col, df_normal[col]))

        self.specs = specs
        return self

    def _fit_one(self, col: str, x_normal: pd.Series) -> FeatureSpec:
        """Decide a binary (``d = 2``) encoding for one numeric column.

        ``x_normal`` is the column restricted to normal training rows.  Walks a
        3-step policy: constant -> quasi-constant -> median split (with a
        degeneracy fallback). Categorical columns never get here.
        """
        x_normal = x_normal.astype(float)
        n_unique_normal = int(x_normal.nunique())

        # (1) constant in normal  ->  same-vs-different
        if n_unique_normal == 1:
            return FeatureSpec(
                name=col, kind="constant_normal", d=2,
                normal_value=float(x_normal.iloc[0]),
            )

        value_counts = x_normal.value_counts()
        mode_value = float(value_counts.index[0])
        mode_share = float(value_counts.iloc[0]) / float(len(x_normal))

        # (2) quasi-constant in normal  ->  same-vs-different
        if mode_share >= self.quasi_constant_threshold:
            return FeatureSpec(
                name=col, kind="constant_normal", d=2, normal_value=mode_value,
            )

        # (3) otherwise  ->  median threshold, unless degenerate on normal
        arr = x_normal.to_numpy()
        median = float(np.median(arr))
        edges = [-np.inf, median, np.inf]
        codes = pd.cut(arr, bins=edges, labels=False, include_lowest=True)
        share_high = float(np.mean(codes == 1))

        if share_high < self.degenerate_eps or share_high > 1.0 - self.degenerate_eps:
            # The median sits at an extreme value; a threshold here would leave
            # the site near-constant on normal traffic. Fall back to the mode
            # same-vs-different rule, which always splits with >= 5% on each side
            # because quasi-constant features were already handled in step (2).
            logger.info(
                "  %s: median split degenerate (share_high=%.4f); "
                "using same-vs-different on mode=%.4g",
                col, share_high, mode_value,
            )
            return FeatureSpec(
                name=col, kind="constant_normal", d=2, normal_value=mode_value,
            )

        return FeatureSpec(name=col, kind="numeric", d=2, edges=edges)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(data_dir: Path, out_dir: Path) -> None:
    """Fit the binary encoder on the train split and write the artefacts.

    Reads ``KDDTrain+.txt`` / ``KDDTest+.txt`` from ``data_dir`` (reusing the
    base module's :func:`load_split`), transforms both splits, sanity-checks
    every encoded column is in ``[0, 2)``, and writes the encoded tensors,
    per-split metadata and ``encoding_schema.json`` into ``out_dir`` (created if
    needed) so the full-resolution encoding in ``data_dir`` is left untouched.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(data_dir / "KDDTrain+.txt")
    test = load_split(data_dir / "KDDTest+.txt")
    logger.info("loaded: %d train rows, %d test rows", len(train), len(test))

    encoder = NSLKDDBinaryEncoder()
    encoder.fit(train)

    train_X = encoder.transform(train)
    test_X = encoder.transform(test)

    physical_dims = encoder.physical_dims
    if any(d != 2 for d in physical_dims):
        raise EncodingError(
            f"binary encoder produced non-binary sites: "
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

    logger.info(
        "schema: %d sites, all d=2  ->  %d physical qubits for the chain",
        len(encoder.specs), len(encoder.specs),
    )
    kind_counts = Counter(s.kind for s in encoder.specs)
    logger.info("  by kind: %s", dict(kind_counts))

    torch.save(train_X, out_dir / "train_X.pt")
    torch.save(test_X, out_dir / "test_X.pt")
    torch.save(build_meta(train), out_dir / "train_meta.pt")
    torch.save(build_meta(test), out_dir / "test_meta.pt")

    schema_json = out_dir / "encoding_schema.json"
    schema_json.write_text(json.dumps(encoder.schema_dict(), indent=2))

    logger.info(
        "wrote artefacts to %s/ "
        "(train_X.pt, test_X.pt, train_meta.pt, test_meta.pt, encoding_schema.json)",
        out_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./nsl_kdd")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "binary"
    main(data_dir, out_dir)
