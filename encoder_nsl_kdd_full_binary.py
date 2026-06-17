"""Full-binary NSL-KDD encoder: every feature is one qubit (d = 2), including
the categoricals.

Extends :class:`encoder_nsl_kdd_binary.NSLKDDBinaryEncoder` (which already maps
every *numeric* feature to a single binary site and drops the categoricals) by
bringing the three string-categorical features -- ``protocol_type``,
``service``, ``flag`` -- back as one binary site each, instead of dropping them.
The chain is therefore 40 sites (41 NSL-KDD features minus the zero-variance
``num_outbound_cmds``), all d = 2.

Both categorical encodings share the same mechanism -- a site is 0 when the
value belongs to a set ``S`` derived from normal traffic, and 1 otherwise -- and
differ only in how ``S`` is built:

* ``categorical_strategy="unknown"``  (seen-in-normal vs UNKNOWN)
    ``S`` = every value observed in normal training traffic. The site is then 0
    for all normal rows and only ever 1 for a value never seen in normal, i.e. a
    *pure novelty detector*: it is constant on the training data (which is all
    normal) and fires only on genuinely unseen categories at test time. This is
    the natural "a service never seen in normal traffic is suspicious" signal.

* ``categorical_strategy="frequency"``  (common vs rare in normal)
    ``S`` = values whose normal frequency is at least ``frequency_threshold``.
    The site is 0 for common values and 1 for rare-but-seen *and* unseen values,
    i.e. a *rarity detector*: it carries within-normal variance (rare normal
    values already read 1), so the model learns a base rate of rarity rather
    than a hard novelty flag.

Comparing the two answers a concrete question: how much of the detection signal
lives in "this exact category" vs merely "this category is uncommon/novel".

Everything else -- numeric binarisation, ``transform``, ``physical_dims``,
``schema_dict``, I/O -- is inherited; the encoded ``S`` is stored in the spec's
``vocab`` field, so the schema round-trips with no extra machinery.

    python encoder_nsl_kdd_full_binary.py ./nsl_kdd ./nsl_kdd_fullbin_unknown unknown
    python encoder_nsl_kdd_full_binary.py ./nsl_kdd ./nsl_kdd_fullbin_freq    frequency
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
    build_meta,
    load_split,
)
from encoder_nsl_kdd_binary import NSLKDDBinaryEncoder

logger = logging.getLogger(__name__)

_CATEGORICAL_BINARY = "categorical_binary"


class NSLKDDFullBinaryEncoder(NSLKDDBinaryEncoder):
    """Binary encoder where the categoricals are also collapsed to one bit each.

    Parameters
    ----------
    categorical_strategy:
        ``"unknown"`` -> site is 0 for any value seen in normal, 1 otherwise.
        ``"frequency"`` -> site is 0 for values at least ``frequency_threshold``
        frequent in normal, 1 otherwise.
    frequency_threshold:
        Minimum normalised frequency in normal traffic for a value to count as
        "common" (only used by the ``"frequency"`` strategy). E.g. ``0.01`` means
        a category must be at least 1% of normal rows to map to 0.
    """

    def __init__(
        self,
        categorical_strategy: str = "unknown",
        frequency_threshold: float = 0.01,
        quasi_constant_threshold: float = 0.95,
        degenerate_eps: float = 0.01,
    ) -> None:
        super().__init__(
            quasi_constant_threshold=quasi_constant_threshold,
            degenerate_eps=degenerate_eps,
        )
        if categorical_strategy not in ("unknown", "frequency"):
            raise ValueError(
                "categorical_strategy must be 'unknown' or 'frequency', "
                f"got {categorical_strategy!r}"
            )
        if not (0.0 < frequency_threshold < 1.0):
            raise ValueError(
                f"frequency_threshold must be in (0, 1), got {frequency_threshold}"
            )
        self.categorical_strategy = categorical_strategy
        self.frequency_threshold = frequency_threshold

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> "NSLKDDFullBinaryEncoder":
        """One binary FeatureSpec per feature, categoricals included.

        Numeric columns go through the inherited binary policy; the three
        categorical columns go through :meth:`_fit_categorical_binary`.
        """
        normal_mask = (df_train["label"].str.rstrip(".") == "normal").to_numpy()
        df_normal = df_train.loc[normal_mask]

        specs = []
        for col in COLUMNS:
            if col in DROP_COLS or col in META_COLS:
                continue
            if col in CATEGORICAL_COLS:
                specs.append(self._fit_categorical_binary(col, df_normal[col]))
            else:
                specs.append(self._fit_one(col, df_normal[col]))

        self.specs = specs
        return self

    def _fit_categorical_binary(self, col: str, x_normal: pd.Series) -> FeatureSpec:
        """Build the binary site for one categorical column.

        ``vocab`` stores the set ``S`` of normal-traffic values that map to 0;
        everything else (rare and/or unseen) maps to 1 at transform time.
        """
        values = x_normal.astype(str)

        if self.categorical_strategy == "unknown":
            normal_set = sorted(values.unique().tolist())
        else:  # "frequency"
            freq = values.value_counts(normalize=True)
            common = freq.index[freq >= self.frequency_threshold].tolist()
            if not common:
                # Threshold above every category's frequency would map all of
                # normal to 1 (a constant, useless site). Keep the most common
                # value so the site retains a 0 level.
                common = [freq.index[0]]
                logger.info(
                    "  %s: frequency_threshold=%.3g excludes every category; "
                    "keeping mode %r as the only common value",
                    col, self.frequency_threshold, common[0],
                )
            normal_set = sorted(common)

        return FeatureSpec(name=col, kind=_CATEGORICAL_BINARY, d=2, vocab=normal_set)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform_one(self, spec: FeatureSpec, x: pd.Series) -> np.ndarray:
        """Route the new binary-categorical kind; defer everything else.

        For ``categorical_binary``: 0 if the (string) value is in ``spec.vocab``
        (the normal set), 1 otherwise. Unseen test values fall to 1 naturally.
        """
        if spec.kind == _CATEGORICAL_BINARY:
            arr = x.astype(str).to_numpy()
            normal = np.asarray(spec.vocab, dtype=object)
            return (~np.isin(arr, normal)).astype(np.int64)
        return super()._transform_one(spec, x)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(data_dir: Path, out_dir: Path, strategy: str) -> None:
    """Fit the full-binary encoder with ``strategy`` and write the artefacts.

    Mirrors the binary encoder's main but adds, per categorical site, a log line
    showing the share of *normal* rows that map to 1 -- a quick sanity check that
    ``"unknown"`` sites are ~0% (constant on normal) while ``"frequency"`` sites
    carry some within-normal variance.
    """
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

    # diagnostics for the categorical sites on the normal training rows
    normal_mask = (train["label"].str.rstrip(".") == "normal").to_numpy()
    for k, spec in enumerate(encoder.specs):
        if spec.kind == _CATEGORICAL_BINARY:
            share_one = float(train_X[normal_mask, k].float().mean().item())
            logger.info(
                "  categorical %-13s (strategy=%s): |S|=%d kept, "
                "%.2f%% of normal rows map to 1",
                spec.name, strategy, len(spec.vocab), 100.0 * share_one,
            )

    logger.info(
        "schema: %d sites, all d=2  ->  %d physical qubits for the chain",
        len(encoder.specs), len(encoder.specs),
    )
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
    strategy = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    default_out = data_dir / f"fullbin_{strategy}"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out
    main(data_dir, out_dir, strategy)
