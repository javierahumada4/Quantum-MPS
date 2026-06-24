#!/usr/bin/env python3
"""
compare_mps_circuit_variants.py
================================

Compara las variantes de realización de un MPS Born machine como circuito:

    1) no-reuse + unitary
    2) no-reuse + isometry
    3) reuse    + unitary
    4) reuse    + isometry

Métricas:
    - número de qubits
    - profundidad high-level
    - profundidad tras transpilar
    - número de puertas de 2 qubits tras transpilar
    - desglose de puertas
    - validación exacta statevector para no-reuse
    - validación estadística por muestreo para reuse
    - CSV + JSON con resultados

Uso típico:
    python compare_mps_circuit_variants.py ./nsl_kdd_qc6

Uso barato:
    python compare_mps_circuit_variants.py ./nsl_kdd_qc6 --shots 2000 --skip-reuse

Uso con base tipo IBM:
    python compare_mps_circuit_variants.py ./nsl_kdd_qc6 --basis cx,rz,sx,x --opt-level 3

Requisitos:
    - estar en el mismo directorio que mps_to_circuit.py, o pasar --module-path
    - qiskit, qiskit-aer, torch, numpy, pandas
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


TWO_QUBIT_GATES = {
    "cx", "cz", "ecr", "rzz", "rxx", "ryy",
    "cy", "cp", "swap", "iswap", "xx_plus_yy"
}


def load_module(module_path: Path):
    """Importa mps_to_circuit.py desde una ruta explícita."""
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(
            f"No encuentro {module_path}. Pon este script en el mismo directorio "
            "que mps_to_circuit.py o usa --module-path /ruta/mps_to_circuit.py"
        )

    spec = importlib.util.spec_from_file_location("mps_to_circuit_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No he podido importar {module_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["mps_to_circuit_local"] = mod
    spec.loader.exec_module(mod)
    return mod


def mps_distribution(mps, N: int) -> np.ndarray:
    """Calcula P_MPS(v) para todos los bitstrings v."""
    all_v = torch.tensor(
        [[(i >> (N - 1 - s)) & 1 for s in range(N)] for i in range(2 ** N)],
        dtype=torch.long,
    )
    with torch.no_grad():
        return torch.exp(mps.log_prob(all_v)).detach().cpu().numpy().astype(np.float64)


def exact_no_reuse_metrics(mps, qc, N: int, b_max: int) -> Dict[str, Any]:
    """Validación exacta para circuitos sin reuse."""
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()

    p_mps = mps_distribution(mps, N)
    p_circ = np.zeros(2 ** N, dtype=np.float64)
    leak = 0.0

    # En mps_to_circuit: registros = bond primero, phys después.
    # En labels de Qiskit: índices más altos a la izquierda.
    # Por tanto: phys bits = primeros N chars; bond bits = últimos b_max chars.
    for bitstr, p in probs.items():
        bits = bitstr.replace(" ", "")
        phys_bits = bits[:N]
        bond_bits = bits[N:]

        if "1" in bond_bits:
            leak += float(p)
            continue

        # phys_bits viene como phys[N-1]..phys[0].
        idx = int(phys_bits[::-1], 2)
        p_circ[idx] += float(p)

    max_abs_err = float(np.max(np.abs(p_mps - p_circ)))
    l1_err = float(np.sum(np.abs(p_mps - p_circ)))
    tvd = 0.5 * l1_err

    return {
        "statevector_leakage": float(leak),
        "statevector_max_abs_err": max_abs_err,
        "statevector_l1_err": l1_err,
        "statevector_tvd": tvd,
        "statevector_exact_match": bool(max_abs_err < 1e-8 and leak < 1e-8),
    }


def add_measurements_no_reuse(qc, N: int, b_max: int):
    """Añade medida solo sobre qubits físicos a un circuito no-reuse."""
    meas = qc.copy()
    creg = ClassicalRegister(N, "v")
    meas.add_register(creg)

    # mps_to_circuit crea: bond[0..b_max-1], phys[0..N-1].
    phys_qubits = [meas.qubits[b_max + k] for k in range(N)]
    meas.measure(phys_qubits, creg)
    return meas


def counts_to_distribution(counts: Dict[str, int], N: int) -> np.ndarray:
    """Convierte counts de Qiskit en distribución P(site0...siteN-1)."""
    shots = sum(counts.values())
    p = np.zeros(2 ** N, dtype=np.float64)
    if shots == 0:
        return p

    for bitstr, c in counts.items():
        clean = bitstr.replace(" ", "")
        # ClassicalRegister imprime c[N-1]..c[0]; cv[k] guarda site k.
        idx = int(clean[::-1], 2)
        p[idx] += c / shots
    return p


def sampled_metrics(mps, qc, N: int, b_max: int, reuse: bool, shots: int) -> Dict[str, Any]:
    """Validación por muestreo."""
    sim = AerSimulator()
    run_qc = qc if reuse else add_measurements_no_reuse(qc, N, b_max)
    tqc = transpile(run_qc, sim, optimization_level=0)
    counts = sim.run(tqc, shots=shots).result().get_counts()

    out: Dict[str, Any] = {
        "shots": int(shots),
        "num_observed_bitstrings": int(len(counts)),
    }

    if N <= 20:
        p_mps = mps_distribution(mps, N)
        p_samp = counts_to_distribution(counts, N)
        out["sample_tvd_vs_mps"] = float(0.5 * np.abs(p_mps - p_samp).sum())
        out["sample_max_abs_err_vs_mps"] = float(np.max(np.abs(p_mps - p_samp)))
        out["sample_mc_noise_scale"] = float(np.sqrt((2 ** N) / shots) / 2)
    else:
        out["sample_tvd_vs_mps"] = None
        out["sample_max_abs_err_vs_mps"] = None
        out["sample_mc_noise_scale"] = None

    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    out["top_counts"] = [
        {
            "bitstring_site_order": bitstr.replace(" ", "")[::-1],
            "count": int(c),
            "frequency": float(c / shots),
        }
        for bitstr, c in top
    ]
    return out


def gate_metrics(qc, basis: Tuple[str, ...], opt_level: int) -> Dict[str, Any]:
    """Transpila y devuelve métricas de profundidad y puertas."""
    tqc = transpile(qc, basis_gates=list(basis), optimization_level=opt_level)
    ops = dict(tqc.count_ops())
    n_2q = int(sum(v for k, v in ops.items() if k in TWO_QUBIT_GATES))

    return {
        "transpiled_depth": int(tqc.depth()),
        "transpiled_size": int(tqc.size()),
        "two_qubit_gates": n_2q,
        "gate_breakdown": ops,
    }


def build_variant(mod, mps, synthesis: str, reuse: bool):
    """Construye una de las cuatro variantes usando mps_to_circuit.py."""
    if synthesis == "isometry":
        return mod.build_circuit_isometry(mps, reuse=reuse)
    if reuse:
        return mod.build_circuit_reuse(mps)
    return mod.build_circuit(mps)


def compare_variants(
    data_dir: Path,
    model_name: str,
    module_path: Path,
    out_dir: Path,
    basis: Tuple[str, ...],
    opt_level: int,
    shots: int,
    max_exact_sites: int,
    skip_reuse: bool,
    skip_sampling: bool,
) -> pd.DataFrame:
    mod = load_module(module_path)

    mps = mod.MPS.load(str(data_dir / model_name))
    print(f"loaded MPS: {mps.num_sites} sites, full_bond_dims={list(mps.full_bond_dims)}")
    mod.prepare_right_canonical(mps)

    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("no_reuse_unitary", "unitary", False),
        ("no_reuse_isometry", "isometry", False),
    ]
    if not skip_reuse:
        variants.extend([
            ("reuse_unitary", "unitary", True),
            ("reuse_isometry", "isometry", True),
        ])

    rows: List[Dict[str, Any]] = []

    for name, synthesis, reuse in variants:
        print("\n" + "=" * 80)
        print(f"VARIANT: {name}")
        print("=" * 80)

        row: Dict[str, Any] = {
            "variant": name,
            "synthesis": synthesis,
            "reuse": bool(reuse),
        }

        # Construcción
        t0 = time.perf_counter()
        try:
            qc, N, b_max = build_variant(mod, mps, synthesis=synthesis, reuse=reuse)
            row["build_error"] = None
        except Exception as exc:
            row["build_error"] = repr(exc)
            row["status"] = "build_failed"
            rows.append(row)
            print(f"BUILD FAILED: {exc!r}")
            continue
        row["build_seconds"] = time.perf_counter() - t0

        row["N_sites"] = int(N)
        row["b_max"] = int(b_max)
        row["num_qubits"] = int(qc.num_qubits)
        row["num_clbits"] = int(qc.num_clbits)
        row["high_level_depth"] = int(qc.depth())
        row["high_level_size"] = int(qc.size())
        row["high_level_ops"] = dict(qc.count_ops())

        print(f"qubits           : {qc.num_qubits}")
        print(f"clbits           : {qc.num_clbits}")
        print(f"high-level depth : {qc.depth()}")
        print(f"high-level ops   : {dict(qc.count_ops())}")

        # Validación exacta solo para no-reuse.
        if not reuse and N <= max_exact_sites:
            print("statevector validation...")
            t0 = time.perf_counter()
            try:
                row.update(exact_no_reuse_metrics(mps, qc, N, b_max))
                row["statevector_seconds"] = time.perf_counter() - t0
                print(f"  leakage       : {row['statevector_leakage']:.2e}")
                print(f"  max abs err   : {row['statevector_max_abs_err']:.2e}")
                print(f"  exact match   : {row['statevector_exact_match']}")
            except Exception as exc:
                row["statevector_error"] = repr(exc)
                print(f"  statevector FAILED: {exc!r}")
        else:
            row["statevector_leakage"] = None
            row["statevector_max_abs_err"] = None
            row["statevector_l1_err"] = None
            row["statevector_tvd"] = None
            row["statevector_exact_match"] = None

        # Transpilación
        print("transpiling...")
        t0 = time.perf_counter()
        try:
            gm = gate_metrics(qc, basis=basis, opt_level=opt_level)
            row.update(gm)
            row["transpile_seconds"] = time.perf_counter() - t0
            print(f"  transpiled depth : {row['transpiled_depth']}")
            print(f"  2q gates         : {row['two_qubit_gates']}")
            print(f"  breakdown        : {row['gate_breakdown']}")
        except Exception as exc:
            row["transpile_error"] = repr(exc)
            print(f"  transpile FAILED: {exc!r}")

        # Muestreo
        if not skip_sampling and shots > 0:
            print("sampling...")
            t0 = time.perf_counter()
            try:
                sm = sampled_metrics(mps, qc, N, b_max, reuse=reuse, shots=shots)
                row.update(sm)
                row["sampling_seconds"] = time.perf_counter() - t0
                if sm.get("sample_tvd_vs_mps") is not None:
                    print(f"  sample TVD       : {sm['sample_tvd_vs_mps']:.4f}")
                    print(f"  MC noise scale   : {sm['sample_mc_noise_scale']:.4f}")
                print(f"  observed strings : {sm['num_observed_bitstrings']}")
            except Exception as exc:
                row["sampling_error"] = repr(exc)
                print(f"  sampling FAILED: {exc!r}")

        row["status"] = "ok"
        rows.append(row)

    df = pd.DataFrame(rows)

    # Guardar CSV plano y JSON completo.
    csv_path = out_dir / "circuit_variant_comparison.csv"
    json_path = out_dir / "circuit_variant_comparison.json"

    flat = df.copy()
    for col in flat.columns:
        if flat[col].map(lambda x: isinstance(x, (dict, list))).any():
            flat[col] = flat[col].map(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
            )
    flat.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    cols = [
        "variant", "num_qubits", "high_level_depth",
        "transpiled_depth", "two_qubit_gates",
        "statevector_max_abs_err", "statevector_leakage",
        "sample_tvd_vs_mps", "build_seconds", "transpile_seconds",
    ]
    shown = [c for c in cols if c in df.columns]
    print(df[shown].to_string(index=False))

    print(f"\nwrote: {csv_path}")
    print(f"wrote: {json_path}")
    return df


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="Directorio con mps_trained.pt")
    ap.add_argument("--model", default="mps_trained.pt")
    ap.add_argument(
        "--module-path",
        type=Path,
        default=Path(__file__).with_name("mps_to_circuit.py"),
        help="Ruta a mps_to_circuit.py",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directorio de salida. Por defecto: <data_dir>/circuit_comparison",
    )
    ap.add_argument(
        "--basis",
        default="cx,rz,sx,x",
        help="Basis gates separadas por coma. Ej: cx,rz,sx,x o cz,rz,sx,x",
    )
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument(
        "--max-exact-sites",
        type=int,
        default=17,
        help="Máximo N para validación statevector enumerando 2^N.",
    )
    ap.add_argument("--skip-reuse", action="store_true")
    ap.add_argument("--skip-sampling", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.data_dir / "circuit_comparison")
    basis = tuple(x.strip() for x in args.basis.split(",") if x.strip())

    compare_variants(
        data_dir=args.data_dir,
        model_name=args.model,
        module_path=args.module_path,
        out_dir=out_dir,
        basis=basis,
        opt_level=args.opt_level,
        shots=args.shots,
        max_exact_sites=args.max_exact_sites,
        skip_reuse=args.skip_reuse,
        skip_sampling=args.skip_sampling,
    )


if __name__ == "__main__":
    main()
