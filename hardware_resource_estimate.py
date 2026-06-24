#!/usr/bin/env python3
"""
hardware_resource_estimate.py
=============================

Cuenta honesta de recursos de las cuatro variantes de circuito MPS->Born
transpilando contra la CONECTIVIDAD REAL de un dispositivo IBM, no solo a una
base abstracta. La diferencia es el coste de routing (SWAPs) que la base
abstracta esconde.

Para cada variante reporta:
    - 2q-gates con base abstracta (suelo optimista, all-to-all)
    - 2q-gates tras enrutar en el dispositivo (lo que de verdad se ejecuta)
    - factor de inflacion = routed_2q / abstract_2q
    - profundidad enrutada y qubits fisicos usados

Topologia del dispositivo:
    --device brisbane   -> FakeBrisbane  (Eagle r3, 127q, heavy-hex, ECR)
    --device torino     -> FakeTorino    (Heron r1, 133q, CZ)
    --device real:NAME  -> backend real via QiskitRuntimeService (necesita cuenta)

Los "fake backends" son fotos de dispositivos IBM reales (coupling map y base
nativa reales), por lo que sus numeros son representativos del hardware sin
necesitar credenciales: ideal para reportar cifras honestas en el paper.

Envio real (opcional):
    --run --device real:ibm_brisbane   ejecuta UNA variante en la QPU y
    devuelve los counts (+ TVD frente al MPS si N es pequeno).

Uso:
    python hardware_resource_estimate.py ./nsl_kdd_qc6 --device brisbane
    python hardware_resource_estimate.py ./nsl_kdd_qc6 --device torino --opt-level 3
    python hardware_resource_estimate.py ./nsl_kdd_qc6 --device real:ibm_torino --run \
        --variant reuse_isometry --shots 4000
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
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


TWO_QUBIT_GATES = {
    "cx", "cz", "ecr", "rzz", "rxx", "ryy",
    "cy", "cp", "swap", "iswap", "xx_plus_yy",
}


def load_module(module_path: Path):
    """Importa mps_to_circuit.py desde una ruta explicita."""
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(
            f"No encuentro {module_path}. Usa --module-path /ruta/mps_to_circuit.py"
        )
    spec = importlib.util.spec_from_file_location("mps_to_circuit_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No he podido importar {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mps_to_circuit_local"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_backend(device: str):
    """Devuelve (backend, descripcion). device in {brisbane,torino,real:NAME}."""
    if device.startswith("real:"):
        from qiskit_ibm_runtime import QiskitRuntimeService

        name = device.split(":", 1)[1]
        service = QiskitRuntimeService()
        backend = service.backend(name)
        return backend, f"real:{name} ({backend.num_qubits}q)"

    from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeTorino

    table = {"brisbane": FakeBrisbane, "torino": FakeTorino}
    if device not in table:
        raise ValueError(f"device debe ser brisbane, torino o real:NAME (no {device!r})")
    backend = table[device]()
    return backend, f"{device} snapshot ({backend.num_qubits}q)"


def two_qubit_count(circuit) -> int:
    ops = dict(circuit.count_ops())
    return int(sum(v for k, v in ops.items() if k in TWO_QUBIT_GATES))


def build_variant(mod, mps, synthesis: str, reuse: bool):
    if synthesis == "isometry":
        return mod.build_circuit_isometry(mps, reuse=reuse)
    if reuse:
        return mod.build_circuit_reuse(mps)
    return mod.build_circuit(mps)


def add_phys_measure_no_reuse(qc, N: int, b_max: int):
    """Mide solo los qubits fisicos (variantes sin reuso) para poder enrutar/ejecutar."""
    meas = qc.copy()
    creg = ClassicalRegister(N, "v")
    meas.add_register(creg)
    phys = [meas.qubits[b_max + k] for k in range(N)]
    meas.measure(phys, creg)
    return meas


def estimate_variant(
    mod, mps, name: str, synthesis: str, reuse: bool,
    backend, abstract_basis: Tuple[str, ...], opt_level: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"variant": name, "synthesis": synthesis, "reuse": bool(reuse)}

    qc, N, b_max = build_variant(mod, mps, synthesis, reuse)
    if not reuse:
        qc = add_phys_measure_no_reuse(qc, N, b_max)

    row["N_sites"] = int(N)
    row["b_max"] = int(b_max)
    row["logical_qubits"] = int(qc.num_qubits)
    row["high_level_depth"] = int(qc.depth())

    # (a) base abstracta: all-to-all, sin routing  -> SUELO OPTIMISTA
    t0 = time.perf_counter()
    abs_qc = transpile(qc, basis_gates=list(abstract_basis), optimization_level=opt_level)
    row["abstract_2q"] = two_qubit_count(abs_qc)
    row["abstract_depth"] = int(abs_qc.depth())
    row["abstract_seconds"] = time.perf_counter() - t0

    # (b) dispositivo real: coupling map + base nativa -> lo que de verdad corre
    t0 = time.perf_counter()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)
    dev_qc = pm.run(qc)
    row["device_2q"] = two_qubit_count(dev_qc)
    row["device_depth"] = int(dev_qc.depth())
    row["device_physical_qubits"] = int(dev_qc.num_qubits)

    # qubits realmente usados (no ociosos) tras el layout
    used = {q for inst in dev_qc.data for q in inst.qubits}
    row["device_qubits_used"] = int(len(used))
    row["device_seconds"] = time.perf_counter() - t0

    if row["abstract_2q"] > 0:
        row["routing_inflation_2q"] = round(row["device_2q"] / row["abstract_2q"], 3)
    else:
        row["routing_inflation_2q"] = None
    row["routing_overhead_2q"] = int(row["device_2q"] - row["abstract_2q"])

    return row, dev_qc


def run_on_hardware(
    mod, mps, dev_qc, backend, N: int, b_max: int, reuse: bool, shots: int,
) -> Dict[str, Any]:
    """Ejecuta UNA variante ya transpilada en la QPU (o fake) y mide TVD si N pequeno."""
    from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

    sampler = RuntimeSampler(backend)
    sampler.options.default_shots = shots
    t0 = time.perf_counter()
    result = sampler.run([dev_qc]).result()
    seconds = time.perf_counter() - t0

    # leer el unico registro clasico
    data = result[0].data
    creg_name = next(iter(vars(data).keys())) if hasattr(data, "__dict__") else "v"
    counts = getattr(data, creg_name).get_counts()

    out: Dict[str, Any] = {
        "shots": int(shots),
        "seconds": float(seconds),
        "num_observed_bitstrings": int(len(counts)),
    }

    if N <= 20:
        all_v = torch.tensor(
            [[(i >> (N - 1 - s)) & 1 for s in range(N)] for i in range(2 ** N)],
            dtype=torch.long,
        )
        with torch.no_grad():
            p_mps = torch.exp(mps.log_prob(all_v)).cpu().numpy().astype(np.float64)
        shots_tot = sum(counts.values())
        p_hw = np.zeros(2 ** N, dtype=np.float64)
        for bitstr, c in counts.items():
            idx = int(bitstr.replace(" ", "")[::-1], 2)
            p_hw[idx] += c / shots_tot
        out["hardware_tvd_vs_mps"] = float(0.5 * np.abs(p_mps - p_hw).sum())
    else:
        out["hardware_tvd_vs_mps"] = None

    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    out["top_counts"] = [
        {"bitstring": b.replace(" ", ""), "count": int(c)} for b, c in top
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--model", default="mps_trained.pt")
    ap.add_argument("--module-path", type=Path,
                    default=Path(__file__).with_name("mps_to_circuit.py"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="brisbane",
                    help="brisbane | torino | real:NAME")
    ap.add_argument("--abstract-basis", default="cx,rz,sx,x")
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--skip-reuse", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="Ejecuta una variante en el backend (fake o real).")
    ap.add_argument("--variant", default="reuse_isometry",
                    help="Variante a ejecutar con --run.")
    ap.add_argument("--shots", type=int, default=4000)
    args = ap.parse_args()

    mod = load_module(args.module_path)
    mps = mod.MPS.load(str(args.data_dir / args.model))
    mod.prepare_right_canonical(mps)
    print(f"loaded MPS: {mps.num_sites} sites, bonds={list(mps.full_bond_dims)}")

    backend, desc = get_backend(args.device)
    print(f"device: {desc}")

    abstract_basis = tuple(x.strip() for x in args.abstract_basis.split(",") if x.strip())

    variants = [
        ("no_reuse_unitary", "unitary", False),
        ("no_reuse_isometry", "isometry", False),
    ]
    if not args.skip_reuse:
        variants += [
            ("reuse_unitary", "unitary", True),
            ("reuse_isometry", "isometry", True),
        ]

    rows: List[Dict[str, Any]] = []
    dev_circuits: Dict[str, Any] = {}
    for name, synth, reuse in variants:
        print(f"\n=== {name} ===")
        try:
            row, dev_qc = estimate_variant(
                mod, mps, name, synth, reuse, backend, abstract_basis, args.opt_level
            )
            dev_circuits[name] = (dev_qc, row["N_sites"], row["b_max"], reuse)
            rows.append(row)
            print(f"  abstract 2q  : {row['abstract_2q']}  (depth {row['abstract_depth']})")
            print(f"  device   2q  : {row['device_2q']}  (depth {row['device_depth']})")
            print(f"  inflation    : x{row['routing_inflation_2q']}  "
                  f"(+{row['routing_overhead_2q']} 2q por routing)")
            print(f"  qubits used  : {row['device_qubits_used']}")
        except Exception as exc:
            print(f"  FAILED: {exc!r}")
            rows.append({"variant": name, "error": repr(exc)})

    df = pd.DataFrame(rows)
    out_dir = args.out_dir or (args.data_dir / "hardware_estimate")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "hardware_resource_estimate.csv", index=False)
    (out_dir / "hardware_resource_estimate.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )

    print("\n" + "=" * 78)
    print(f"RESOURCE SUMMARY on {desc}")
    print("=" * 78)
    cols = ["variant", "logical_qubits", "device_qubits_used",
            "abstract_2q", "device_2q", "routing_inflation_2q", "device_depth"]
    shown = [c for c in cols if c in df.columns]
    print(df[shown].to_string(index=False))

    if args.run:
        if args.variant not in dev_circuits:
            print(f"\nNo puedo ejecutar {args.variant!r}; no se construyo.")
        else:
            dev_qc, N, b_max, reuse = dev_circuits[args.variant]
            print(f"\nEjecutando {args.variant} en {desc} ({args.shots} shots)...")
            run_out = run_on_hardware(mod, mps, dev_qc, backend, N, b_max, reuse, args.shots)
            (out_dir / f"hardware_run_{args.variant}.json").write_text(
                json.dumps(run_out, indent=2, ensure_ascii=False)
            )
            print(f"  observed bitstrings: {run_out['num_observed_bitstrings']}")
            if run_out.get("hardware_tvd_vs_mps") is not None:
                print(f"  TVD(hardware, MPS) : {run_out['hardware_tvd_vs_mps']:.4f}")
            print(f"  top counts         : {run_out['top_counts']}")

    print(f"\nwrote: {out_dir/'hardware_resource_estimate.csv'}")


if __name__ == "__main__":
    main()
