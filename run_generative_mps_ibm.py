#!/usr/bin/env python3
"""
run_generative_mps_ibm.py
=========================

Ejecuta un modelo generativo MPS (Born machine) como circuito cuantico y
MUESTREA de la distribucion aprendida, en simulador ideal, en un backend
"fake" ruidoso o en una QPU real de IBM. Evalua la fidelidad distribucional de
las muestras frente a la distribucion exacta del MPS.

Idea: el circuito prepara |Psi> con P_MPS(v)=|Psi(v)|^2; medir los qubits
fisicos genera muestras v ~ P_MPS. En hardware, el ruido degrada esa
distribucion; este script lo cuantifica.

Backends:
    --backend aer            simulador ideal (referencia, v ~ P_MPS exacto)
    --backend fake:brisbane  foto ruidosa de un dispositivo real (Eagle)
    --backend fake:torino    foto ruidosa de un dispositivo real (Heron)
    --backend real           QPU real (menos ocupada) via QiskitRuntimeService
    --backend real:ibm_torino   QPU real concreta

Metricas (escalan a N grande):
    - TVD frente al MPS (exacto si N<=16; si no, sobre el soporte observado)
    - fidelidad clasica (Bhattacharyya) y distancia de Hellinger
    - marginales por sitio P(s_k=1): hardware vs ideal vs MPS
    - solapamiento de los k bitstrings mas probables (top-k)
    - correlacion de frecuencias hardware-vs-ideal en el soporte comun

Uso:
    # ensayo en seco (ideal, sin gastar QPU)
    python run_generative_mps_ibm.py ./nsl_kdd_qc6 --backend aer --shots 8000

    # ensayo con ruido realista del dispositivo
    python run_generative_mps_ibm.py ./nsl_kdd_qc6 --backend fake:torino

    # ejecucion REAL en IBM (requiere cuenta guardada o --token)
    python run_generative_mps_ibm.py ./nsl_kdd_qc6 --backend real:ibm_torino \
        --shots 4000 --variant reuse_isometry

Credenciales IBM (una vez):
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum", token="TU_TOKEN")
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from qiskit import ClassicalRegister, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator


# ----------------------------------------------------------------------
def load_module(module_path: Path):
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"No encuentro {module_path}.")
    spec = importlib.util.spec_from_file_location("mps_to_circuit_local", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mps_to_circuit_local"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_circuit(mod, mps, variant: str):
    """Construye el circuito con medida de los N qubits fisicos.

    Variantes: reuse_isometry (recomendada para hardware), reuse_unitary,
    no_reuse_isometry, no_reuse_unitary.
    """
    if variant == "reuse_isometry":
        qc, N, b_max = mod.build_circuit_isometry(mps, reuse=True)
        return qc, N, b_max, True
    if variant == "reuse_unitary":
        qc, N, b_max = mod.build_circuit_reuse(mps)
        return qc, N, b_max, True
    if variant == "no_reuse_isometry":
        qc, N, b_max = mod.build_circuit_isometry(mps, reuse=False)
    elif variant == "no_reuse_unitary":
        qc, N, b_max = mod.build_circuit(mps)
    else:
        raise ValueError(f"variante desconocida: {variant!r}")
    # anyadir medida fisica
    creg = ClassicalRegister(N, "v")
    qc.add_register(creg)
    phys = [qc.qubits[b_max + k] for k in range(N)]
    qc.measure(phys, creg)
    return qc, N, b_max, False


def counts_to_site_order(counts: Dict[str, int]) -> Dict[str, int]:
    """Reordena el bitstring de Qiskit (c[N-1]..c[0]) a orden de sitio (site0..)."""
    out: Dict[str, int] = {}
    for bitstr, c in counts.items():
        site = bitstr.replace(" ", "")[::-1]
        out[site] = out.get(site, 0) + int(c)
    return out


# ----------------------------------------------------------------------
# Backends y muestreo
# ----------------------------------------------------------------------
def get_target(backend_spec: str):
    """Devuelve (sampler_kind, backend_obj_or_None, descripcion)."""
    if backend_spec == "aer":
        return "aer_ideal", None, "Aer ideal (referencia)"
    if backend_spec.startswith("fake:"):
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeTorino
        name = backend_spec.split(":", 1)[1]
        be = {"brisbane": FakeBrisbane, "torino": FakeTorino}[name]()
        return "fake", be, f"fake:{name} ({be.num_qubits}q, ruido de dispositivo real)"
    if backend_spec.startswith("real"):
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        if ":" in backend_spec:
            be = service.backend(backend_spec.split(":", 1)[1])
        else:
            be = service.least_busy(operational=True, simulator=False)
        return "real", be, f"real:{be.name} ({be.num_qubits}q)"
    raise ValueError(f"backend desconocido: {backend_spec!r}")


def sample_ideal(qc, shots: int, opt_basis: Tuple[str, ...]) -> Dict[str, int]:
    sim = AerSimulator()
    tqc = transpile(qc, sim, basis_gates=list(opt_basis), optimization_level=1)
    return sim.run(tqc, shots=shots).result().get_counts()


def sample_backend(qc, backend, shots: int, opt_level: int, kind: str) -> Tuple[Dict[str, int], Any]:
    """Transpila contra la topologia del backend y muestrea con SamplerV2."""
    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)
    tqc = pm.run(qc)

    if kind == "real":
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
        sampler = RuntimeSampler(backend)
    else:  # fake -> SamplerV2 de Aer construido desde el backend (incluye su ruido)
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        sampler = AerSamplerV2.from_backend(backend)
    # En algunas versiones de SamplerV2, `options.default_shots` no afecta a
    # AerSamplerV2.from_backend y se queda en el default interno (1024).
    # Pasar `shots` directamente a run(...) hace que fake/real usen exactamente
    # el mismo numero de shots que Aer ideal. Mantenemos el fallback por
    # compatibilidad con versiones que no acepten el argumento keyword.
    try:
        sampler.options.default_shots = shots
    except Exception:
        pass

    try:
        result = sampler.run([tqc], shots=shots).result()
    except TypeError:
        result = sampler.run([tqc]).result()
    data = result[0].data
    creg = next(iter(data.__dict__.keys()))
    counts = getattr(data, creg).get_counts()
    return counts, tqc


# ----------------------------------------------------------------------
# Metricas de fidelidad generativa
# ----------------------------------------------------------------------
def exact_mps_distribution(mps, N: int) -> np.ndarray:
    # enumeracion vectorizada (rapida hasta N~20): site0 es el MSB del indice
    idx = torch.arange(2 ** N, dtype=torch.long)
    shifts = torch.arange(N - 1, -1, -1, dtype=torch.long)
    all_v = ((idx.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).long()
    with torch.no_grad():
        return torch.exp(mps.log_prob(all_v)).cpu().numpy().astype(np.float64)


def exact_mps_site_marginals(p_mps: np.ndarray, N: int) -> np.ndarray:
    """Marginales por sitio P(s_k=1) a partir de la distribucion exacta del MPS."""
    idx = np.arange(2 ** N)
    m = np.empty(N, dtype=np.float64)
    for k in range(N):
        bit = (idx >> (N - 1 - k)) & 1  # site k
        m[k] = float(p_mps[bit == 1].sum())
    return m


def mps_prob_of_bitstrings(mps, site_bitstrings: List[str]) -> np.ndarray:
    """P_MPS exacto para una lista de bitstrings en orden de sitio (escala a N grande)."""
    V = torch.tensor([[int(b) for b in s] for s in site_bitstrings], dtype=torch.long)
    with torch.no_grad():
        return torch.exp(mps.log_prob(V)).cpu().numpy().astype(np.float64)


def dist_from_counts_full(site_counts: Dict[str, int], N: int) -> np.ndarray:
    shots = sum(site_counts.values())
    p = np.zeros(2 ** N, dtype=np.float64)
    for s, c in site_counts.items():
        p[int(s, 2)] += c / shots  # s en orden de sitio: site0 es el MSB
    return p


def site_marginals(site_counts: Dict[str, int], N: int) -> np.ndarray:
    shots = sum(site_counts.values())
    m = np.zeros(N, dtype=np.float64)
    for s, c in site_counts.items():
        for k, ch in enumerate(s):
            if ch == "1":
                m[k] += c
    return m / shots


def generative_metrics(
    mps, N: int, site_counts: Dict[str, int], ref_counts: Dict[str, int]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"shots": int(sum(site_counts.values()))}

    # marginales por sitio (siempre, escalan a cualquier N)
    m_hw = site_marginals(site_counts, N)
    m_ref = site_marginals(ref_counts, N)
    out["marginal_L1_hw_vs_ref"] = float(np.abs(m_hw - m_ref).sum())
    out["marginal_max_abs_hw_vs_ref"] = float(np.abs(m_hw - m_ref).max())
    out["site_marginals_hw"] = m_hw.round(4).tolist()
    out["site_marginals_ref"] = m_ref.round(4).tolist()

    # top-k solapamiento (hardware vs referencia ideal)
    k = min(10, len(ref_counts))
    top_hw = {s for s, _ in sorted(site_counts.items(), key=lambda kv: -kv[1])[:k]}
    top_ref = {s for s, _ in sorted(ref_counts.items(), key=lambda kv: -kv[1])[:k]}
    out["topk_k"] = int(k)
    out["topk_overlap"] = float(len(top_hw & top_ref) / k) if k else None

    # soporte comun: correlacion de frecuencias hw vs ref
    sh = sum(site_counts.values()); sr = sum(ref_counts.values())
    common = set(site_counts) & set(ref_counts)
    if len(common) >= 3:
        fh = np.array([site_counts[s] / sh for s in common])
        fr = np.array([ref_counts[s] / sr for s in common])
        out["freq_corr_hw_vs_ref"] = float(np.corrcoef(fh, fr)[0, 1])
    else:
        out["freq_corr_hw_vs_ref"] = None

    if N <= 20:
        # distribucion completa: metricas exactas frente al MPS
        p_mps = exact_mps_distribution(mps, N)
        p_hw = dist_from_counts_full(site_counts, N)
        p_ref = dist_from_counts_full(ref_counts, N)
        out["tvd_hw_vs_mps"] = float(0.5 * np.abs(p_hw - p_mps).sum())
        out["tvd_ref_vs_mps"] = float(0.5 * np.abs(p_ref - p_mps).sum())
        out["tvd_hw_vs_ref"] = float(0.5 * np.abs(p_hw - p_ref).sum())
        bc = float(np.sum(np.sqrt(p_hw * p_mps)))
        out["fidelity_hw_vs_mps"] = bc ** 2          # Bhattacharyya/clasica
        out["hellinger_hw_vs_mps"] = float(np.sqrt(max(0.0, 1.0 - bc)))
        out["mc_noise_scale"] = float(np.sqrt((2 ** N) / sh) / 2)
        # marginales frente al MPS EXACTO (metrica de fidelidad principal)
        m_mps = exact_mps_site_marginals(p_mps, N)
        out["marginal_L1_hw_vs_mps"] = float(np.abs(m_hw - m_mps).sum())
        out["marginal_max_abs_hw_vs_mps"] = float(np.abs(m_hw - m_mps).max())
        out["site_marginals_mps"] = m_mps.round(4).tolist()
    else:
        # N grande: TVD restringido al soporte observado + P_MPS exacto por muestra
        strings = list(site_counts.keys())
        p_mps_obs = mps_prob_of_bitstrings(mps, strings)
        f_hw = np.array([site_counts[s] / sh for s in strings])
        out["tvd_hw_vs_mps_observed_support"] = float(0.5 * np.abs(f_hw - p_mps_obs).sum())
        out["mps_mass_on_observed_support"] = float(p_mps_obs.sum())
        out["tvd_hw_vs_mps"] = None
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--model", default="mps_trained.pt")
    ap.add_argument("--module-path", type=Path,
                    default=Path(__file__).with_name("mps_to_circuit.py"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--variant", default="reuse_isometry",
                    choices=["reuse_isometry", "reuse_unitary",
                             "no_reuse_isometry", "no_reuse_unitary"])
    ap.add_argument("--backend", default="aer",
                    help="aer | fake:brisbane | fake:torino | real | real:NAME")
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--basis", default="cx,rz,sx,x", help="base para el muestreo ideal")
    ap.add_argument("--shots", type=int, default=8000)
    ap.add_argument("--ref-shots", type=int, default=0,
                    help="shots para la referencia ideal (0 = usar --shots)")
    args = ap.parse_args()

    mod = load_module(args.module_path)
    mps = mod.MPS.load(str(args.data_dir / args.model))
    mod.prepare_right_canonical(mps)
    print(f"MPS: {mps.num_sites} sites, bonds={list(mps.full_bond_dims)}")

    qc, N, b_max, reuse = build_circuit(mod, mps, args.variant)
    print(f"circuito: variante={args.variant}, qubits={qc.num_qubits}, "
          f"N={N}, b_max={b_max}, reuse={reuse}")

    basis = tuple(x.strip() for x in args.basis.split(",") if x.strip())
    ref_shots = args.ref_shots or args.shots

    # 1) referencia ideal (v ~ P_MPS exacto)
    print(f"\nmuestreo IDEAL (Aer) con {ref_shots} shots...")
    ref_counts = counts_to_site_order(sample_ideal(qc, ref_shots, basis))

    # 2) backend objetivo
    kind, backend, desc = get_target(args.backend)
    print(f"backend objetivo: {desc}")
    t0 = time.perf_counter()
    if kind == "aer_ideal":
        tgt_counts_raw = sample_ideal(qc, args.shots, basis)
        tqc = None
    else:
        tgt_counts_raw, tqc = sample_backend(qc, backend, args.shots, args.opt_level, kind)
    seconds = time.perf_counter() - t0
    tgt_counts = counts_to_site_order(tgt_counts_raw)
    print(f"  muestreo completado en {seconds:.1f}s, "
          f"{len(tgt_counts)} bitstrings observados")
    if tqc is not None:
        n2q = sum(v for g, v in tqc.count_ops().items()
                  if g in ("cx", "cz", "ecr", "rzz"))
        print(f"  circuito ejecutado: profundidad {tqc.depth()}, {n2q} puertas 2q")

    # 3) metricas de fidelidad generativa
    metrics = generative_metrics(mps, N, tgt_counts, ref_counts)
    metrics["backend"] = desc
    metrics["variant"] = args.variant
    metrics["seconds"] = seconds
    if tqc is not None:
        metrics["executed_depth"] = int(tqc.depth())
        metrics["executed_2q_gates"] = int(n2q)

    print("\n" + "=" * 70)
    print(f"FIDELIDAD GENERATIVA  ({desc})")
    print("=" * 70)
    if metrics.get("tvd_hw_vs_mps") is not None:
        print(f"  fidelidad clasica vs MPS : {metrics['fidelity_hw_vs_mps']:.4f}")
        print(f"  marginal L1   vs MPS     : {metrics['marginal_L1_hw_vs_mps']:.4f}  <- metrica principal")
        print(f"  TVD(hardware, MPS)       : {metrics['tvd_hw_vs_mps']:.4f}")
        print(f"  TVD(ideal,    MPS)       : {metrics['tvd_ref_vs_mps']:.4f}  (suelo MC)")
        print(f"  TVD(hardware, ideal)     : {metrics['tvd_hw_vs_ref']:.4f}  (solo ruido)")
    else:
        print(f"  TVD(hw,MPS) soporte obs. : {metrics['tvd_hw_vs_mps_observed_support']:.4f}")
        print(f"  masa MPS en soporte obs. : {metrics['mps_mass_on_observed_support']:.4f}")
        print(f"  marginal L1 (hw vs ideal): {metrics['marginal_L1_hw_vs_ref']:.4f}  <- metrica principal")
    print(f"  (aux) top-{metrics['topk_k']} overlap      : {metrics['topk_overlap']}  "
          f"| corr frec.: {metrics['freq_corr_hw_vs_ref']}")

    out_dir = args.out_dir or (args.data_dir / "hardware_generation")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.backend.replace(":", "_")
    (out_dir / f"generation_{tag}_{args.variant}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote: {out_dir / f'generation_{tag}_{args.variant}.json'}")


if __name__ == "__main__":
    main()