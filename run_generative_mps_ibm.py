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


def transpile_ideal(qc, opt_basis: Tuple[str, ...], seed_transpiler: Optional[int] = 1234):
    """Transpila el circuito a la base ideal UNA vez (sin ruido, sin topologia)."""
    sim = AerSimulator()
    return transpile(qc, sim, basis_gates=list(opt_basis), optimization_level=1,
                     seed_transpiler=seed_transpiler)


def run_ideal(tqc, shots: int, seed: Optional[int] = None) -> Dict[str, int]:
    """Muestrea el circuito ideal ya transpilado. ``seed`` -> reproducible."""
    sim = AerSimulator(seed_simulator=seed) if seed is not None else AerSimulator()
    return sim.run(tqc, shots=shots).result().get_counts()


def transpile_backend(qc, backend, opt_level: int, seed_transpiler: Optional[int] = 1234):
    """Transpila contra la topologia real del backend UNA vez.

    ``seed_transpiler`` fijo => el routing (estocastico en heavy-hex) es
    determinista, asi el circuito es identico entre invocaciones y al variar la
    semilla de muestreo solo cambia el ruido de disparo (barras de error limpias).
    """
    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level,
                                      seed_transpiler=seed_transpiler)
    return pm.run(qc)


def run_backend(tqc, backend, shots: int, kind: str,
                seed: Optional[int] = None) -> Dict[str, int]:
    """Muestrea un circuito ya transpilado con SamplerV2. ``seed`` -> reproducible.

    Para fake (Aer), la semilla se fija en el constructor:
    ``AerSamplerV2.from_backend(backend, seed=seed)`` (verificado reproducible).
    Para hardware real no hay semilla: el muestreo lo produce el dispositivo.
    """
    if kind == "real":
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
        sampler = RuntimeSampler(backend)
    else:  # fake -> SamplerV2 de Aer con el modelo de ruido del backend
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        if seed is not None:
            sampler = AerSamplerV2.from_backend(backend, seed=int(seed))
        else:
            sampler = AerSamplerV2.from_backend(backend)
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
    return getattr(data, creg).get_counts()


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
def _aggregate(per_list: List[Dict[str, Any]], idx_key: str,
               keep_per_item: bool = True) -> Dict[str, Any]:
    """Agrega una lista de dicts de metricas: claves planas = MEDIA, mas
    ``<clave>_std`` (desv. tipica muestral) por escalar. ``idx_key`` es el
    nombre del indice (``seed`` o ``boot``). Las claves no escalares se toman
    del primer elemento.
    """
    # claves de contabilidad o constantes: no tiene sentido darles media/std
    BOOKKEEPING = {idx_key, "seed", "boot", "seconds", "topk_k",
                   "mc_noise_scale", "shots"}
    base = dict(per_list[0])
    scalar_keys = [k for k, v in per_list[0].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and k not in BOOKKEEPING]
    n = len(per_list)
    for k in scalar_keys:
        vals = [s[k] for s in per_list if isinstance(s.get(k), (int, float))]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        base[k] = float(arr.mean())
        base[f"{k}_std"] = float(arr.std(ddof=1)) if n >= 2 else 0.0
    base["n_" + idx_key + "s"] = n
    if keep_per_item:
        base[idx_key + "s"] = [s.get(idx_key) for s in per_list]
        base["per_" + idx_key] = [
            {kk: s[kk] for kk in ([idx_key] + scalar_keys) if kk in s}
            for s in per_list
        ]
    return base


def _aggregate_over_seeds(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _aggregate(per_seed, "seed", keep_per_item=True)


def bootstrap_metrics(mps, N: int, tgt_counts: Dict[str, int],
                      ref_counts: Dict[str, int], shots: int,
                      B: int, seed: int = 0) -> Dict[str, Any]:
    """Barra de error por bootstrap de los counts del objetivo.

    Trata ``tgt_counts`` como la distribucion ruidosa devuelta por el backend y
    remuestrea ``shots`` extracciones B veces (multinomial), recomputando las
    metricas. Da media y ``<clave>_std`` del RUIDO DE DISPARO. Funciona igual
    para circuitos estaticos y dinamicos (reuse), donde el re-muestreo de Aer no
    genera ruido real. Reproducible via ``seed``.
    """
    rng = np.random.default_rng(seed)
    keys = list(tgt_counts)
    probs = np.array([tgt_counts[k] for k in keys], dtype=np.float64)
    probs = probs / probs.sum()
    per_boot: List[Dict[str, Any]] = []
    for b in range(B):
        draw = rng.multinomial(shots, probs)
        resampled = {k: int(d) for k, d in zip(keys, draw) if d > 0}
        m = generative_metrics(mps, N, resampled, ref_counts)
        m["boot"] = b
        per_boot.append(m)
    return _aggregate(per_boot, "boot", keep_per_item=False)


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
    ap.add_argument("--seed", type=int, default=None,
                    help="semilla de MUESTREO (reproducible). None = aleatoria.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="lista de semillas separadas por coma, p.ej. 0,1,2,3,4. "
                         "Transpila UNA vez y repite solo el muestreo por semilla; "
                         "la salida da media y <clave>_std. Tiene prioridad sobre --seed.")
    ap.add_argument("--transpile-seed", type=int, default=1234,
                    help="semilla del transpilador (routing). Fija => circuito "
                         "determinista entre invocaciones.")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="si >0, barra de error por bootstrap de los counts (B "
                         "remuestreos). RECOMENDADO: es el unico metodo que da "
                         "ruido de disparo real tambien en circuitos reuse "
                         "(dinamicos), donde Aer no lo genera. Tiene prioridad "
                         "sobre --seeds para las columnas _std.")
    args = ap.parse_args()

    # semillas efectivas
    if args.seeds:
        seeds: List[Optional[int]] = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [None]  # una corrida sin semilla (comportamiento previo)

    mod = load_module(args.module_path)
    mps = mod.MPS.load(str(args.data_dir / args.model))
    mod.prepare_right_canonical(mps)
    print(f"MPS: {mps.num_sites} sites, bonds={list(mps.full_bond_dims)}")

    qc, N, b_max, reuse = build_circuit(mod, mps, args.variant)
    print(f"circuito: variante={args.variant}, qubits={qc.num_qubits}, "
          f"N={N}, b_max={b_max}, reuse={reuse}")

    basis = tuple(x.strip() for x in args.basis.split(",") if x.strip())
    ref_shots = args.ref_shots or args.shots

    # --- TRANSPILAR UNA SOLA VEZ (asi solo varia la semilla de muestreo) ---
    tqc_ideal = transpile_ideal(qc, basis, seed_transpiler=args.transpile_seed)
    kind, backend, desc = get_target(args.backend)
    print(f"backend objetivo: {desc}")
    if kind == "aer_ideal":
        tqc_dev = None
        n2q = None
    else:
        tqc_dev = transpile_backend(qc, backend, args.opt_level,
                                    seed_transpiler=args.transpile_seed)
        n2q = sum(v for g, v in tqc_dev.count_ops().items()
                  if g in ("cx", "cz", "ecr", "rzz"))
        print(f"  circuito ejecutado (transpilado 1 vez): "
              f"profundidad {tqc_dev.depth()}, {n2q} puertas 2q")

    print(f"\nmuestreo con {len(seeds)} semilla(s): {seeds}")
    per_seed: List[Dict[str, Any]] = []
    first_tgt_counts: Optional[Dict[str, int]] = None
    first_ref_counts: Optional[Dict[str, int]] = None
    for sd in seeds:
        t0 = time.perf_counter()
        # referencia ideal con semilla INDEPENDIENTE del objetivo (si no, en el
        # modo aer ref==tgt y las metricas hw-vs-ideal serian triviales).
        ref_seed = None if sd is None else sd + 10_000
        ref_counts = counts_to_site_order(run_ideal(tqc_ideal, ref_shots, seed=ref_seed))
        if kind == "aer_ideal":
            tgt_counts = counts_to_site_order(run_ideal(tqc_ideal, args.shots, seed=sd))
        else:
            tgt_counts = counts_to_site_order(
                run_backend(tqc_dev, backend, args.shots, kind, seed=sd))
        seconds = time.perf_counter() - t0

        if first_tgt_counts is None:
            first_tgt_counts, first_ref_counts = tgt_counts, ref_counts

        m = generative_metrics(mps, N, tgt_counts, ref_counts)
        m["seed"] = sd
        m["seconds"] = seconds
        per_seed.append(m)
        fid = m.get("fidelity_hw_vs_mps")
        print(f"  seed={sd}: fidelidad={fid if fid is None else round(fid,4)}, "
              f"{len(tgt_counts)} bitstrings, {seconds:.1f}s")

    # --- agregado / barra de error ---
    error_bar_method = "none"
    if args.bootstrap > 0:
        # Bootstrap de los counts del primer muestreo: ruido de disparo real,
        # uniforme para todas las variantes (incl. reuse dinamico).
        boot_seed = seeds[0] if seeds[0] is not None else 0
        print(f"\nbootstrap: {args.bootstrap} remuestreos de los counts "
              f"(ruido de disparo, valido tambien en reuse)...")
        metrics = bootstrap_metrics(mps, N, first_tgt_counts, first_ref_counts,
                                    args.shots, args.bootstrap, seed=boot_seed)
        error_bar_method = "bootstrap"
    elif len(per_seed) == 1:
        metrics = dict(per_seed[0])
    else:
        metrics = _aggregate_over_seeds(per_seed)
        error_bar_method = "seeds"
    metrics["error_bar_method"] = error_bar_method

    metrics["backend"] = desc
    metrics["variant"] = args.variant
    metrics["N_sites"] = int(N)
    metrics["b_max"] = int(b_max)
    if tqc_dev is not None:
        metrics["executed_depth"] = int(tqc_dev.depth())
        metrics["executed_2q_gates"] = int(n2q)

    print("\n" + "=" * 70)
    tag_seeds = f"{len(seeds)} semillas" if len(seeds) > 1 else f"seed={seeds[0]}"
    print(f"FIDELIDAD GENERATIVA  ({desc}, {tag_seeds})")
    print("=" * 70)
    if metrics.get("tvd_hw_vs_mps") is not None:
        std = metrics.get("fidelity_hw_vs_mps_std")
        std_s = f" +/- {std:.4f}" if std is not None else ""
        print(f"  fidelidad clasica vs MPS : {metrics['fidelity_hw_vs_mps']:.4f}{std_s}")
        tstd = metrics.get("tvd_hw_vs_mps_std")
        tstd_s = f" +/- {tstd:.4f}" if tstd is not None else ""
        print(f"  TVD(hardware, MPS)       : {metrics['tvd_hw_vs_mps']:.4f}{tstd_s}")
        print(f"  marginal L1   vs MPS     : {metrics['marginal_L1_hw_vs_mps']:.4f}")
        print(f"  TVD(ideal,    MPS)       : {metrics['tvd_ref_vs_mps']:.4f}  (suelo MC)")
        print(f"  TVD(hardware, ideal)     : {metrics['tvd_hw_vs_ref']:.4f}  (solo ruido)")
    else:
        print(f"  TVD(hw,MPS) soporte obs. : {metrics['tvd_hw_vs_mps_observed_support']:.4f}")
        print(f"  masa MPS en soporte obs. : {metrics['mps_mass_on_observed_support']:.4f}")
        print(f"  marginal L1 (hw vs ideal): {metrics['marginal_L1_hw_vs_ref']:.4f}")
    if len(seeds) > 1:
        print(f"  (agregado sobre {len(seeds)} semillas; las claves _std son la desv. tipica)")

    out_dir = args.out_dir or (args.data_dir / "hardware_generation")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.backend.replace(":", "_")
    (out_dir / f"generation_{tag}_{args.variant}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote: {out_dir / f'generation_{tag}_{args.variant}.json'}")


if __name__ == "__main__":
    main()