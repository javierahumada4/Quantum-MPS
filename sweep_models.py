#!/usr/bin/env python3
"""
sweep_models.py
===============

Driver de experimentos para comparar modelos generativos MPS de DISTINTO TAMANYO
(distinto numero de features k) de cara a su ejecucion en hardware.

Para cada (k, D_max) de la rejilla --k-list x --bond-list:
    1. selecciona las k features mejor ordenadas (feature selection),
    2. entrena un MPS de ese tamanyo y capacidad (D_max),
    3. corre compare_mps_circuit_variants.py   (correccion exacta + coste abstracto),
    4. corre hardware_resource_estimate.py      (coste con conectividad real),
    5. para cada variante de --variants, corre run_generative_mps_ibm.py en aer y
       fake:DEVICE  (fidelidad vs MPS),
y junta TODO en una unica tabla comparativa (sweep_summary.csv/json), priorizando
RECURSOS (device_2q, profundidad, qubits) y FIDELIDAD (fidelidad clasica y
marginal_L1 frente al MPS exacto). NO usa la QPU real: solo aer y fake.

Tres barridos en uno:
    - features:  --k-list 8,11,14,17  --bond-list 4
    - capacidad: --k-list 8  --bond-list 2,4,8        (D_max => b_max=log2 D_max)
    - variantes: --k-list 8  --bond-list 4  --variants no_reuse_isometry,reuse_unitary,reuse_isometry,no_reuse_unitary

Reusa los tres scripts tal cual (via subprocess) y solo importa el entrenador
para fijar el bond maximo.

Orden de features
-----------------
  --feature-order order.json   lista JSON de indices de sitio, del MAS al MENOS
                               importante. Para cada k se toman los k primeros.
  (por defecto)                proxy generativo: entropia marginal por sitio.

Uso:
    # barrido de features (D_max fijo)
    python sweep_models.py ./nsl_kdd --k-list 8,11,14,17 --bond-list 4 \
        --device torino --variants no_reuse_isometry --shots 8000

    # barrido de capacidad (k fijo)
    python sweep_models.py ./nsl_kdd --k-list 8 --bond-list 2,4,8 --shots 8000

    # comparar variantes (k y D_max fijos)
    python sweep_models.py ./nsl_kdd --k-list 8 --bond-list 4 \
        --variants no_reuse_isometry,reuse_unitary,reuse_isometry,no_reuse_unitary

    # smoke test rapido
    python sweep_models.py ./nsl_kdd --k-list 8 --bond-list 2,4 --num-loops 5 --shots 2000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------
def load_full_artifacts(data_dir: Path):
    schema = json.loads((data_dir / "encoding_schema.json").read_text())
    train_X = torch.load(data_dir / "train_normal_X.pt", weights_only=True).long()
    val_path = data_dir / "val_normal_X.pt"
    val_X = (torch.load(val_path, weights_only=True).long()
             if val_path.exists() else None)
    return schema, train_X, val_X


def entropy_ranking(train_X: torch.Tensor) -> List[int]:
    """Orden por entropia marginal por sitio (mas informativo primero)."""
    p1 = train_X.float().mean(dim=0).clamp(1e-6, 1 - 1e-6).numpy()
    H = -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))
    return list(np.argsort(-H))  # descendente


def load_feature_order(path: Optional[Path], train_X: torch.Tensor) -> List[int]:
    if path is None:
        order = entropy_ranking(train_X)
        print(f"feature order: proxy por entropia marginal (pasa --feature-order "
              f"para usar tu RFE)")
        return order
    raw = json.loads(Path(path).read_text())
    # admite [i,j,...] o {"order":[...]} o registros con clave 'site'
    if isinstance(raw, dict) and "order" in raw:
        order = list(raw["order"])
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
        order = [int(r["site"]) for r in raw]
    else:
        order = [int(x) for x in raw]
    print(f"feature order: {path}")
    return order


def write_reduced_dir(out_dir: Path, schema: Dict, train_X, val_X, sites: List[int]):
    """Escribe un data_dir reducido con las columnas (sitios) seleccionadas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = torch.tensor(sites, dtype=torch.long)
    torch.save(train_X.index_select(1, idx).cpu(), out_dir / "train_normal_X.pt")
    if val_X is not None:
        torch.save(val_X.index_select(1, idx).cpu(), out_dir / "val_normal_X.pt")

    red = dict(schema)
    red["physical_dims"] = [int(schema["physical_dims"][s]) for s in sites]
    if "feature_names" in schema:
        red["feature_names"] = [schema["feature_names"][s] for s in sites]
    red["selected_sites"] = [int(s) for s in sites]
    (out_dir / "encoding_schema.json").write_text(json.dumps(red, indent=2))


def train_reduced(out_dir: Path, max_bond_dim: int, num_loops: Optional[int],
                  fixed_bond: bool = False):
    """Entrena un MPS en out_dir fijando el bond maximo (import + override).

    Si ``fixed_bond`` es True, desactiva el crecimiento adaptativo y la
    truncacion para que el eje de capacidad sea exacto y reproducible:
      * init_bond_cap = max_bond_dim   -> sin crecimiento (el cap nace al maximo)
      * discarded_weight_threshold = 0 -> no se descarta dimension por peso
      * svd_cutoff = 0                 -> no se recorta por valor singular
    Asi todos los bonds (que la estructura permita) valen exactamente max_bond_dim
    y b_max = log2(max_bond_dim) queda fijado por el tope, no por el entrenamiento.
    """
    import importlib
    import train_mps_nsl_kdd as t
    importlib.reload(t)  # estado limpio entre puntos del barrido
    t.CONFIG.max_bond_dim = int(max_bond_dim)

    if fixed_bond:
        t.CONFIG.init_bond_cap = int(max_bond_dim)        # cap = max -> sin crecimiento
        t.CONFIG.discarded_weight_threshold = 0.0          # sin recorte por peso
        t.CONFIG.svd_cutoff = 0                     # sin recorte por valor singular
        t.INIT_BOND_DIM = min(int(t.INIT_BOND_DIM), int(max_bond_dim))
    else:
        t.CONFIG.init_bond_cap = min(int(t.CONFIG.init_bond_cap), int(max_bond_dim))
        t.INIT_BOND_DIM = min(int(t.INIT_BOND_DIM), int(t.CONFIG.init_bond_cap))

    if num_loops is not None:
        t.CONFIG.num_loops = int(num_loops)
    t.main(out_dir)


def run_script(script: str, args: List[str], scripts_dir: Path) -> int:
    cmd = [sys.executable, str(scripts_dir / script)] + args
    print("    $ " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    [WARN] {script} salio con codigo {res.returncode}")
        tail = (res.stderr or res.stdout).strip().splitlines()[-4:]
        for ln in tail:
            print(f"      | {ln}")
    return res.returncode


def read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def pick_variant(rows: List[Dict], variant: str) -> Dict:
    for r in (rows or []):
        if r.get("variant") == variant:
            return r
    return {}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="dir con artefactos completos del encoder")
    ap.add_argument("--k-list", default="8,11,14,17")
    ap.add_argument("--bond-list", default="4",
                    help="lista de D_max a barrer, p.ej. 2,4,8. b_max=log2(D_max).")
    ap.add_argument("--num-loops", type=int, default=None,
                    help="override de bucles DMRG (None = el del CONFIG)")
    ap.add_argument("--fixed-bond", action="store_true",
                    help="desactiva crecimiento adaptativo y truncacion: cada bond vale "
                         "exactamente D_max (eje de capacidad exacto y reproducible).")
    ap.add_argument("--feature-order", type=Path, default=None)
    ap.add_argument("--device", default="torino", choices=["torino", "brisbane"])
    ap.add_argument("--variants", default="no_reuse_isometry",
                    help="lista de variantes a evaluar en fidelidad, separadas por coma. "
                         "p.ej. no_reuse_isometry,reuse_unitary,reuse_isometry,no_reuse_unitary")
    ap.add_argument("--shots", type=int, default=8000)
    ap.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    bond_list = [int(x) for x in args.bond_list.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    sd = args.scripts_dir
    sweep_root = args.out_dir or (args.data_dir / "sweep")
    sweep_root.mkdir(parents=True, exist_ok=True)

    schema, train_X, val_X = load_full_artifacts(args.data_dir)
    N_full = train_X.shape[1]
    order = load_feature_order(args.feature_order, train_X)
    order = [s for s in order if 0 <= s < N_full]

    rows: List[Dict[str, Any]] = []
    for k in k_list:
        if k > N_full:
            print(f"\n[skip] k={k} > N_full={N_full}")
            continue
        sites = sorted(order[:k])  # preserva la localidad de la cadena MPS
        for bond in bond_list:
            print("\n" + "=" * 78)
            print(f"k = {k} features   |   D_max = {bond} (b_max<=log2(D_max)={int(np.log2(bond))})")
            print("=" * 78)
            kdir = sweep_root / f"k{k:02d}_D{bond:02d}"
            write_reduced_dir(kdir, schema, train_X, val_X, sites)

            print("  [1] entrenando MPS...")
            try:
                train_reduced(kdir, bond, args.num_loops, fixed_bond=args.fixed_bond)
            except Exception as exc:
                print(f"    [ERROR] entrenamiento fallo: {exc!r}")
                rows.append({"k": k, "D_max": bond, "error": f"train: {exc!r}"})
                continue

            print("  [2] compare_mps_circuit_variants...")
            run_script("compare_mps_circuit_variants.py",
                       [str(kdir), "--max-exact-sites", str(k),
                        "--shots", str(min(args.shots, 4000))], sd)
            comp = read_json(kdir / "circuit_comparison" / "circuit_variant_comparison.json")

            print("  [3] hardware_resource_estimate...")
            run_script("hardware_resource_estimate.py",
                       [str(kdir), "--device", args.device, "--opt-level", "3"], sd)
            hw = read_json(kdir / "hardware_estimate" / "hardware_resource_estimate.json")

            for variant in variants:
                print(f"  [4] run_generative ({variant}) aer + fake:{args.device}...")
                run_script("run_generative_mps_ibm.py",
                           [str(kdir), "--backend", "aer", "--variant", variant,
                            "--shots", str(args.shots)], sd)
                gen_aer = read_json(kdir / "hardware_generation" /
                                    f"generation_aer_{variant}.json")
                run_script("run_generative_mps_ibm.py",
                           [str(kdir), "--backend", f"fake:{args.device}", "--variant", variant,
                            "--shots", str(args.shots)], sd)
                gen_fake = read_json(kdir / "hardware_generation" /
                                     f"generation_fake_{args.device}_{variant}.json")

                cvar = pick_variant(comp, variant)
                hvar = pick_variant(hw, variant)
                row = {
                    "k": k,
                    "D_max": bond,
                    "b_max": cvar.get("b_max"),
                    "variant": variant,
                    "logical_qubits": cvar.get("num_qubits"),
                    "statevector_max_abs_err": cvar.get("statevector_max_abs_err"),
                    "abstract_2q": cvar.get("two_qubit_gates"),
                    "device_2q": hvar.get("device_2q"),
                    "routing_inflation": hvar.get("routing_inflation_2q"),
                    "device_depth": hvar.get("device_depth"),
                    "ideal_fidelity": (gen_aer or {}).get("fidelity_hw_vs_mps"),
                    "hw_fidelity": (gen_fake or {}).get("fidelity_hw_vs_mps"),
                    "hw_marginal_L1_vs_mps": (gen_fake or {}).get("marginal_L1_hw_vs_mps"),
                    "hw_tvd_vs_mps": (gen_fake or {}).get("tvd_hw_vs_mps"),
                    "hw_tvd_vs_ideal": (gen_fake or {}).get("tvd_hw_vs_ref"),
                    "hw_topk_overlap": (gen_fake or {}).get("topk_overlap"),
                }
                rows.append(row)
                print(f"    -> {variant}: device_2q={row['device_2q']}, "
                      f"hw_fidelity={row['hw_fidelity']}, "
                      f"marginal_L1={row['hw_marginal_L1_vs_mps']}")

    # --- guardar y resumir ---
    (sweep_root / "sweep_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(sweep_root / "sweep_summary.csv", index=False)
        cols = ["k", "D_max", "b_max", "variant", "logical_qubits",
                "device_2q", "device_depth", "routing_inflation",
                "hw_fidelity", "hw_marginal_L1_vs_mps", "hw_tvd_vs_mps"]
        shown = [c for c in cols if c in df.columns]
        print("\n" + "=" * 78)
        print(f"SWEEP SUMMARY (device={args.device})  [recursos + fidelidad vs MPS]")
        print("=" * 78)
        print(df[shown].to_string(index=False))
    except Exception as exc:
        print(f"[warn] no pude formatear con pandas: {exc!r}")

    print(f"\nwrote: {sweep_root / 'sweep_summary.csv'}")
    print(f"wrote: {sweep_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()