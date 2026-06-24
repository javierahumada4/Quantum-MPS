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
import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
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


def jsonable(value: Any) -> Any:
    """Convierte numpy/torch/path/etc. a tipos seguros para JSON y CSV."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def json_dumps(data: Any, **kwargs) -> str:
    return json.dumps(jsonable(data), ensure_ascii=False, **kwargs)


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-._")
    return (text or "sweep")[:max_len]


def make_run_id(args, k_list: List[int], bond_list: List[int], variants: List[str],
                timestamp_utc: str) -> str:
    """Identificador estable y legible para aislar cada ejecución del barrido."""
    if args.sweep_name:
        return slugify(args.sweep_name)
    payload = {
        "data_dir": str(args.data_dir.resolve()),
        "k_list": k_list,
        "bond_list": bond_list,
        "variants": variants,
        "device": args.device,
        "shots": args.shots,
        "fixed_bond": args.fixed_bond,
        "num_loops": args.num_loops,
        "feature_order": str(args.feature_order) if args.feature_order else None,
    }
    digest = hashlib.sha1(json_dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    compact_ts = timestamp_utc.replace("-", "").replace(":", "").replace("+00:00", "Z")
    return slugify(f"{compact_ts}_k{'-'.join(map(str, k_list))}_D{'-'.join(map(str, bond_list))}_{digest}")


def load_existing_summary(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
        return raw if isinstance(raw, list) else []
    except Exception as exc:
        print(f"[WARN] no pude leer resumen existente {path}: {exc!r}; empiezo sin acumularlo")
        return []


def flatten_rows_for_csv(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        out = {}
        for key, value in jsonable(row).items():
            if isinstance(value, (dict, list)):
                out[key] = json_dumps(value)
            else:
                out[key] = value
        flat_rows.append(out)
    return flat_rows


def write_summary_files(rows: List[Dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    """Escribe JSON completo y CSV plano con las mismas filas."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_dumps(rows, indent=2))

    flat_rows = flatten_rows_for_csv(rows)
    fieldnames: List[str] = []
    seen = set()
    for row in flat_rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)


def summarize_training(data_dir: Path) -> Dict[str, Any]:
    """Extrae lo necesario para diagnosticar convergencia sin abrir train_history.json."""
    hist = read_json(data_dir / "train_history.json") or []
    if not isinstance(hist, list) or not hist:
        return {"train_history_rows": 0}

    def finite_number(x):
        return isinstance(x, (int, float)) and np.isfinite(float(x))

    first = hist[0]
    last = hist[-1]
    train_vals = [(r.get("loop"), r.get("train_nll")) for r in hist if finite_number(r.get("train_nll"))]
    val_vals = [(r.get("loop"), r.get("val_nll")) for r in hist if finite_number(r.get("val_nll"))]
    best_train_loop, best_train = min(train_vals, key=lambda kv: kv[1]) if train_vals else (None, None)
    best_val_loop, best_val = min(val_vals, key=lambda kv: kv[1]) if val_vals else (None, None)

    final_bonds = last.get("bond_dims") or []
    if isinstance(final_bonds, list) and final_bonds:
        final_bond_min = int(min(final_bonds))
        final_bond_max = int(max(final_bonds))
        final_bond_mean = float(np.mean(final_bonds))
    else:
        final_bond_min = final_bond_max = final_bond_mean = None

    total_updates = sum(int(r.get("num_updates") or 0) for r in hist)
    total_skipped = sum(int(r.get("num_skipped_nan") or 0) for r in hist)
    wallclock = last.get("wallclock_s")
    elapsed_sum = sum(float(r.get("elapsed_s") or 0.0) for r in hist)

    out: Dict[str, Any] = {
        "train_history_rows": len(hist),
        "train_first_loop": first.get("loop"),
        "train_final_loop": last.get("loop"),
        "train_first_train_nll": first.get("train_nll"),
        "train_final_train_nll": last.get("train_nll"),
        "train_best_train_nll": best_train,
        "train_best_train_loop": best_train_loop,
        "train_final_val_nll": last.get("val_nll"),
        "train_best_val_nll": best_val,
        "train_best_val_loop": best_val_loop,
        "train_final_generalization_gap": (
            last.get("val_nll") - last.get("train_nll")
            if finite_number(last.get("val_nll")) and finite_number(last.get("train_nll")) else None
        ),
        "train_final_lr": last.get("lr"),
        "train_final_bond_dims": final_bonds,
        "train_final_bond_min": final_bond_min,
        "train_final_bond_max": final_bond_max,
        "train_final_bond_mean": final_bond_mean,
        "train_final_cap": last.get("max_bond_dim_cap"),
        "train_max_gradient_norm": max((float(r.get("max_gradient_norm") or 0.0) for r in hist), default=None),
        "train_max_discarded_weight": max((float(r.get("max_discarded_weight") or 0.0) for r in hist), default=None),
        "train_total_updates": total_updates,
        "train_total_skipped_nan": total_skipped,
        "train_wallclock_s": wallclock,
        "train_elapsed_s_sum": elapsed_sum,
    }
    return out


def feature_names_for_sites(schema: Dict[str, Any], sites: List[int]) -> List[str]:
    names = schema.get("feature_names") or []
    out = []
    for s in sites:
        if 0 <= int(s) < len(names):
            out.append(str(names[int(s)]))
        else:
            out.append(f"site_{int(s)}")
    return out


def add_prefixed_fields(row: Dict[str, Any], prefix: str, data: Optional[Dict[str, Any]],
                        *, skip: Optional[set] = None) -> None:
    """Copia métricas de un JSON hijo a la fila con prefijo para hacerlo autocontenido."""
    if not data:
        return
    skip = skip or set()
    for key, value in data.items():
        if key in skip:
            continue
        row[f"{prefix}_{key}"] = value


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
    ap.add_argument("--sweep-name", default=None,
                    help="nombre legible para este barrido; si se omite se genera uno con timestamp")
    ap.add_argument("--summary-mode", default="append", choices=["append", "overwrite"],
                    help="append acumula en sweep_summary.*; overwrite reinicia el resumen acumulado")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="directorio para los artefactos de ESTA ejecucion; por defecto sweep/runs/<run_id>")
    args = ap.parse_args()

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    bond_list = [int(x) for x in args.bond_list.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    sd = args.scripts_dir
    sweep_root = args.out_dir or (args.data_dir / "sweep")
    sweep_root.mkdir(parents=True, exist_ok=True)

    timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    run_id = make_run_id(args, k_list, bond_list, variants, timestamp_utc)
    run_root = args.run_dir or (sweep_root / "runs" / run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    summary_json = sweep_root / "sweep_summary.json"
    summary_csv = sweep_root / "sweep_summary.csv"
    existing_rows = [] if args.summary_mode == "overwrite" else load_existing_summary(summary_json)

    schema, train_X, val_X = load_full_artifacts(args.data_dir)
    N_full = int(train_X.shape[1])
    order = load_feature_order(args.feature_order, train_X)
    order = [int(s) for s in order if 0 <= int(s) < N_full]

    run_meta: Dict[str, Any] = {
        "sweep_run_id": run_id,
        "sweep_name": args.sweep_name or run_id,
        "sweep_timestamp_utc": timestamp_utc,
        "sweep_command": " ".join(shlex.quote(x) for x in sys.argv),
        "sweep_root": str(sweep_root),
        "sweep_run_root": str(run_root),
        "source_data_dir": str(args.data_dir),
        "scripts_dir": str(sd),
        "summary_mode": args.summary_mode,
        "device": args.device,
        "shots": int(args.shots),
        "num_loops_override": args.num_loops,
        "fixed_bond": bool(args.fixed_bond),
        "k_list_arg": k_list,
        "bond_list_arg": bond_list,
        "variants_arg": variants,
        "feature_order_path": str(args.feature_order) if args.feature_order else None,
        "feature_order_strategy": "file" if args.feature_order else "entropy_marginal_proxy",
        "N_full": N_full,
        "n_train_normal_rows": int(len(train_X)),
        "n_val_normal_rows": int(0 if val_X is None else len(val_X)),
    }

    rows: List[Dict[str, Any]] = []

    def persist_progress() -> None:
        combined = existing_rows + rows
        write_summary_files(combined, summary_json, summary_csv)
        write_summary_files(rows, run_root / "sweep_summary_this_run.json",
                            run_root / "sweep_summary_this_run.csv")
    for k in k_list:
        if k > N_full:
            print(f"\n[skip] k={k} > N_full={N_full}")
            continue
        topk_sites_by_importance = [int(s) for s in order[:k]]
        sites = sorted(topk_sites_by_importance)  # preserva la localidad de la cadena MPS
        selected_feature_names = feature_names_for_sites(schema, sites)
        topk_feature_names = feature_names_for_sites(schema, topk_sites_by_importance)
        for bond in bond_list:
            print("\n" + "=" * 78)
            print(f"k = {k} features   |   D_max = {bond} (b_max<=log2(D_max)={int(np.log2(bond))})")
            print("=" * 78)
            kdir = run_root / f"k{k:02d}_D{bond:02d}"
            point_meta: Dict[str, Any] = {
                **run_meta,
                "k": int(k),
                "D_max": int(bond),
                "requested_b_max_log2_D": int(np.log2(bond)) if bond > 0 else None,
                "experiment_dir": str(kdir),
                "selected_sites_chain_order": sites,
                "selected_features_chain_order": selected_feature_names,
                "selected_sites_importance_order": topk_sites_by_importance,
                "selected_features_importance_order": topk_feature_names,
            }
            write_reduced_dir(kdir, schema, train_X, val_X, sites)

            print("  [1] entrenando MPS...")
            try:
                train_reduced(kdir, bond, args.num_loops, fixed_bond=args.fixed_bond)
            except Exception as exc:
                print(f"    [ERROR] entrenamiento fallo: {exc!r}")
                rows.append({**point_meta, "stage": "train", "status": "error",
                             "error": f"train: {exc!r}"})
                persist_progress()
                continue

            train_summary = summarize_training(kdir)

            print("  [2] compare_mps_circuit_variants...")
            compare_rc = run_script("compare_mps_circuit_variants.py",
                       [str(kdir), "--max-exact-sites", str(k),
                        "--shots", str(min(args.shots, 4000))], sd)
            comp = read_json(kdir / "circuit_comparison" / "circuit_variant_comparison.json")

            print("  [3] hardware_resource_estimate...")
            hardware_rc = run_script("hardware_resource_estimate.py",
                       [str(kdir), "--device", args.device, "--opt-level", "3"], sd)
            hw = read_json(kdir / "hardware_estimate" / "hardware_resource_estimate.json")

            for variant in variants:
                print(f"  [4] run_generative ({variant}) aer + fake:{args.device}...")
                gen_aer_rc = run_script("run_generative_mps_ibm.py",
                           [str(kdir), "--backend", "aer", "--variant", variant,
                            "--shots", str(args.shots)], sd)
                gen_aer = read_json(kdir / "hardware_generation" /
                                    f"generation_aer_{variant}.json")
                gen_fake_rc = run_script("run_generative_mps_ibm.py",
                           [str(kdir), "--backend", f"fake:{args.device}", "--variant", variant,
                            "--shots", str(args.shots)], sd)
                gen_fake = read_json(kdir / "hardware_generation" /
                                     f"generation_fake_{args.device}_{variant}.json")

                cvar = pick_variant(comp, variant)
                hvar = pick_variant(hw, variant)
                row = {
                    **point_meta,
                    **train_summary,
                    "stage": "variant",
                    "status": "ok",
                    "variant": variant,
                    "synthesis": cvar.get("synthesis", hvar.get("synthesis")),
                    "reuse": cvar.get("reuse", hvar.get("reuse")),
                    "compare_return_code": compare_rc,
                    "hardware_return_code": hardware_rc,
                    "gen_aer_return_code": gen_aer_rc,
                    "gen_fake_return_code": gen_fake_rc,

                    # Alias historicos / columnas principales para leer rapido.
                    "b_max": cvar.get("b_max", hvar.get("b_max")),
                    "logical_qubits": cvar.get("num_qubits", hvar.get("logical_qubits")),
                    "statevector_max_abs_err": cvar.get("statevector_max_abs_err"),
                    "abstract_2q": cvar.get("two_qubit_gates", hvar.get("abstract_2q")),
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

                # Métricas completas autocontenidas: no hace falta abrir subcarpetas.
                add_prefixed_fields(row, "circuit", cvar)
                add_prefixed_fields(row, "routed", hvar)
                add_prefixed_fields(row, "aer", gen_aer)
                add_prefixed_fields(row, "fake", gen_fake)

                rows.append(row)
                persist_progress()
                print(f"    -> {variant}: device_2q={row['device_2q']}, "
                      f"hw_fidelity={row['hw_fidelity']}, "
                      f"marginal_L1={row['hw_marginal_L1_vs_mps']}")

    # --- guardar y resumir ---
    persist_progress()
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        all_df = pd.DataFrame(existing_rows + rows)
        cols = ["sweep_run_id", "k", "D_max", "b_max", "variant", "logical_qubits",
                "device_2q", "device_depth", "routing_inflation",
                "hw_fidelity", "hw_marginal_L1_vs_mps", "hw_tvd_vs_mps",
                "train_best_val_nll", "train_final_val_nll"]
        shown = [c for c in cols if c in df.columns]
        print("\n" + "=" * 78)
        print(f"SWEEP SUMMARY THIS RUN (device={args.device})  [recursos + fidelidad vs MPS]")
        print("=" * 78)
        if shown and len(df):
            print(df[shown].to_string(index=False))
        print(f"\nfilas en este barrido: {len(rows)}")
        print(f"filas acumuladas en sweep_summary: {len(all_df)}")
    except Exception as exc:
        print(f"[warn] no pude formatear con pandas: {exc!r}")

    print(f"\nwrote cumulative: {summary_csv}")
    print(f"wrote cumulative: {summary_json}")
    print(f"wrote this run : {run_root / 'sweep_summary_this_run.csv'}")
    print(f"wrote this run : {run_root / 'sweep_summary_this_run.json'}")


if __name__ == "__main__":
    main()