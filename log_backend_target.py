# log_backend_target.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def get_backend(spec: str):
    if spec == "fake:fez":
        from qiskit_ibm_runtime.fake_provider import FakeFez
        return FakeFez(), "fake:fez"

    if spec == "fake:marrakesh":
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        return FakeMarrakesh(), "fake:marrakesh"

    if spec == "fake:kingston":
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        return FakeKingston(), "fake:kingston"

    if spec.startswith("real:"):
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        name = spec.split(":", 1)[1]
        return service.backend(name), spec

    raise ValueError(f"Backend no soportado: {spec}")


def target_summary(backend, spec: str) -> dict:
    target = backend.target

    operation_names = sorted(str(x) for x in target.operation_names)

    per_operation = {}
    two_qubit_operations = {}

    for op_name in operation_names:
        try:
            qarg_map = target[op_name]
        except Exception:
            continue

        qargs_serialized = []
        twoq_qargs = []

        for qargs in qarg_map:
            if qargs is None:
                qargs_serialized.append("global")
                continue

            qargs_tuple = tuple(int(q) for q in qargs)
            qargs_serialized.append(list(qargs_tuple))

            if len(qargs_tuple) == 2:
                twoq_qargs.append(list(qargs_tuple))

        per_operation[op_name] = {
            "num_qargs": len(qargs_serialized),
            "qargs": qargs_serialized[:50],
            "qargs_truncated": len(qargs_serialized) > 50,
        }

        if twoq_qargs:
            two_qubit_operations[op_name] = {
                "num_edges": len(twoq_qargs),
                "edges_sample": twoq_qargs[:50],
                "edges_truncated": len(twoq_qargs) > 50,
            }

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend_spec": spec,
        "backend_name": getattr(backend, "name", None),
        "backend_version": getattr(backend, "backend_version", None),
        "num_qubits": getattr(backend, "num_qubits", None),
        "operation_names": operation_names,
        "basis_gates_backend_attr": getattr(backend, "basis_gates", None),
        "two_qubit_operation_names": sorted(two_qubit_operations.keys()),
        "two_qubit_operations": two_qubit_operations,
        "per_operation": per_operation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="fake:fez")
    parser.add_argument("--out", default="./nsl_kdd/backend_target_fake_fez.json")
    args = parser.parse_args()

    backend, spec = get_backend(args.backend)
    summary = target_summary(backend, spec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Backend: {summary['backend_name']}")
    print(f"Qubits: {summary['num_qubits']}")
    print(f"Operation names: {', '.join(summary['operation_names'])}")
    print(f"Two-qubit operations: {', '.join(summary['two_qubit_operation_names'])}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()