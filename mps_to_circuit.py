"""
mps_to_circuit.py
=================

Turn a trained Born-machine MPS (yours: physical dim 2 everywhere, bond dims =
powers of two, open boundaries D_0 = D_N = 1) into an *exact* quantum circuit and
run it on the Aer statevector / sampler simulator.

The circuit prepares

        U |0...0>_phys  (x) |0...0>_bond   =   |Psi>_phys (x) |0>_bond

so that measuring the N physical qubits in the computational basis samples
v ~ |Psi(v)|^2 / Z = P(v): exactly the distribution your DMRG model learned.
Anomaly score -log P(v) can then be estimated from shot frequencies, or computed
exactly from the statevector for validation.

Construction (Schoen-Solano-Verstraete-Wolf-Cirac sequential preparation):
  1. right-canonicalise  -> every site tensor A^[k] is a right isometry
        sum_{s,b} A[a,s,b] conj(A[a',s,b]) = delta(a,a')
  2. embed each isometry into a unitary on (bond register) (x) (one physical qubit)
  3. apply them as a staircase; because D_0 = D_N = 1 the bond register starts and
     ends in |0>, leaving a pure product with the physical register.

Because all bonds are powers of two, bond k needs exactly log2(D_k) qubits with no
padding waste -- which is the whole point of `restrict_bond_to_pow2` in training.

Usage:
    python mps_to_circuit.py /path/to/nsl_kdd        # loads mps_trained.pt
    python mps_to_circuit.py /path/to/nsl_kdd --shots 100000
    python mps_to_circuit.py /path/to/nsl_kdd --circuit-png circuit.png  # draw the circuit
    python mps_to_circuit.py /path/to/nsl_kdd --reuse   # qubit-reuse variant (b_max+1 qubits)
    python mps_to_circuit.py /path/to/nsl_kdd --transpile            # report native gate counts
    python mps_to_circuit.py /path/to/nsl_kdd --reuse --transpile --basis cz,rz,sx,x
    python mps_to_circuit.py /path/to/nsl_kdd --synthesis isometry --transpile  # fewer CX

Two synthesis choices for the A_k boxes:
  * unitary (default): each A_k padded to a generic (b_max+1)-qubit unitary; simple.
  * isometry: Iten et al. decomposition on the minimal qubits per site; exact and
    markedly cheaper on hardware (~50% fewer CX on a tapered bond profile).

Two circuit variants:
  * no-reuse (default): N + b_max qubits, no measurement until the end; has a full
    statevector, so it is verified exactly against P(v).
  * --reuse: b_max + 1 qubits via mid-circuit measurement + reset of one physical
    qubit. Exact too, but inherently a sampler (validated by sampled frequencies).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from mps import MPS

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator, Statevector


# ----------------------------------------------------------------------
def prepare_right_canonical(mps: MPS) -> MPS:
    """Normalise and put the MPS into exact right-canonical form (in place)."""
    mps.normalize_state()
    mps.right_canonicalize(truncate=False)   # QR, lossless -> right isometries
    mps.normalize_state()                    # fold the norm sitting on site 0
    mps.right_canonicalize(truncate=False)
    return mps


def bond_qubit_counts(mps: MPS) -> list[int]:
    counts = []
    for D in mps.full_bond_dims:
        bq = int(round(np.log2(D)))
        if 2 ** bq != D:
            raise ValueError(f"bond dim {D} is not a power of two; "
                             "train with restrict_bond_to_pow2=True")
        counts.append(bq)
    return counts


def site_unitary(A: torch.Tensor, b_max: int) -> np.ndarray:
    """Right-isometry tensor A:(Dl,2,Dr) -> unitary of size 2^(b_max+1).

    Gate register = |bond (b_max qubits)> (x) |phys (1 qubit)>; the meaningful
    bond index lives in the LOW bond qubits, the physical qubit is the LSB.
    """
    Dl, d, Dr = A.shape
    assert d == 2, "physical dimension must be 2 (binarised features)"
    A = A.detach().cpu().numpy().astype(np.complex128)
    nreg = 2 ** (b_max + 1)

    def reg_index(bond, phys):
        return (bond << 1) | phys

    Iso = np.zeros((nreg, Dl), dtype=np.complex128)
    for alpha in range(Dl):
        for s in range(d):
            for beta in range(Dr):
                Iso[reg_index(beta, s), alpha] = A[alpha, s, beta]

    gram = Iso.conj().T @ Iso
    if not np.allclose(gram, np.eye(Dl), atol=1e-8):
        raise RuntimeError(f"site tensor is not a right isometry "
                           f"(max dev {np.abs(gram - np.eye(Dl)).max():.2e}); "
                           "did you right-canonicalise?")

    in_cols = [reg_index(alpha, 0) for alpha in range(Dl)]
    U = np.zeros((nreg, nreg), dtype=np.complex128)
    for alpha in range(Dl):
        U[:, in_cols[alpha]] = Iso[:, alpha]

    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(np.concatenate(
        [Iso, rng.standard_normal((nreg, nreg - Dl))
              + 1j * rng.standard_normal((nreg, nreg - Dl))], axis=1))
    complement = q[:, Dl:]
    remaining = [c for c in range(nreg) if c not in in_cols]
    for j, c in enumerate(remaining):
        U[:, c] = complement[:, j]
    return U


def build_circuit(mps: MPS) -> tuple[QuantumCircuit, int, int]:
    """Return (circuit, N_physical, b_max_bond_qubits)."""
    N = mps.num_sites
    bcounts = bond_qubit_counts(mps)
    b_max = max(bcounts)

    bond = QuantumRegister(b_max, "bond")
    phys = QuantumRegister(N, "phys")
    qc = QuantumCircuit(bond, phys, name="mps_born_machine")

    # one isometry-unitary per site, acting on the whole bond register + phys[k]
    for k in range(N):
        U = site_unitary(mps.site_tensors[k].data, b_max)
        gate = Operator(U)
        # Qiskit is little-endian: the LSB of the gate register is qubit list[0].
        # site_unitary uses phys as LSB, then bond low->high -> [phys[k], bond[0..]].
        qubits = [phys[k]] + list(bond)
        qc.unitary(gate, qubits, label=f"A{k}")
    return qc, N, b_max


# ----------------------------------------------------------------------
def verify_statevector(mps: MPS, qc: QuantumCircuit, N: int, b_max: int) -> None:
    """Compare circuit probabilities to the MPS Born probabilities exactly."""
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()  # keys are bitstrings 'phys... bond...' (q0 rightmost)

    # qubit order in the statevector label (left->right) is high index .. low index:
    # phys[N-1]..phys[0], bond[b_max-1]..bond[0].  We want bond == 0.
    all_v = torch.tensor([[(i >> (N - 1 - s)) & 1 for s in range(N)]
                          for i in range(2 ** N)], dtype=torch.long)
    with torch.no_grad():
        p_mps = torch.exp(mps.log_prob(all_v)).numpy()

    # Qiskit is little-endian: in the label, the LEFTMOST char is the highest-index
    # qubit. Registers were added bond (q0..) then phys (q b_max..), so phys qubits
    # are the high indices -> leftmost N chars; bond qubits are the rightmost b_max.
    p_circ = np.zeros(2 ** N)
    leak = 0.0
    for bitstr, p in probs.items():
        bits = bitstr.replace(" ", "")
        phys_bits = bits[:N]              # phys[N-1] .. phys[0]
        bond_bits = bits[N:]             # bond[b_max-1] .. bond[0]
        if "1" in bond_bits:
            leak += p
            continue
        v_index = int(phys_bits[::-1], 2)  # phys[k] holds site k; all_v has site 0 as MSB
        p_circ[v_index] += p

    print(f"  bond-register leakage     : {leak:.2e}   (should be ~0)")
    print(f"  sum P_mps / sum P_circ    : {p_mps.sum():.6f} / {p_circ.sum():.6f}")
    print(f"  max |P_mps - P_circ|      : {np.abs(p_mps - p_circ).max():.2e}")
    ok = np.abs(p_mps - p_circ).max() < 1e-8 and leak < 1e-8
    print("  -> EXACT MATCH" if ok else "  -> MISMATCH (check endianness / canonical form)")


def sample(qc: QuantumCircuit, N: int, b_max: int, shots: int) -> None:
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    meas = qc.copy()
    creg = ClassicalRegister(N, "v")
    meas.add_register(creg)
    meas.measure([meas.qubits[b_max + k] for k in range(N)], creg)  # phys qubits only
    sim = AerSimulator()
    result = sim.run(transpile(meas, sim), shots=shots).result()
    counts = result.get_counts()
    print(f"\n  sampled {shots} shots; {len(counts)} distinct configurations")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    for bitstr, c in top:
        v = bitstr.replace(" ", "")[::-1]   # creg[N-1]..creg[0] -> site 0..N-1
        print(f"    v={v}  freq={c/shots:.4f}  -log P_hat={-np.log(c/shots):.3f}")


# ----------------------------------------------------------------------
def build_circuit_reuse(mps: MPS) -> tuple[QuantumCircuit, int, int]:
    """Qubit-reuse variant: only (b_max + 1) qubits via mid-circuit measurement.

    The true Schoen et al. sequential preparation on hardware. The bond register
    stays coherent throughout; a SINGLE physical qubit is reused: at each site it
    is taken from |0>, gets the emitted feature written onto it by A_k, is measured
    (recording s_k), and reset to |0> for the next site.

    This is exact, not approximate: no gate touches the physical qubit after its
    own A_k, so measuring mid-circuit gives the same distribution as measuring at
    the end, and the reset only frees the qubit. It samples v ~ P(v) using
    (b_max + 1) qubits instead of (N + b_max) -- e.g. 5 instead of 18 for k14.

    Note this circuit is inherently a *sampler* (the physical register is measured
    away), so it has no full statevector to read; validate it by comparing sampled
    frequencies to P(v) -- see sample_reuse.
    """
    N = mps.num_sites
    b_max = max(bond_qubit_counts(mps))
    bond = QuantumRegister(b_max, "bond")
    phys = QuantumRegister(1, "p")           # single physical qubit, reused
    cv = ClassicalRegister(N, "v")
    qc = QuantumCircuit(bond, phys, cv, name="mps_born_machine_reuse")
    for k in range(N):
        U = site_unitary(mps.site_tensors[k].data, b_max)
        qc.unitary(Operator(U), [phys[0]] + list(bond), label=f"A{k}")
        qc.measure(phys[0], cv[k])           # record s_k
        if k < N - 1:
            qc.reset(phys[0])                # back to |0> for the next site
    return qc, N, b_max


def sample_reuse(qc: QuantumCircuit, N: int, shots: int, mps: MPS = None) -> None:
    """Sample the qubit-reuse circuit and (if feasible) check it against P(v)."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator()
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
    print(f"\n  reuse-sampled {shots} shots; {len(counts)} distinct configurations")

    if mps is not None and N <= 20:
        all_v = torch.tensor([[(i >> (N - 1 - s)) & 1 for s in range(N)]
                              for i in range(2 ** N)], dtype=torch.long)
        with torch.no_grad():
            p_mps = torch.exp(mps.log_prob(all_v)).numpy()
        p_circ = np.zeros(2 ** N)
        for bitstr, c in counts.items():
            p_circ[int(bitstr.replace(" ", "")[::-1], 2)] = c / shots
        tvd = 0.5 * np.abs(p_mps - p_circ).sum()
        noise = np.sqrt(2 ** N / shots) / 2
        print(f"  total-variation vs P(v)  : {tvd:.4f}   (MC noise scale ~ {noise:.4f})")

    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    for bitstr, c in top:
        v = bitstr.replace(" ", "")[::-1]
        print(f"    v={v}  freq={c/shots:.4f}  -log P_hat={-np.log(c/shots):.3f}")


# ----------------------------------------------------------------------
def _site_isometry(A: torch.Tensor, n_act: int):
    """Build the local isometry matrix for one site, in the qubit ordering
    [out-bond (LSB..), phys (MSB)] acting on n_act qubits. Returns a
    2^n_act x 2^Dl array whose columns (incoming bond alpha) are orthonormal.
    """
    A = A.detach().cpu().numpy().astype(np.complex128)
    Dl, d, Dr = A.shape
    M = np.zeros((2 ** n_act, Dl), dtype=np.complex128)
    phys_bit = n_act - 1
    for alpha in range(Dl):
        for s in range(d):
            for beta in range(Dr):
                M[(s << phys_bit) | beta, alpha] = A[alpha, s, beta]
    return M


def build_circuit_isometry(mps: MPS, reuse: bool = False):
    """Isometry-synthesised circuit (Iten et al.): each A_k is decomposed as an
    isometry on the *minimal* number of qubits rather than padded to a generic
    (b_max+1)-qubit unitary. Exact, and markedly cheaper at the narrow bonds
    (~50% fewer CX on a tapered bond profile).

    Bond qubits are carried on a small pool that grows/shrinks with the bond
    dimension; the physical qubit is fresh per site (no-reuse) or a single
    measured-and-reset qubit (reuse). Returns (circuit, N, b_max).
    """
    from qiskit.circuit.library import Isometry
    N = mps.num_sites
    b = bond_qubit_counts(mps)
    b_max = max(b)
    bond = QuantumRegister(b_max, "bond")
    if reuse:
        phys = QuantumRegister(1, "p")
        cv = ClassicalRegister(N, "v")
        qc = QuantumCircuit(bond, phys, cv, name="mps_iso_reuse")
    else:
        phys = QuantumRegister(N, "phys")
        qc = QuantumCircuit(bond, phys, name="mps_iso")

    active: list = []                    # bond wires holding the incoming bond
    free = list(bond)
    for k in range(N):
        bl, br = b[k], b[k + 1]
        if br >= bl:
            grown = [free.pop(0) for _ in range(br - bl)]
            bond_part = active + grown   # out-bond wires (len br)
        else:
            bond_part = active           # low br are out-bond, the rest freed -> |0>
        pq = phys[0] if reuse else phys[k]
        acted = bond_part + [pq]
        n_act = len(bond_part) + 1
        qc.append(Isometry(_site_isometry(mps.site_tensors[k].data, n_act), 0, 0), acted)
        if reuse:
            qc.measure(phys[0], cv[k])
            if k < N - 1:
                qc.reset(phys[0])
        if br >= bl:
            active = bond_part
        else:
            active, freed = active[:br], active[br:]
            free = freed + free
    return qc, N, b_max


# ----------------------------------------------------------------------
def draw_circuit_png(qc: QuantumCircuit, path: Path, dpi: int = 300) -> None:
    """Render the high-level sequential-preparation circuit as a PNG.

    A staircase of isometry gates A_k, each acting on the shared bond register
    and one physical qubit -- the Schoen et al. construction drawn directly
    (the transpiled gate-level circuit would be hundreds of CX and unreadable).
    Needs matplotlib (and pylatexenc for the drawer's labels).

    The gate names A_k are moved ABOVE each box; Qiskit draws them inside by
    default, where they collide with the per-wire port indices.
    """
    import matplotlib.pyplot as plt
    fig = qc.draw(output="mpl", fold=-1, style={"name": "bw"})
    ax = fig.axes[0]

    # Every box's top edge sits on the top wire (y = 0). Lift the gate-name text
    # artists to a single row just above it, keeping each one's column (x).
    import re
    is_gate_name = re.compile(r"^A\d+$").match
    label_row_y = 0.9
    for txt in list(ax.texts):
        if is_gate_name(txt.get_text()):
            x, _ = txt.get_position()
            txt.set_position((x, label_row_y))
            txt.set_va("bottom")
            txt.set_ha("center")
            txt.set_fontsize(11)
            txt.set_fontweight("bold")
    ax.set_ylim(top=max(ax.get_ylim()[1], label_row_y + 0.8))

    fig.suptitle("MPS Born machine as a sequential-preparation circuit", y=1.04)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------
_TWO_QUBIT_GATES = {"cx", "cz", "ecr", "rzz", "rxx", "ryy", "cy", "cp", "swap", "iswap"}


def report_gate_counts(qc: QuantumCircuit, basis_gates=("cx", "rz", "sx", "x"),
                       opt_level: int = 3):
    """Transpile to a native gate set and print how many gates the circuit needs.

    The A_k boxes are multi-qubit unitaries; real hardware only runs 1- and
    2-qubit gates, so they must be decomposed. The two-qubit (entangling) count
    and the depth are what actually gate the fidelity on a device.

    Note: Qiskit's default synthesis treats each A_k as a *generic* unitary, so
    this count is an upper bound -- isometry-aware synthesis (each A_k is an
    isometry, not a full unitary) would lower it, especially at the wide bonds.
    """
    from qiskit import transpile
    basis = list(basis_gates)
    t = transpile(qc, basis_gates=basis, optimization_level=opt_level)
    ops = dict(t.count_ops())
    n_2q = sum(v for k, v in ops.items() if k in _TWO_QUBIT_GATES)

    print(f"  transpiled to {basis} (optimization_level={opt_level}):")
    print(f"    two-qubit (entangling) gates : {n_2q}")
    print(f"    circuit depth                : {t.depth()}")
    print(f"    qubits                       : {t.num_qubits}")
    print("    full gate breakdown          :")
    for g in sorted(ops):
        print(f"      {g:<10}: {ops[g]}")
    return t


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--model", default="mps_trained.pt")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--circuit-png", type=Path, default=None,
                    help="if set, render the circuit diagram to this PNG path")
    ap.add_argument("--reuse", action="store_true",
                    help="use the qubit-reuse circuit (b_max+1 qubits, mid-circuit "
                         "measurement) instead of the N+b_max no-reuse circuit")
    ap.add_argument("--transpile", action="store_true",
                    help="transpile to a native gate set and report the gate counts")
    ap.add_argument("--basis", default="cx,rz,sx,x",
                    help="comma-separated native basis for --transpile "
                         "(e.g. 'cx,rz,sx,x' or 'cz,rz,sx,x')")
    ap.add_argument("--synthesis", choices=("unitary", "isometry"), default="unitary",
                    help="how to build each A_k: 'unitary' (generic, simple) or "
                         "'isometry' (Iten et al., minimal qubits, fewer CX on hardware)")
    ap.add_argument("--opt-level", type=int, default=3,
                    help="transpiler optimization level (0-3) for --transpile")
    args = ap.parse_args()

    mps = MPS.load(str(args.data_dir / args.model))
    print(f"loaded MPS: {mps.num_sites} sites, bond dims {mps.full_bond_dims}")
    prepare_right_canonical(mps)

    basis = tuple(args.basis.split(","))
    iso = args.synthesis == "isometry"

    if args.reuse:
        qc, N, b_max = (build_circuit_isometry(mps, reuse=True) if iso
                        else build_circuit_reuse(mps))
        print(f"reuse circuit [{args.synthesis}]: {b_max} bond + 1 physical = "
              f"{qc.num_qubits} qubits (vs {N + b_max} no-reuse), {N} sites via "
              f"mid-circuit measurement")
        if args.circuit_png is not None:
            print("\ndrawing circuit:")
            draw_circuit_png(qc, args.circuit_png)
        if args.transpile:
            print("\ntranspiling:")
            report_gate_counts(qc, basis_gates=basis, opt_level=args.opt_level)
        shots = args.shots if args.shots > 0 else 100_000
        sample_reuse(qc, N, shots, mps=mps)
        return

    qc, N, b_max = (build_circuit_isometry(mps, reuse=False) if iso
                    else build_circuit(mps))
    print(f"circuit [{args.synthesis}]: {N} physical qubits + {b_max} bond qubits "
          f"= {qc.num_qubits} total, {N} site-isometries")

    print("\nverifying against the MPS Born distribution (statevector):")
    verify_statevector(mps, qc, N, b_max)

    if args.circuit_png is not None:
        print("\ndrawing circuit:")
        draw_circuit_png(qc, args.circuit_png)

    if args.transpile:
        print("\ntranspiling:")
        report_gate_counts(qc, basis_gates=basis, opt_level=args.opt_level)

    if args.shots > 0:
        sample(qc, N, b_max, args.shots)


if __name__ == "__main__":
    main()