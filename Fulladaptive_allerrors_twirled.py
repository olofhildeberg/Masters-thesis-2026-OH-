from __future__ import annotations
from typing import Dict, Tuple, Literal
import math
import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import pauli_error


# -----------------------------
# Noise models
# -----------------------------
Target = Literal["data", "anc", "all"]


def t1_tphi_to_pauli_probs(dt: float, T1: float, Tphi: float) -> Tuple[float, float, float]:
    """
    Pauli-twirled idle decoherence channel for one tick of duration dt.

    Inputs:
      T1   : relaxation time (in the same units as dt)
      Tphi : pure dephasing time (in the same units as dt)

    Use float("inf") for limiting cases:
      - pure dephasing only: T1 = inf
      - relaxation only    : Tphi = inf

    Returns:
      (pX, pY, pZ)

    Channel applied on each delay tick:
      I with probability 1 - pX - pY - pZ
      X with probability pX
      Y with probability pY
      Z with probability pZ

    This is a stabilizer-compatible Pauli approximation to the combined
    amplitude-damping + pure-dephasing idle channel.
    """
    exp_t1 = 1.0 if math.isinf(T1) else math.exp(-dt / T1)
    exp_phi = 1.0 if math.isinf(Tphi) else math.exp(-2.0 * dt / Tphi)

    gamma = 1.0 - exp_t1
    lam = exp_t1 * (1.0 - exp_phi)

    pX = gamma / 4.0
    pY = gamma / 4.0
    pZ = 0.5 - gamma / 4.0 - 0.5 * math.sqrt(max(0.0, 1.0 - gamma - lam))

    # Numerical safety
    pX = max(0.0, pX)
    pY = max(0.0, pY)
    pZ = max(0.0, pZ)

    if pX + pY + pZ > 1.0 + 1e-12:
        raise ValueError("Computed idle Pauli probabilities are invalid")

    return pX, pY, pZ


def make_noise_model(
    N: int,
    n_anc: int,
    dt: float = 1.0,
    T1: float = float("inf"),
    Tphi: float = float("inf"),
    p_cx_local: float = 0.0,
    target: Target = "all",
) -> NoiseModel:
    """
    Combined noise model:

    1) On each delay tick:
       - apply a single idle Pauli channel derived from T1 and Tphi
         through a Pauli-twirled approximation

    2) After each CX:
       - each of the two participating qubits independently gets a 1-qubit
         depolarizing Pauli error with probability p_cx_local
       - this means:
            single-qubit CX faults are O(p)
            two-qubit correlated faults are O(p^2)

    target controls which qubits receive delay noise:
      - "data": data qubits only
      - "anc":  ancillas only
      - "all":  all qubits
    """
    nm = NoiseModel()

    # -------------------------
    # Delay noise: single twirled idle decoherence channel
    # -------------------------
    pX, pY, pZ = t1_tphi_to_pauli_probs(dt=dt, T1=T1, Tphi=Tphi)
    pI = 1.0 - pX - pY - pZ

    if pX > 0 or pY > 0 or pZ > 0:
        delay_err = pauli_error([
            ("I", pI),
            ("X", pX),
            ("Y", pY),
            ("Z", pZ),
        ])

        if target == "data":
            qubits = range(0, N)
        elif target == "anc":
            qubits = range(N, N + n_anc)
        elif target == "all":
            qubits = range(0, N + n_anc)
        else:
            raise ValueError("target must be one of: 'data', 'anc', 'all'")

        for q in qubits:
            nm.add_quantum_error(delay_err, "delay", [q])

    # -------------------------
    # CX noise: independent 1-qubit Pauli channel on each CX participant
    # -------------------------
    if p_cx_local > 0:
        if p_cx_local > 1.0:
            raise ValueError("p_cx_local must be <= 1")

        single_q_pauli = pauli_error([
            ("I", 1 - p_cx_local),
            ("X", p_cx_local / 3),
            ("Y", p_cx_local / 3),
            ("Z", p_cx_local / 3),
        ])

        # independent error on control and target
        cx_err = single_q_pauli.tensor(single_q_pauli)
        nm.add_all_qubit_quantum_error(cx_err, "cx")

    return nm


# -----------------------------
# Metrics
# -----------------------------
def parity_expectation_from_counts(counts: Dict[str, int]) -> float:
    """E[(-1)^{wt(bitstring)}] from counts over N-bit outputs."""
    shots = sum(counts.values())
    acc = 0
    for bitstring, c in counts.items():
        acc += c * (1 if (bitstring.count("1") % 2 == 0) else -1)
    return acc / shots if shots else 0.0


def p0N_p1N(counts: Dict[str, int], N: int) -> Tuple[float, float]:
    shots = sum(counts.values())
    return counts.get("0" * N, 0) / shots, counts.get("1" * N, 0) / shots


# -----------------------------
# Full adaptive circuit (no in-circuit corrections)
# -----------------------------
def build_full_adaptive_no_corrections_time_steps(N: int, basis: str = "Z") -> QuantumCircuit:
    """
    Full adaptive GHZ fusion circuit WITHOUT in-circuit corrections.

    Idea:
    - Start all data qubits in |0>
    - Apply H to every data qubit, giving |+>^N
    - Use ancillas to measure Z_i Z_{i+1} parity between neighboring data qubits
    - Measure ancillas into classical register m
    - Measure data in basis 'Z' or 'X' into classical register out
    - Insert explicit delay(1) ticks after each layer

    For this version:
    - n_anc = N - 1
    - ancilla a[k] probes the bond between d[k] and d[k+1]
    """
    if N < 2:
        raise ValueError("N must be >= 2.")

    n_anc = N - 1

    d = QuantumRegister(N, "d")
    a = QuantumRegister(n_anc, "a")
    m = ClassicalRegister(n_anc, "m")
    out = ClassicalRegister(N, "out")
    qc = QuantumCircuit(d, a, m, out)

    def tick_all() -> None:
        for q in range(N):
            qc.delay(1, d[q])
        for k in range(n_anc):
            qc.delay(1, a[k])

    # L1: prepare |+> on all data qubits
    for q in range(N):
        qc.h(d[q])
    tick_all()

    # L2: first half of ZZ parity extraction (left data qubit -> ancilla)
    for k in range(n_anc):
        qc.cx(d[k], a[k])
    tick_all()

    # L3: second half of ZZ parity extraction (right data qubit -> ancilla)
    for k in range(n_anc):
        qc.cx(d[k + 1], a[k])
    tick_all()

    # L4: measure ancillas
    for k in range(n_anc):
        qc.measure(a[k], m[k])
    tick_all()

    # Final basis choice for data qubits
    b = basis.upper()
    if b == "X":
        qc.h(d)
    elif b != "Z":
        raise ValueError("basis must be 'Z' or 'X'")

    qc.measure(d, out)
    return qc


# -----------------------------
# Post-processing (Z-basis only)
# -----------------------------
def apply_pauli_frame_from_m_full_adaptive(out_bits: str, m_bits: str) -> str:
    """
    Z-basis correction rule for the full adaptive protocol.

    Ancilla m[k] stores the parity between qubits k and k+1.
    If the cumulative XOR of m[0..j-1] is 1, then qubit j must be X-flipped
    in the Z-basis readout to align it with qubit 0.

    So:
      - qubit 0 is the reference
      - qubit j (j >= 1) is flipped iff XOR(m[0], ..., m[j-1]) == 1
    """
    N = len(out_bits)
    bits = list(out_bits)

    prefix = 0
    for j in range(1, N):
        prefix ^= int(m_bits[j - 1])
        if prefix == 1:
            bits[j] = "1" if bits[j] == "0" else "0"

    return "".join(bits)


def flip_measurement_bits(bitstring: str, p_meas: float, rng: random.Random) -> str:
    bits = list(bitstring)
    for i in range(len(bits)):
        if rng.random() < p_meas:
            bits[i] = "1" if bits[i] == "0" else "0"
    return "".join(bits)


def marginalize_and_correct(
    counts: Dict[str, int],
    N: int,
    basis: str,
    p_meas_anc: float = 0.0,
    rng: random.Random | None = None,
) -> Dict[str, int]:
    """
    Input keys from Aer look like: "<out_bits> <m_bits>".
    - basis='Z': apply X-frame bit-flip corrections based on m_bits
    - basis='X': do not apply those bit flips
    - p_meas_anc: ancilla measurement bit-flip probability
    Returns counts over N-bit output strings only.
    """
    corrected: Dict[str, int] = {}
    b = basis.upper()

    if rng is None:
        rng = random.Random()

    for key, c in counts.items():
        parts = key.split(" ")
        out_bits = parts[0]
        m_bits = parts[1] if len(parts) > 1 else ""

        # corrupt the measured ancilla syndrome
        m_bits_noisy = flip_measurement_bits(m_bits, p_meas_anc, rng)

        if b == "Z":
            out_corr = apply_pauli_frame_from_m_full_adaptive(out_bits, m_bits_noisy)
        elif b == "X":
            out_corr = out_bits
        else:
            raise ValueError("basis must be 'Z' or 'X'")

        corrected[out_corr] = corrected.get(out_corr, 0) + c

    return corrected


# -----------------------------
# Simulation helper
# -----------------------------
def run_counts(
    qc: QuantumCircuit,
    shots: int,
    noise_model: NoiseModel | None,
    seed: int | None = None,
    debug_ticks: bool = False,
    label: str = "",
) -> Dict[str, int]:
    sim = AerSimulator(method="stabilizer", noise_model=noise_model, seed_simulator=seed)
    tqc = transpile(qc, sim, optimization_level=0)

    if debug_ticks:
        ops = tqc.count_ops()
        nq = tqc.num_qubits
        dly = int(ops.get("delay", 0))
        print(f"\n--- {label} ---")
        print("num_qubits:", nq)
        print("delay_count:", dly)
        print("delay_per_qubit (ticks):", dly / nq if nq else None)
        print("depth:", tqc.depth())

    return sim.run(tqc, shots=shots).result().get_counts()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    N = 16
    shots = 1100
    dt = 1.0                 # one tick = one layer
    #T1 = 500.0               # relaxation time in ticks; use float('inf') for no relaxation
    Tphi = 250.0             # pure dephasing time in ticks; use float('inf') for no pure dephasing
    #p_meas_err = 0.001       # ancilla readout bit-flip in postprocessing
    #p_cx = 0.01              # total probability of a nontrivial 2-qubit Pauli after each CX
    T1 = float("inf") 
    # Tphi = float("inf")
    p_meas_err = 0.000
    p_cx = 0.00   
    seed = 1234

    n_anc = N - 1

    qcZ = build_full_adaptive_no_corrections_time_steps(N, basis="Z")
    qcX = build_full_adaptive_no_corrections_time_steps(N, basis="X")

    # Print the effective per-tick idle probabilities once
    pX_idle, pY_idle, pZ_idle = t1_tphi_to_pauli_probs(dt=dt, T1=T1, Tphi=Tphi)
    print("Idle Pauli-twirled probabilities per tick:")
    print(f"  pX = {pX_idle:.8f}")
    print(f"  pY = {pY_idle:.8f}")
    print(f"  pZ = {pZ_idle:.8f}")
    print(f"  pI = {1.0 - pX_idle - pY_idle - pZ_idle:.8f}")

    def eval_case(tag: str, nm: NoiseModel) -> None:
        rawZ = run_counts(qcZ, shots, nm, seed=seed, debug_ticks=True, label=f"{tag} / Z")
        rawX = run_counts(qcX, shots, nm, seed=seed, debug_ticks=True, label=f"{tag} / X")

        rng = random.Random(999)

        Zc = marginalize_and_correct(rawZ, N, basis="Z", p_meas_anc=p_meas_err, rng=rng)
        Xc = marginalize_and_correct(rawX, N, basis="X", p_meas_anc=0, rng=rng)

        Xexp = parity_expectation_from_counts(Xc)
        p0, p1 = p0N_p1N(Zc, N)
        F = 0.5 * (p0 + p1 + Xexp)

        print(f"{tag}:  P(0^N)={p0:.6f}, P(1^N)={p1:.6f}, <X^N>={Xexp:.6f}, F~={F:.6f}")

    nm_data = make_noise_model(
        N, n_anc,
        dt=dt,
        T1=T1,
        Tphi=Tphi,
        p_cx_local=p_cx,
        target="data",
    )
    nm_anc = make_noise_model(
        N, n_anc,
        dt=dt,
        T1=T1,
        Tphi=Tphi,
        p_cx_local=p_cx,
        target="anc",
    )
    nm_all = make_noise_model(
        N, n_anc,
        dt=dt,
        T1=T1,
        Tphi=Tphi,
        p_cx_local=p_cx,
        target="all",
    )

    eval_case("DATA only", nm_data)
    eval_case("ANC only ", nm_anc)
    eval_case("ALL qubits", nm_all)
