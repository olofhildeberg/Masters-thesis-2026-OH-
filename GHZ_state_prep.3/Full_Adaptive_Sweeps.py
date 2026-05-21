from __future__ import annotations
from typing import Dict, Tuple
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import pauli_error


# -----------------------------
# Timing helpers
# -----------------------------
def ns_to_ticks(duration_ns: float, tick_ns: float) -> int:
    if duration_ns < 0:
        raise ValueError("duration_ns must be >= 0")
    if tick_ns <= 0:
        raise ValueError("tick_ns must be > 0")
    return max(0, math.ceil(duration_ns / tick_ns))


# -----------------------------
# Noise model
# -----------------------------
def t1_tphi_to_pauli_probs(dt: float, T1: float, Tphi: float) -> Tuple[float, float, float]:
    """Calculate twirled idle Pauli probabilities for one delay tick of duration dt."""
    if math.isinf(T1) and math.isinf(Tphi):
        T2 = float("inf")
    elif math.isinf(T1):
        T2 = Tphi
    elif math.isinf(Tphi):
        T2 = 2.0 * T1
    else:
        T2 = 1.0 / (1.0 / (2.0 * T1) + 1.0 / Tphi)

    e1 = 1.0 if math.isinf(T1) else math.exp(-dt / T1)
    e2 = 1.0 if math.isinf(T2) else math.exp(-dt / T2)

    pX = (1.0 - e1) / 4.0
    pY = (1.0 - e1) / 4.0
    pZ = (1.0 - e2) / 2.0 - (1.0 - e1) / 4.0

    pX = max(0.0, pX)
    pY = max(0.0, pY)
    pZ = max(0.0, pZ)

    if pX + pY + pZ > 1.0 + 1e-12:
        raise ValueError("Computed idle Pauli probabilities are invalid")

    return pX, pY, pZ



def make_noise_model(
    N: int,
    n_anc: int,
    dt: float = 300.0,
    T1: float = float("inf"),
    Tphi: float = float("inf"),
    p_cx_local: float = 0.0,
) -> NoiseModel:
    """
    Combined noise model:

    1) On each delay tick:
       - apply a single idle Pauli channel derived from T1 and Tphi
         through a Pauli-twirled approximation
       - applied to ALL qubits: data + ancillas

    2) After each CX:
       - each of the two participating qubits independently gets a 1-qubit
         depolarizing Pauli error with probability p_cx_local
    """
    nm = NoiseModel()

    pX, pY, pZ = t1_tphi_to_pauli_probs(dt=dt, T1=T1, Tphi=Tphi)
    pI = 1.0 - pX - pY - pZ

    if pX > 0 or pY > 0 or pZ > 0:
        delay_err = pauli_error([
            ("I", pI),
            ("X", pX),
            ("Y", pY),
            ("Z", pZ),
        ])

        for q in range(N + n_anc):
            nm.add_quantum_error(delay_err, "delay", [q])

    if p_cx_local > 0:
        if p_cx_local > 1.0:
            raise ValueError("p_cx_local must be <= 1")

        single_q_pauli = pauli_error([
            ("I", 1 - p_cx_local),
            ("X", p_cx_local / 3),
            ("Y", p_cx_local / 3),
            ("Z", p_cx_local / 3),
        ])

        cx_err = single_q_pauli.tensor(single_q_pauli)
        nm.add_all_qubit_quantum_error(cx_err, "cx")

    return nm


# -----------------------------
# Metrics
# -----------------------------
def parity_expectation_from_counts(counts: Dict[str, int]) -> float:
    shots = sum(counts.values())
    acc = 0.0
    for bitstring, c in counts.items():
        acc += c * (1.0 if (bitstring.count("1") % 2 == 0) else -1.0)
    return acc / shots if shots else 0.0



def p0N_p1N(counts: Dict[str, int], N: int) -> Tuple[float, float]:
    shots = sum(counts.values())
    if shots == 0:
        return 0.0, 0.0
    return counts.get("0" * N, 0) / shots, counts.get("1" * N, 0) / shots


# -----------------------------
# Full adaptive circuit
# -----------------------------
def build_full_adaptive_no_corrections_time_steps(
    N: int,
    basis: str = "Z",
    tick_ns: float = 300.0,
    entangling_layer_ns: float = 300.0,
    measurement_ns: float = 2100.0,
    classical_feedforward_ns: float = 900.0,
) -> QuantumCircuit:
    """
    Full adaptive GHZ fusion circuit with a coarse-grained hardware timing model.

    Timing model:
    - one delay tick = tick_ns
    - each entangling layer contributes entangling_layer_ns of idle time
    - physical ancilla measurement time is modeled as idle time BEFORE the
      idealized Qiskit measure instruction
    - classical feed-forward / post-processing time is modeled as idle time
      AFTER the idealized Qiskit measure instruction
    - the initial 1-qubit Hadamard layer is neglected in the idle-time model

    Default values:
    - tick_ns = 300 ns
    - entangling_layer_ns = 300 ns          -> 1 tick after each CX layer
    - measurement_ns = 2100 ns             -> 7 ticks before ancilla measurement
    - classical_feedforward_ns = 900 ns    -> 3 ticks after ancilla measurement
    """
    if N < 2:
        raise ValueError("N must be >= 2.")

    n_anc = N - 1

    d = QuantumRegister(N, "d")
    a = QuantumRegister(n_anc, "a")
    m = ClassicalRegister(n_anc, "m")
    out = ClassicalRegister(N, "out")
    qc = QuantumCircuit(d, a, m, out)

    entangling_ticks = ns_to_ticks(entangling_layer_ns, tick_ns)
    measurement_ticks = ns_to_ticks(measurement_ns, tick_ns)
    classical_feedforward_ticks = ns_to_ticks(classical_feedforward_ns, tick_ns)

    def tick_all(num_ticks: int) -> None:
        """Delay all data and ancilla qubits for num_ticks delay ticks."""
        for _ in range(num_ticks):
            for q in range(N):
                qc.delay(1, d[q])
            for k in range(n_anc):
                qc.delay(1, a[k])

    def tick_data_only(num_ticks: int) -> None:
        """Delay only data qubits for num_ticks delay ticks."""
        for _ in range(num_ticks):
            for q in range(N):
                qc.delay(1, d[q])

    # Initial parallel H layer
    for q in range(N):
        qc.h(d[q])

    # First entangling layer
    for k in range(n_anc):
        qc.cx(d[k], a[k])
    tick_all(entangling_ticks)

    # Second entangling layer
    for k in range(n_anc):
        qc.cx(d[k + 1], a[k])
    tick_all(entangling_ticks)

    # Physical ancilla measurement duration.
    # This comes before the idealized Qiskit measure instruction, because the
    # ancillas are still physically involved during the measurement pulse.
    tick_all(measurement_ticks)

    # Idealized mid-circuit ancilla readout instruction.
    for k in range(n_anc):
        qc.measure(a[k], m[k])

    # Classical feed-forward / post-processing latency.
    # After the ancillas are measured, only data qubits are still part of the
    # final GHZ state, so this delay is applied only to the data qubits.
    tick_data_only(classical_feedforward_ticks)

    b = basis.upper()
    if b == "X":
        qc.h(d)
    elif b != "Z":
        raise ValueError("basis must be 'Z' or 'X'")

    qc.measure(d, out)
    return qc


# -----------------------------
# Post-processing
# -----------------------------
def apply_pauli_frame_from_m_full_adaptive(out_bits: str, m_bits: str) -> str:
    """Z-basis correction rule for the full adaptive protocol."""
    bits = list(out_bits)
    prefix = 0

    for j in range(1, len(out_bits)):
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
# Simulation helpers
# -----------------------------
def run_counts(
    tqc: QuantumCircuit,
    shots: int,
    noise_model: NoiseModel | None,
    seed: int | None = None,
) -> Dict[str, int]:
    sim = AerSimulator(method="stabilizer", noise_model=noise_model, seed_simulator=seed)
    return sim.run(tqc, shots=shots).result().get_counts()



def fidelity_from_params(
    tqcZ: QuantumCircuit,
    tqcX: QuantumCircuit,
    N: int,
    n_anc: int,
    shots: int,
    dt: float,
    T1: float,
    Tphi: float,
    p_meas_err: float,
    p_cx: float,
    seed: int,
) -> float:
    nm = make_noise_model(
        N=N,
        n_anc=n_anc,
        dt=dt,
        T1=T1,
        Tphi=Tphi,
        p_cx_local=p_cx,
    )

    rawZ = run_counts(tqcZ, shots, nm, seed=seed)
    rawX = run_counts(tqcX, shots, nm, seed=seed + 1)

    rng = random.Random(seed + 999)
    Zc = marginalize_and_correct(rawZ, N, basis="Z", p_meas_anc=p_meas_err, rng=rng)
    Xc = marginalize_and_correct(rawX, N, basis="X", p_meas_anc=0.0, rng=rng)

    Xexp = parity_expectation_from_counts(Xc)
    p0, p1 = p0N_p1N(Zc, N)
    return 0.5 * (p0 + p1 + Xexp)


# -----------------------------
# Sweep utilities
# -----------------------------
def linspace(start: float, stop: float, n: int) -> list[float]:
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]



def geomspace(start: float, stop: float, n: int) -> list[float]:
    if n == 1:
        return [start]
    ratio = (stop / start) ** (1 / (n - 1))
    vals = [start]
    for _ in range(1, n):
        vals.append(vals[-1] * ratio)
    return vals


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # ===== User settings =====
    N = 32
    shots = 1000
    seed = 1234

    # Coarse-grained timing model in ns
    tick_ns = 300.0
    entangling_layer_ns = 300.0

    # Split the old 3000 ns measurement+feedback block into:
    # - physical measurement duration before the idealized measure instruction
    # - classical feed-forward latency after the idealized measure instruction
    measurement_ns = 2100.0
    classical_feedforward_ns = 900.0

    # Each delay tick in the noise model corresponds to 300 ns
    dt = tick_ns

    p_vals = linspace(0.001, 0.03, 18)

    # Sweep T1 and Tphi over a realistic superconducting-qubit range:
    # 5 microseconds to 300 microseconds, expressed in ns.
    T_vals = geomspace(20_000.0, 500_000.0, 18)
    # =========================

    BASE_DIR = Path(__file__).resolve().parent
    outdir = BASE_DIR / "Sweep_results"
    outdir.mkdir(parents=True, exist_ok=True)

    n_anc = N - 1

    qcZ = build_full_adaptive_no_corrections_time_steps(
        N,
        basis="Z",
        tick_ns=tick_ns,
        entangling_layer_ns=entangling_layer_ns,
        measurement_ns=measurement_ns,
        classical_feedforward_ns=classical_feedforward_ns,
    )
    qcX = build_full_adaptive_no_corrections_time_steps(
        N,
        basis="X",
        tick_ns=tick_ns,
        entangling_layer_ns=entangling_layer_ns,
        measurement_ns=measurement_ns,
        classical_feedforward_ns=classical_feedforward_ns,
    )

    base_sim = AerSimulator(method="stabilizer")
    tqcZ = transpile(qcZ, base_sim, optimization_level=0)
    tqcX = transpile(qcX, base_sim, optimization_level=0)

    print(f"Running full-adaptive sweeps with N={N}, shots={shots}")
    print(f"Saving results in: {outdir}")
    print(
        f"Timing model: tick={tick_ns} ns, entangling layer={entangling_layer_ns} ns, "
        f"measurement={measurement_ns} ns before measure, "
        f"classical feed-forward={classical_feedforward_ns} ns after measure"
    )
    print("Initial 1-qubit Hadamard layer neglected in idle-time model.")

    # -------------------------
    # Sweep 1: measurement and CX errors
    # -------------------------
    meas_fids = []
    cx_fids = []

    for i, p in enumerate(p_vals):
        F_meas = fidelity_from_params(
            tqcZ=tqcZ,
            tqcX=tqcX,
            N=N,
            n_anc=n_anc,
            shots=shots,
            dt=dt,
            T1=float("inf"),
            Tphi=float("inf"),
            p_meas_err=p,
            p_cx=0.0,
            seed=seed + 10 * i,
        )
        meas_fids.append(F_meas)
        print(f"p_meas={p:.4f} -> F={F_meas:.6f}")

        F_cx = fidelity_from_params(
            tqcZ=tqcZ,
            tqcX=tqcX,
            N=N,
            n_anc=n_anc,
            shots=shots,
            dt=dt,
            T1=float("inf"),
            Tphi=float("inf"),
            p_meas_err=0.0,
            p_cx=p,
            seed=seed + 1000 + 10 * i,
        )
        cx_fids.append(F_cx)
        print(f"p_cx  ={p:.4f} -> F={F_cx:.6f}")

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(p_vals, meas_fids, marker="o", label="Measurement error only")
    plt.plot(p_vals, cx_fids, marker="s", label="CNOT/CZ error only")
    plt.xlabel("Error probability")
    plt.ylabel("GHZ fidelity")
    plt.title(f"Full adaptive: fidelity vs measurement and entangling-gate error (shots={shots}, N={N})")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig1 = outdir / "full_adaptive_fidelity_vs_meas_and_cx.png"
    plt.savefig(fig1, dpi=180)
    plt.close()

    # -------------------------
    # Sweep 2: Tphi and T1
    # -------------------------
    tphi_fids = []
    t1_fids = []

    for i, T in enumerate(T_vals):
        F_tphi = fidelity_from_params(
            tqcZ=tqcZ,
            tqcX=tqcX,
            N=N,
            n_anc=n_anc,
            shots=shots,
            dt=dt,
            T1=float("inf"),
            Tphi=T,
            p_meas_err=0.0,
            p_cx=0.0,
            seed=seed + 2000 + 10 * i,
        )
        tphi_fids.append(F_tphi)
        print(f"Tphi={T:.3f} ns -> F={F_tphi:.6f}")

        F_t1 = fidelity_from_params(
            tqcZ=tqcZ,
            tqcX=tqcX,
            N=N,
            n_anc=n_anc,
            shots=shots,
            dt=dt,
            T1=T,
            Tphi=float("inf"),
            p_meas_err=0.0,
            p_cx=0.0,
            seed=seed + 3000 + 10 * i,
        )
        t1_fids.append(F_t1)
        print(f"T1  ={T:.3f} ns -> F={F_t1:.6f}")

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(T_vals, tphi_fids, marker="o", label=r"Pure dephasing only ($T_\phi$)")
    plt.plot(T_vals, t1_fids, marker="s", label=r"Relaxation only ($T_1$)")
    plt.xscale("log")
    plt.xlabel("Time constant (ns)")
    plt.ylabel("GHZ fidelity")
    plt.title(fr"Full adaptive: fidelity vs $T_\phi$ and $T_1$ (shots={shots}, N={N})")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig2 = outdir / "full_adaptive_fidelity_vs_tphi_and_t1.png"
    plt.savefig(fig2, dpi=180)
    plt.close()

    print("\nSaved:")
    print(fig1)
    print(fig2)


if __name__ == "__main__":
    main()
