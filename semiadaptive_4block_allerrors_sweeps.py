from __future__ import annotations
from typing import Dict, Tuple, Literal
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


Target = Literal["data", "anc", "all"]


# -----------------------------
# Noise models
# -----------------------------
def t1_tphi_to_pauli_probs(dt: float, T1: float, Tphi: float) -> Tuple[float, float, float]:
    if dt < 0:
        raise ValueError("dt must be >= 0")
    if T1 <= 0 or Tphi <= 0:
        if not math.isinf(T1) and T1 <= 0:
            raise ValueError("T1 must be > 0 or float('inf')")
        if not math.isinf(Tphi) and Tphi <= 0:
            raise ValueError("Tphi must be > 0 or float('inf')")

    exp_t1 = 1.0 if math.isinf(T1) else math.exp(-dt / T1)
    exp_phi = 1.0 if math.isinf(Tphi) else math.exp(-2.0 * dt / Tphi)

    gamma = 1.0 - exp_t1
    lam = exp_t1 * (1.0 - exp_phi)

    pX = gamma / 4.0
    pY = gamma / 4.0
    pZ = 0.5 - gamma / 4.0 - 0.5 * math.sqrt(max(0.0, 1.0 - gamma - lam))

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
    nm = NoiseModel()

    # Idle decoherence from twirled T1/Tphi channel
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

    # Local independent Pauli error on both participants after each CX
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
    acc = 0
    for bitstring, c in counts.items():
        acc += c * (1 if (bitstring.count("1") % 2 == 0) else -1)
    return acc / shots if shots else 0.0



def p0N_p1N(counts: Dict[str, int], N: int) -> Tuple[float, float]:
    shots = sum(counts.values())
    if shots == 0:
        return 0.0, 0.0
    return counts.get("0" * N, 0) / shots, counts.get("1" * N, 0) / shots


# -----------------------------
# Semi-adaptive 4-block circuit
# -----------------------------
def build_adaptive_4block_no_corrections_time_steps(N: int, basis: str = "Z") -> QuantumCircuit:
    if N < 4 or (N % 4) != 0:
        raise ValueError("N must be a multiple of 4 and at least 4.")

    n_blocks = N // 4
    n_anc = n_blocks - 1

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

    # Build 4-qubit GHZ blocks in parallel
    for b in range(n_blocks):
        q0 = 4 * b
        qc.h(d[q0])
    tick_all()

    for b in range(n_blocks):
        q0 = 4 * b
        q1 = q0 + 1
        qc.cx(d[q0], d[q1])
    tick_all()

    for b in range(n_blocks):
        q0 = 4 * b
        q1 = q0 + 1
        q2 = q0 + 2
        q3 = q0 + 3
        qc.cx(d[q0], d[q2])
        qc.cx(d[q1], d[q3])
    tick_all()

    # Fuse neighboring blocks
    for k in range(n_anc):
        left = 4 * k + 3
        qc.cx(d[left], a[k])
    tick_all()

    for k in range(n_anc):
        right = 4 * k + 4
        qc.cx(d[right], a[k])
    tick_all()

    for k in range(n_anc):
        qc.measure(a[k], m[k])
    tick_all()

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
def apply_pauli_frame_from_m_4block(out_bits: str, m_bits: str) -> str:
    n_blocks = len(out_bits) // 4
    bits = list(out_bits)

    prefix = 0
    for j in range(1, n_blocks):
        prefix ^= int(m_bits[j - 1])
        if prefix == 1:
            q0 = 4 * j
            for q in (q0, q0 + 1, q0 + 2, q0 + 3):
                bits[q] = "1" if bits[q] == "0" else "0"

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
            out_corr = apply_pauli_frame_from_m_4block(out_bits, m_bits_noisy)
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
    target: Target,
    seed: int,
) -> float:
    nm = make_noise_model(
        N=N,
        n_anc=n_anc,
        dt=dt,
        T1=T1,
        Tphi=Tphi,
        p_cx_local=p_cx,
        target=target,
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
# Sweep helpers
# -----------------------------
def linspace(start: float, stop: float, n: int) -> list[float]:
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]



def logspace(start: float, stop: float, n: int) -> list[float]:
    if start <= 0 or stop <= 0:
        raise ValueError("logspace requires positive start and stop")
    if n == 1:
        return [start]
    ls = math.log10(start)
    le = math.log10(stop)
    step = (le - ls) / (n - 1)
    return [10 ** (ls + i * step) for i in range(n)]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # ===== User settings =====
    N = 32                      # must be divisible by 4
    shots = 1000                 # raise to 2048+ for smoother curves
    dt = 1.0
    target: Target = "all"
    seed = 1234

    n_p = 18
    p_min = 0.001
    p_max = 0.03

    n_t = 18
    t_min = 40.0
    t_max = 2000.0

    outdir = Path(r"C:\Users\Olof hildeberg\Documents\Karriär\Skola\Masterarbete\Demoncode\Allerrors")
    outdir.mkdir(parents=True, exist_ok=True)
    # =========================

    if N % 4 != 0:
        raise ValueError("N must be divisible by 4 for the semi-adaptive 4-block protocol.")

    n_blocks = N // 4
    n_anc = n_blocks - 1

    sim = AerSimulator(method="stabilizer")
    qcZ = build_adaptive_4block_no_corrections_time_steps(N, basis="Z")
    qcX = build_adaptive_4block_no_corrections_time_steps(N, basis="X")
    tqcZ = transpile(qcZ, sim, optimization_level=0)
    tqcX = transpile(qcX, sim, optimization_level=0)

    p_vals = linspace(p_min, p_max, n_p)
    t_vals = logspace(t_min, t_max, n_t)

    meas_F: list[float] = []
    cx_F: list[float] = []
    tphi_F: list[float] = []
    t1_F: list[float] = []

    print(f"Running semi-adaptive 4-block sweeps with N={N}, shots={shots}, target={target}")

    print("\nSweep 1/4: fidelity vs measurement error")
    for i, p in enumerate(p_vals, start=1):
        F = fidelity_from_params(
            tqcZ, tqcX, N, n_anc, shots,
            dt=dt,
            T1=float("inf"),
            Tphi=float("inf"),
            p_meas_err=p,
            p_cx=0.0,
            target=target,
            seed=seed + 1000 + i,
        )
        meas_F.append(F)
        print(f"  [{i:02d}/{len(p_vals)}] p_meas={p:.5f}  F~={F:.6f}")

    print("\nSweep 2/4: fidelity vs CX error")
    for i, p in enumerate(p_vals, start=1):
        F = fidelity_from_params(
            tqcZ, tqcX, N, n_anc, shots,
            dt=dt,
            T1=float("inf"),
            Tphi=float("inf"),
            p_meas_err=0.0,
            p_cx=p,
            target=target,
            seed=seed + 2000 + i,
        )
        cx_F.append(F)
        print(f"  [{i:02d}/{len(p_vals)}] p_cx={p:.5f}  F~={F:.6f}")

    print("\nSweep 3/4: fidelity vs Tphi (pure dephasing only)")
    for i, Tphi in enumerate(t_vals, start=1):
        F = fidelity_from_params(
            tqcZ, tqcX, N, n_anc, shots,
            dt=dt,
            T1=float("inf"),
            Tphi=Tphi,
            p_meas_err=0.0,
            p_cx=0.0,
            target=target,
            seed=seed + 3000 + i,
        )
        tphi_F.append(F)
        print(f"  [{i:02d}/{len(t_vals)}] Tphi={Tphi:.3f}  F~={F:.6f}")

    print("\nSweep 4/4: fidelity vs T1 (relaxation only, twirled)")
    for i, T1 in enumerate(t_vals, start=1):
        F = fidelity_from_params(
            tqcZ, tqcX, N, n_anc, shots,
            dt=dt,
            T1=T1,
            Tphi=float("inf"),
            p_meas_err=0.0,
            p_cx=0.0,
            target=target,
            seed=seed + 4000 + i,
        )
        t1_F.append(F)
        print(f"  [{i:02d}/{len(t_vals)}] T1={T1:.3f}  F~={F:.6f}")

    # Figure 1: measurement and CX errors
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(p_vals, meas_F, marker="o", linewidth=2, label="Measurement error")
    plt.plot(p_vals, cx_F, marker="s", linewidth=2, label="CX error")
    plt.xlabel("Error probability")
    plt.ylabel("Fidelity estimate")
    plt.title(f"Semi-adaptive 4-block GHZ: Fidelity vs measurement/CX error (N={N})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig1 = outdir / "semiadaptive_fidelity_vs_meas_and_cx.png"
    plt.savefig(fig1, dpi=180, bbox_inches="tight")
    plt.close()

    # Figure 2: Tphi and T1 sweeps
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(t_vals, tphi_F, marker="o", linewidth=2, label=r"$T_\phi$")
    plt.plot(t_vals, t1_F, marker="s", linewidth=2, label=r"$T_1$")
    plt.xscale("log")
    plt.xlabel("Time constant (ticks)")
    plt.ylabel("Fidelity estimate")
    plt.title(f"Semi-adaptive 4-block GHZ: Fidelity vs $T_\\phi$ and $T_1$ (N={N})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig2 = outdir / "semiadaptive_fidelity_vs_tphi_and_t1.png"
    plt.savefig(fig2, dpi=180, bbox_inches="tight")
    plt.close()

    print("\nSaved plots:")
    print(f"  {fig1}")
    print(f"  {fig2}")


if __name__ == "__main__":
    main()
