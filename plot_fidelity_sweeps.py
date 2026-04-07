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


def t1_tphi_to_pauli_probs(dt: float, T1: float, Tphi: float) -> Tuple[float, float, float]:
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



def build_full_adaptive_no_corrections_time_steps(N: int, basis: str = "Z") -> QuantumCircuit:
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

    for q in range(N):
        qc.h(d[q])
    tick_all()

    for k in range(n_anc):
        qc.cx(d[k], a[k])
    tick_all()

    for k in range(n_anc):
        qc.cx(d[k + 1], a[k])
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



def apply_pauli_frame_from_m_full_adaptive(out_bits: str, m_bits: str) -> str:
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



def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return


def main() -> None:
    # -------------------------
    # User-facing settings
    # -------------------------
    N = 32
    shots = 1000            # lowered from 1100 for faster sweeps
    dt = 1.0               # one tick = one layer
    target: Target = "all"  # change to "data" or "anc" if you want
    seed = 1234

    # Sweep ranges
    p_vals = linspace(0.001, 0.03, 18)
    T_vals = geomspace(40.0, 2000.0, 18)

    outdir = Path(r"C:\Users\Olof hildeberg\Documents\Karriär\Skola\Masterarbete\Demoncode\Allerrors")
    outdir.mkdir(parents=True, exist_ok=True)

    n_anc = N - 1

    qcZ = build_full_adaptive_no_corrections_time_steps(N, basis="Z")
    qcX = build_full_adaptive_no_corrections_time_steps(N, basis="X")

    base_sim = AerSimulator(method="stabilizer")
    tqcZ = transpile(qcZ, base_sim, optimization_level=0)
    tqcX = transpile(qcX, base_sim, optimization_level=0)

    print(f"Running sweeps with N={N}, shots={shots}, target={target}")

    # -------------------------
    # Sweep 1: measurement and CX errors (one at a time)
    # -------------------------
    meas_rows = []
    cx_rows = []
    meas_fids = []
    cx_fids = []

    for i, p in enumerate(p_vals):
        F_meas = fidelity_from_params(
            tqcZ, tqcX, N, n_anc, shots, dt,
            T1=float("inf"), Tphi=float("inf"),
            p_meas_err=p, p_cx=0.0,
            target=target, seed=seed + 10 * i,
        )
        meas_fids.append(F_meas)
        meas_rows.append({"parameter": "p_meas", "value": p, "fidelity": F_meas})
        print(f"p_meas={p:.4f} -> F={F_meas:.6f}")

        F_cx = fidelity_from_params(
            tqcZ, tqcX, N, n_anc, shots, dt,
            T1=float("inf"), Tphi=float("inf"),
            p_meas_err=0.0, p_cx=p,
            target=target, seed=seed + 1000 + 10 * i,
        )
        cx_fids.append(F_cx)
        cx_rows.append({"parameter": "p_cx", "value": p, "fidelity": F_cx})
        print(f"p_cx  ={p:.4f} -> F={F_cx:.6f}")

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(p_vals, meas_fids, marker="o", label="Measurement error only")
    plt.plot(p_vals, cx_fids, marker="s", label="CNOT error only")
    plt.xlabel("Error probability")
    plt.ylabel("GHZ fidelity")
    plt.title(f"Fidelity vs measurement and CNOT error (shots={shots})")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig1 = outdir / "fidelity_vs_meas_and_cx.png"
    plt.savefig(fig1, dpi=180)
    plt.close()

    # -------------------------
    # Sweep 2: Tphi and T1 (one at a time)
    # -------------------------
    tphi_rows = []
    t1_rows = []
    tphi_fids = []
    t1_fids = []

    for i, T in enumerate(T_vals):
        F_tphi = fidelity_from_params(
            tqcZ, tqcX, N, n_anc, shots, dt,
            T1=float("inf"), Tphi=T,
            p_meas_err=0.0, p_cx=0.0,
            target=target, seed=seed + 2000 + 10 * i,
        )
        tphi_fids.append(F_tphi)
        tphi_rows.append({"parameter": "Tphi", "value": T, "fidelity": F_tphi})
        print(f"Tphi={T:.3f} -> F={F_tphi:.6f}")

        F_t1 = fidelity_from_params(
            tqcZ, tqcX, N, n_anc, shots, dt,
            T1=T, Tphi=float("inf"),
            p_meas_err=0.0, p_cx=0.0,
            target=target, seed=seed + 3000 + 10 * i,
        )
        t1_fids.append(F_t1)
        t1_rows.append({"parameter": "T1", "value": T, "fidelity": F_t1})
        print(f"T1  ={T:.3f} -> F={F_t1:.6f}")

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(T_vals, tphi_fids, marker="o", label=r"Pure dephasing only ($T_\phi$ finite)")
    plt.plot(T_vals, t1_fids, marker="s", label=r"Relaxation only ($T_1$ finite)")
    plt.xscale("log")
    plt.xlabel("Time constant (ticks)")
    plt.ylabel("GHZ fidelity")
    plt.title(fr"Fidelity vs $T_\phi$ and $T_1$ (shots={shots})")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig2 = outdir / "fidelity_vs_tphi_and_t1.png"
    plt.savefig(fig2, dpi=180)
    plt.close()

    print("\nSaved:")
    print(fig1)
    print(fig2)


if __name__ == "__main__":
    main()
