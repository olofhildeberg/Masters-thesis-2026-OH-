from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import pauli_error


# -----------------------------
# Shared utilities
# -----------------------------


def ns_to_ticks(duration_ns: float, tick_ns: float) -> int:
    if duration_ns < 0:
        raise ValueError("duration_ns must be >= 0")
    if tick_ns <= 0:
        raise ValueError("tick_ns must be > 0")
    return max(0, math.ceil(duration_ns / tick_ns))
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



def t1_tphi_to_pauli_probs(dt: float, T1: float, Tphi: float) -> Tuple[float, float, float]:
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



def make_noise_model_all_qubits(
    N: int,
    n_anc: int,
    dt: float = 300.0,
    T1: float = float("inf"),
    Tphi: float = float("inf"),
    p_cx_local: float = 0.0,
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
        for q in range(0, N + n_anc):
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



def make_noise_model_data_only(
    dt: float = 300.0,
    T1: float = float("inf"),
    Tphi: float = float("inf"),
    p_cx_local: float = 0.0,
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
        nm.add_all_qubit_quantum_error(delay_err, "delay")

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



def run_counts(
    tqc: QuantumCircuit,
    shots: int,
    noise_model: NoiseModel | None,
    seed: int | None = None,
) -> Dict[str, int]:
    sim = AerSimulator(method="stabilizer", noise_model=noise_model, seed_simulator=seed)
    return sim.run(tqc, shots=shots).result().get_counts()



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
# Full adaptive
# -----------------------------
def build_full_adaptive_no_corrections_time_steps(
    N: int,
    basis: str = "Z",
    tick_ns: float = 300.0,
    measurement_ns: float = 2100.0,
    classical_feedforward_ns: float = 900.0,
) -> QuantumCircuit:
    n_anc = N - 1

    d = QuantumRegister(N, "d")
    a = QuantumRegister(n_anc, "a")
    m = ClassicalRegister(n_anc, "m")
    out = ClassicalRegister(N, "out")
    qc = QuantumCircuit(d, a, m, out)

    def tick_all(num_ticks: int = 1) -> None:
        for _ in range(num_ticks):
            for q in range(N):
                qc.delay(1, d[q])
            for k in range(n_anc):
                qc.delay(1, a[k])

    for q in range(N):
        qc.h(d[q])

    for k in range(n_anc):
        qc.cx(d[k], a[k])
    tick_all(1)

    for k in range(n_anc):
        qc.cx(d[k + 1], a[k])
    tick_all(1)

    measurement_ticks = ns_to_ticks(measurement_ns, tick_ns)
    classical_feedforward_ticks = ns_to_ticks(classical_feedforward_ns, tick_ns)

    # Physical readout duration: ancillas and data wait before the measurement result exists.
    tick_all(measurement_ticks)

    for k in range(n_anc):
        qc.measure(a[k], m[k])

    # Classical feed-forward / Pauli-frame update after readout.
    # Ancillas have already been measured, so only data qubits keep idling.
    for _ in range(classical_feedforward_ticks):
        for q in range(N):
            qc.delay(1, d[q])

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



def marginalize_and_correct_full(
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



def fidelity_full_adaptive(
    tqcZ: QuantumCircuit,
    tqcX: QuantumCircuit,
    N: int,
    shots: int,
    dt: float,
    T1: float,
    Tphi: float,
    p_meas_err: float,
    p_cx: float,
    seed: int,
) -> float:
    nm = make_noise_model_all_qubits(
        N=N,
        n_anc=N - 1,
        dt=dt,
        T1=T1,
        Tphi=Tphi,
        p_cx_local=p_cx,
    )

    rawZ = run_counts(tqcZ, shots, nm, seed=seed)
    rawX = run_counts(tqcX, shots, nm, seed=seed + 1)

    rng = random.Random(seed + 999)
    Zc = marginalize_and_correct_full(rawZ, N, basis="Z", p_meas_anc=p_meas_err, rng=rng)
    Xc = marginalize_and_correct_full(rawX, N, basis="X", p_meas_anc=0.0, rng=rng)

    Xexp = parity_expectation_from_counts(Xc)
    p0, p1 = p0N_p1N(Zc, N)
    return 0.5 * (p0 + p1 + Xexp)


# -----------------------------
# Semi-adaptive 4-block
# -----------------------------
def build_adaptive_4block_no_corrections_time_steps(
    N: int,
    basis: str = "Z",
    tick_ns: float = 300.0,
    measurement_ns: float = 2100.0,
    classical_feedforward_ns: float = 900.0,
) -> QuantumCircuit:
    if N < 4 or (N % 4) != 0:
        raise ValueError("N must be a multiple of 4 and at least 4.")

    n_blocks = N // 4
    n_anc = n_blocks - 1

    d = QuantumRegister(N, "d")
    a = QuantumRegister(n_anc, "a")
    m = ClassicalRegister(n_anc, "m")
    out = ClassicalRegister(N, "out")
    qc = QuantumCircuit(d, a, m, out)

    def tick_all(num_ticks: int = 1) -> None:
        for _ in range(num_ticks):
            for q in range(N):
                qc.delay(1, d[q])
            for k in range(n_anc):
                qc.delay(1, a[k])

    for b in range(n_blocks):
        q0 = 4 * b
        qc.h(d[q0])

    for b in range(n_blocks):
        q0 = 4 * b
        q1 = q0 + 1
        qc.cx(d[q0], d[q1])
    tick_all(1)

    for b in range(n_blocks):
        q0 = 4 * b
        q1 = q0 + 1
        q2 = q0 + 2
        q3 = q0 + 3
        qc.cx(d[q0], d[q2])
        qc.cx(d[q1], d[q3])
    tick_all(1)

    for k in range(n_anc):
        left = 4 * k + 3
        qc.cx(d[left], a[k])
    tick_all(1)

    for k in range(n_anc):
        right = 4 * k + 4
        qc.cx(d[right], a[k])
    tick_all(1)

    measurement_ticks = ns_to_ticks(measurement_ns, tick_ns)
    classical_feedforward_ticks = ns_to_ticks(classical_feedforward_ns, tick_ns)

    # Physical readout duration: ancillas and data wait before the measurement result exists.
    tick_all(measurement_ticks)

    for k in range(n_anc):
        qc.measure(a[k], m[k])

    # Classical feed-forward / Pauli-frame update after readout.
    # Ancillas have already been measured, so only data qubits keep idling.
    for _ in range(classical_feedforward_ticks):
        for q in range(N):
            qc.delay(1, d[q])

    b = basis.upper()
    if b == "X":
        qc.h(d)
    elif b != "Z":
        raise ValueError("basis must be 'Z' or 'X'")

    qc.measure(d, out)
    return qc



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



def marginalize_and_correct_semi(
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



def fidelity_semi_adaptive(
    tqcZ: QuantumCircuit,
    tqcX: QuantumCircuit,
    N: int,
    shots: int,
    dt: float,
    T1: float,
    Tphi: float,
    p_meas_err: float,
    p_cx: float,
    seed: int,
) -> float:
    n_blocks = N // 4
    n_anc = n_blocks - 1

    nm = make_noise_model_all_qubits(
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
    Zc = marginalize_and_correct_semi(rawZ, N, basis="Z", p_meas_anc=p_meas_err, rng=rng)
    Xc = marginalize_and_correct_semi(rawX, N, basis="X", p_meas_anc=0.0, rng=rng)

    Xexp = parity_expectation_from_counts(Xc)
    p0, p1 = p0N_p1N(Zc, N)
    return 0.5 * (p0 + p1 + Xexp)


# -----------------------------
# Non-adaptive automatic IBM-style GHZ schedule
# -----------------------------
def heavy_hex_like_graph(nr: int, nc: int) -> Dict[int, List[int]]:
    """
    Synthetic heavy-hex-like graph for local/offline simulation.
    For real IBM hardware, use graph_from_backend instead.
    """
    keep = [[True for _ in range(nc)] for _ in range(nr)]

    for r in range(nr):
        if r % 2 == 1:
            phase = 0 if ((r // 2) % 2 == 0) else 2
            for c in range(nc):
                if c % 4 != phase:
                    keep[r][c] = False

    coord_to_id = {}
    id_to_coord = []
    next_id = 0
    for r in range(nr):
        for c in range(nc):
            if keep[r][c]:
                coord_to_id[(r, c)] = next_id
                id_to_coord.append((r, c))
                next_id += 1

    g = {i: [] for i in range(next_id)}
    for i, (r, c) in enumerate(id_to_coord):
        for dr, dc in [(1, 0), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nr and 0 <= cc < nc and keep[rr][cc]:
                j = coord_to_id[(rr, cc)]
                g[i].append(j)
                g[j].append(i)

    return {u: sorted(set(vs)) for u, vs in g.items()}



def graph_to_coupling_map(g: Dict[int, List[int]]) -> CouplingMap:
    edges = set()
    for u, nbrs in g.items():
        for v in nbrs:
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b))

    directed = []
    for a, b in sorted(edges):
        directed.append((a, b))
        directed.append((b, a))

    return CouplingMap(directed)



def graph_from_coupling_edges(edges: Iterable[Tuple[int, int]]) -> Dict[int, List[int]]:
    g: Dict[int, Set[int]] = {}
    for u, v in edges:
        g.setdefault(int(u), set()).add(int(v))
        g.setdefault(int(v), set()).add(int(u))
    return {u: sorted(vs) for u, vs in g.items()}



def graph_from_backend(backend) -> Dict[int, List[int]]:
    """
    Extract an undirected connectivity graph from an IBM backend.
    This does not pass a custom coupling_map to transpile when backend is used.
    """
    if getattr(backend, "coupling_map", None) is not None:
        cm = backend.coupling_map
        if hasattr(cm, "get_edges"):
            return graph_from_coupling_edges(cm.get_edges())
        return graph_from_coupling_edges(cm)

    # Fallback through target, useful for newer Qiskit objects.
    edges = []
    target = backend.target
    for qargs in target.get("cx", {}):
        if len(qargs) == 2:
            edges.append((qargs[0], qargs[1]))
    if not edges:
        for name in ("ecr", "cz"):
            for qargs in target.get(name, {}):
                if len(qargs) == 2:
                    edges.append((qargs[0], qargs[1]))
    if not edges:
        raise ValueError("Could not extract two-qubit connectivity from backend.")
    return graph_from_coupling_edges(edges)



def connected_component_containing(g: Dict[int, List[int]], start: int) -> Set[int]:
    seen = {start}
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for v in g[u]:
            if v not in seen:
                seen.add(v)
                dq.append(v)
    return seen



def largest_connected_component(g: Dict[int, List[int]]) -> Dict[int, List[int]]:
    unseen = set(g)
    best: Set[int] = set()
    while unseen:
        start = next(iter(unseen))
        comp = connected_component_containing(g, start)
        if len(comp) > len(best):
            best = comp
        unseen -= comp
    return {u: [v for v in g[u] if v in best] for u in sorted(best)}



def all_pairs_distances_limited(g: Dict[int, List[int]]) -> Dict[int, Dict[int, int]]:
    dists: Dict[int, Dict[int, int]] = {}
    for s in g:
        dist = {s: 0}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in g[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        dists[s] = dist
    return dists



def pick_candidate_roots(g: Dict[int, List[int]], n_roots: int) -> List[int]:
    """
    Pick central, high-degree candidate roots.
    """
    dists = all_pairs_distances_limited(g)
    nodes = list(g)
    scored = []
    for u in nodes:
        ecc = max(dists[u].values())
        avg = sum(dists[u].values()) / len(dists[u])
        deg = len(g[u])
        scored.append((ecc, avg, -deg, u))
    scored.sort()
    return [u for *_unused, u in scored[: min(n_roots, len(scored))]]


@dataclass
class GHZSchedule:
    source_list: List[int]          # logical source qubits
    target_list: List[int]          # logical target qubits
    layer_list: List[int]           # CX layer index for each edge
    physical_layout: List[int]      # logical qubit i -> physical qubit physical_layout[i]
    root_physical: int
    num_layers: int



def greedy_broadcast_once(
    g: Dict[int, List[int]],
    root: int,
    N: int,
    rng: random.Random,
    mode: int,
) -> Optional[Tuple[List[Tuple[int, int, int]], List[int]]]:
    """
    Build one GHZ broadcast schedule on the physical graph.

    At each layer, active qubits may each create one new active neighbor, and
    every qubit can appear in at most one CX in that layer. This directly mimics
    the IBM-style hand-written source/target/layer lists, but generated from a graph.
    """
    active: Set[int] = {root}
    activation_order: List[int] = [root]
    physical_edges_with_layers: List[Tuple[int, int, int]] = []
    layer = 0

    while len(active) < N:
        candidates = []
        for u in active:
            for v in g[u]:
                if v not in active:
                    # Several simple scores. Different modes generate different schedules.
                    inactive_deg = sum(1 for w in g[v] if w not in active)
                    total_deg = len(g[v])
                    if mode == 0:
                        score = (inactive_deg, total_deg, rng.random())
                    elif mode == 1:
                        score = (total_deg, inactive_deg, rng.random())
                    elif mode == 2:
                        score = (inactive_deg - len(g[u]), total_deg, rng.random())
                    elif mode == 3:
                        score = (rng.random(), inactive_deg, total_deg)
                    else:
                        score = (inactive_deg, rng.random(), total_deg)
                    candidates.append((score, u, v))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        used: Set[int] = set()
        chosen: List[Tuple[int, int]] = []
        chosen_targets: Set[int] = set()

        for _score, u, v in candidates:
            if len(active) + len(chosen_targets) >= N:
                break
            if u in used or v in used or v in active or v in chosen_targets:
                continue
            chosen.append((u, v))
            chosen_targets.add(v)
            used.add(u)
            used.add(v)

        if not chosen:
            return None

        for u, v in chosen:
            physical_edges_with_layers.append((u, v, layer))
            activation_order.append(v)

        active.update(chosen_targets)
        layer += 1

    return physical_edges_with_layers, activation_order[:N]



def find_low_depth_ghz_schedule(
    g: Dict[int, List[int]],
    N: int,
    num_roots: int = 80,
    trials_per_root: int = 40,
    seed: int = 2026,
) -> GHZSchedule:
    """
    Generate IBM-style source/target/layer lists from a hardware graph.

    This is not guaranteed globally optimal, but it automates the same kind of
    object used in the IBM challenge notebook: explicit CX sources, CX targets,
    layer labels, and a physical layout.
    """
    g = largest_connected_component(g)
    if len(g) < N:
        raise ValueError(f"Largest connected component has only {len(g)} qubits, but N={N}.")

    roots = pick_candidate_roots(g, num_roots)
    rng_master = random.Random(seed)

    best_edges: Optional[List[Tuple[int, int, int]]] = None
    best_order: Optional[List[int]] = None
    best_root: Optional[int] = None
    best_layers = 10**9
    best_parallel_score = -1

    for root in roots:
        for trial in range(trials_per_root):
            rng = random.Random(rng_master.randint(0, 2**31 - 1))
            mode = trial % 5
            result = greedy_broadcast_once(g, root=root, N=N, rng=rng, mode=mode)
            if result is None:
                continue
            edges, order = result
            num_layers = 1 + max(layer for *_uv, layer in edges)
            layer_sizes = [0] * num_layers
            for *_uv, layer in edges:
                layer_sizes[layer] += 1
            parallel_score = sum(s * s for s in layer_sizes)

            if (num_layers < best_layers) or (
                num_layers == best_layers and parallel_score > best_parallel_score
            ):
                best_edges = edges
                best_order = order
                best_root = root
                best_layers = num_layers
                best_parallel_score = parallel_score

    if best_edges is None or best_order is None or best_root is None:
        raise RuntimeError("Could not find a GHZ schedule for this graph and N.")

    # The logical circuit has qubits 0..N-1. logical i maps to physical_layout[i].
    physical_layout = best_order
    phys_to_logical = {p: i for i, p in enumerate(physical_layout)}

    source_list = []
    target_list = []
    layer_list = []
    for u_phys, v_phys, layer in best_edges:
        if u_phys not in phys_to_logical or v_phys not in phys_to_logical:
            continue
        source_list.append(phys_to_logical[u_phys])
        target_list.append(phys_to_logical[v_phys])
        layer_list.append(layer)

    return GHZSchedule(
        source_list=source_list,
        target_list=target_list,
        layer_list=layer_list,
        physical_layout=physical_layout,
        root_physical=best_root,
        num_layers=best_layers,
    )


# ============================================================
# Circuit construction
# ============================================================
def build_layered_ghz_from_schedule(
    schedule: GHZSchedule,
    add_delay_after_each_cx_layer: bool = True,
) -> QuantumCircuit:
    """
    Build a GHZ circuit from explicit IBM-style source/target/layer lists.

    No delay is inserted after the initial H layer. One delay tick is inserted
    after each entangling layer, matching the 300 ns coarse timing model.
    """
    N = len(schedule.physical_layout)
    qc = QuantumCircuit(N)

    root_logical = schedule.physical_layout.index(schedule.root_physical)
    qc.h(root_logical)

    edges_by_layer: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(schedule.num_layers)}
    for s, t, layer in zip(schedule.source_list, schedule.target_list, schedule.layer_list):
        edges_by_layer[layer].append((s, t))

    for layer in range(schedule.num_layers):
        used = set()
        for s, t in edges_by_layer[layer]:
            if s in used or t in used:
                raise ValueError(f"Invalid schedule: qubit reused in CX layer {layer}.")
            used.add(s)
            used.add(t)
            qc.cx(s, t)

        if add_delay_after_each_cx_layer:
            for q in range(N):
                qc.delay(1, q)

    return qc



def measure_in_basis(prep: QuantumCircuit, basis: str) -> QuantumCircuit:
    N = prep.num_qubits
    qc = QuantumCircuit(N, N)
    qc.compose(prep, inplace=True)

    if basis.upper() == "X":
        qc.h(range(N))
    elif basis.upper() != "Z":
        raise ValueError("basis must be 'Z' or 'X'")

    qc.measure(range(N), range(N))
    return qc


def fidelity_non_adaptive(
    tqcZ: QuantumCircuit,
    tqcX: QuantumCircuit,
    N: int,
    shots: int,
    dt: float,
    T1: float,
    Tphi: float,
    p_cx: float,
    seed: int,
) -> float:
    nm = make_noise_model_data_only(
        dt=dt,
        T1=T1,
        Tphi=Tphi,
        p_cx_local=p_cx,
    )

    countsZ = run_counts(tqcZ, shots, nm, seed=seed)
    countsX = run_counts(tqcX, shots, nm, seed=seed + 1)

    Xexp = parity_expectation_from_counts(countsX)
    p0, p1 = p0N_p1N(countsZ, N)
    return 0.5 * (p0 + p1 + Xexp)


# -----------------------------
# Main comparison sweeps
# -----------------------------
def main() -> None:
    N = 128
    shots = 10000

    # Coarse timing model in ns
    tick_ns = 300.0
    measurement_ns = 2100.0
    classical_feedforward_ns = 900.0

    # Each delay tick in the noise model corresponds to tick_ns.
    dt = tick_ns
    seed = 1234

    # Pure dephasing sweep values, expressed in ns.
    T_vals = geomspace(10_000.0, 400_000.0, 18)

    base_dir = Path(__file__).resolve().parent
    outdir = base_dir / "Sweep_results"
    outdir.mkdir(parents=True, exist_ok=True)

    base_sim = AerSimulator(method="stabilizer")

    # Build/transpile full adaptive.
    qc_full_Z = build_full_adaptive_no_corrections_time_steps(
        N, basis="Z", tick_ns=tick_ns,
        measurement_ns=measurement_ns,
        classical_feedforward_ns=classical_feedforward_ns,
    )
    qc_full_X = build_full_adaptive_no_corrections_time_steps(
        N, basis="X", tick_ns=tick_ns,
        measurement_ns=measurement_ns,
        classical_feedforward_ns=classical_feedforward_ns,
    )
    tqc_full_Z = transpile(qc_full_Z, base_sim, optimization_level=0)
    tqc_full_X = transpile(qc_full_X, base_sim, optimization_level=0)

    # Build/transpile semi-adaptive.
    qc_semi_Z = build_adaptive_4block_no_corrections_time_steps(
        N, basis="Z", tick_ns=tick_ns,
        measurement_ns=measurement_ns,
        classical_feedforward_ns=classical_feedforward_ns,
    )
    qc_semi_X = build_adaptive_4block_no_corrections_time_steps(
        N, basis="X", tick_ns=tick_ns,
        measurement_ns=measurement_ns,
        classical_feedforward_ns=classical_feedforward_ns,
    )
    tqc_semi_Z = transpile(qc_semi_Z, base_sim, optimization_level=0)
    tqc_semi_X = transpile(qc_semi_X, base_sim, optimization_level=0)

    # Build/transpile non-adaptive.
    num_roots_to_try = 80
    greedy_trials_per_root = 40
    layout_search_seed = 2026

    side = math.ceil(math.sqrt(8 * N / 5)) + 8
    while True:
        full_graph = heavy_hex_like_graph(side, side)
        if len(largest_connected_component(full_graph)) >= N:
            break
        side += 2

    coupling = graph_to_coupling_map(full_graph)

    non_schedule = find_low_depth_ghz_schedule(
        full_graph,
        N=N,
        num_roots=num_roots_to_try,
        trials_per_root=greedy_trials_per_root,
        seed=layout_search_seed,
    )

    non_layer_sizes = [0] * non_schedule.num_layers
    for layer in non_schedule.layer_list:
        non_layer_sizes[layer] += 1

    print("\nGenerated automatic IBM-style non-adaptive GHZ schedule")
    print(f"Physical root qubit:          {non_schedule.root_physical}")
    print(f"Number of CX gates:           {len(non_schedule.source_list)}")
    print(f"Number of CX layers:          {non_schedule.num_layers}")
    print(f"CX gates per layer:           {non_layer_sizes}")
    print(f"Modeled idle time:            {non_schedule.num_layers * dt:.1f} ns")

    prep_non = build_layered_ghz_from_schedule(
        non_schedule,
        add_delay_after_each_cx_layer=True,
    )
    qc_non_Z = measure_in_basis(prep_non, "Z")
    qc_non_X = measure_in_basis(prep_non, "X")

    tqc_non_Z = transpile(
        qc_non_Z,
        coupling_map=coupling,
        initial_layout=non_schedule.physical_layout,
        optimization_level=0,
        routing_method="none",
    )
    tqc_non_X = transpile(
        qc_non_X,
        coupling_map=coupling,
        initial_layout=non_schedule.physical_layout,
        optimization_level=0,
        routing_method="none",
    )

    print(f"Running Tphi-only comparison sweep with shots={shots}, N={N}")
    print(
        f"Adaptive timing model: tick={tick_ns} ns, "
        f"measurement={measurement_ns} ns before ancilla measure, "
        f"classical feed-forward={classical_feedforward_ns} ns after ancilla measure"
    )

    # Tphi sweep only. Measurement and entangling-gate errors are fixed to zero.
    full_tphi = []
    semi_tphi = []
    non_tphi = []

    for i, T in enumerate(T_vals):
        F_full = fidelity_full_adaptive(
            tqc_full_Z, tqc_full_X, N, shots, dt,
            T1=float("inf"), Tphi=T,
            p_meas_err=0.0, p_cx=0.0,
            seed=seed + 5000 + 10 * i,
        )
        F_semi = fidelity_semi_adaptive(
            tqc_semi_Z, tqc_semi_X, N, shots, dt,
            T1=float("inf"), Tphi=T,
            p_meas_err=0.0, p_cx=0.0,
            seed=seed + 6000 + 10 * i,
        )
        F_non = fidelity_non_adaptive(
            tqc_non_Z, tqc_non_X, N, shots, dt,
            T1=float("inf"), Tphi=T,
            p_cx=0.0,
            seed=seed + 7000 + 10 * i,
        )

        full_tphi.append(F_full)
        semi_tphi.append(F_semi)
        non_tphi.append(F_non)
        print(f"Tphi={T:.3f} ns -> full={F_full:.6f}, semi={F_semi:.6f}, non={F_non:.6f}")

    plt.figure(figsize=(7.5, 3.2))
    plt.plot(T_vals, full_tphi, marker="o", label="Full adaptive")
    plt.plot(T_vals, semi_tphi, marker="s", label="Semi-adaptive")
    plt.plot(T_vals, non_tphi, marker="^", label="Non-adaptive")
    plt.xscale("log")
    plt.xlabel(r"$T_\phi$ (ns)")
    plt.ylabel("GHZ fidelity")
    plt.title(fr"Fidelity vs $T_\phi$ (shots={shots}, N={N})")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="upper left")
    plt.tight_layout()

    fig_tphi = outdir / f"protocol_comparison_fidelity_vs_tphi_only_{N}N.png"
    plt.savefig(fig_tphi, dpi=180)
    plt.close()

    print("\nSaved:")
    print(fig_tphi)


if __name__ == "__main__":
    main()
