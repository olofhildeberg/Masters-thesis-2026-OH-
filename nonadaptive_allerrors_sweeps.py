from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import pauli_error


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
# Twirled idle decoherence + local CX noise
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
    dt: float = 1.0,
    T1: float = float("inf"),
    Tphi: float = float("inf"),
    p_cx_local: float = 0.0,
) -> NoiseModel:
    nm = NoiseModel()

    # idle decoherence on each delay tick
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

    # independent local Pauli error on both CX participants
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
# Hardware graphs
# -----------------------------
from typing import Dict, List

def grid_graph(nr: int, nc: int) -> Dict[int, List[int]]:
    N = nr * nc
    g = {i: [] for i in range(N)}
    for r in range(nr):
        for c in range(nc):
            u = r * nc + c
            if r + 1 < nr:
                v = (r + 1) * nc + c
                g[u].append(v)
                g[v].append(u)
            if c + 1 < nc:
                v = r * nc + (c + 1)
                g[u].append(v)
                g[v].append(u)
    return g


def heavy_hex_like_graph(nr: int, nc: int) -> Dict[int, List[int]]:
    """
    Build a staggered superhex-like graph on an underlying nr x nc square lattice.

    Construction:
      - even rows: keep all qubits
      - odd rows: keep only every 4th qubit
      - the kept columns on odd rows alternate with a horizontal shift of 2
        from one odd row to the next

    Example (1-based indexing):
      row 2 keeps cols 1,5,9,...
      row 4 keeps cols 3,7,11,...
      row 6 keeps cols 1,5,9,...
      ...

    The returned graph is COMPACTLY RELABELED to nodes 0..M-1 in row-major order
    over the kept sites only.
    """
    # Step 1: decide which lattice sites to keep
    keep = [[True for _ in range(nc)] for _ in range(nr)]

    for r in range(nr):
        if r % 2 == 1:
            # odd rows are sparse bridge rows
            # alternate phase 0,2,0,2,... as we go downward
            phase = 0 if ((r // 2) % 2 == 0) else 2
            for c in range(nc):
                if c % 4 != phase:
                    keep[r][c] = False

    # Step 2: compact relabeling of kept sites
    coord_to_id = {}
    id_to_coord = []
    next_id = 0
    for r in range(nr):
        for c in range(nc):
            if keep[r][c]:
                coord_to_id[(r, c)] = next_id
                id_to_coord.append((r, c))
                next_id += 1

    # Step 3: build nearest-neighbor graph on kept sites only
    g = {i: [] for i in range(next_id)}

    for i, (r, c) in enumerate(id_to_coord):
        for dr, dc in [(1, 0), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nr and 0 <= cc < nc and keep[rr][cc]:
                j = coord_to_id[(rr, cc)]
                g[i].append(j)
                g[j].append(i)

    return g



def graph_to_coupling_map(g: Dict[int, List[int]]) -> CouplingMap:
    edges = set()
    for u, nbrs in g.items():
        for v in nbrs:
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b))
    directed = []
    for a, b in edges:
        directed.append((a, b))
        directed.append((b, a))
    return CouplingMap(directed)


# -----------------------------
# BFS spanning tree + GHZ schedule
# -----------------------------
def bfs_tree_parents(g: Dict[int, List[int]], root: int) -> Tuple[List[Optional[int]], List[int]]:
    N = len(g)
    parent: List[Optional[int]] = [None] * N
    depth: List[int] = [-1] * N

    q = deque([root])
    depth[root] = 0
    while q:
        u = q.popleft()
        for v in g[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                parent[v] = u
                q.append(v)

    if any(d == -1 for d in depth):
        raise ValueError("Graph is disconnected for the chosen N/root")
    return parent, depth



def choose_centerish_root(g: Dict[int, List[int]]) -> int:
    N = len(g)
    sample = list(range(0, N, max(1, N // 16)))
    best = 0
    best_score = float("inf")
    for cand in range(N):
        dist = [-1] * N
        dist[cand] = 0
        dq = deque([cand])
        while dq:
            u = dq.popleft()
            for v in g[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        score = sum(dist[s] for s in sample)
        if score < best_score:
            best_score = score
            best = cand
    return best

def ghz_on_bfs_tree_with_time_steps(
    g: Dict[int, List[int]],
    root: Optional[int] = None,
    add_time_step_after_each_layer: bool = True,
    print_schedule_once: bool = False,
) -> QuantumCircuit:
    N = len(g)
    if N < 2:
        raise ValueError("Need at least 2 qubits")

    if root is None:
        root = choose_centerish_root(g)

    parent, depth = bfs_tree_parents(g, root)
    qc = QuantumCircuit(N)

    # Build children list of the BFS tree
    children = {u: [] for u in range(N)}
    for v in range(N):
        p = parent[v]
        if p is not None:
            children[p].append(v)

    # Prioritize children with larger downstream subtree first
    subtree_height = [0] * N
    order = sorted(range(N), key=lambda x: depth[x], reverse=True)
    for u in order:
        if children[u]:
            subtree_height[u] = 1 + max(subtree_height[c] for c in children[u])

    for u in range(N):
        children[u].sort(key=lambda c: subtree_height[c], reverse=True)

    # Root activation
    activated = {root}
    qc.h(root)

    # Decide whether to print on this call
    do_print = (
        print_schedule_once
        and not getattr(ghz_on_bfs_tree_with_time_steps, "_has_printed_schedule", False)
    )

    if do_print:
        print("\n--- GHZ schedule debug (printed once) ---")
        print(f"Root: {root}")
        print("Parent array:", parent)
        print("Depth array: ", depth)
        print("Children:")
        for u in range(N):
            if children[u]:
                print(f"  {u} -> {children[u]}")

    layer_count = 0

    if add_time_step_after_each_layer:
        for q in range(N):
            qc.delay(1, q)

    # Ready edges: children of already-activated qubits
    ready = [(root, c) for c in children[root]]

    while ready:
        used = set()
        sublayer = []
        next_ready = []

        # Greedy scheduling of non-conflicting ready edges
        for p, v in ready:
            if p in activated and (p not in used) and (v not in used):
                sublayer.append((p, v))
                used.add(p)
                used.add(v)
            else:
                next_ready.append((p, v))

        if not sublayer:
            raise RuntimeError("No schedulable edges found; scheduling got stuck.")

        # Print exact CX gates in this sublayer
        if do_print:
            layer_count += 1
            print(f"CX layer {layer_count}: {sublayer}")

        newly_activated = []
        for p, v in sublayer:
            qc.cx(p, v)
            newly_activated.append(v)

        for v in newly_activated:
            activated.add(v)

        # Newly activated qubits can now start spawning children
        for v in newly_activated:
            for c in children[v]:
                next_ready.append((v, c))

        if add_time_step_after_each_layer:
            for q in range(N):
                qc.delay(1, q)

        ready = next_ready

    if do_print:
        print(f"Total CX layers: {layer_count}")
        ghz_on_bfs_tree_with_time_steps._has_printed_schedule = True

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


# -----------------------------
# Simulation helpers
# -----------------------------
def run_counts(
    tqc: QuantumCircuit,
    shots: int,
    noise_model: Optional[NoiseModel],
    seed: Optional[int] = None,
) -> Dict[str, int]:
    sim = AerSimulator(method="stabilizer", noise_model=noise_model, seed_simulator=seed)
    return sim.run(tqc, shots=shots).result().get_counts()



def fidelity_from_params(
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
    nm = make_noise_model(dt=dt, T1=T1, Tphi=Tphi, p_cx_local=p_cx)
    countsZ = run_counts(tqcZ, shots=shots, noise_model=nm, seed=seed)
    countsX = run_counts(tqcX, shots=shots, noise_model=nm, seed=seed + 1)

    Xexp = parity_expectation_from_counts(countsX)
    p0, p1 = p0N_p1N(countsZ, N)
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
    N = 32
    shots = 2000
    dt = 1.0
    seed = 1234

    n_p = 12
    p_min = 0.001
    p_max = 0.03

    n_t = 12
    t_min = 40.0
    t_max = 2000.0

    outdir = Path(r"C:\Users\Olof hildeberg\Documents\Karriär\Skola\Masterarbete\Demoncode\Allerrors")
    outdir.mkdir(parents=True, exist_ok=True)
    # =========================

    side = math.ceil(math.sqrt(8 * N / 5)) + 2
    nr = nc = side
    full = heavy_hex_like_graph(nr, nc)
    g = {i: [v for v in full[i] if v < N] for i in range(N)}
    coupling = graph_to_coupling_map(g)

    prep = ghz_on_bfs_tree_with_time_steps(
    g,
    root=None,
    add_time_step_after_each_layer=True,
    print_schedule_once=True,
    )

    sim = AerSimulator(method="stabilizer")
    qcZ = measure_in_basis(prep, "Z")
    qcX = measure_in_basis(prep, "X")

    

    tqcZ = transpile(
        qcZ,
        sim,
        coupling_map=coupling,
        optimization_level=0,
        layout_method="trivial",
        routing_method="none",
    )

    ops = tqcZ.count_ops()

    print("\n--- Circuit summary ---")
    print("Depth:", tqcZ.depth())
    print("CX gates:", ops.get("cx",0))
    print("Delays:", ops.get("delay",0))

    tqcX = transpile(
        qcX,
        sim,
        coupling_map=coupling,
        optimization_level=0,
        layout_method="trivial",
        routing_method="none",
    )

    

    p_vals = linspace(p_min, p_max, n_p)
    t_vals = logspace(t_min, t_max, n_t)

    cx_F: list[float] = []
    tphi_F: list[float] = []
    t1_F: list[float] = []

    print(f"Running non-adaptive hardware-tree sweeps with N={N}, shots={shots}")

    print("\nSweep 1/3: fidelity vs CX error")
    for i, p in enumerate(p_vals, start=1):
        F = fidelity_from_params(
            tqcZ, tqcX, N, shots,
            dt=dt,
            T1=float("inf"),
            Tphi=float("inf"),
            p_cx=p,
            seed=seed + 1000 + i,
        )
        cx_F.append(F)
        print(f"  [{i:02d}/{len(p_vals)}] p_cx={p:.5f}  F~={F:.6f}")

    print("\nSweep 2/3: fidelity vs Tphi (pure dephasing only)")
    for i, Tphi in enumerate(t_vals, start=1):
        F = fidelity_from_params(
            tqcZ, tqcX, N, shots,
            dt=dt,
            T1=float("inf"),
            Tphi=Tphi,
            p_cx=0.0,
            seed=seed + 2000 + i,
        )
        tphi_F.append(F)
        print(f"  [{i:02d}/{len(t_vals)}] Tphi={Tphi:.3f}  F~={F:.6f}")

    print("\nSweep 3/3: fidelity vs T1 (relaxation only, twirled)")
    for i, T1 in enumerate(t_vals, start=1):
        F = fidelity_from_params(
            tqcZ, tqcX, N, shots,
            dt=dt,
            T1=T1,
            Tphi=float("inf"),
            p_cx=0.0,
            seed=seed + 3000 + i,
        )
        t1_F.append(F)
        print(f"  [{i:02d}/{len(t_vals)}] T1={T1:.3f}  F~={F:.6f}")

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(p_vals, cx_F, marker="o", linewidth=2, label="CX error")
    plt.xlabel("Error probability")
    plt.ylabel("Fidelity estimate")
    plt.title(f"Non-adaptive GHZ: Fidelity vs CX error (N={N})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig1 = outdir / "nonadaptive_fidelity_vs_cx.png"
    plt.savefig(fig1, dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(t_vals, tphi_F, marker="o", linewidth=2, label=r"$T_\phi$")
    plt.plot(t_vals, t1_F, marker="s", linewidth=2, label=r"$T_1$")
    plt.xscale("log")
    plt.xlabel("Time constant (ticks)")
    plt.ylabel("Fidelity estimate")
    plt.title(f"Non-adaptive GHZ: Fidelity vs $T_\\phi$ and $T_1$ (N={N})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig2 = outdir / "nonadaptive_fidelity_vs_tphi_and_t1.png"
    plt.savefig(fig2, dpi=180, bbox_inches="tight")
    plt.close()

    print("\nSaved plots:")
    print(f"  {fig1}")
    print(f"  {fig2}")


if __name__ == "__main__":
    main()
