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
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import pauli_error


# ============================================================
# Metrics
# ============================================================
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


# ============================================================
# Twirled idle decoherence + local entangling-gate noise
# ============================================================
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



def make_noise_model(
    dt: float = 300.0,
    T1: float = float("inf"),
    Tphi: float = float("inf"),
    p_cx_local: float = 0.0,
) -> NoiseModel:
    """
    Noise model used for the simulator sweeps.

    1) Each explicit delay instruction represents dt ns of idle time.
    2) After each CX, both participating qubits independently receive a
       one-qubit depolarizing Pauli error with probability p_cx_local.
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


# ============================================================
# Graph utilities
# ============================================================
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


# ============================================================
# Simulation helpers
# ============================================================
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
    shots: int,
    dt: float,
    T1: float,
    Tphi: float,
    p_cx: float,
    seed: int,
) -> float:
    nm = make_noise_model(
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


# ============================================================
# Backend/simulation setup
# ============================================================
def get_backend_if_requested(use_real_backend: bool, backend_name: str):
    if not use_real_backend:
        return None

    # Import only when needed, so local simulations do not require this package.
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    return service.backend(backend_name)



def make_synthetic_graph_for_N(N: int) -> Dict[int, List[int]]:
    side = math.ceil(math.sqrt(8 * N / 5)) + 8
    while True:
        g = heavy_hex_like_graph(side, side)
        if len(largest_connected_component(g)) >= N:
            return g
        side += 2



def transpile_for_mode(
    qc: QuantumCircuit,
    backend,
    coupling: Optional[CouplingMap],
    physical_layout: List[int],
) -> QuantumCircuit:
    """
    Backend-safe transpilation.

    If backend is a real IBM backend, do NOT also pass coupling_map or basis_gates.
    That avoids invalidating backend durations/error rates.
    """
    if backend is not None:
        return transpile(
            qc,
            backend=backend,
            initial_layout=physical_layout,
            optimization_level=0,
        )

    assert coupling is not None
    return transpile(
        qc,
        coupling_map=coupling,
        initial_layout=physical_layout,
        optimization_level=0,
        routing_method="none",
    )



def connectivity_diagnostics(
    graph: Dict[int, List[int]],
    schedule: GHZSchedule,
) -> None:
    """
    Print connectivity diagnostics for the physical qubits used by the schedule.

    This is useful for checking that the selected patch has the expected heavy-hex-like
    structure. In particular, it reports whether two selected neighboring physical
    qubits both have degree >= 3 inside the selected patch.
    """
    selected = set(schedule.physical_layout)
    phys_to_logical = {p: i for i, p in enumerate(schedule.physical_layout)}

    full_degree = {p: len(graph[p]) for p in selected}
    selected_neighbors = {
        p: sorted(v for v in graph[p] if v in selected)
        for p in selected
    }
    selected_degree = {p: len(selected_neighbors[p]) for p in selected}

    print("\nConnectivity diagnostics for selected physical qubits")
    print("Format: logical -> physical | degree in full graph | degree inside selected patch | selected neighbors")

    for logical, physical in enumerate(schedule.physical_layout):
        nbrs_phys = selected_neighbors[physical]
        nbrs_log = [phys_to_logical[n] for n in nbrs_phys]
        print(
            f"q[{logical:>3}] -> phys {physical:>4} | "
            f"full degree = {full_degree[physical]} | "
            f"selected degree = {selected_degree[physical]} | "
            f"neighbors phys = {nbrs_phys} | neighbors logical = {nbrs_log}"
        )

    selected_edges = []
    for u in selected:
        for v in selected_neighbors[u]:
            if u < v:
                selected_edges.append((u, v))

    high_high_selected = [
        (u, v)
        for u, v in selected_edges
        if selected_degree[u] >= 3 and selected_degree[v] >= 3
    ]
    high_high_full = [
        (u, v)
        for u, v in selected_edges
        if full_degree[u] >= 3 and full_degree[v] >= 3
    ]

    print("\nSelected patch summary")
    print(f"Number of selected physical qubits: {len(selected)}")
    print(f"Number of selected physical edges:  {len(selected_edges)}")
    print(f"Max degree inside selected patch:   {max(selected_degree.values()) if selected_degree else 0}")
    print(f"Max degree in full graph/backend:   {max(full_degree.values()) if full_degree else 0}")

    if high_high_selected:
        print("\nWARNING: Adjacent selected qubits both have selected-patch degree >= 3:")
        for u, v in high_high_selected:
            print(
                f"  phys {u} -- phys {v} "
                f"(logical {phys_to_logical[u]} -- logical {phys_to_logical[v]}), "
                f"selected degrees {selected_degree[u]} and {selected_degree[v]}"
            )
    else:
        print("\nOK: No adjacent selected qubits both have selected-patch degree >= 3.")

    if high_high_full:
        print("\nNote: Adjacent selected qubits both have full-graph/backend degree >= 3:")
        for u, v in high_high_full:
            print(
                f"  phys {u} -- phys {v} "
                f"(logical {phys_to_logical[u]} -- logical {phys_to_logical[v]}), "
                f"full degrees {full_degree[u]} and {full_degree[v]}"
            )
    else:
        print("No adjacent selected qubits both have full-graph/backend degree >= 3.")




def print_parent_child_tree(schedule: GHZSchedule) -> None:
    """
    Print the generated GHZ tree as parent -> children.

    Here parent and child refer to the logical qubits in the generated circuit.
    The corresponding physical qubits are also printed, using schedule.physical_layout.
    """
    N = len(schedule.physical_layout)
    children: Dict[int, List[Tuple[int, int]]] = {q: [] for q in range(N)}

    for parent, child, layer in zip(schedule.source_list, schedule.target_list, schedule.layer_list):
        children[parent].append((child, layer))

    print("\nParent -> children GHZ tree")
    print("Format: logical parent / physical parent -> logical child / physical child (CX layer)")

    root_logical = schedule.physical_layout.index(schedule.root_physical)
    print(f"Root: logical {root_logical} / physical {schedule.root_physical}")

    for parent in range(N):
        if not children[parent]:
            continue

        parent_phys = schedule.physical_layout[parent]
        child_strings = []
        for child, layer in sorted(children[parent], key=lambda item: (item[1], item[0])):
            child_phys = schedule.physical_layout[child]
            child_strings.append(
                f"logical {child} / physical {child_phys} (layer {layer})"
            )

        print(
            f"logical {parent} / physical {parent_phys} -> "
            + ", ".join(child_strings)
        )

    print("\nParent-child summary")
    num_parents = sum(1 for q in range(N) if children[q])
    max_children = max((len(v) for v in children.values()), default=0)
    parents_with_multiple_children = [q for q in range(N) if len(children[q]) > 1]

    print(f"Number of parent qubits:          {num_parents}")
    print(f"Maximum children of one parent:   {max_children}")
    print(f"Parents with >1 child:            {parents_with_multiple_children}")
# ============================================================
# Main
# ============================================================
def main() -> None:
    # -------------------------
    # User-facing settings
    # -------------------------
    use_real_backend = False
    ibm_backend_name = "ibm_brisbane"

    N = 21
    shots = 100
    seed = 1234

    # Coarse timing model used in the thesis simulations.
    # One explicit delay instruction represents one entangling-gate layer.
    delay_tick_ns = 300.0

    # Coherence sweep range: 20 us to 500 us, expressed in ns.
    T_min_ns = 20_000.0
    T_max_ns = 500_000.0
    n_sweep_points = 18

    p_min = 0.001
    p_max = 0.03

    # Search controls for the automatic IBM-style GHZ schedule.
    # Increase these to spend more classical time searching for fewer CX layers.
    num_roots_to_try = 80
    greedy_trials_per_root = 40
    random_seed_for_layout_search = 2026
    # =========================

    backend = get_backend_if_requested(use_real_backend, ibm_backend_name)

    if backend is None:
        graph = make_synthetic_graph_for_N(N)
        coupling = graph_to_coupling_map(graph)
        print("Using synthetic heavy-hex-like graph for simulation.")
    else:
        graph = graph_from_backend(backend)
        coupling = None
        print(f"Using real backend topology from: {ibm_backend_name}")

    schedule = find_low_depth_ghz_schedule(
        graph,
        N=N,
        num_roots=num_roots_to_try,
        trials_per_root=greedy_trials_per_root,
        seed=random_seed_for_layout_search,
    )

    layer_sizes = [0] * schedule.num_layers
    for layer in schedule.layer_list:
        layer_sizes[layer] += 1

    print("\nGenerated IBM-style GHZ schedule")
    print(f"N logical qubits:             {N}")
    print(f"Physical root qubit:          {schedule.root_physical}")
    print(f"Physical layout:              {schedule.physical_layout}")
    print(f"Number of CX gates:           {len(schedule.source_list)}")
    print(f"Number of CX layers:          {schedule.num_layers}")
    print(f"CX gates per layer:           {layer_sizes}")
    print(f"Modeled idle time:            {schedule.num_layers * delay_tick_ns:.1f} ns")
    print(f"source_list = {schedule.source_list}")
    print(f"target_list = {schedule.target_list}")
    print(f"layer_list  = {schedule.layer_list}\n")

    print_parent_child_tree(schedule)
    connectivity_diagnostics(graph, schedule)

    prep = build_layered_ghz_from_schedule(schedule)
    qcZ = measure_in_basis(prep, "Z")
    qcX = measure_in_basis(prep, "X")

    tqcZ = transpile_for_mode(qcZ, backend, coupling, schedule.physical_layout)
    tqcX = transpile_for_mode(qcX, backend, coupling, schedule.physical_layout)

    base_dir = Path(__file__).resolve().parent
    outdir = base_dir / "Sweep_results"
    outdir.mkdir(parents=True, exist_ok=True)

    p_vals = linspace(p_min, p_max, n_sweep_points)
    T_vals = geomspace(T_min_ns, T_max_ns, n_sweep_points)

    print(f"Running non-adaptive automatic IBM-style sweeps with shots={shots}, N={N}")

    # -------------------------
    # Sweep 1: entangling-gate error only
    # -------------------------
    cx_fids = []
    for i, p in enumerate(p_vals):
        F_cx = fidelity_from_params(
            tqcZ, tqcX, N, shots, delay_tick_ns,
            T1=float("inf"), Tphi=float("inf"),
            p_cx=p,
            seed=seed + 1000 + 10 * i,
        )
        cx_fids.append(F_cx)
        print(f"p_ent={p:.4f} -> F={F_cx:.6f}")

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(p_vals, cx_fids, marker="s", color="tab:orange", label="Entangling-gate error only")
    plt.xlabel("Entangling-gate error probability")
    plt.ylabel("GHZ fidelity")
    plt.title(f"Auto IBM-style non-adaptive fidelity vs entangling-gate error (shots={shots}, N={N})")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="lower left")
    plt.tight_layout()
    fig1 = outdir / f"auto_ibm_non_adaptive_fidelity_vs_entangling_error_{N}N.png"
    plt.savefig(fig1, dpi=180)
    plt.close()

    # -------------------------
    # Sweep 2: Tphi and T1
    # -------------------------
    tphi_fids = []
    t1_fids = []

    for i, T in enumerate(T_vals):
        F_tphi = fidelity_from_params(
            tqcZ, tqcX, N, shots, delay_tick_ns,
            T1=float("inf"), Tphi=T,
            p_cx=0.0,
            seed=seed + 2000 + 10 * i,
        )
        tphi_fids.append(F_tphi)
        print(f"Tphi={T:.3f} ns -> F={F_tphi:.6f}")

        F_t1 = fidelity_from_params(
            tqcZ, tqcX, N, shots, delay_tick_ns,
            T1=T, Tphi=float("inf"),
            p_cx=0.0,
            seed=seed + 3000 + 10 * i,
        )
        t1_fids.append(F_t1)
        print(f"T1  ={T:.3f} ns -> F={F_t1:.6f}")

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(T_vals, tphi_fids, marker="o", label=r"Pure dephasing only ($T_\phi$ finite)")
    plt.plot(T_vals, t1_fids, marker="s", label=r"Relaxation only ($T_1$ finite)")
    plt.xscale("log")
    plt.xlabel("Time constant (ns)")
    plt.ylabel("GHZ fidelity")
    plt.title(fr"Auto IBM-style non-adaptive fidelity vs $T_\phi$ and $T_1$ (shots={shots}, N={N})")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig2 = outdir / f"auto_ibm_non_adaptive_fidelity_vs_tphi_and_t1_{N}N.png"
    plt.savefig(fig2, dpi=180)
    plt.close()

    print("\nSaved:")
    print(fig1)
    print(fig2)


if __name__ == "__main__":
    main()
