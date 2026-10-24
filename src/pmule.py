from jupyter_tikz import TexFragment
from numpyarray_to_latex.jupyter import to_jup
from numpyarray_to_latex import to_ltx
from IPython.display import display, Image, SVG, Latex, Markdown, Math, HTML
from functools import lru_cache, reduce
from itertools import chain, product, islice
import math
from math import ceil
from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def highlight_first_minimum(row):
    styles = pd.Series('', index=row.index)
    
    # 2. Find the index (column label) of the first minimum value
    # idxmin() returns the index label of the first occurrence of the minimum value
    first_min_col_label = row.idxmin()
    
    # 3. Use the found index label to apply the highlight style to only that position
    highlight = "background-color: yellow"
    styles[first_min_col_label] = highlight
    
    # 4. Return the styles as a list
    return styles.tolist()


def colorea_minimo_fila(df):
    return df.style.apply(highlight_first_minimum, axis=1).format('{:.0f}', na_rep='')


def highlight_maximum(column):
    highlight = "background-color: greenyellow;"
    default = ""

    maximum_in_column = column.max()

    # must return one string per cell in this column
    return [highlight if v == maximum_in_column else default for v in column]


def highlight_minimum(row):
    highlight = "background-color: yellow;"
    default = ""

    minimum_in_row = row.min()

    # must return one string per cell in this column
    return [highlight if v == minimum_in_row else default for v in row]


def highlight_cur(value):
    return "background-color: white;  border:1px solid black;"


def truncate_and_pad(value, n_decimal_places):
    def truncate_float(value, n_digits):
        if n_digits < 0:
            raise ValueError("n_digits must be non-negative.")
        #multiplier = 10**n_digits
        #truncated_value = math.trunc(value * multiplier) / multiplier
        rounded_value = round(value, n_digits)
        return int(rounded_value) if n_digits == 0 else rounded_value
        #return int(truncated_value) if n_digits == 0 else truncated_value

    if isinstance(value, str):
        if not value.strip():          # casilla vacía en filas/columnas extra
            return value
        value = float(value)
    elif (
        isinstance(value, np.int64)
        or isinstance(value, np.int32)
        or isinstance(value, np.int16)
        or isinstance(value, np.int8)
    ):
        value = int(value)
    elif (
        value is None
        or np.isnan(value)
    ):
        value = 0.0

    truncated = truncate_float(value, n_decimal_places)
    format_string = f"{{:.{n_decimal_places}f}}"
    return format_string.format(truncated)


def read_rcp(filename):
    data = {
        "num_activities": 0,
        "activities": [],
        "num_resources": 0,
        "resource_availability": [],
        "resources": None,
        "predecessors": {},
    }

    with open(filename, "r") as file:
        lines = file.readlines()

        first_line = lines[0].strip().split()
        data["num_activities"] = int(first_line[0])
        data["num_resources"] = int(first_line[1])

        # Calculate the number of digits required for zero-padding
        num_digits = len(str(data["num_activities"]))

        data["predecessors"] = {
            f"A{str(iteration + 1).zfill(num_digits)}": []
            for iteration in range(data["num_activities"])
        }
        second_line = lines[1].strip().split()

        resource_columns = [f"Resource {i + 1}" for i in range(data["num_resources"])]
        data["resource_availability"] = pd.DataFrame(
            [int(avail) for avail in second_line],
            columns=["Availability"],
            index=resource_columns,
        )
        resource_data = []

        # Following lines: Activity details
        line_index = 2
        while line_index < len(lines):
            line = lines[line_index].strip().split()
            if line:
                # Zero-padded activity name
                activity_name = f"A{str(len(data['activities']) + 1).zfill(num_digits)}"

                # Parse activity details
                duration = int(line[0])
                resource_requirements = [
                    int(req) for req in line[1 : data["num_resources"] + 1]
                ]
                num_successors = int(line[data["num_resources"] + 1])
                successors = [
                    int(line[i]) for i in range(data["num_resources"] + 2, len(line))
                ]

                # Store activity details
                activity = {
                    "name": activity_name,
                    "duration": duration,
                    "resource_requirements": resource_requirements,
                    "successors": successors,
                }
                data["activities"].append(activity)

                # Calculate predecessors based on successors
                for succ in successors:
                    successor_name = f"A{str(succ).zfill(num_digits)}"
                    if successor_name not in data["predecessors"]:
                        data["predecessors"][successor_name] = []
                    data["predecessors"][successor_name].append(activity_name)

                # Append resource requirements for DataFrame
                resource_data.append(resource_requirements)

            line_index += 1

        # Create resource DataFrame
        data["resources"] = pd.DataFrame(
            resource_data,
            columns=resource_columns,
            index=[activity["name"] for activity in data["activities"]],
        )

        # Convert predecessors values to comma-separated strings
        for key in data["predecessors"]:
            data["predecessors"][key] = ",".join(data["predecessors"][key])

        activity_durations = {
            activity["name"]: activity["duration"] for activity in data["activities"]
        }
        data["duration"] = pd.DataFrame.from_dict(
            activity_durations, orient="index", columns=["duration"]
        )

    return data


def significant_values(values, threshold):
    values = np.absolute(values)
    percents = values.cumsum() / values.sum()
    return int(np.argmax(percents > threshold))


# ---------------------------------------------------------------------------
# Fondahl/Elmaghraby direct AoA construction
# ---------------------------------------------------------------------------

from collections import defaultdict as _defaultdict


def fondahl_construct(immediate_predecessors: dict) -> tuple:
    """Build an Activity-on-Arrow PERT network using the Fondahl (1961) / Elmaghraby (1977)
    direct construction algorithm.

    Unlike the Roy/augmented approach (one dummy per predecessor link followed by
    simplification), Fondahl builds a network with few dummies directly in a single
    topological forward pass (not the minimum: that problem is NP-hard, and
    ``_simplify_local`` usually removes many more afterwards):

    * Activities with a single predecessor are attached directly to that predecessor's
      head node (no dummy needed) whenever that node has not yet been used by another
      activity as its tail — Rule 1 (direct attachment).
    * Activities whose predecessor set has already been seen get the *same* convergence
      node, sharing it without an extra dummy — Rule 2 (convergence node reuse).
    * Otherwise a fresh convergence node is created and one dummy per predecessor is
      added (the minimum necessary to represent the convergence).

    Parameters
    ----------
    immediate_predecessors : dict[str, set[str]]
        Mapping of activity name → set of its *immediate* (direct) predecessors.
        Activities with no predecessors map to the empty set.
        The dict need not be pre-sorted; topological order is computed internally
        using Kahn's algorithm in O(V + E).

    Returns
    -------
    nodes : set
        Integer node ids used in the graph (SOURCE = 0, SINK = 1, others ≥ 2).
    real_arcs : list of (tail_node, head_node, activity_name)
        One entry per actual activity.
    dummy_arcs : list of (tail_node, head_node)
        One entry per dummy arrow (including source/sink connectors).
    """
    # --- Kahn's topological sort (O(V + E)) ---------------------------------
    successors: dict = _defaultdict(list)
    for act, preds in immediate_predecessors.items():
        for p in preds:
            successors[p].append(act)
    in_degree = {a: len(p) for a, p in immediate_predecessors.items()}
    queue = [a for a, d in in_degree.items() if d == 0]
    topo = []
    while queue:
        a = queue.pop()
        topo.append(a)
        for b in successors[a]:
            in_degree[b] -= 1
            if in_degree[b] == 0:
                queue.append(b)

    # --- node counter --------------------------------------------------------
    _next_node = iter(range(10_000))
    SOURCE = next(_next_node)   # node 0
    SINK = next(_next_node)     # node 1

    head: dict = {}             # activity  → its head node
    tail: dict = {}             # activity  → its tail node
    real_arcs: list = []
    dummy_arcs: list = []
    convergence_node: dict = {} # frozenset(preds) → shared convergence node
    head_users: dict = _defaultdict(set)  # node → set of activities using it as tail

    def get_tail(preds: frozenset) -> int:
        """Return the tail node for an activity with the given predecessor set."""
        if not preds:
            return SOURCE

        # Reuse an existing convergence node for this exact predecessor set
        if preds in convergence_node:
            return convergence_node[preds]

        # Rule 1 – single predecessor: reuse its head node if still free
        if len(preds) == 1:
            (p,) = preds
            candidate = head[p]
            if not head_users[candidate]:   # node not yet used as a tail
                convergence_node[preds] = candidate
                return candidate

        # Rule 2 – create a fresh convergence node + one dummy per predecessor
        node = next(_next_node)
        convergence_node[preds] = node
        for p in preds:
            dummy_arcs.append((head[p], node))
        return node

    for act in topo:
        preds = frozenset(immediate_predecessors[act])
        t = get_tail(preds)
        h = next(_next_node)
        tail[act] = t
        head[act] = h
        head_users[t].add(act)
        real_arcs.append((t, h, act))

    # Connect dangling head nodes to SINK
    all_tails = set(tail.values()) | {n for n, _ in dummy_arcs}
    for act in topo:
        if head[act] not in all_tails:
            dummy_arcs.append((head[act], SINK))

    nodes = {SOURCE, SINK}
    for t, h, _ in real_arcs:
        nodes |= {t, h}
    for t, h in dummy_arcs:
        nodes |= {t, h}

    return nodes, real_arcs, dummy_arcs


def _simplify_sweep(project: "ProjectGraph") -> tuple:
    """Single non-resetting pass over all dummies.

    Unlike ProjectGraph.simplify() which resets the dummy list on every
    successful removal (O(D²) total dummy work), this function iterates
    the snapshot dummy list exactly once and returns the updated project
    together with the number of dummies removed.

    Run in a loop until the count is 0 (usually 1-2 passes after Fondahl).

    Returns
    -------
    (updated_project, n_removed)
    """
    removed = 0
    for dummy in list(project.dummies):   # snapshot — no reset
        updated, project = ProjectGraph.assess_dummy(dummy, project)
        if updated:
            removed += 1
    return project, removed


def _simplify_legacy(project: "ProjectGraph") -> "ProjectGraph":
    """Simplificación ANTIGUA (hasta 2026-10): _simplify_sweep hasta punto fijo.

    Ya no se usa en create_project_fondahl (sustituida por _simplify_local).
    Se conserva como referencia y para comparar resultados: es correcta pero
    lenta, porque cada ficticia se valida recalculando las predecesoras
    distantes de todo el proyecto dos veces (contraer / eliminar), lo que da
    un coste del orden de n³ (120 actividades ≈ 100 s).
    """
    removed = 1
    while removed:
        project, removed = _simplify_sweep(project)
    return project


def _simplify_local(project: "ProjectGraph") -> "ProjectGraph":
    """Elimina o contrae ficticias redundantes con comprobaciones LOCALES.

    Toma exactamente las mismas decisiones, en el mismo orden, que
    _simplify_legacy (y por tanto produce el mismo grafo), pero valida cada
    ficticia con operaciones locales en lugar de recalcular las predecesoras
    distantes de todo el proyecto. Pasa de ~n³ a casi lineal (500
    actividades: horas → décimas de segundo).

    Qué hacía la versión antigua (y se reproduce aquí)
    -------------------------------------------------
    Pasadas sobre las ficticias en orden alfabético de nombre, hasta que una
    pasada no cambia nada. Para cada ficticia d = (u → v):
      1) intenta CONTRAERLA: nx.contracted_nodes funde u en v (las entradas y
         salidas de u pasan a v; si ya existía un arco con los mismos
         extremos se conserva el existente y se descarta el de u);
      2) si no vale, intenta ELIMINARLA (borrar el arco);
    y acepta la operación si el grafo sigue siendo acíclico, con un único
    origen y un único sumidero, y las predecesoras distantes de cada
    actividad real no cambian.

    Idea clave — el conjunto A(n)
    -----------------------------
    Para cada nudo n:

        A(n) = actividades REALES que deben haber terminado para alcanzar n
             = ⋃ sobre los arcos (p → n):  A(p) ∪ ({actividad del arco} si es real)

    Las predecesoras distantes de una actividad real X son exactamente
    A(cola(X)). Por tanto, una operación conserva el proyecto si y solo si
    no cambia A(n) en ningún nudo del que salga una actividad real. A se
    calcula una vez en orden topológico y se guarda como bitset (un entero
    de Python: bit i = actividad real i), así que uniones, diferencias y
    comparaciones son operaciones de bits.

    Comprobación de la CONTRACCIÓN de d = (u → v)
    ---------------------------------------------
      a) Ninguna actividad real puede perderse: si un arco real de u choca
         con uno ya existente de v (mismo vecino), contracted_nodes lo
         descartaría → rechazar.
      b) Acíclico: si hay otro camino u ⇝ v además de d, fundir u y v cierra
         un ciclo → rechazar. Se busca con un DFS desde los sucesores de u,
         podado a nudos anteriores a v en el orden topológico. Sin ese otro
         camino tampoco pueden aparecer orígenes ni sumideros nuevos.
      c) Semántica: A(v) no cambia (lo que llega a u ya llegaba a v por d).
         Lo que cambia es lo que sale de u, que pasa a salir de v:
           - una actividad real que salía de u pasa a tener A(v) en vez de
             A(u) → hace falta A(u) == A(v);
           - una ficticia (u → w) aporta ahora A(v) a w: A(w) puede crecer.
         Ese crecimiento se propaga aguas abajo (solo puede crecer, nunca
         menguar); si llega a un nudo del que sale una actividad real →
         rechazar. Si no, se acepta y se actualizan los A que crecieron.

    Comprobación de la ELIMINACIÓN de d = (u → v)
    ---------------------------------------------
      a) u necesita otra salida y v otra entrada (si no, aparece un origen o
         un sumidero nuevo). Borrar un arco nunca crea ciclos.
      b) Semántica: A'(v) = unión de las demás entradas de v. Si es menor
         que A(v), la disminución se propaga aguas abajo (recalculando cada
         nudo afectado desde sus entradas, en orden topológico); si llega a
         un nudo del que sale una actividad real → rechazar.

    Las propagaciones se detienen en cuanto un nudo no cambia, así que en la
    práctica tocan muy pocos nudos. Tras cada operación aceptada se recalcula
    el orden topológico y el índice nombre → arco (O(V + E)).

    El problema de mínimo número de ficticias es NP-duro: esto es un voraz
    que llega a un mínimo local y su resultado depende del orden (por eso se
    respeta el orden alfabético de la versión antigua).
    """
    import heapq

    G = project.pert_graph.copy()

    def is_dummy_name(name: str) -> bool:
        return name.startswith("@")

    real = sorted({d["activity"] for _, _, d in G.edges(data=True)
                   if not is_dummy_name(d["activity"])})
    bit = {name: 1 << i for i, name in enumerate(real)}   # ficticias → 0

    def contribution(p, n, a_p) -> int:
        """Lo que el arco (p → n) aporta a A(n), si A(p) vale a_p."""
        return a_p | bit.get(G.edges[p, n]["activity"], 0)

    def has_real_out(n) -> bool:
        return any(not is_dummy_name(d) for _, _, d in G.out_edges(n, data="activity"))

    pos, where = {}, {}

    def reindex():
        pos.clear()
        pos.update({n: i for i, n in enumerate(nx.topological_sort(G))})
        where.clear()
        where.update({d: (s, t) for s, t, d in G.edges(data="activity")})

    reindex()
    A = {}
    for n in sorted(pos, key=pos.get):
        acc = 0
        for p in G.predecessors(n):
            acc |= contribution(p, n, A[p])
        A[n] = acc

    def propagate_growth(seeds: dict):
        """seeds: nudo → A nuevo (⊇ A actual). Devuelve los A nuevos o None."""
        new = {}
        heap = []
        for n, val in seeds.items():
            if val != A[n]:
                new[n] = val
                heapq.heappush(heap, (pos[n], n))
        while heap:
            _, n = heapq.heappop(heap)
            if has_real_out(n):
                return None
            for s in G.successors(n):
                val = new.get(s, A[s]) | contribution(n, s, new[n])
                if val != new.get(s, A[s]):
                    if s not in new:
                        heapq.heappush(heap, (pos[s], s))
                    new[s] = val
        return new

    def fit_merged(u, v):
        """Tras fundir u en v, el orden topologico sigue valiendo si v ocupa el puesto de v o el de u
        entre sus predecesores y sucesores; si no, se recalcula entero (raro)."""
        pos_u = pos.pop(u)
        for x in (pos[v], pos_u):
            if all(pos[p] < x for p in G.predecessors(v)) and all(x < pos[w] for w in G.successors(v)):
                pos[v] = x
                return
        pos.clear()
        pos.update({n: i for i, n in enumerate(nx.topological_sort(G))})

    def try_contract(u, v) -> bool:
        # (a) ningún arco real de u puede chocar con uno existente de v
        for p in G.predecessors(u):
            if G.has_edge(p, v) and not is_dummy_name(G.edges[p, u]["activity"]):
                return False
        for w in G.successors(u):
            if w != v and G.has_edge(v, w) and not is_dummy_name(G.edges[u, w]["activity"]):
                return False
        # (b) otro camino u ⇝ v → ciclo
        stack = [w for w in G.successors(u) if w != v]
        seen = set(stack)
        while stack:
            n = stack.pop()
            if n == v:
                return False
            for s in G.successors(n):
                if s not in seen and pos[s] <= pos[v]:
                    seen.add(s)
                    stack.append(s)
        # (c) semántica
        seeds = {}
        for w in G.successors(u):
            if w == v:
                continue
            if is_dummy_name(G.edges[u, w]["activity"]):
                seeds[w] = seeds.get(w, A[w]) | A[v]
            elif A[u] != A[v]:
                return False
        new = propagate_growth(seeds)
        if new is None:
            return False
        # nombres de los arcos de u y v antes de fundirlos (un arco de u puede pisar a uno de v)
        antes = [d for x in (u, v) for _, _, d in G.in_edges(x, data="activity")]
        antes += [d for x in (u, v) for _, _, d in G.out_edges(x, data="activity")]
        nx.contracted_nodes(G, v, u, self_loops=False, copy=False)
        A.update(new)
        del A[u]
        for d in antes:
            where.pop(d, None)
        for s_, t_, d in G.in_edges(v, data="activity"):
            where[d] = (s_, t_)
        for s_, t_, d in G.out_edges(v, data="activity"):
            where[d] = (s_, t_)
        fit_merged(u, v)
        return True

    def try_remove(u, v) -> bool:
        if G.out_degree(u) < 2 or G.in_degree(v) < 2:
            return False
        # Cada nudo se recalcula al sacarlo del heap (orden topológico), cuando
        # ya están resueltos todos sus predecesores afectados.
        new, queued, heap = {}, {v}, [(pos[v], v)]
        while heap:
            _, n = heapq.heappop(heap)
            val = 0
            for p in G.predecessors(n):
                if (p, n) != (u, v):
                    val |= contribution(p, n, new.get(p, A[p]))
            if val == A[n]:
                continue                 # no cambia: la propagación se corta aquí
            if has_real_out(n):
                return False
            new[n] = val
            for s in G.successors(n):
                if s not in queued:
                    queued.add(s)
                    heapq.heappush(heap, (pos[s], s))
        G.remove_edge(u, v)
        A.update(new)
        return True

    changed = True
    while changed:
        changed = False
        for name in sorted(where):
            if not is_dummy_name(name) or name not in where:
                continue
            u, v = where[name]
            if try_contract(u, v):
                changed = True
            elif try_remove(u, v):
                del where[name]              # quitar un arco no invalida el orden topologico
                changed = True
    return ProjectGraph(G)


class PredecessorTable:
    @staticmethod
    def from_dict_of_strings(data: Dict[str, str], simplify=True):
        def replace_and_split(x):
            result = set(
                x.replace(" ", "").replace("-", "").replace(";", ",").split(",")
            ) - {""}
            return result

        dict_of_sets = {key: replace_and_split(value) for key, value in data.items()}
        return PredecessorTable.from_dict_of_sets(data=dict_of_sets, simplify=simplify)

    @staticmethod
    def from_dict_of_sets(
        data: Dict[str, Set[str]], simplify=True
    ) -> "PredecessorTable":
        names = chain.from_iterable(data.values())
        expanded_dict = {key: set() for key in names}
        expanded_dict.update(data)
        return PredecessorTable(data=expanded_dict, simplify=simplify)

    @staticmethod
    def from_dataframe_of_strings(
        data: pd.DataFrame, activity: str, predecessor: str, simplify=True
    ) -> "PredecessorTable":
        dataframe = data.reset_index().copy(deep=True)
        miniframe = dataframe.loc[:, [activity, predecessor]]
        miniframe.set_index(activity, inplace=True)
        dict_of_sets = miniframe.to_dict()
        return PredecessorTable.from_dict_of_strings(
            data=dict_of_sets[predecessor], simplify=simplify
        )

    @staticmethod
    def from_project(project, dummies=False) -> "PredecessorTable":
        new_data = project.distant_predecessor(dummies=dummies)
        return PredecessorTable.from_dict_of_sets(data=new_data, simplify=True)

    def __init__(self, data, simplify=True):
        self.immediate_predecessor = dict(sorted(data.items()))
        self._check_acyclic()
        # distant_predecessor MUST be calculated before immediate_predecessor. It is used by calculate_immediate_predecessor_of
        # Las clausuras se calculan con enteros de Python como conjuntos de bits (una sola vez) y se convierten a conjuntos al final.
        nombres, masks = self._closure_masks(self.immediate_predecessor.keys())
        self.distant_predecessor = dict(
            sorted(
                {n: self._bits_to_set(m, nombres) for n, m in masks.items()}.items()
            )
        )
        if simplify:
            # inmediatas = ancestros que no son ancestros de otro predecesor directo
            # (la union de los ancestros de todos los ancestros coincide con la de los predecesores directos)
            inmediatas = {}
            for key, directas in self.immediate_predecessor.items():
                cubiertas = 0
                for a in directas:
                    cubiertas |= masks[a]
                inmediatas[key] = self._bits_to_set(masks[key] & ~cubiertas, nombres)
            self.immediate_predecessor = dict(sorted(inmediatas.items()))

    def _check_acyclic(self) -> None:
        """Lanza ValueError si la tabla de precedencias contiene un ciclo
        (incluida una actividad predecesora de sí misma)."""
        g = nx.DiGraph()
        g.add_nodes_from(self.immediate_predecessor.keys())
        for activity, predecessors in self.immediate_predecessor.items():
            for ancestor in predecessors:
                g.add_edge(ancestor, activity)
        if nx.is_directed_acyclic_graph(g):
            return
        cycle = nx.find_cycle(g)
        chain_str = " → ".join([cycle[0][0]] + [t for _, t in cycle])
        raise ValueError(
            f"La tabla de precedencias contiene un ciclo: {chain_str}. "
            "Revisa las prelaciones: ninguna actividad puede preceder(se) a sí misma."
        )

    @property
    def activity_names(self) -> Set[str]:
        return sorted(
            list(
                set(chain.from_iterable(self.immediate_predecessor.values()))
                | set(self.immediate_predecessor.keys())
            )
        )

    def _closure_masks(self, names):
        """Ancestros de cada actividad como enteros de Python usados como conjuntos de bits.

        Devuelve ``(nombres, memo)``: ``nombres[i]`` es la actividad del bit ``i`` (orden alfabético) y
        ``memo[actividad]`` la máscara de todos sus ancestros. Recorrido iterativo con memoria: cada
        actividad se resuelve una sola vez (un OR por arista, O(aristas·n/64)), de modo que el coste no
        crece con el número de rutas ni con el tamaño de los conjuntos de ancestros. No usa recursión
        (sin límite por la longitud de la cadena más larga) y detecta ciclos con ``ValueError``.
        """
        nombres = sorted(self.immediate_predecessor)
        bit = {n: i for i, n in enumerate(nombres)}
        memo: Dict[str, int] = {}
        visiting: Set[str] = set()
        for root in names:
            if root in memo:
                continue
            stack = [root]
            while stack:
                node = stack[-1]
                if node in memo:
                    stack.pop()
                    continue
                predecessors = self.immediate_predecessor[node]
                pending = [p for p in predecessors if p not in memo]
                if pending:
                    if node in visiting or any(p in visiting for p in pending):
                        raise ValueError(
                            f"La tabla de precedencias contiene un ciclo que pasa por '{node}'."
                        )
                    visiting.add(node)
                    stack.extend(pending)
                    continue
                mask = 0
                for ancestor in predecessors:
                    mask |= (1 << bit[ancestor]) | memo[ancestor]
                memo[node] = mask
                visiting.discard(node)
                stack.pop()
        return nombres, memo

    @staticmethod
    def _bits_to_set(mask: int, nombres) -> Set[str]:
        """Conjunto de actividades cuyos bits están activos en ``mask``."""
        return _bits_a_conjunto(mask, nombres)

    def _distant_closure(self, names) -> Dict[str, Set[str]]:
        """Conjunto de todos los ancestros de cada actividad de ``names`` (y de los ancestros que se
        hayan resuelto por el camino). Los conjuntos devueltos son independientes entre sí
        (se pueden modificar sin afectar a los demás)."""
        nombres, memo = self._closure_masks(names)
        return {n: self._bits_to_set(m, nombres) for n, m in memo.items()}

    def calculate_distant_predecessor_of(self, name: str) -> Set[str]:
        return set(self._distant_closure([name])[name])

    def calculate_immediate_predecessor_of(self, name: str) -> Dict[str, str]:
        predecessors = self.distant_predecessor[name]
        cubiertas = set().union(*(self.distant_predecessor[a] for a in predecessors))
        return predecessors - cubiertas

    @property
    def start_activities(self) -> Set[str]:
        result = {
            key for key, value in self.immediate_predecessor.items() if value == set()
        }
        return result

    @property
    def end_activities(self) -> Set[str]:
        predecessors = set(chain.from_iterable(self.immediate_predecessor.values()))
        result = {
            key
            for key, value in self.immediate_predecessor.items()
            if key not in predecessors
        }
        return result

    @property
    def augmented_predecessor(self) -> Dict[str, str]:
        result = {key: set() for key in self.immediate_predecessor.keys()}

        for activity, predecessors in self.immediate_predecessor.items():
            if predecessors == set():
                result[activity].add(f"@∇StartOfProject⤑Δ{activity}")
            else:
                for ancestor in predecessors:
                    dummy_name = f"@∇{ancestor}⤑Δ{activity}"
                    result[activity].add(dummy_name)
                    result[dummy_name] = {ancestor}
        for activity in self.end_activities:
            result[activity].add(f"@∇{activity}⤑ΔEndOfProject")
        return result

    @property
    def dummies(self):
        result = (
            self.augmented_predecessor.keys()
            | chain.from_iterable(self.augmented_predecessor.values())
        ) - set(self.activity_names)
        return result

    @property
    def nodes(self) -> Set[str]:
        result = list()
        for activity, predecessors in self.augmented_predecessor.items():
            result.append(activity)
            result.extend(predecessors)
        result = set(
            chain.from_iterable(name.split("⤑") for name in result if name[0] == "@")
        )
        return result

    def copy(self):
        return PredecessorTable(data=self.immediate_predecessor.copy())

    def create_project_roy(self, simplify=True):
        """Build a PERT graph using Roy's augmented-predecessor method + simplify.

        Each predecessor link becomes a dummy activity; SOURCE and SINK
        connectors are added; then simplify() contracts or removes safe
        dummies.  Produces correct results for most inputs but generates a
        directed cycle (incorrectly) when two activities share an identical
        non-empty predecessor set.  Use create_project_fondahl (or the
        default create_project) to avoid that bug.

        Parameters
        ----------
        simplify : bool, default True
            Apply simplify() to remove redundant dummies after construction.
        """
        def create_edge_info_for_dummy(name):
            return name.replace("@", "").split("⤑") + [{"activity": name}]

        edgelist_dummy_activities = list(
            create_edge_info_for_dummy(name)
            for name in chain.from_iterable(self.augmented_predecessor.values())
            if name[0] == "@"
        )

        edgelist_actual_activities = list(
            (f"Δ{name}", f"∇{name}", {"activity": name}) for name in self.activity_names
        )
        edgelist = edgelist_dummy_activities + edgelist_actual_activities

        pert_graph = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
        for source, target, edge_attributes in edgelist:
            pert_graph.edges[source, target]["activity"] = edge_attributes["activity"]
        if not simplify:
            return ProjectGraph(pert_graph)
        return ProjectGraph(pert_graph).simplify()

    def create_project(self, simplify=True):
        """Build a PERT graph — delegates to create_project_fondahl (default).

        Fondahl's direct construction never creates redundant dummies in the
        first place and avoids the cycle bug present in the original Roy method.
        The Roy method is still available as create_project_roy.

        Parameters
        ----------
        simplify : bool, default True
            Remove/contract redundant dummies with ``_simplify_local``.
        """
        return self.create_project_fondahl(simplify=simplify)

    def create_project_fondahl(self, simplify=True):
        """Build a PERT graph using the Fondahl/Elmaghraby direct construction.

        Unlike the Roy+simplify approach this method never creates redundant
        dummies in the first place, avoiding the cycle-creation bug that
        affects ``create_project`` when two activities share all predecessors.

        Fondahl does NOT give the minimum number of dummies: the optional
        post-construction simplification (``_simplify_local``) usually removes
        30-50 % of them.  It validates each dummy with local bitset checks
        (see its docstring), so it is almost linear in the graph size.  The
        previous implementation (``_simplify_legacy``: ``_simplify_sweep``
        until a fixed point) gives the same kind of result but is ~n³.

        Parameters
        ----------
        simplify : bool, default True
            Remove/contract redundant dummies with ``_simplify_local``.
        """
        nodes, real_arcs, dummy_arcs = fondahl_construct(self.immediate_predecessor)

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for tail, head, act_name in real_arcs:
            G.add_edge(tail, head, activity=act_name)
        for tail, head in dummy_arcs:
            dummy_name = f"@{tail}→{head}"   # e.g. "@3→7"
            G.add_edge(tail, head, activity=dummy_name)

        # Relabel integer nodes to strings immediately so that
        # nodes_from_edges (which f-strings node IDs) and contract_activities
        # work correctly during simplification.
        G = nx.relabel_nodes(
            G,
            {node: str(i + 1)
             for i, node in enumerate(nx.topological_sort(G))},
        )
        project = ProjectGraph(G)

        if simplify:
            project = _simplify_local(project)   # antes: _simplify_legacy(project)

        # Final relabeling to get clean sequential node labels after simplification.
        new_graph = nx.relabel_nodes(
            project.pert_graph,
            {node: str(i + 1)
             for i, node in enumerate(nx.topological_sort(project.pert_graph))},
        )
        # Rename dummy edges to reflect final node IDs after relabeling.
        for source, target, data in new_graph.edges(data=True):
            if data['activity'].startswith('@'):
                data['activity'] = f'@{source}→{target}'
        return ProjectGraph(new_graph)

    @property
    def immediate_linkage_matrix(self):
        activities_name = sorted(list(self.activity_names))
        linkage_matrix = pd.DataFrame(
            data=False, columns=activities_name, index=activities_name, dtype=bool
        )
        linkage_matrix.index.name = "activities"
        for activity, predecessors in self.immediate_predecessor.items():
            for ancestor in predecessors:
                linkage_matrix.loc[activity, ancestor] = True
        return linkage_matrix

    @property
    def distant_linkage_matrix(self):
        """Matriz de enlace de predecesoras *distantes* (clausura transitiva).

        .. note::
           Por compatibilidad con notebooks existentes devuelve un ``Styler``
           (coloreado), no un DataFrame como ``immediate_linkage_matrix``.
           Para operar con los datos usa ``.data`` o
           ``distant_linkage_dataframe``. Para mostrarla coloreada de forma
           explícita usa ``display_distant_linkage_matrix()``.
        """
        return self._estilo_matriz(self.distant_linkage_dataframe)

    @property
    def distant_linkage_dataframe(self) -> pd.DataFrame:
        """Matriz booleana de predecesoras distantes como DataFrame."""
        activities_name = sorted(list(self.activity_names))
        linkage_matrix = pd.DataFrame(
            data=False, columns=activities_name, index=activities_name, dtype=bool
        )
        linkage_matrix.index.name = "activities"
        for activity, predecessors in self.distant_predecessor.items():
            for ancestor in predecessors:
                linkage_matrix.loc[activity, ancestor] = True
        return linkage_matrix

    @staticmethod
    def _estilo_matriz(data: pd.DataFrame):
        """``Styler`` de una matriz booleana de enlace: cuadrícula fina y suave, celdas
        cuadradas de igual ancho, sin rayado de filas y las celdas verdaderas con un
        tono azul y una marca ✓."""
        data = data.copy()
        data.columns.name = "precedente →"
        data.index.name = "actividad ↓"
        celda = ("text-align: center; padding: 0.3em 0; border: 1px solid #d5dbe3; "
                 "box-shadow: none; ")
        return (data.style
                .format(lambda v: "✓" if v else "")
                .map(lambda v: celda + ("background-color: #ffe5cc; color: #cc6600; "
                                        "font-weight: bold;" if v else "background-color: white;"))
                .set_table_styles([
                    {"selector": "", "props": "border-collapse: collapse; table-layout: fixed; "
                                              "width: auto; margin: 0.5em auto;"},
                    {"selector": "th", "props": "text-align: center; border: 1px solid #d5dbe3; "
                                                "background-color: #f4f6f9; box-shadow: none; "
                                                "padding: 0.3em 0; width: 2.4em;"},
                    {"selector": "th.row_heading", "props": "text-align: right; width: 7em; "
                                                            "padding: 0.3em 0.7em 0.3em 0;"},
                    {"selector": "th.index_name, th.col_heading.level0:first-child",
                     "props": "font-style: italic; font-weight: normal; width: 7em; "
                              "color: #5b6573;"},
                ]))

    def display_distant_linkage_matrix(self, ordenar: bool = False):
        """Matriz de predecesoras distantes como figura (``MatrizEnlace``). Con
        ``ordenar=True`` filas y columnas van en orden de precedencia."""
        return MatrizEnlace(self.distant_linkage_dataframe, ordenar=ordenar)

    def display_immediate_linkage_matrix(self, ordenar: bool = False):
        """Matriz de predecesoras inmediatas (cuadro de prelaciones expandido) como
        figura TikZ (``MatrizEnlace``); ``.data`` es el ``DataFrame`` booleano. Con
        ``ordenar=True`` filas y columnas van en orden de precedencia (triangular
        inferior); por defecto se conserva el orden de la tabla."""
        return MatrizEnlace(self.immediate_linkage_matrix, ordenar=ordenar)


class ProjectGraph:
    def __init__(self, graph):
        if graph is None:
            self.pert_graph = nx.DiGraph()
        else:
            self.pert_graph = graph.copy()

    def copy(self):
        return ProjectGraph(self.pert_graph)

    def __repr__(self) -> str:
        return (f"ProjectGraph(activities={len(self.actual_activities)}, "
                f"dummies={len(self.dummies)}, nodes={self.pert_graph.number_of_nodes()})")

    def __len__(self) -> int:
        """Número de actividades reales (sin ficticias)."""
        return len(self.actual_activities)

    def __contains__(self, activity) -> bool:
        return activity in self.nodes_from_edges

    def __eq__(self, other) -> bool:
        """Dos proyectos son iguales si tienen las mismas actividades reales y
        la misma estructura de precedencias (independientemente de la
        numeración de nodos o de las ficticias concretas)."""
        if not isinstance(other, ProjectGraph):
            return NotImplemented
        return (self.actual_activities == other.actual_activities
                and self.distant_predecessor() == other.distant_predecessor())

    __hash__ = None  # mutable; la igualdad es estructural

    @staticmethod
    def from_dict_of_strings(data: Dict[str, str], simplify=True):
        # Simplify is for create_project to decide
        # whether or not to deliver a non-simplified project with plenty of dummies
        # The simplify for PredecessorTable is for reducing the data to the immediate_predecessor predecessors
        # I think it is better to assume that the data will always be reduced to the immediate_predecessor predecessors
        # and leave the simplify parameter for create_project
        result_table = PredecessorTable.from_dict_of_strings(data=data, simplify=True)
        result_project = result_table.create_project(simplify=simplify)
        return result_project

    @staticmethod
    def from_dict_of_sets(
        data: Dict[str, Set[str]], simplify=True
    ) -> "PredecessorTable":
        # for simplify see the reasons in from_dict_of_strings
        result_table = PredecessorTable.from_dict_of_sets(data=data, simplify=True)
        result_project = result_table.create_project(simplify=simplify)
        return result_project

    @staticmethod
    def from_dataframe_of_strings(
        data: pd.DataFrame, activity: str, predecessor: str, simplify=True
    ) -> "PredecessorTable":
        # for simplify see the reasons in from_dict_of_strings
        result_table = PredecessorTable.from_dataframe_of_strings(
            data=data, activity=activity, predecessor=predecessor, simplify=simplify
        )
        result_project = result_table.create_project(simplify=simplify)
        return result_project

    def paths(self, dummies=True):
        first_node, last_node = self.nodes[0], self.nodes[-1]
        edges_from_nodes = self.edges_from_nodes
        result = []
        for index, path in enumerate(
            nx.all_simple_edge_paths(
                self.pert_graph, source=first_node, target=last_node
            ),
            1,
        ):
            new_item = list(edges_from_nodes[source, target] for source, target in path)
            if not dummies:
                new_item = list(filter(lambda x: x[0] != "@", new_item))
            result.append(new_item)

        result = {
            "Route_" + str(i + 1): value for i, value in enumerate(sorted(result))
        }
        return result

    def path_matrix(self, dummies=True):
        paths = self.paths(dummies=dummies)
        activities = self.activities if dummies else self.actual_activities
        result = pd.DataFrame(0, index=activities, columns=paths.keys())

        for path_name, path in paths.items():
            result.loc[path, path_name] = 1
        return result.T

    def path_matrix_SVD(self, dummies=True, compute_uv=True):
        path_matrix = self.path_matrix(dummies=dummies)
        result = np.linalg.svd(path_matrix, full_matrices=False, compute_uv=compute_uv)
        return result

    def display_path_matrix(self, dummies=True):
        def highlight_true(value):
            if value:
                return "background-color: cyan;  border: 1px solid black;"
            else:
                return "background-color: white;  border:1px solid black;"

        data = self.path_matrix(dummies=dummies)
        try:     # figura TikZ en el estilo del cuadro de prelaciones, con un 1 en cada celda
            return MatrizEnlace(data.astype(bool), valor="1", valor_cero="0",
                                rotulo=r"actividad $\to$\\ ruta $\downarrow$")
        except (FileNotFoundError, RuntimeError):
            return data.replace(False, "").style.map(highlight_true)

    @staticmethod
    def _is_single_source_sink(project) -> bool:
        """True si el grafo PERT es acíclico y tiene exactamente un nodo
        origen y un sumidero (invariantes que toda simplificación debe
        preservar)."""
        G = project.pert_graph
        n_sources = sum(1 for n in G if G.in_degree(n) == 0)
        n_sinks = sum(1 for n in G if G.out_degree(n) == 0)
        return n_sources == 1 and n_sinks == 1 and nx.is_directed_acyclic_graph(G)

    @staticmethod
    def assess_dummy(name, project):
        """Decide si la ficticia ``name`` puede contraerse o eliminarse sin
        alterar el proyecto.

        Una ficticia es prescindible si, tras contraerla (o eliminarla), (a) el
        conjunto de predecesoras distantes de cada actividad real no cambia y
        (b) el grafo sigue teniendo un único origen y un único sumidero.  La
        condición (b) evita eliminar el conector al sumidero de una actividad
        final que comparte nodo de partida con otra (p. ej. I y J con las
        mismas predecesoras): esa ficticia es la que impide que ambas sean
        arcos paralelos entre los mismos dos nodos.  (c) El grafo sigue siendo
        acíclico: contraer una ficticia de convergencia cuando dos actividades
        comparten el mismo conjunto de predecesoras redirigía la cola de la
        otra y creaba un ciclo (bug histórico del método de Roy).
        """
        original_copy = project.copy()
        reference = project.distant_predecessor(dummies=False)

        other_project = project.contract_activities([name])
        if (other_project.distant_predecessor(dummies=False) == reference
                and ProjectGraph._is_single_source_sink(other_project)):
            return (True, other_project)

        other_project = project.remove_activities([name])
        if (other_project.distant_predecessor(dummies=False) == reference
                and ProjectGraph._is_single_source_sink(other_project)):
            return (True, other_project)

        return (False, original_copy)

    def simplify(self):
        new_project = self.copy()
        dummies = new_project.dummies
        while dummies:
            dummy = dummies.pop()
            update_needed, new_project = ProjectGraph.assess_dummy(
                name=dummy, project=new_project
            )
            if update_needed:  # dummy_was_removed_and_the_rest_changed_their_names,
                dummies = new_project.dummies

        # new_graph = nx.convert_node_labels_to_integers(new_project.pert_graph, ordering='increasing degree', first_label=1)
        # new_graph = nx.relabel_nodes(new_graph, {id: str(id) for id in new_graph.nodes})
        new_graph = nx.relabel_nodes(
            new_project.pert_graph,
            {
                id: str(indice + 1)
                for indice, id in enumerate(nx.topological_sort(new_project.pert_graph))
            },
        )
        return ProjectGraph(new_graph)

    @property
    def nodes_from_edges(
        self,
    ) -> Dict[str, Tuple[str, str]]:  # activity_name: (source, target)
        # Un solo recorrido, O(aristas). (Antes se fusionaban diccionarios de uno en uno
        # con reduce, O(aristas^2) en cada consulta.) Misma semantica: si una clave
        # se repite, gana la ultima arista.
        return {
            attributes["activity"]: (f"{source}", f"{target}")
            for (source, target, attributes) in self.pert_graph.edges(data=True)
        }

    @property
    def edges_from_nodes(self):
        return {
            (f"{source}", f"{target}"): attributes["activity"]
            for (source, target, attributes) in self.pert_graph.edges(data=True)
        }

    def contract_activities(self, names):
        def contract_activity(name: str, project: ProjectGraph):
            if name not in project.activities:
                # print(f'Activity {name} not found in project')
                return project.copy()
            source, target = project.nodes_from_edges[name]
            new_graph = nx.contracted_nodes(
                project.pert_graph, target, source, self_loops=False, copy=True
            )
            return ProjectGraph(new_graph)

        return reduce(
            lambda project, name: contract_activity(name=name, project=project),
            names,
            self,
        )

    def remove_activities(self, names):
        def remove_activity(project_graph, name: str):
            if name not in self.activities:
                return project_graph
            if name in self.nodes_from_edges:
                project_graph.remove_edge(*self.nodes_from_edges[name])
            return project_graph

        new_graph = self.pert_graph.copy()
        new_graph = reduce(
            lambda graph, name: remove_activity(graph, name), names, new_graph
        )
        return ProjectGraph(new_graph)

    def calculate_distant_predecessor_of(self, activity, dummies=False,
                                         _nodes_from_edges=None, _edges_from_nodes=None):
        # Los dos diccionarios privados permiten reutilizarlos al calcular todas las
        # actividades (distant_predecessor) sin reconstruirlos en cada una.
        nodes_from_edges_dict = (
            self.nodes_from_edges if _nodes_from_edges is None else _nodes_from_edges
        )
        source, target = nodes_from_edges_dict[activity]
        edges_from_nodes_dict = (
            self.edges_from_nodes if _edges_from_nodes is None else _edges_from_nodes
        )
        predecessors = [
            edges_from_nodes_dict[(s, t)]
            for s, t, reverse in nx.edge_dfs(
                self.pert_graph, source=source, orientation="reverse"
            )
        ]

        result = (
            predecessors if dummies else filter(lambda x: x[0] != "@", predecessors)
        )
        return set(result)

    def calculate_immediate_predecessor_of(self, activity, dummies=False):
        """Predecesoras inmediatas de ``activity``.

        dummies=True  → nombres de los arcos que entran en el nodo origen
                        (pueden ser ficticias ``@…``).
        dummies=False → se resuelven las cadenas de ficticias hasta llegar a
                        actividades reales.
        """
        source, _ = self.nodes_from_edges[activity]
        edges_dict = self.edges_from_nodes
        G = self.pert_graph
        if dummies:
            return {edges_dict[(f"{s}", f"{t}")] for s, t in G.in_edges(source)}
        result, stack, seen = set(), [source], set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for s, t in G.in_edges(node):
                name = edges_dict[(f"{s}", f"{t}")]
                if name.startswith("@"):
                    stack.append(s)
                else:
                    result.add(name)
        return result

    def _cierre_en_bits(self):
        """Ancestros e inmediatas reales de cada arco (actividad o ficticia), como enteros de Python usados
        como conjuntos de bits (el bit ``i`` es ``self.activities[i]``). Una sola pasada en orden topologico:
        cuesta O(aristas * actividades / 64) en lugar de un recorrido por actividad.

        Devuelve ``(todas, reales, ancestros, inmediatas)`` o ``None`` si el grafo tiene ciclos.
        ``ancestros[x]``: arcos que deben haber terminado antes de ``x`` (ficticias incluidas);
        ``inmediatas[x]``: actividades reales ancestros de ``x`` que no lo son de otra actividad real
        ancestro de ``x`` (las predecesoras del grafo de Roy, su reduccion transitiva)."""
        G = self.pert_graph
        if not nx.is_directed_acyclic_graph(G):
            return None
        todas = self.activities
        bit = {a: 1 << i for i, a in enumerate(todas)}
        reales = 0
        for a in todas:
            if a[0] != "@":
                reales |= bit[a]
        llegan, tapadas = {}, {}      # por nudo: arcos que llegan, y ancestros de las actividades reales que llegan
        for node in nx.topological_sort(G):
            acumulado = cubiertas = 0
            for s, _t, a in G.in_edges(node, data="activity"):
                acumulado |= bit[a] | llegan[s]
                cubiertas |= tapadas[s] | (llegan[s] if a[0] != "@" else 0)
            llegan[node], tapadas[node] = acumulado, cubiertas
        origen = {a["activity"]: u for u, _v, a in G.edges(data=True)}
        ancestros = {k: llegan[origen[k]] for k in todas}
        inmediatas = {k: llegan[origen[k]] & reales & ~tapadas[origen[k]] for k in todas}
        return todas, reales, ancestros, inmediatas

    def distant_predecessor(self, dummies=False, format=None):
        todas = self.activities
        activities = todas if dummies else [x for x in todas if x[0] != "@"]
        cierre = self._cierre_en_bits()
        if cierre is not None:
            _, reales, ancestros, _ = cierre
            resultado = {
                key: _bits_a_conjunto(ancestros[key] if dummies else ancestros[key] & reales, todas)
                for key in activities
            }
        else:
            # Grafo con ciclos (p. ej. contracciones de prueba en assess_dummy): se
            # conserva el recorrido por actividad, que tolera ciclos.
            nfe, efn = self.nodes_from_edges, self.edges_from_nodes
            resultado = {
                key: self.calculate_distant_predecessor_of(
                    key, dummies=dummies, _nodes_from_edges=nfe, _edges_from_nodes=efn
                )
                for key in activities
            }
        if format == "DataFrame":
            resultado = pd.Series(
                {key: ", ".join(sorted(value)) for key, value in resultado.items()},
                name="predecessors",
            ).replace("", "----").to_frame()
        return resultado

    @property
    def nodes(self):
        return [f"{nodo}" for nodo in list(nx.topological_sort(self.pert_graph))]

    @property
    def activities(self):
        return sorted(
            [
                attributes["activity"]
                for source, target, attributes in self.pert_graph.edges(data=True)
            ]
        )

    @property
    def dummies(self):
        return sorted([name for name in self.activities if name[0] == "@"])

    @property
    def actual_activities(self):
        return sorted([name for name in self.activities if name[0] != "@"])

    def _normalize_durations(self, durations, dummies: bool = True,
                             warn_unknown: bool = True) -> pd.Series:
        """Normaliza duraciones a una Series indexada por las actividades.

        - Acepta dict, Series o cualquier Mapping.
        - Las actividades ausentes reciben duración 0 (las ficticias siempre).
        - dtype ``int`` si todos los valores son enteros, ``float`` en otro caso
          (evita el FutureWarning de pandas por asignación de dtype incompatible).
        - Duraciones negativas → ``ValueError``.
        - Claves que no son actividades del proyecto → ``UserWarning``
          (probable error tipográfico), salvo ``warn_unknown=False``.

        Parameters
        ----------
        dummies : bool
            True → índice = ``self.activities`` (con ficticias);
            False → índice = ``self.actual_activities``.
        """
        if isinstance(durations, pd.Series):
            provided = durations.to_dict()
        elif isinstance(durations, pd.DataFrame):
            raise TypeError("durations debe ser dict o Series, no DataFrame; "
                            "selecciona la columna (p.ej. data['duration']).")
        else:
            provided = dict(durations) if durations is not None else {}

        index = self.activities if dummies else self.actual_activities
        known = set(self.activities)
        unknown = [k for k in provided if k not in known]
        if unknown and warn_unknown:
            import warnings
            warnings.warn(
                f"Duraciones para actividades que no están en el proyecto: "
                f"{sorted(map(str, unknown))}. Se ignoran.", UserWarning, stacklevel=3)
        values = [provided.get(a, 0) for a in index]
        values = [0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                  for v in values]
        negative = [a for a, v in zip(index, values) if v < 0]
        if negative:
            raise ValueError(f"Duraciones negativas no permitidas: {negative}")
        all_int = all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            for v in values
        )
        dtype = int if all_int else float
        result = pd.Series(values, index=index, dtype=dtype)
        # Conservar el nombre del índice de entrada (p. ej. 'activity'): se
        # propaga a calendar() y a las tablas que derivan de él.
        if isinstance(durations, pd.Series) and durations.index.name is not None:
            result.index.name = durations.index.name
        return result

    def calculate_pert(self, durations: Dict[str, float]):
        dur_series = self._normalize_durations(durations, dummies=True)
        duraciones = dur_series.to_dict()
        dtype = dur_series.dtype.type if dur_series.dtype != object else float
        nodos = self.nodes
        indice_dataframe = sorted(nodos, key=int)
        edges_from_nodes = self.edges_from_nodes
        tempranos = pd.Series(0, index=indice_dataframe, dtype=dtype)
        tardios = pd.Series(0, index=indice_dataframe, dtype=dtype)
        H_total = pd.Series(0, index=self.activities, dtype=dtype)
        for nodo_id in nodos[1:]:
            tempranos[nodo_id] = max(
                tempranos[source] + duraciones.get(edges_from_nodes[(source, target)])
                for (source, target) in self.pert_graph.in_edges(nodo_id)
            )
        tardios[nodos[-1]] = tempranos[nodos[-1]]
        for nodo_id in nodos[-2::-1]:
            tardios[nodo_id] = min(
                tardios[target] - duraciones.get(edges_from_nodes[(source, target)])
                for (source, target) in self.pert_graph.out_edges(nodo_id)
            )

        for source, target in self.pert_graph.edges():
            activity_name = edges_from_nodes[(source, target)]
            H_total[activity_name] = (
                tardios[target] - duraciones.get(activity_name) - tempranos[source]
            )

        return dict(
            nodes=pd.DataFrame(dict(early=tempranos, late=tardios)),
            activities=pd.DataFrame(dict(H_total=H_total)),
        )

    def calendar(self, duraciones):
        """Calendario de cada actividad real: inicios/fines más tempranos y
        tardíos, holgura total y duración.

        ``duraciones`` puede ser dict o Series; las actividades ausentes se
        toman con duración 0.
        """
        orden_entrada = list(duraciones.index) if isinstance(duraciones, pd.Series) else None
        duraciones = self._normalize_durations(duraciones, dummies=False)
        if orden_entrada is not None:
            # Respetar el orden de las actividades tal como las dio el usuario
            # (las que falten se añaden al final)
            orden = [a for a in orden_entrada if a in duraciones.index]
            orden += [a for a in duraciones.index if a not in orden]
            duraciones = duraciones.reindex(orden)
        calendario = pd.DataFrame(
            0,
            index=duraciones.index,
            columns=[
                "inicio_mas_temprano",
                "inicio_mas_tardio",
                "fin_mas_temprano",
                "fin_mas_tardio",
                "H_libre",
                "H_independiente",
            ],
            dtype=duraciones.dtype,
        )

        resultados_pert = self.calculate_pert(duraciones)
        calendario["H_total"] = resultados_pert["activities"]["H_total"]
        calendario["duracion"] = duraciones

        tempranos = resultados_pert["nodes"]["early"]
        tardios = resultados_pert["nodes"]["late"]

        for nodo_inicial, nodo_final in self.pert_graph.edges:
            activity_name = self.pert_graph.edges[nodo_inicial, nodo_final]["activity"]
            if activity_name[0] == "@":
                continue
            calendario.loc[activity_name, "inicio_mas_temprano"] = tempranos[
                nodo_inicial
            ]
            calendario.loc[activity_name, "inicio_mas_tardio"] = tardios[
                nodo_final
            ] - duraciones.get(activity_name)
            calendario.loc[activity_name, "fin_mas_temprano"] = tempranos[
                nodo_inicial
            ] + duraciones.get(activity_name)
            calendario.loc[activity_name, "fin_mas_tardio"] = tardios[nodo_final]
            # Holgura libre: margen sin retrasar el inicio más temprano de
            # las sucesoras.  Holgura independiente: margen garantizado aunque
            # las predecesoras acaben lo más tarde posible.  Se definen por
            # actividades (no por los sucesos del dibujo): las ficticias a la salida
            # del suceso final no deben "absorber" la holgura.
            d = duraciones.get(activity_name)
            sucesoras, predecesoras = self._vecinas_de_actividad(activity_name)
            fin_proyecto = tempranos.max()
            nat = {f"{n}": n for n in tempranos.index}
            primero_sucesoras = min(
                (tempranos[nat[self.nodes_from_edges[b][0]]] for b in sucesoras), default=fin_proyecto
            )
            ultimo_predecesoras = max(
                (tardios[nat[self.nodes_from_edges[b][1]]] for b in predecesoras), default=0
            )
            calendario.loc[activity_name, "H_libre"] = (
                primero_sucesoras - tempranos[nodo_inicial] - d
            )
            calendario.loc[activity_name, "H_independiente"] = max(
                0, primero_sucesoras - ultimo_predecesoras - d
            )

        return calendario

    def _vecinas_de_actividad(self, actividad):
        """Actividades reales sucesoras y predecesoras inmediatas de ``actividad``,
        deducidas del PERT: las que arrancan en el suceso final de ``actividad`` o en uno
        al que se llega desde él solo por ficticias (y, al revés, las que terminan en el
        suceso inicial o en uno desde el que se llega a él solo por ficticias)."""
        if not hasattr(self, "_cache_vecinas"):
            G = self.pert_graph
            ficticias = nx.DiGraph([(u, v) for u, v, d in G.edges(data=True)
                                    if d.get("activity", "").startswith("@")])
            ficticias.add_nodes_from(G.nodes)
            alcance = {n: {n} | nx.descendants(ficticias, n) for n in G.nodes}
            reales = {a: uv for a, uv in self.nodes_from_edges.items() if not a.startswith("@")}
            vecinas = {}
            for a, (u, v) in reales.items():
                sucesoras = [b for b, (x, _) in reales.items() if x in alcance[v]]
                predecesoras = [b for b, (_, y) in reales.items() if u in alcance[y]]
                vecinas[a] = (sucesoras, predecesoras)
            self._cache_vecinas = vecinas
        return self._cache_vecinas[actividad]

    def duration(self, durations):
        resultados_pert = self.calculate_pert(durations)
        duraciones = resultados_pert["nodes"]["early"].max()
        return duraciones

    def critical_path(self, durations):
        """
        def calculate_path_float(path, floats):
                path_floats = [floats[activity] for activity in path]
                return sum(path_floats)

        duraciones = {key: 0 for key in self.activities}
        duraciones.update(durations)
        resultados_pert = self.calculate_pert(duraciones)
        total_floats = resultados_pert['activities']['H_total']

        result = {name: path for name, path in self.paths().items(
        ) if calculate_path_float(path, total_floats) == 0}
        """
        rutas = self.path_matrix(dummies=False)
        dur = self._normalize_durations(durations, dummies=False)
        duraciones_rutas = pd.Series(
            rutas.values @ dur.reindex(rutas.columns).values, index=rutas.index
        )
        duracion_proyecto = duraciones_rutas.max()
        result = {
            name: list(rutas.columns[rutas.loc[name] == 1])
            for name in rutas.index
            if duraciones_rutas[name] == duracion_proyecto
        }
        return result

    def display_critical_path(self, durations):
        critical_path = self.critical_path(durations)
        for ruta,actividades in critical_path.items():
            display(Markdown(f"- {ruta} :\t" + ", ".join(actividades) + " <br> "))

    def zaderenko(self, durations: Dict[str, float]):
        duraciones = self._normalize_durations(durations, dummies=True).to_dict()
        resultados_pert = self.calculate_pert(duraciones)["nodes"]
        lista_de_nodos_ordenada = sorted(self.nodes, key=int)
        z = pd.DataFrame(
            np.nan, index=lista_de_nodos_ordenada, columns=lista_de_nodos_ordenada
        )

        for name, (source, target) in self.nodes_from_edges.items():
            z.loc[source, target] = duraciones[name]

        z["early"] = resultados_pert["early"]
        z = pd.concat([z, resultados_pert["late"].to_frame().T], axis=0).fillna("")
        return z

    def pert(
        self,
        filename=None,
        durations=False,
        size=None,
        orientation="landscape",
        rankdir="LR",
        ordering="out",
        ranksep=0.5,
        nodesep=0.5,
        rotate=0,
        motor="tikz",
        **kwargs,
    ):
        """Dibuja el grafo PERT. Por defecto (``motor="tikz"``) con TikZ y la disposición
        propia de pmule (:meth:`pert_tikz`: sin graphviz, el camino crítico en rojo; los
        argumentos de disposición ``ranksep``, ``nodesep``… no se usan); con
        ``motor="graphviz"``, el dibujo de graphviz de siempre (pygraphviz)."""
        if motor == "tikz":
            return self.pert_tikz(filename, durations)
        if motor != "graphviz":
            raise ValueError("motor debe ser 'tikz' o 'graphviz'")
        if filename is None:
            filename = "output_pert_figure.svg"

        if isinstance(durations, pd.Series) or durations:
            duraciones = {key: 0 for key in self.activities}
            duraciones.update(durations)
            resultados_pert = self.calculate_pert(duraciones)
            tempranos = resultados_pert["nodes"]["early"]
            tardios = resultados_pert["nodes"]["late"]
            H_total = resultados_pert["activities"]["H_total"]

        try:
            import pygraphviz as pgv
        except ImportError as exc:
            raise ImportError(
                "plot_pert() requiere pygraphviz. "
                "Instálalo con: conda install -c conda-forge pygraphviz"
            ) from exc
        dot_graph = pgv.AGraph(
            size=size,
            orientation=orientation,
            rankdir=rankdir,
            ordering=ordering,
            ranksep=ranksep,
            nodesep=nodesep,
            rotate=rotate,
            directed=True,
            **kwargs,
        )

        dot_graph.node_attr["shape"] = "Mrecord"
        dot_graph.add_edges_from(self.pert_graph.edges)

        for node_number in dot_graph.nodes():
            current_node = dot_graph.get_node(node_number)

            if isinstance(durations, pd.Series) or durations:
                current_node.attr["label"] = (
                    f"{node_number} | {{ "
                    f"<early> {tempranos[node_number]} | "
                    f"<last>  {tardios[node_number]} }}"
                )
            else:
                current_node.attr["label"] = (
                    f"{node_number} | {{ <early>  | <last>   }}"
                )

        for origin, destination in dot_graph.edges_iter():
            current_edge = dot_graph.get_edge(origin, destination)
            current_edge.attr["headport"] = "early"
            current_edge.attr["tailport"] = "last"

            activity_name = self.edges_from_nodes[(origin, destination)]
            if isinstance(durations, pd.Series) or durations:
                current_edge.attr["label"] = (
                    f"{activity_name}({duraciones[activity_name]})"
                )
                if H_total[activity_name] == 0:
                    current_edge.attr["color"] = "red:red"
                    current_edge.attr["fontcolor"] = "red"
                    current_edge.attr["style"] = (
                        "dashed" if activity_name[0] == "@" else "bold"
                    )

                else:
                    current_edge.attr["color"] = (
                        "azure4"  # if activity_name[0] == '@' else 'black'
                    )
                    current_edge.attr["fontcolor"] = "azure4"
                    current_edge.attr["style"] = (
                        "dashed" if activity_name[0] == "@" else "solid"
                    )
            else:
                current_edge.attr["label"] = f"{activity_name}"
                current_edge.attr["fontcolor"] = (
                    "forestgreen" if activity_name[0] == "@" else "black"
                )
                current_edge.attr["style"] = (
                    "dashed" if activity_name[0] == "@" else "solid"
                )
                current_edge.attr["color"] = (
                    "forestgreen" if activity_name[0] == "@" else "black"
                )

        dot_graph.draw(filename, prog="dot")
        if filename.lower().endswith(".svg"):
            return SVG(filename)
        elif filename.lower().endswith((".pdf")):
            filename = filename.replace(".pdf", ".svg")
            dot_graph.draw(filename, prog="dot")  # Create additional SVG file
            return SVG(filename)
        return Image(filename)

    def gantt(
        self,
        data,
        duration_label,
        resource_label=None,
        total=None,
        acumulado=False,
        holguras=False,
        cuadrados=False,
        tikz=False,
        params=None,
        extra_rows = None,
        extra_cols = None
    ):
        if params is None:
            params = dict()
        
        user_cols = extra_cols.copy() if extra_cols is not None else None
        user_rows = extra_rows.copy() if extra_rows is not None else None
        # Las filas/columnas del usuario se añaden al final (user_*); si no se
        # vacían aquí, con total=None se dibujaban dos veces en el TikZ.
        extra_cols = extra_rows = None

        my_data = data.copy()
        my_data[duration_label] = my_data[duration_label].astype(int)
        if resource_label is None:
            resource_label = "resources"
            my_data[resource_label] = "  "

        if isinstance(resource_label, str) and resource_label == "names":
            resource_label = "resources"
            my_data[resource_label] = my_data.index

        resultados_pert = self.calculate_pert(my_data.loc[:, duration_label])
        tempranos = (resultados_pert["nodes"]["early"]).astype(int)
        duracion_proyecto = tempranos.max()
        periodos = range(1, ceil(duracion_proyecto) + 1)
        actividades_con_duracion = [
            nombre
            for nombre in self.activities
            if my_data.loc[:, duration_label].get(nombre, 0) != 0
        ]
        actividades_con_duracion.sort()
        if tikz:
            gantt_data = pd.DataFrame(
                index=actividades_con_duracion,
                columns=["start", "duration"],
                dtype="int64",
            )
        gantt = pd.DataFrame("", index=actividades_con_duracion, columns=periodos, dtype=object)

        for edge in self.pert_graph.edges:
            activity_name = self.pert_graph.edges[edge]["activity"]
            if activity_name[0] == "@":
                continue
            duracion_tarea = my_data.loc[:, duration_label].get(activity_name, 0)
            if duracion_tarea != 0:
                comienzo_tarea = tempranos[edge[0]]
                gantt.loc[
                    activity_name,
                    (comienzo_tarea + 1) : (comienzo_tarea + duracion_tarea),
                ] = my_data.loc[activity_name, resource_label]
                if tikz:
                    gantt_data.loc[activity_name, ["start"]] = comienzo_tarea
                    gantt_data.loc[activity_name, ["duration"]] = duracion_tarea
                    gantt_data.loc[activity_name, ["resource"]] = my_data.loc[
                        activity_name, resource_label
                    ]

        def color_gantt(val):
            background = "white" if val == "" else "sandybrown"
            style = f"background-color: {background}"
            return style

        # Set CSS properties for th elements in dataframe
        th_props = [
            ("font-size", "11px"),
            ("text-align", "center"),
            ("font-weight", "bold"),
            ("color", "#6d6d6d"),
            ("background-color", "#f7f7f9"),
            ("border", "1px solid"),
        ]

        # Set CSS properties for td elements in dataframe
        td_props = [
            ("font-size", "11px"),
            ("border-color", "#c0c0c0"),
            ("border", "1px solid"),
        ]

        # Set table styles
        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
            dict(selector="", props=[("border", "3px solid")]),
        ]

        def summary(df, fn=np.sum, axis=0, name="Total"):
            df_total = df.replace("", 0)
            total = df_total.apply(fn, axis=axis).to_frame(name)
            if axis == 0:
                total = total.T
            out = pd.concat([df, total], axis=axis)
            return out



        if total is None:
            mat = gantt
        elif total == "columna":
            mat = summary(gantt, axis=1)
            extra_cols = mat.copy().loc[actividades_con_duracion, ["Total"]]
        elif total == "fila":
            mat = summary(gantt, axis=0)
            extra_rows = mat.copy().loc[["Total"],]
            if acumulado:
                fila_acumulado = mat.loc["Total"].cumsum()
                fila_acumulado.name = "Acumulado"
                mat = pd.concat([mat, fila_acumulado.to_frame().T], axis=0).fillna("")
                extra_rows = mat.copy().loc[["Total", "Acumulado"],]

            if cuadrados:
                fila_cuadrados = mat.loc["Total"] ** 2
                fila_cuadrados.name = "Cuadrados"
                mat = pd.concat([mat, fila_cuadrados.to_frame().T], axis=0).fillna("")
                extra_rows = mat.copy().loc[["Total", "Cuadrados"],]
        elif total == "ambas":
            mat = summary(summary(gantt, axis=0), axis=1)
            extra_rows = mat.copy().loc[["Total"],]
            extra_cols = mat.copy().loc[actividades_con_duracion, ["Total"]]
            if acumulado:
                fila_acumulado = mat.loc["Total"].drop("Total").cumsum()
                fila_acumulado.name = "Acumulado"
                mat = pd.concat([mat, fila_acumulado.to_frame().T], axis=0).fillna("")
                extra_rows = mat.copy().loc[["Total", "Acumulado"],]
                extra_cols = mat.copy().loc[actividades_con_duracion, ["Total"]]

            if cuadrados:
                fila_cuadrados = mat.loc["Total"] ** 2
                fila_cuadrados.name = "Cuadrados"
                mat = pd.concat([mat, fila_cuadrados.to_frame().T], axis=0).fillna("")
                extra_rows = mat.copy().loc[["Total", "Cuadrados"],]

        if holguras:
            mat["H_total"] = resultados_pert["activities"]["H_total"]
            if total in ["fila", "ambas"]:
                mat.loc["Total", "H_total"] = None

        if user_cols is not None:
            mat = pd.concat([user_cols, mat], axis=1)
        if user_rows is not None:
            mat = pd.concat([user_rows, mat], axis=0)

        resultado = (
            mat.style.set_table_styles(styles)
            .map(color_gantt)
            .apply(
                lambda x: [
                    (
                        "background: #f7f7f9"
                        if x.name in ["Total", "Acumulado", "H_total", "Cuadrados"]
                        else ""
                    )
                    for i in x
                ],
                axis=0,
            )
            .apply(
                lambda x: [
                    (
                        "background: #f7f7f9"
                        if x.name in ["Total", "Acumulado", "H_total", "Cuadrados"]
                        else ""
                    )
                    for i in x
                ],
                axis=1,
            )
        )
        dibujo = ""
        if tikz:
            gantt_data["Htotal"] = resultados_pert["activities"]["H_total"].astype(
                "Int64"
            )
            if holguras:
                extra_cols = pd.concat(
                    [extra_cols, gantt_data[["Htotal"]]], axis=1
                ).copy()
            params["inner_text"] = params.get("inner_text", "resource")
            if user_cols is not None:
                extra_cols = pd.concat([user_cols, extra_cols], axis=1)
            if user_rows is not None:
                extra_rows = pd.concat([user_rows, extra_rows], axis=0) 
            precedences = (
                self._gantt_precedences(set(gantt_data.index))
                if params.get("arrows", True) else None
            )
            dibujo = make_gantt_tikz(
                gantt_data=gantt_data,
                extra_cols=extra_cols,
                extra_rows=extra_rows,
                params=params,
                precedences=precedences,
            )
        return resultado, dibujo

    def _gantt_precedences(self, present: set) -> list:
        """Pares (predecesora, sucesora) entre las barras del Gantt.

        Solo actividades reales (las ficticias se resuelven); si una
        predecesora inmediata no tiene barra (duración 0), se salta hasta las
        suyas, de modo que la prelación no se pierde.
        """
        pairs = set()
        for act in present:
            stack = list(self.calculate_immediate_predecessor_of(act))
            seen = set()
            while stack:
                pred = stack.pop()
                if pred in seen:
                    continue
                seen.add(pred)
                if pred in present:
                    pairs.add((pred, act))
                elif pred in self.nodes_from_edges:
                    stack.extend(self.calculate_immediate_predecessor_of(pred))
        return sorted(pairs)

    def roy(
        self,
        filename=None,
        durations=False,
        size=None,
        orientation="landscape",
        rankdir="LR",
        ordering="out",
        ranksep=0.5,
        nodesep=0.5,
        rotate=0,
        dpi=300,
        motor="tikz",
        **kwargs,
    ):
        """Dibuja el grafo Roy. Por defecto (``motor="tikz"``) con TikZ y la disposición
        propia de pmule (:meth:`roy_tikz`, sin graphviz y con el camino crítico en rojo);
        con ``motor="graphviz"``, el dibujo de graphviz (pygraphviz). En ambos, los nodos
        al estilo del PMBOK: ``ES | d | EF`` arriba, el nombre en la banda central y
        ``LS | H | LF`` abajo (origen de tiempos en 0)."""
        if motor == "tikz":
            return self.roy_tikz(filename, durations)
        if motor != "graphviz":
            raise ValueError("motor debe ser 'tikz' o 'graphviz'")
        if isinstance(durations, (pd.Series, pd.DataFrame)):
            durations = durations.copy()
        df = PredecessorTable.from_project(self).immediate_linkage_matrix
        rows_to_keep = ~df.index.str.startswith("@")
        columns_to_keep = ~df.columns.str.startswith("@")
        df_cleaned = df.loc[rows_to_keep, columns_to_keep]

        self.roy_graph = make_Roy(df_cleaned)

        if filename is None:
            filename = "output_roy_figure.png"

        if durations is None:
            durations = False

        if durations is not False:
            calendario = self.calendar(durations)
            inicio_mas_temprano = calendario["inicio_mas_temprano"].copy()
            inicio_mas_tardio = calendario["inicio_mas_tardio"].copy()
            durations["start"] = 0
            durations["finish"] = 0
            inicio_mas_temprano["start"] = 0
            inicio_mas_tardio["start"] = 0

            inicio_mas_temprano["finish"] = calendario['fin_mas_temprano'].max()
            inicio_mas_tardio["finish"] = inicio_mas_temprano["finish"] 
            critical_nodes = ["start", "finish"] + [activity for activity in self.actual_activities if calendario.loc[activity, "H_total"] == 0]


        try:
            import pygraphviz as pgv
        except ImportError as exc:
            raise ImportError(
                "plot_roy() requiere pygraphviz. "
                "Instálalo con: conda install -c conda-forge pygraphviz"
            ) from exc
        dot_graph = pgv.AGraph(
            size=size,
            orientation=orientation,
            rankdir=rankdir,
            ordering=ordering,
            ranksep=ranksep,
            nodesep=nodesep,
            rotate=rotate,
            directed=True,
            **kwargs,
        )

        dot_graph.node_attr["shape"] = "plain"
        dot_graph.add_edges_from(self.roy_graph.edges)

        def etiqueta(nombre, valores=None, critico=False):
            """Nodo al estilo del PMBOK (tabla HTML de graphviz): ``ES | d | EF`` arriba, el
            nombre en la banda central y ``LS | H | LF`` abajo."""
            color = "red" if critico else "black"
            relleno = "#FDE8E8" if critico else "white"
            v = valores or {}
            celda = lambda k: (f'<TD WIDTH="34" BGCOLOR="{relleno}">'
                               f'{"<B>" + v[k] + "</B>" if (k in v and critico) else v.get(k, "&nbsp;")}</TD>')
            return (f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5" COLOR="{color}">'
                    f'<TR>{celda("ES")}{celda("d")}{celda("EF")}</TR>'
                    f'<TR><TD COLSPAN="3" BGCOLOR="{color if critico else relleno}"><FONT COLOR="{"white" if critico else "black"}"><B>{nombre}</B></FONT></TD></TR>'
                    f'<TR>{celda("LS")}{celda("H")}{celda("LF")}</TR></TABLE>>')

        for nodo in dot_graph.nodes():
            current_node = dot_graph.get_node(nodo)
            if durations is not False:
                es, d, ls = (inicio_mas_temprano[str(nodo)], durations[str(nodo)],
                             inicio_mas_tardio[str(nodo)])
                f = self._num_paso
                current_node.attr["label"] = etiqueta(
                    nodo, dict(ES=f(es), d=f(d), EF=f(es + d), LS=f(ls), H=f(ls - es), LF=f(ls + d)),
                    critico=str(nodo) in critical_nodes)
            else:
                current_node.attr["label"] = etiqueta(nodo)

        # --- 2. Style Edges in NetworkX ---
        if durations is not False:
            for u, v in dot_graph.edges():
                u_name = str(u)
                v_name = str(v)
                
                if u_name in critical_nodes and v_name in critical_nodes:
                    current_edge = dot_graph.get_edge(u, v)
                    current_edge.attr["color"] = "red"
                    current_edge.attr["penwidth"] = 2.0
                else:
                    # OPTIONAL: Set non-critical edge attributes (defaults)
                    current_edge = dot_graph.get_edge(u, v)
                    current_edge.attr["color"] = "black"
                    current_edge.attr["penwidth"] = 1.0 # Standard width

        dot_graph.graph_attr['dpi'] = str(dpi)
        dot_graph.draw(filename, prog="dot", format="png")
        return Image(filename)





    def gantt_cargas(
        self, data, duration_label, resource_label, report=True, tikz=False, params=None
    ):
        my_data = data.copy()

        gantt, dibujo = self.gantt(
            my_data,
            duration_label,
            resource_label,
            total="fila",
            acumulado=False,
            holguras=True,
            cuadrados=True,
            tikz=tikz,
            params=params,
        )
        gantt.data.loc["Total", "H_total"] = np.nan
        suma_cuadrados = gantt.data.loc["Cuadrados", :].sum()
        gantt.data.loc["Cuadrados", "H_total"] = suma_cuadrados
        if report:
            print("Suma de cuadrados:", suma_cuadrados, "\n")
        return gantt, dibujo

    def desplazar(
        self,
        data,
        duration_label,
        resource_label,
        report=True,
        tikz=False,
        holguras=True,
        cuadrados=True,
        params=None,
        **activity_shifts,
    ):
        """Desplaza el inicio de una o varias actividades (``**activity_shifts``
        = {actividad: periodos}) insertando una actividad de espera ``💤X``.

        .. warning::
           Este método **muta** ``self.pert_graph`` in place: añade un nodo
           auxiliar y el arco ``💤X`` por cada actividad desplazada. Si
           necesitas conservar el proyecto original, trabaja sobre
           ``self.copy()`` (como hace ``evaluar_desplazamiento``).

        Returns
        -------
        (data_modificada, gantt_df, dibujo_tikz)
        """
        if resource_label not in data.columns:
            raise ValueError("Resource label not in data columns")

        my_data = data.copy()
        dibujo = ""
        gantt_df = None
        for actividad, duracion in activity_shifts.items():
            slide_label = "💤" + actividad
            if slide_label in my_data.index:
                my_data.loc[slide_label, duration_label] += duracion
            else:
                nueva_fila = (
                    pd.Series(
                        {duration_label: duracion},
                        name=slide_label,
                        index=my_data.columns,
                    )
                    .fillna(0)
                    .astype(int)
                )
                my_data = pd.concat([my_data, nueva_fila.to_frame().T], axis=0)

        lista_edges = list(self.pert_graph.edges)
        for edge in lista_edges:
            activity_name = self.pert_graph.edges[edge]["activity"]
            slide_label = "💤" + activity_name

            if activity_name in activity_shifts and slide_label not in self.activities:
                self.pert_graph.remove_edge(edge[0], edge[1])
                nodo_auxiliar_str = str(
                    max([int(n) for n in self.pert_graph.nodes]) + 1
                )

                self.pert_graph.add_node(nodo_auxiliar_str, id=nodo_auxiliar_str)
                self.pert_graph.add_edge(
                    edge[0], nodo_auxiliar_str, activity="💤" + activity_name
                )
                self.pert_graph.add_edge(
                    nodo_auxiliar_str, edge[1], activity=activity_name
                )

        lista_nodos = list(nx.topological_sort(self.pert_graph))
        nx.set_node_attributes(
            self.pert_graph,
            {nodo: {"id": (id + 1)} for id, nodo in enumerate(lista_nodos)},
        )

        if cuadrados:
            gantt_df, dibujo = self.gantt_cargas(
                my_data,
                duration_label,
                resource_label,
                report=report,
                tikz=tikz,
                params=params,
            )
        else:
            gantt_df, dibujo = self.gantt(
                my_data,
                duration_label,
                resource_label,
                total="fila",
                holguras=holguras,
                cuadrados=cuadrados,
                tikz=tikz,
                params=params,
            )
        return my_data, gantt_df, dibujo

    def evaluar_desplazamiento(
        self,
        data,
        duration_label,
        resource_label,
        report=True,
        tikz=True,
        **desplazamientos,
    ):
        proyecto = self.copy()
        my_data = data.copy()
        dibujo = ""
        new_data, gantt_df, dibujo = proyecto.desplazar(
            my_data,
            duration_label=duration_label,
            resource_label=resource_label,
            **desplazamientos,
            report=report,
        )
        gantt_df, dibujo = proyecto.gantt_cargas(
            new_data, duration_label, resource_label, report=False, tikz=tikz
        )
        return gantt_df.data.loc["Cuadrados", "H_total"], dibujo

    def evaluar_rango_de_desplazamientos(
        self, data, duration_label, resource_label, activity, report=True, tikz=True
    ):
        my_data = data.copy()
        dibujo = ""

        minimo = 0
        maximo = int(
            self.calculate_pert(my_data.loc[:, duration_label])["activities"].loc[
                activity, "H_total"
            ]
        )
        suma_cuadrados = pd.DataFrame(
            0, index=range(minimo, maximo + 1), columns=["Suma_de_cuadrados"]
        )
        if report:
            print("Sin desplazar:")
        gantt_df, dibujo = self.gantt_cargas(
            my_data, duration_label, resource_label, report=report, tikz=tikz
        )
        suma_cuadrados.loc[0, "Suma_de_cuadrados"] = gantt_df.data.loc[
            "Cuadrados", "H_total"
        ]
        for slide in range(minimo + 1, maximo + 1):
            if report:
                print("Desplazamiento:", slide)
            carga2, dibujo = self.evaluar_desplazamiento(
                my_data,
                duration_label,
                resource_label,
                **{activity: slide},
                report=report,
                tikz=tikz,
            )
            suma_cuadrados.loc[slide, "Suma_de_cuadrados"] = carga2
        return suma_cuadrados, dibujo

    def nivelar(self, data, duration_label, resource_label, report=False):
        """Nivelación de recursos por mínimos cuadrados: para cada actividad
        no crítica (ordenadas por fin más temprano) prueba todos los
        desplazamientos dentro de su holgura y aplica el que minimiza la suma
        de cuadrados de las cargas.

        .. warning::
           Este método **muta** ``self.pert_graph`` (llama a ``desplazar``
           sobre ``self``). Usa ``self.copy().nivelar(...)`` para preservar
           el proyecto original.

        Returns
        -------
        (desplazamientos, cuadrados_styler, data_final, gantt_df, dibujo)
        """
        calendario = self.calendar(data[duration_label])
        calendario = calendario.loc[calendario["H_total"] > 0, :]
        actividades = calendario.sort_values(by=["fin_mas_temprano", 'H_total'], ascending=[True,False])
        maxima_holgura = int(calendario["H_total"].max()) if len(calendario) else 0
        cuadrados = pd.DataFrame(0, columns=actividades.index, index=range(maxima_holgura + 1) )
        desplazamientos = dict()
        gantt_df, dibujo = self.gantt_cargas(data, duration_label, resource_label,
                                              report=False, tikz=True)
        my_data = data[[duration_label, resource_label]]
        for actividad in actividades.index:
            suma_de_cuadrados, dibujo = self.evaluar_rango_de_desplazamientos(
                data=my_data,
                duration_label=duration_label,
                resource_label=resource_label,
                activity=actividad,
                report=False,
                tikz=False,
            )
            cuadrados[actividad] = suma_de_cuadrados

            desplazamientos[actividad] = suma_de_cuadrados["Suma_de_cuadrados"].idxmin()
            #print(f"Actividad: {actividad}. Desplazamiento óptimo: {desplazamientos[actividad]}")
            if desplazamientos[actividad] > 0:
                if report:
                    print(f"Actividad: {actividad}. Desplazamiento óptimo: {desplazamientos[actividad]}")
                my_data, gantt_df, dibujo = self.desplazar(
                    data=my_data,
                    duration_label=duration_label,
                    resource_label=resource_label,
                    tikz=False,
                    cuadrados=True,
                    report=False,
                    **{actividad: desplazamientos[actividad]},
                )
        gantt_df, dibujo = self.gantt_cargas(my_data, duration_label, resource_label,
                                              report=False, tikz=True)
        desplazamientos = pd.Series(desplazamientos).to_frame(name="desplazamientos")
        cuadrados = colorea_minimo_fila(cuadrados.T)
        return desplazamientos, cuadrados, my_data, gantt_df, dibujo


    def nivelar_pasos_tikz_html(self, data, duration_label, resource_label, escala: float = 1.0) -> str:
        """Visor paso a paso de la nivelación por mínimos cuadrados (:meth:`nivelar`) en TikZ:
        arriba, el cuadro de la suma de cuadrados de cada actividad en cada desplazamiento
        posible; abajo, el Gantt con las cargas correspondiente. El paso 0 es la solución;
        después, para cada actividad no crítica (por fin más temprano), un subpaso por cada
        desplazamiento probado y un paso final con el desplazamiento elegido (el de menor
        suma de cuadrados, que ya queda aplicado para las siguientes actividades).
        No modifica el proyecto."""
        import base64, html as _h, re as _re
        p = self.copy()
        cal = p.calendar(data[duration_label])
        cal = cal.loc[cal["H_total"] > 0, :]
        orden = list(cal.sort_values(by=["fin_mas_temprano", "H_total"], ascending=[True, False]).index)
        mis_datos = data[[duration_label, resource_label]]
        filas = []                                   # una por actividad: H, sumas, gantts, elegido
        for act in orden:
            H = int(p.calculate_pert(mis_datos[duration_label])["activities"].loc[act, "H_total"])
            g, dib = p.gantt_cargas(mis_datos, duration_label, resource_label, report=False, tikz=True)
            sumas, gantts = {0: g.data.loc["Cuadrados", "H_total"]}, {0: dib}
            for desp in range(1, H + 1):
                q, dib = p.evaluar_desplazamiento(mis_datos, duration_label, resource_label,
                                                  report=False, tikz=True, **{act: desp})
                sumas[desp], gantts[desp] = q, dib
            elegido = int(pd.Series(sumas).idxmin())
            if elegido > 0:
                mis_datos, _, _ = p.desplazar(data=mis_datos, duration_label=duration_label,
                                              resource_label=resource_label, tikz=False,
                                              cuadrados=True, report=False, **{act: elegido})
            filas.append(dict(act=act, H=H, sumas=sumas, gantts=gantts, elegido=elegido))
        _, gantt_final = p.gantt_cargas(mis_datos, duration_label, resource_label, report=False, tikz=True)

        n = len(filas)
        Hmax = max((f["H"] for f in filas), default=0)
        CW, RH, X0, GAP = 1.25, 0.62, 2.6, 0.9
        ancho_tabla = X0 + (Hmax + 1) * CW

        def num(v):
            return f"{int(v)}" if float(v) == int(v) else f"{v:.1f}"

        def tex(t):
            for a, b in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("💤", "zz ")):
                t = t.replace(a, b)
            return t

        def tabla(i=None, desp=None, final=False):
            """``i``: fila en curso; ``desp``: último desplazamiento probado; ``final``: fila i completa."""
            fondo, texto, rayas = [], [], []
            cy = lambda r: -(r + 0.5) * RH
            cx = lambda c: X0 + (c + 0.5) * CW
            GRIS = "black!30"

            def celda(x, y, txt, color=None, fondo_c=None, bold=False):
                if fondo_c:
                    fondo.append(rf"\fill[{fondo_c}, rounded corners=1.5pt] ({x - CW / 2 + .04:.3f},{y - RH / 2 + .04:.3f}) "
                                 rf"rectangle ({x + CW / 2 - .04:.3f},{y + RH / 2 - .04:.3f});")
                if txt != "":
                    est = ("text=" + color + "," if color else "") + ("font=\\bfseries," if bold else "")
                    texto.append(rf"\node[{est}] at ({x:.3f},{y:.3f}) {{{txt}}};")

            celda(X0 / 2, cy(0), r"\textit{actividad}")
            for c in range(Hmax + 1):
                celda(cx(c), cy(0), str(c), bold=True)
            texto.append(rf"\node[anchor=west,font=\scriptsize] at ({X0},{RH * 0.35:.3f}) {{desplazamiento}};")
            for j, f in enumerate(filas):
                y = cy(1 + j)
                hecha = i is None or j < i or (j == i and final)
                en_curso = i is not None and j == i and not final
                visible_hasta = f["H"] if hecha else (desp if en_curso else -1)
                viejo = i is not None and j < i
                celda(X0 / 2, y, tex(str(f["act"])) + rf"\,\scriptsize({f['H']})",
                      GRIS if viejo else None, bold=not viejo)
                minimo = min(f["sumas"].values())
                for c in range(0, min(visible_hasta, f["H"]) + 1):
                    v = f["sumas"][c]
                    if hecha and c == f["elegido"]:
                        fc = "green!15" if viejo else "green!45!lime!60"
                    elif en_curso and c == desp:
                        fc = "orange!30"
                    else:
                        fc = None
                    celda(cx(c), y, num(v), GRIS if viejo else None, fc)
            alto = (1 + n) * RH
            rayas.append(rf"\draw[black!70] (0,{-RH}) -- ({ancho_tabla},{-RH});")
            rayas.append(rf"\draw[black!70] ({X0 - 0.04},0) -- ({X0 - 0.04},{-alto});")
            for r in range(2, n + 1):
                rayas.append(rf"\draw[black!12] (0,{-r * RH}) -- ({ancho_tabla},{-r * RH});")
            return "\n".join(fondo + rayas + texto), alto

        def fotograma(dibujo_gantt, **kw):
            cuerpo, alto = tabla(**kw)
            dibujo_gantt = dibujo_gantt.replace("💤", "zz")      # pdflatex no admite el emoji
            interior = _re.sub(r"^\s*\\begin\{tikzpicture\}[^\n]*\n", "", dibujo_gantt.strip())
            interior = _re.sub(r"\\end\{tikzpicture\}\s*$", "", interior)
            xs = [float(v) for v in _re.findall(r"at \(\s*([\d.]+),", dibujo_gantt)]
            ancho = max([ancho_tabla] + [x + 1 for x in xs])
            desplazar_y = alto + 1.4
            return (rf"\begin{{tikzpicture}}[x={escala}cm,y={escala}cm,>=latex,font=\small]" + "\n"
                    rf"\useasboundingbox (-4.6,0.8) rectangle ({ancho + 2.6:.3f},{-(desplazar_y + 8.6):.3f});" + "\n"
                    + cuerpo + "\n"
                    rf"\begin{{scope}}[shift={{(0,{-desplazar_y:.3f})}}]" + "\n" + interior + "\n\\end{scope}\n\\end{tikzpicture}")

        codigos, pies = [fotograma(gantt_final)], []
        pies.append("Nivelación por mínimos cuadrados: solución. Cuadro de la suma de cuadrados de las cargas de cada "
                    "actividad (entre paréntesis, su holgura total) para cada desplazamiento, con el mínimo resaltado, "
                    "y diagrama de Gantt con cargas resultante. Orden seguido: actividades no críticas por fin más temprano "
                    "creciente (a igualdad, la de mayor holgura primero), " + ", ".join(str(f["act"]) for f in filas) +
                    "; cada desplazamiento elegido queda fijado antes de estudiar la siguiente. "
                    "Pulse «siguiente» para ver cómo se obtiene.")
        for i, f in enumerate(filas):
            for desp in range(f["H"] + 1):
                codigos.append(fotograma(f["gantts"][desp], i=i, desp=desp))
                pies.append(f"Actividad {f['act']} (holgura {f['H']}): desplazada {desp} "
                            f"{'periodo' if desp == 1 else 'periodos'}, la suma de cuadrados de las cargas es {num(f['sumas'][desp])}.")
            codigos.append(fotograma(f["gantts"][f["elegido"]], i=i, desp=f["elegido"], final=True))
            pies.append(f"Actividad {f['act']}: se elige el desplazamiento {f['elegido']} "
                        f"(suma de cuadrados mínima, {num(f['sumas'][f['elegido']])}); queda aplicado para las siguientes actividades.")
        fotos = []
        for svg, pie in zip(_svgs_de_tikz(codigos, "calc,arrows.meta,patterns"), pies):
            b64 = base64.b64encode(svg.encode("utf8")).decode("ascii")
            fotos.append('<img alt="paso" style="display:block;margin:0 auto;width:100%;max-width:1000px;height:auto" '
                         f'src="data:image/svg+xml;base64,{b64}">'
                         f'<p style="text-align:center;margin-top:.5em">{_h.escape(pie)}</p>')
        return _html_pasos(fotos, inicio=0)

    def display_nivelar_pasos_tikz(self, data, duration_label, resource_label, escala: float = 1.0) -> None:
        """Muestra en un cuaderno la nivelación paso a paso (:meth:`nivelar_pasos_tikz_html`)."""
        display(HTML(self.nivelar_pasos_tikz_html(data, duration_label, resource_label, escala)))

    def asignar(self, data, duration_label='duration', resource_label='resources', maximo=None, report=True, params=None):
        # maximo is indexed starting at 1
        if maximo is None:
            raise ValueError("Error: Debe introducir un valor para Máximo. Si es un vector use una serie indexada desde 1")
        if duration_label not in data.columns:
            raise ValueError("Error: Duration label not in data columns")
        if resource_label not in data.columns:
            raise ValueError("Error: Resource label not in data columns")
        if params is None:
            params = dict() 
        my_data = data.copy()
        empezadas = pd.Series({actividad: False for actividad in my_data.index}).to_frame(name='Empezadas')
        
        gantt_df, dibujo = self.gantt(data=my_data, 
                                        duration_label=duration_label,
                                        resource_label=resource_label,
                                        holguras=True,
                                        total='fila',
                                        tikz=False)
        
        duracion_proyecto = gantt_df.data.columns.drop('H_total').max()
        if isinstance(maximo, (int, float)):
            maximo = pd.Series([maximo] * (duracion_proyecto), index=range(1, duracion_proyecto+1))
        if isinstance(maximo, list):
            maximo = pd.Series(maximo, index=range(1, len(maximo)+1))
        if isinstance(maximo, pd.Series) and maximo.index.min() != 1:
                raise ValueError("Error: maximo debe estar indexado empezando en el periodo 1")
        if data[resource_label].max() > max(maximo):
            raise ValueError("Error: Hay actividades que requieren más recursos que el máximo permitido")
        
        periodo = 1
        while periodo <= duracion_proyecto:
            if periodo <= len(maximo):
                current_max = maximo[periodo]
            else:
                current_max = maximo.iloc[-1]
                maximo[periodo] = current_max
            if report:
                display(Markdown(f"<br> Periodo: {periodo}. "))
            df = pd.concat([gantt_df.data, my_data[duration_label], empezadas], axis=1)
            total = df.loc['Total', periodo]
            df = (df.loc[:, [periodo, 'H_total', duration_label, 'Empezadas']]
                    .replace('', None)
                    .dropna(subset=periodo)
                    .sort_values(by=['Empezadas', 'H_total', duration_label], ascending=[False, True, True])
                    .drop(labels='Total', axis=0)
                    )
            elegibles = df.index.to_list()
            
            if total > current_max:
                """
                acumulada = df[periodo].cumsum()
                display(Markdown(f"Acumulada: {acumulada}"))
                se_programan = set(df.loc[acumulada <= current_max,:].index)
                if report:
                    display(Markdown(f"Periodo: {periodo}. Total: {total}. Se programan: {list(se_programan)}"))
                se_retrasan = set(elegibles) - se_programan
                if report:
                    display(Markdown(f"Periodo: {periodo}. Se desplazan: {list(se_retrasan)}"))
                empezadas.loc[list(se_programan), 'Empezadas'] = True
                """
                acumulada = 0
                se_programan = set()
                se_retrasan = set()
                for actividad in elegibles:
                    if acumulada + df.loc[actividad, periodo] <= current_max:
                        acumulada += df.loc[actividad, periodo]
                        se_programan.add(actividad)
                        empezadas.loc[actividad, 'Empezadas'] = True
                    else:
                        se_retrasan.add(actividad)
                if report:
                    display(Markdown(f"Periodo: {periodo}. Total: {total}. Se programan: {list(se_programan)}"))
                    display(Markdown(f"Periodo: {periodo}. Se desplazan: {list(se_retrasan)}"))
                if se_programan == set():
                    if report:
                        display(Markdown(f"¡Atención! No se puede programar ninguna actividad en el periodo {periodo} con el máximo de recursos disponible ({current_max}). El proyecto no es factible con estos recursos."))
                    break


                my_data, gantt_df, dibujo = self.desplazar(
                                            data = my_data, 
                                            duration_label = duration_label,
                                            resource_label = resource_label,
                                            tikz = False,
                                            report=False,
                                            **{actividad: 1 for actividad in sorted(se_retrasan)}
                )
                
            else:
                acumulada = df[periodo].cumsum()
                se_programan = set(df.loc[acumulada <= current_max,:].index)
                empezadas.loc[list(se_programan), 'Empezadas'] = True
                if report:
                    display(Markdown(f"Periodo: {periodo}. Total: {total}. Se programan: {list(se_programan)}"))
                
            
            periodo += 1
            gantt_df, dibujo = self.gantt(data=my_data, 
                                        duration_label=duration_label,
                                        resource_label=resource_label,
                                        holguras=True,
                                        total='fila',
                                        tikz=False)
            duracion_proyecto = gantt_df.data.columns.drop('H_total').max()
        maximo = maximo.to_frame(name='Máximo')
        gantt_df, dibujo = self.gantt(data=my_data, 
                                    duration_label=duration_label,
                                    resource_label=resource_label,
                                    holguras=True,
                                    total='fila',
                                    tikz=True,
                                    params=params,
                                    extra_rows=maximo.T)
        return my_data, gantt_df, dibujo

    def asignar_pasos(self, data, duration_label='duration', resource_label='resources', maximo=None):
        """Registro, periodo a periodo, del algoritmo de :meth:`asignar` (no modifica el
        proyecto). Devuelve ``(pasos, datos_finales)`` con una entrada por periodo evaluado:
        ``periodo``, ``total`` (demanda), ``limite``, ``candidatas`` (``DataFrame`` ordenado por
        prioridad con la carga del periodo, ``Empezadas``, ``H_total`` y la duración),
        ``se_programan``, ``se_retrasan``, ``antes`` y ``despues`` (los datos con las duraciones
        antes y después de postergar), ``proy_antes`` y ``proy_despues`` (el proyecto en esos
        momentos, con las actividades de retraso añadidas) e ``infactible``. Sigue exactamente el criterio de
        :meth:`asignar`: primero las actividades ya empezadas, luego menor holgura total y, a
        igualdad, menor duración; se admiten mientras la carga acumulada no supere el límite."""
        if maximo is None:
            raise ValueError("Error: Debe introducir un valor para Máximo.")
        p = self.copy()
        my_data = data.copy()
        empezadas = pd.Series({a: False for a in my_data.index}).to_frame(name='Empezadas')
        gantt_df, _ = p.gantt(data=my_data, duration_label=duration_label, resource_label=resource_label,
                              holguras=True, total='fila', tikz=False)
        duracion_proyecto = gantt_df.data.columns.drop('H_total').max()
        if isinstance(maximo, (int, float)):
            maximo = pd.Series([maximo] * duracion_proyecto, index=range(1, duracion_proyecto + 1))
        if isinstance(maximo, list):
            maximo = pd.Series(maximo, index=range(1, len(maximo) + 1))
        maximo = maximo.copy()
        pasos, periodo = [], 1
        while periodo <= duracion_proyecto:
            if periodo <= len(maximo):
                limite = maximo[periodo]
            else:
                limite = maximo.iloc[-1]
                maximo[periodo] = limite
            df = pd.concat([gantt_df.data, my_data[duration_label], empezadas], axis=1)
            total = df.loc['Total', periodo]
            df = (df.loc[:, [periodo, 'H_total', duration_label, 'Empezadas']]
                    .replace('', None).dropna(subset=periodo)
                    .sort_values(by=['Empezadas', 'H_total', duration_label], ascending=[False, True, True])
                    .drop(labels='Total', axis=0))
            elegibles = df.index.to_list()
            paso = dict(periodo=periodo, total=total, limite=limite, candidatas=df.copy(),
                        antes=my_data.copy(), proy_antes=p.copy(), infactible=False)
            if total > limite:
                acumulada, se_programan, se_retrasan, acum_en = 0, set(), set(), {}
                for a in elegibles:
                    if acumulada + df.loc[a, periodo] <= limite:
                        acumulada += df.loc[a, periodo]
                        se_programan.add(a)
                        empezadas.loc[a, 'Empezadas'] = True
                    acum_en[a] = acumulada
                    if a not in se_programan:
                        se_retrasan.add(a)
                paso.update(se_programan=se_programan, se_retrasan=se_retrasan, acumulada=acum_en)
                if not se_programan:
                    paso.update(infactible=True, despues=my_data.copy(), proy_despues=p.copy())
                    pasos.append(paso)
                    break
                my_data, gantt_df, _ = p.desplazar(data=my_data, duration_label=duration_label,
                                                   resource_label=resource_label, tikz=False, report=False,
                                                   **{a: 1 for a in sorted(se_retrasan)})
            else:
                acumulada = df[periodo].cumsum()
                se_programan = set(df.loc[acumulada <= limite, :].index)
                empezadas.loc[list(se_programan), 'Empezadas'] = True
                paso.update(se_programan=se_programan, se_retrasan=set(),
                            acumulada=acumulada.to_dict())
            paso['despues'] = my_data.copy()
            paso['proy_despues'] = p.copy()
            pasos.append(paso)
            periodo += 1
            gantt_df, _ = p.gantt(data=my_data, duration_label=duration_label, resource_label=resource_label,
                                  holguras=True, total='fila', tikz=False)
            duracion_proyecto = gantt_df.data.columns.drop('H_total').max()
        return pasos, my_data, maximo, p

    def asignar_pasos_tikz_html(self, data, duration_label='duration', resource_label='resources',
                                maximo=None, escala: float = 1.0) -> str:
        """Visor paso a paso del algoritmo de asignación (:meth:`asignar`) en TikZ. El paso 0
        es el resultado final. Después, para cada periodo evaluado: si la demanda cabe en el
        límite, un fotograma; si no, tres: (1) las actividades candidatas numeradas por orden
        de prioridad y el criterio que se aplica, (2) cuáles se ejecutan (verde) y cuáles se
        posponen (rojo) al ir acumulando la carga hasta el límite, y (3) el diagrama tras
        desplazar un periodo las pospuestas. No modifica el proyecto."""
        import base64, html as _h, re as _re
        p = self.copy()
        pasos, finales, maximo, p_fin = p.asignar_pasos(data, duration_label, resource_label, maximo)

        def gantt_tikz(proy, datos, periodos):
            m = maximo.loc[1:periodos].to_frame(name='Máximo').T
            g, dib = proy.gantt(data=datos, duration_label=duration_label, resource_label=resource_label,
                             holguras=True, total='fila', tikz=True, params=dict(period_width=1), extra_rows=m)
            return g, dib.replace("💤", "zz")

        def filas_y_periodos(proy, datos):
            g, _ = proy.gantt(data=datos, duration_label=duration_label, resource_label=resource_label,
                           holguras=True, total='fila', tikz=False)
            return ([f for f in g.data.index if f not in ('Total', 'Máximo') and not str(f).startswith("💤")],
                    int(g.data.columns.drop('H_total').max()))

        filas_fin, P_fin = filas_y_periodos(p_fin, finales)
        n_fin = len(filas_fin)
        BX0, BX1 = -2.2, P_fin + 2 + 5.6
        BY0, BY1 = 1.1, -(n_fin + 2) * 0.8 - 0.3

        def fotograma(proy, datos, periodo, marcas=None, notas=None, pasado=True, hasta=None):
            filas, P = filas_y_periodos(proy, datos)
            P = max(P, periodo)
            _, dib = gantt_tikz(proy, datos, P)
            n = len(filas)
            ov = []
            if pasado and periodo > 1:
                ov.append(rf"\fill[black!12,opacity=0.35] (0,0.8) rectangle ({periodo - 1},{-(n + 2) * 0.8});")
            if periodo is not None:
                fin_ = periodo if hasta is None else hasta
                ov.append(rf"\fill[orange!35,opacity=0.35] ({periodo - 1},0.8) rectangle ({fin_},{-(n + 2) * 0.8});")
                ov.append(rf"\draw[orange!80!black,very thick] ({periodo - 1},0.8) rectangle ({fin_},{-(n + 2) * 0.8});")
            col = {"cand": "orange!80!black", "ok": "green!50!black", "no": "red!75!black"}
            for a, (rank, est) in (marcas or {}).items():
                if a in filas:
                    r = filas.index(a)
                    ov.append(rf"\node[circle,fill={col[est]},text=white,font=\scriptsize\bfseries,inner sep=1pt,"
                              rf"minimum size=0.36cm] at ({periodo - 0.8},{-0.8 * r - 0.2}) {{{rank}}};")
            for a, txt in (notas or {}).items():
                if a in filas:
                    r = filas.index(a)
                    ov.append(rf"\node[anchor=west,font=\scriptsize,{'green!40!black' if txt.startswith('ejec') else 'red!70!black'}]"
                              rf" at ({P + 2.15},{-0.8 * r - 0.4}) {{{txt}}};")
            lineas = dib.strip().split("\n")
            assert lineas[0].lstrip().startswith(r"\begin{tikzpicture}")
            cuerpo = "\n".join(lineas[1:-1])
            return (lineas[0].replace("scale=1", f"scale={escala}") + "\n"
                    rf"\useasboundingbox ({BX0},{BY0}) rectangle ({BX1:.2f},{BY1:.2f});" + "\n"
                    + cuerpo + "\n" + "\n".join(ov) + "\n" + r"\end{tikzpicture}")

        f = lambda v: f"{v:g}"
        codigos, pies = [], []
        # paso 0: resultado final
        P0 = P_fin
        _, dib0 = gantt_tikz(p_fin, finales, P0)
        lineas = dib0.strip().split("\n")
        codigos.append(lineas[0].replace("scale=1", f"scale={escala}") + "\n"
                       rf"\useasboundingbox ({BX0},{BY0}) rectangle ({BX1:.2f},{BY1:.2f});" + "\n"
                       + "\n".join(lineas[1:-1]) + "\n\\end{tikzpicture}")
        pies.append("Resultado del algoritmo de asignación con los límites de recursos de la fila «Máximo». "
                    "En cada periodo, si la demanda supera el límite, se ejecutan primero las actividades ya empezadas, "
                    "después las de menor holgura total y, a igualdad, las más cortas, mientras la carga acumulada "
                    "no supere el límite; las demás se posponen un periodo (💤 en el diagrama). "
                    "Pulse «siguiente» para ver cómo se decide periodo a periodo.".replace("💤", "zz"))
        grupos, run = [], []
        for ps in pasos:
            if ps["total"] <= ps["limite"]:
                # periodos consecutivos sin conflicto y con las mismas candidatas: un solo fotograma
                if run and tuple(run[-1]["candidatas"].index) != tuple(ps["candidatas"].index):
                    grupos.append(run); run = []
                run.append(ps)
                continue
            if run:
                grupos.append(run); run = []
            grupos.append(ps)
        if run:
            grupos.append(run)
        for ps in grupos:
            if isinstance(ps, list):
                ps0, ps1 = ps[0], ps[-1]
                if len(ps) == 1:
                    ps = ps0
                else:
                    demandas = ", ".join(f(q["total"]) for q in ps)
                    codigos.append(fotograma(ps0["proy_antes"], ps0["antes"], ps0["periodo"], hasta=ps1["periodo"]))
                    pies.append(f"Periodos {ps0['periodo']} a {ps1['periodo']}: la demanda ({demandas}) no supera el "
                                f"máximo en ninguno ({', '.join(f(q['limite']) for q in ps)}), con las mismas actividades "
                                f"({', '.join(str(a) for a in ps0['candidatas'].index)}); se ejecutan todas y no hay nada "
                                "que decidir.")
                    continue
            per, tot, lim, cand = ps["periodo"], ps["total"], ps["limite"], ps["candidatas"]
            etiqueta = {a: ("ya empezada" if bool(cand.loc[a, "Empezadas"]) else "no empezada") for a in cand.index}
            orden_txt = "; ".join(
                f"{i + 1}.º {a} (carga {f(cand.loc[a, per])}, {etiqueta[a]}, H={f(cand.loc[a, 'H_total'])}, d={f(cand.loc[a, duration_label])})"
                for i, a in enumerate(cand.index))
            rank = {a: i + 1 for i, a in enumerate(cand.index)}
            if tot <= lim:
                codigos.append(fotograma(ps["proy_antes"], ps["antes"], per, {a: (rank[a], "ok") for a in cand.index},
                                         {a: "ejecuta" for a in cand.index}))
                pies.append(f"Periodo {per}: la demanda total ({f(tot)}) no supera el máximo ({f(lim)}). "
                            f"Se ejecutan todas las actividades previstas: {', '.join(str(a) for a in cand.index)}.")
                continue
            codigos.append(fotograma(ps["proy_antes"], ps["antes"], per, {a: (rank[a], "cand") for a in cand.index}))
            pies.append(f"Periodo {per}: la demanda total ({f(tot)}) supera el máximo ({f(lim)}); hay que decidir qué "
                        "actividades se ejecutan. Criterio de prioridad: (1) las ya empezadas, que no se interrumpen; "
                        "(2) menor holgura total H, las más críticas; (3) menor duración d. Orden resultante: "
                        + orden_txt + ".")
            marcas = {a: (rank[a], "ok" if a in ps["se_programan"] else "no") for a in cand.index}
            notas = {}
            for a in cand.index:
                if a in ps["se_programan"]:
                    notas[a] = rf"ejecuta (acum. {f(ps['acumulada'][a])} $\le$ {f(lim)})"
                else:
                    notas[a] = rf"pospone ({f(prev_acum(ps, a, cand))}+{f(cand.loc[a, per])} $>$ {f(lim)})"
            codigos.append(fotograma(ps["proy_antes"], ps["antes"], per, marcas, notas))
            ejec = [str(a) for a in cand.index if a in ps["se_programan"]]
            pos_ = [str(a) for a in cand.index if a in ps["se_retrasan"]]
            if ps.get("infactible"):
                pies.append(f"Periodo {per}: no cabe ninguna actividad con el máximo disponible ({f(lim)}): "
                            "el proyecto no es factible con estos recursos.")
                continue
            pies.append(f"Periodo {per}: se recorren las actividades por orden de prioridad acumulando carga. "
                        f"Se ejecutan {', '.join(ejec)}; se posponen {', '.join(pos_)} porque, al añadirlas, la carga "
                        f"acumulada superaría el máximo ({f(lim)}).")
            codigos.append(fotograma(ps["proy_despues"], ps["despues"], per))
            pies.append(f"Periodo {per}: las actividades pospuestas ({', '.join(pos_)}) se desplazan un periodo, "
                        "con las que dependen de ellas; este es el diagrama de partida del siguiente periodo.")
        fotos = []
        for svg, pie in zip(_svgs_de_tikz(codigos, "calc,arrows.meta,patterns"), pies):
            b64 = base64.b64encode(svg.encode("utf8")).decode("ascii")
            fotos.append('<img alt="paso" style="display:block;margin:0 auto;width:100%;max-width:1100px;height:auto" '
                         f'src="data:image/svg+xml;base64,{b64}">'
                         f'<p style="text-align:center;margin-top:.5em">{_h.escape(pie)}</p>')
        return _html_pasos(fotos, inicio=0)

    def display_asignar_pasos_tikz(self, data, duration_label='duration', resource_label='resources',
                                   maximo=None, escala: float = 1.0) -> None:
        """Muestra en un cuaderno la asignación paso a paso (:meth:`asignar_pasos_tikz_html`)."""
        display(HTML(self.asignar_pasos_tikz_html(data, duration_label, resource_label, maximo, escala)))

    # ------------------------------------------------------------------
    # Asignación con varios recursos limitados
    # ------------------------------------------------------------------

    def _real_predecessors(self):
        """Predecesoras inmediatas de cada actividad real, deducidas del grafo.

        Una actividad real ``a`` (arco u→v) necesita terminadas todas las
        actividades reales que terminan en u, atravesando las ficticias
        (``@…``) que llegan a u.  No modifica el grafo.
        """
        G = self.pert_graph
        finishing = {}
        for node in nx.topological_sort(G):
            acc = set()
            for u, _, attr in G.in_edges(node, data=True):
                act = attr["activity"]
                if act[0] == "@":
                    acc |= finishing[u]
                else:
                    acc.add(act)
            finishing[node] = acc
        preds = {}
        for u, _, attr in G.edges(data=True):
            act = attr["activity"]
            if act[0] != "@":
                preds[act] = set(finishing[u])
        return preds

    @staticmethod
    def _limit_vector(spec, label):
        """Límite de un recurso por periodo → (vector desde el periodo 1, n)."""
        if isinstance(spec, pd.Series):
            if spec.index.min() != 1:
                raise ValueError(
                    f"Error: el límite de '{label}' debe estar indexado desde el periodo 1")
            vec = spec.sort_index().reindex(range(1, int(spec.index.max()) + 1)).ffill()
            return vec.to_numpy(dtype=float)
        if isinstance(spec, (list, tuple, np.ndarray)):
            return np.asarray(spec, dtype=float)
        return np.array([float(spec)])

    def _vectores_limite(self, resource_labels, maximos):
        """{recurso: vector de límites desde el periodo 1} a partir de ``maximos``
        (escalar, lista con un valor por recurso, dict o DataFrame; ver ``asignar_multi``)."""
        K = len(resource_labels)
        if isinstance(maximos, pd.DataFrame):
            specs = {rl: maximos[rl] for rl in resource_labels}
        elif isinstance(maximos, dict):
            falta = [rl for rl in resource_labels if rl not in maximos]
            if falta:
                raise ValueError(f"Error: faltan límites para {falta}")
            specs = {rl: maximos[rl] for rl in resource_labels}
        elif isinstance(maximos, (list, tuple, np.ndarray)):
            if len(maximos) != K:
                raise ValueError(
                    "Error: una lista de maximos debe tener un valor por recurso; "
                    "para límites variables en el tiempo use un dict o un DataFrame")
            specs = dict(zip(resource_labels, maximos))
        else:
            specs = {rl: maximos for rl in resource_labels}
        vectors = {rl: self._limit_vector(specs[rl], rl) for rl in resource_labels}
        return vectors

    def asignar_multi(self, data, duration_label="duration", resource_labels=None,
                      maximos=None, report=False, tikz=False, params=None, dibujar=True):
        """Asignación de recursos con límites en VARIOS recursos a la vez.

        Esquema en paralelo (parallel schedule generation), sin interrupciones:
        se recorren los periodos 1, 2, 3… y en cada uno se consideran las
        actividades no empezadas cuyas predecesoras ya han terminado, por
        prioridad (menor ``LS`` del CPM sin recursos, luego menor duración,
        luego nombre).  Una actividad empieza si, **para todos los recursos y
        en todos los periodos que ocuparía**, la carga ya comprometida más su
        demanda no supera el límite; si algún recurso se satura, espera.

        No modifica ``self``.

        Parameters
        ----------
        data : DataFrame con la duración y una columna por recurso.
        resource_labels : lista de columnas de recurso.
        maximos : límite por recurso.  Un escalar (mismo límite para todos),
            una lista/tupla/array con un límite por recurso (en el orden de
            ``resource_labels``), un dict ``{recurso: escalar | lista | Series}``
            o un DataFrame (columnas = recursos, índice desde el periodo 1) para
            límites que varían con el tiempo.  Una lista o Series por recurso
            se interpreta desde el periodo 1; el último valor se prolonga.
        report : si True, muestra la traza con Markdown (como ``asignar``).
        tikz : si True, dibuja también el Gantt TikZ de cada recurso.
        dibujar : si False no construye los Gantt (``gantts``/``dibujos`` vacíos, ``proyecto``
            None y ``data`` sin filas de espera): mucho más rápido para bucles y simulaciones.

        Returns
        -------
        AsignacionMulti con ``data`` (duraciones con filas de espera ``💤X``),
        ``inicios``, ``fines``, ``esperas``, ``duracion``, ``cargas``,
        ``limites``, ``traza`` (decisión y recurso limitante por periodo y
        actividad), ``gantts`` y ``dibujos`` (dict por recurso).
        """
        if isinstance(resource_labels, str):
            resource_labels = [resource_labels]
        if not resource_labels:
            raise ValueError("Error: Debe indicar resource_labels (lista de recursos)")
        resource_labels = list(resource_labels)
        if maximos is None:
            raise ValueError("Error: Debe introducir maximos (un límite por recurso)")
        if duration_label not in data.columns:
            raise ValueError("Error: Duration label not in data columns")
        for rl in resource_labels:
            if rl not in data.columns:
                raise ValueError(f"Error: Resource label '{rl}' not in data columns")
        K = len(resource_labels)

        vectors = self._vectores_limite(resource_labels, maximos)

        preds = self._real_predecessors()
        acts = sorted(preds)
        faltan = [a for a in acts if a not in data.index]
        if faltan:
            raise ValueError(f"Error: faltan filas en data para {faltan}")
        dur = {a: int(data.loc[a, duration_label]) for a in acts}
        dem = {a: np.array([float(data.loc[a, rl]) for rl in resource_labels]) for a in acts}
        for a in acts:
            for k, rl in enumerate(resource_labels):
                if dem[a][k] > vectors[rl].max():
                    raise ValueError(
                        f"Error: la actividad {a} requiere más '{rl}' ({dem[a][k]:g}) "
                        f"que el máximo permitido ({vectors[rl].max():g})")

        # planificador en paralelo (prioridad: LS del CPM sin recursos)
        orden = _orden_topologico(preds)
        n0 = max(len(v) for v in vectors.values())
        horizonte = n0 + sum(dur.values()) + 1
        maxd = max(dur.values(), default=0)
        ancho = horizonte + maxd + 1
        limite = _matriz_limite([vectors[rl] for rl in resource_labels], ancho)
        inicio, fin, traza, carga = _planificar_paralelo(
            preds, orden, dur, dem, limite, horizonte, resource_labels)

        duracion = max(fin.values(), default=0)
        traza = pd.DataFrame(traza, columns=["periodo", "actividad", "decision", "bloqueada_por"])
        inicios = pd.Series(inicio).sort_index()
        fines = pd.Series(fin).sort_index()
        esperas = pd.Series({
            a: inicio[a] - max((fin[p] for p in preds[a]), default=0) for a in acts
        }).sort_index()
        esperas = esperas[esperas > 0]

        cargas = pd.DataFrame(carga[:, :duracion].T, index=range(1, duracion + 1),
                              columns=resource_labels)
        limites = pd.DataFrame(limite[:, :duracion].T, index=range(1, duracion + 1),
                               columns=resource_labels)
        cargas.index.name = limites.index.name = "periodo"

        proyecto, my_data, gantts, dibujos = None, data.copy(), {}, {}
        if dibujar:
            # Gantt con las esperas, sobre una copia (desplazar muta el grafo)
            proyecto = self.copy()
            if len(esperas):
                # orden cronológico, y alfabético dentro de cada periodo (como asignar)
                shifts = {a: int(esperas[a]) for a in sorted(esperas.index, key=lambda x: (inicio[x], x))}
                my_data, _, _ = proyecto.desplazar(
                    my_data, duration_label, resource_labels[0], report=False,
                    cuadrados=False, **shifts)
            for rl in resource_labels:
                gantts[rl], dibujos[rl] = proyecto.gantt(
                    data=my_data, duration_label=duration_label, resource_label=rl,
                    holguras=True, total="fila", tikz=tikz, params=params,
                    extra_rows=limites[[rl]].T.rename(index={rl: "Máximo"}))

        if report:
            for periodo, grupo in traza.groupby("periodo"):
                emp = list(grupo.loc[grupo.decision == "empieza", "actividad"])
                esp = [f"{r.actividad} ({r.bloqueada_por})"
                       for r in grupo.loc[grupo.decision == "espera"].itertuples()]
                display(Markdown(f"Periodo {periodo}. Empiezan: {emp}. Esperan: {esp}"))

        return AsignacionMulti(my_data, inicios, fines, esperas, duracion, cargas,
                               limites, traza, gantts, dibujos, proyecto)

    # ------------------------------------------------------------------
    # Crashing con recursos limitados
    # ------------------------------------------------------------------

    def _preparar_crashing(self, data, duration_label, resource_labels, maximos,
                           min_durations, costs, demanda_extra):
        """Datos normalizados para ``crashing_recursos`` y ``crashing_recursos_exacto``."""
        resource_labels = [resource_labels] if isinstance(resource_labels, str) else list(resource_labels)
        if not resource_labels:
            raise ValueError("Error: Debe indicar resource_labels")
        if maximos is None:
            raise ValueError("Error: Debe introducir maximos (un límite por recurso)")
        for col in [duration_label] + resource_labels:
            if col not in data.columns:
                raise ValueError(f"Error: la columna '{col}' no está en data")
        K = len(resource_labels)
        vectors = self._vectores_limite(resource_labels, maximos)
        preds = self._real_predecessors()
        orden = _orden_topologico(preds)
        faltan = [a for a in orden if a not in data.index]
        if faltan:
            raise ValueError(f"Error: faltan filas en data para {faltan}")
        dur0 = {a: int(data.loc[a, duration_label]) for a in orden}
        min_durations = pd.Series(min_durations, dtype=float)
        costs = pd.Series(costs, dtype=float)
        mind = {a: int(min_durations.get(a, dur0[a])) for a in orden}
        ucr = {a: float(costs.get(a, np.inf)) for a in orden}
        for a in orden:
            if mind[a] > dur0[a]:
                raise ValueError(f"Error: la duración mínima de {a} supera a la normal")
        base = {a: np.array([float(data.loc[a, rl]) for rl in resource_labels]) for a in orden}
        extra = {a: np.zeros(K) for a in orden}
        if demanda_extra is not None:
            if isinstance(demanda_extra, pd.DataFrame):
                demanda_extra = {a: demanda_extra.loc[a, resource_labels].to_numpy(dtype=float)
                                 for a in demanda_extra.index}
            for a, v in dict(demanda_extra).items():
                if a not in extra:
                    continue
                if isinstance(v, dict):
                    extra[a] = np.array([float(v.get(rl, 0.0)) for rl in resource_labels])
                else:
                    extra[a] = np.asarray(v, dtype=float).reshape(K)
        maxlim = np.array([vectors[rl].max() for rl in resource_labels])
        for a in orden:
            if np.any(base[a] > maxlim + 1e-9):
                raise ValueError(f"Error: la actividad {a} requiere más recurso que el máximo permitido")
        n0 = max(len(v) for v in vectors.values())
        horizonte = n0 + sum(dur0.values()) + 1
        ancho = horizonte + max(dur0.values(), default=0) + 1
        limite = _matriz_limite([vectors[rl] for rl in resource_labels], ancho)
        return dict(etiquetas=resource_labels, K=K, preds=preds, orden=orden, dur0=dur0, mind=mind,
                    ucr=ucr, base=base, extra=extra, maxlim=maxlim, horizonte=horizonte,
                    limite=limite, vectors=vectors)

    @staticmethod
    def _corte_minimo(arcos, criticas, fuentes, sumideros, coste, reducible):
        """Corte de coste mínimo (por actividades) de la red de arcos ajustados."""
        INF = 1e12
        G = nx.DiGraph()
        for a in criticas:
            G.add_edge((a, "in"), (a, "out"), capacity=coste[a] if reducible(a) else INF)
        for b, antes in arcos.items():
            for a in antes:
                G.add_edge((a, "out"), (b, "in"), capacity=INF)
        for a in fuentes:
            G.add_edge("fuente", (a, "in"), capacity=INF)
        for a in sumideros:
            G.add_edge((a, "out"), "sumidero", capacity=INF)
        valor, (S, T_) = nx.minimum_cut(G, "fuente", "sumidero")
        if valor >= INF / 2:
            return None
        return sorted(a for a in criticas if (a, "in") in S and (a, "out") in T_)

    def cadena_critica_recursos(self, data, duration_label, resource_labels, maximos):
        """Cadena crítica con recursos del plan que produce ``asignar_multi``.

        Son las actividades desde las que se llega al final del proyecto por «arcos ajustados»:
        la siguiente empieza justo cuando termina la anterior, porque la primera es su predecesora
        o porque libera el recurso por el que la segunda esperaba.  Devuelve un dict con
        ``duracion`` (con recursos), ``duracion_sin_recursos`` (CPM), ``actividades`` (lista) y
        ``arcos`` (DataFrame ``de``, ``a``, ``motivo``: 'precedencia' o 'recurso').
        """
        resource_labels = [resource_labels] if isinstance(resource_labels, str) else list(resource_labels)
        vectors = self._vectores_limite(resource_labels, maximos)
        preds = self._real_predecessors()
        orden = _orden_topologico(preds)
        dur = {a: int(data.loc[a, duration_label]) for a in orden}
        dem = {a: np.array([float(data.loc[a, rl]) for rl in resource_labels]) for a in orden}
        n0 = max(len(v) for v in vectors.values())
        horizonte = n0 + sum(dur.values()) + 1
        ancho = horizonte + max(dur.values(), default=0) + 1
        limite = _matriz_limite([vectors[rl] for rl in resource_labels], ancho)
        ini, fin, traza, _ = _planificar_paralelo(preds, orden, dur, dem, limite, horizonte, resource_labels)
        T = max(fin.values(), default=0)
        criticas, arcos = _cadena_critica(preds, ini, fin, traza, dem, T, resource_labels)
        filas = [dict(de=a, a=b, motivo="precedencia" if a in preds[b] else "recurso")
                 for b in sorted(arcos) for a in sorted(arcos[b])]
        return dict(duracion=T, duracion_sin_recursos=_cpm_puro(preds, dur, orden)[2],
                    actividades=sorted(criticas),
                    arcos=pd.DataFrame(filas, columns=["de", "a", "motivo"]))

    def plan_cadena_critica(self, data, agresiva="agresiva", segura="segura", recurso=None,
                            metodo="cortar_pegar", metodo_alimentacion=None,
                            secuencia_recursos=None):
        """Plan de cadena crítica (CCPM, Goldratt) con estimaciones agresivas y seguras.

        Sigue los pasos de la lección 12 del curso:

        1. **Conflictos de recursos.** Cada recurso de la columna ``recurso`` es *una unidad*
           (una cuadrilla, una persona): dos actividades que lo usan no pueden solaparse. Se
           programan con el planificador en paralelo de ``asignar_multi`` (prioridad: menor
           inicio más tardío del CPM sin recursos, menor duración, nombre) con las duraciones
           agresivas, y el orden en que cada recurso atiende a sus actividades se añade a la red
           como **enlaces de recurso** (``secuencia_recursos`` = ``{recurso: [actividades en
           orden]}`` lo impone a mano). La red queda *nivelada*: cualquier programación que
           respete sus prelaciones respeta los recursos.
        2. **Cadena crítica**: la ruta más larga de la red nivelada con las duraciones
           agresivas. Si empatan varias se toma la de mayor suma de márgenes y, después, la
           primera por orden alfabético (``cadenas_empatadas`` dice cuántas había).
        3. **Colchón del proyecto** (CP), al final de la cadena, con los márgenes
           ``segura − agresiva`` de sus actividades: ``"cortar_pegar"`` (la mitad de la suma) o
           ``"raiz"`` (raíz de la suma de los cuadrados), redondeado al entero superior.
        4. **Colchones de alimentación** (CA) en cada arco de la red nivelada que entra en la
           cadena desde una actividad que no es de ella (o al final de la cadena, si esa
           actividad no tiene sucesoras). Se dimensionan con los márgenes de la cadena de
           alimentación más larga que llega a ese arco (método ``metodo_alimentacion``, por
           defecto el mismo) y se adopta el menor entre el colchón requerido y la holgura
           disponible: la cadena crítica no se retrasa; si el colchón no cabe, la cadena de
           alimentación empieza en su inicio más temprano y el colchón se reduce. Las
           actividades de alimentación se programan **lo más tarde posible**, con su colchón
           por delante de la cadena.
        5. **Colchones de recursos**: avisos (no tiempo) antes de cada actividad de la cadena que
           usa un recurso que no usaba su predecesora en la cadena.

        ``data``: DataFrame indexado por actividad con las columnas ``agresiva`` y ``segura``
        (enteras) y, opcionalmente, ``recurso`` (nombre del recurso; varios separados por comas;
        vacío o ``'---'`` si no usa ninguno). No modifica ``self``. Devuelve un
        :class:`PlanCadenaCritica`.
        """
        if metodo not in ("cortar_pegar", "raiz"):
            raise ValueError("Error: metodo debe ser 'cortar_pegar' o 'raiz'")
        metodo_alimentacion = metodo_alimentacion or metodo
        if metodo_alimentacion not in ("cortar_pegar", "raiz"):
            raise ValueError("Error: metodo_alimentacion debe ser 'cortar_pegar' o 'raiz'")
        for col in (agresiva, segura) + ((recurso,) if recurso else ()):
            if col not in data.columns:
                raise ValueError(f"Error: la columna '{col}' no está en data")
        preds = self._real_predecessors()
        orden = _orden_topologico(preds)
        faltan = [a for a in orden if a not in data.index]
        if faltan:
            raise ValueError(f"Error: faltan filas en data para {faltan}")
        d = {a: int(data.loc[a, agresiva]) for a in orden}
        s = {a: int(data.loc[a, segura]) for a in orden}
        negativos = [a for a in orden if s[a] < d[a]]
        if negativos:
            raise ValueError(f"Error: la estimación segura es menor que la agresiva en {negativos}")
        usa = {a: (_recursos_unitarios(data.loc[a, recurso]) if recurso else ()) for a in orden}
        nombres = sorted(set().union(*(set(u) for u in usa.values()))) if orden else []

        # 1. Conflictos de recursos -> enlaces de recurso (red nivelada)
        traza, secuencias = [], {}
        if nombres:
            dem = {a: np.array([1.0 if r in usa[a] else 0.0 for r in nombres]) for a in orden}
            horizonte = sum(d.values()) + 1
            limite = np.ones((len(nombres), horizonte + max(d.values(), default=0) + 1))
            ini, _, traza, _ = _planificar_paralelo(preds, orden, d, dem, limite, horizonte, nombres)
            secuencias = {r: sorted((a for a in orden if r in usa[a] and d[a] > 0), key=lambda a: (ini[a], a))
                          for r in nombres}
            for r, seq in (secuencia_recursos or {}).items():
                if r not in secuencias or set(seq) != set(secuencias[r]):
                    raise ValueError(f"Error: la secuencia del recurso {r} debe contener exactamente "
                                     f"{sorted(secuencias.get(r, []))}")
                secuencias[r] = list(seq)
        G = nx.DiGraph()
        G.add_nodes_from(orden)
        G.add_edges_from((p, a) for a in orden for p in preds[a])
        enlaces = []
        for r in nombres:
            seq = secuencias[r]
            for x, y in zip(seq, seq[1:]):
                if nx.has_path(G, y, x):
                    raise ValueError(f"Error: la secuencia del recurso {r} contradice las prelaciones ({x} → {y})")
                if not nx.has_path(G, x, y):
                    enlaces.append((x, y, r))
                    G.add_edge(x, y)
        preds_niv = {a: set(preds[a]) for a in orden}
        for x, y, _ in enlaces:
            preds_niv[y].add(x)
        orden_niv = _orden_topologico(preds_niv)
        succ_niv = {a: set() for a in orden_niv}
        for a in orden_niv:
            for p in preds_niv[a]:
                succ_niv[p].add(a)
        ES, LS, T = _cpm_puro(preds_niv, d, orden_niv)
        margen = {a: s[a] - d[a] for a in orden}

        # 2. Cadena crítica
        criticas = [a for a in orden_niv if ES[a] == LS[a] and d[a] > 0]
        Gc = nx.DiGraph()
        Gc.add_nodes_from(criticas)
        Gc.add_edges_from((p, a) for a in criticas for p in preds_niv[a]
                          if p in Gc and ES[p] + d[p] == ES[a])
        fuentes = [a for a in criticas if ES[a] == 0]
        sumideros = [a for a in criticas if ES[a] + d[a] == T]
        caminos = []
        for f in fuentes:
            for t in sumideros:
                if f == t:
                    caminos.append([f])
                else:
                    caminos.extend(islice(nx.all_simple_paths(Gc, f, t), 2000))
        caminos.sort(key=lambda c: (-sum(margen[a] for a in c), c))
        cadena = caminos[0] if caminos else []
        C = set(cadena)

        def tam(ms, met):
            bruto = 0.5 * sum(ms) if met == "cortar_pegar" else math.sqrt(sum(m * m for m in ms))
            return bruto, int(math.ceil(bruto - 1e-9))

        # 3. Colchón del proyecto
        ms_cadena = [margen[a] for a in cadena]
        calculo_cp = {"cortar_pegar": tam(ms_cadena, "cortar_pegar")[0], "raiz": tam(ms_cadena, "raiz")[0]}
        cp = tam(ms_cadena, metodo)[1]

        # 4. Colchones de alimentación: cadena de alimentación más larga que termina en cada actividad
        mejor = {}
        for a in orden_niv:
            if a in C:
                continue
            cands = [mejor[q] for q in preds_niv[a] if q not in C]
            if cands:
                b = sorted(cands, key=lambda c: (-c[0], -c[1], c[2]))[0]
                mejor[a] = (b[0] + d[a], b[1] + margen[a], b[2] + [a])
            else:
                mejor[a] = (d[a], margen[a], [a])
        posicion = {a: k for k, a in enumerate(cadena)}
        fusiones = [(p, c) for c in cadena for p in sorted(preds_niv[c] - C)]
        fusiones += [(p, None) for p in orden_niv if p not in C and not succ_niv[p]]
        fusiones.sort(key=lambda pc: ((ES[pc[1]], posicion[pc[1]]) if pc[1] is not None else (T, len(cadena)), pc[0]))
        filas, fb = [], {}
        for k, (p, c) in enumerate(fusiones, 1):
            cad = mejor[p][2]
            ms = [margen[x] for x in cad]
            bruto, requerido = tam(ms, metodo_alimentacion)
            entrada = ES[c] if c is not None else T
            disponible = entrada - (ES[p] + d[p])
            adoptado = max(0, min(requerido, disponible))
            fb[(p, c)] = adoptado
            filas.append(dict(nombre=f"CA{k}", desde=p, antes_de=c if c is not None else "fin",
                              cadena_alimentacion=" - ".join(cad), margenes=" + ".join(str(m) for m in ms),
                              bruto=round(bruto, 2), requerido=requerido, disponible=disponible,
                              adoptado=adoptado, inicio=entrada - adoptado, fin=entrada))
        alimentacion = pd.DataFrame(filas, columns=["nombre", "desde", "antes_de", "cadena_alimentacion",
                                                    "margenes", "bruto", "requerido", "disponible",
                                                    "adoptado", "inicio", "fin"])
        alimentacion = alimentacion.set_index("nombre") if len(alimentacion) else alimentacion

        # Programación: cadena lo antes posible, alimentación lo más tarde posible
        inicio = {a: ES[a] for a in cadena}
        for a in reversed(orden_niv):
            if a in C:
                continue
            lims = [(ES[y] - fb[(a, y)]) if y in C else inicio[y] for y in succ_niv[a]]
            if not succ_niv[a]:
                lims.append(T - fb[(a, None)])
            inicio[a] = min(lims) - d[a]

        # 5. Colchones de recursos
        avisos = []
        for x, y in zip(cadena, cadena[1:]):
            nuevos = [r for r in usa[y] if r not in usa[x]]
            if nuevos:
                avisos.append(dict(antes_de=y, recursos=", ".join(nuevos), despues_de=x, inicio_previsto=ES[y]))

        enlaces_cadena = pd.DataFrame(
            [dict(de=x, a=y, motivo="precedencia" if x in preds[y] else "recurso",
                  recurso=", ".join(r for r in usa[y] if r in usa[x])) for x, y in zip(cadena, cadena[1:])],
            columns=["de", "a", "motivo", "recurso"])
        ES_s, _, T_s = _cpm_puro(preds_niv, s, orden_niv)
        return PlanCadenaCritica(
            cadena=cadena, enlaces_cadena=enlaces_cadena,
            enlaces_recurso=pd.DataFrame(enlaces, columns=["de", "a", "recurso"]),
            predecesoras=preds_niv, predecesoras_originales=preds, orden=orden_niv,
            agresiva=pd.Series(d, name=agresiva).reindex(sorted(d)),
            segura=pd.Series(s, name=segura).reindex(sorted(s)),
            recursos={a: tuple(usa[a]) for a in orden},
            duracion_cadena=T, margen=pd.Series(margen, name="margen").reindex(sorted(margen)),
            colchon_proyecto=cp, calculo_colchon_proyecto=calculo_cp, metodo=metodo,
            metodo_alimentacion=metodo_alimentacion, alimentacion=alimentacion,
            colchones_recursos=pd.DataFrame(avisos, columns=["antes_de", "recursos", "despues_de", "inicio_previsto"]),
            inicios=pd.Series(inicio, name="inicio").reindex(sorted(inicio)).astype(int),
            inicio_temprano=pd.Series(ES, name="inicio_temprano").reindex(sorted(ES)).astype(int),
            duracion_tradicional=T_s,
            inicio_tradicional=pd.Series(ES_s).reindex(sorted(ES_s)).astype(int),
            cadenas_empatadas=len(caminos), traza=traza, secuencias=secuencias,
            colchones_alimentacion_arco=fb)

    def crashing_recursos(self, data, duration_label, resource_labels, maximos,
                          min_durations, costs, objetivo=None, reduccion=None,
                          demanda_extra=None, max_pasos=200, combinaciones=2,
                          dibujar=False):
        """Reducción de la duración (crashing) de un proyecto con recursos limitados.

        Con límites de recursos, el camino crítico deja de ser lo que manda: la duración
        la fija la **cadena crítica con recursos** (caminos de precedencia más esperas por
        recurso).  Acortar la actividad de menor coste unitario de reducción (Ackoff)
        puede no acortar nada.  En cada paso este método:

        1. planifica con ``asignar_multi`` (planificador en paralelo),
        2. halla la cadena crítica con recursos,
        3. evalúa como candidatos el corte de coste mínimo de esa cadena, cada actividad
           reducible por separado y (``combinaciones=2``) cada pareja,
        4. se queda con el candidato más barato que de verdad acorta la duración
           **ya replanificada**; si ninguno la acorta, se detiene.

        Dos modelos de reducción: con dinero y la demanda constante (por defecto: horas
        extra, mejor equipo) o con recurso (``demanda_extra``: unidades más de cada
        recurso por periodo recortado, de modo que acortar consume el recurso escaso).

        Parameters
        ----------
        data : DataFrame con la duración normal y una columna por recurso (demanda normal).
        min_durations, costs : duración mínima y coste unitario de reducción por actividad
            (``inf`` = no reducible).
        objetivo | reduccion : duración a alcanzar, o nº de periodos a recortar.  Si no se
            da ninguno se reduce mientras sea posible.
        demanda_extra : DataFrame (actividades × recursos), dict {act: {recurso: u}} o
            {act: lista}; demanda adicional por periodo recortado.

        Returns
        -------
        CrashingRecursos
        """
        if objetivo is not None and reduccion is not None:
            raise ValueError("Indique solo objetivo o reduccion")
        m = self._preparar_crashing(data, duration_label, resource_labels, maximos,
                                    min_durations, costs, demanda_extra)
        etiquetas, preds, orden = m["etiquetas"], m["preds"], m["orden"]
        dur0, mind, ucr, base, extra = m["dur0"], m["mind"], m["ucr"], m["base"], m["extra"]
        acts = list(orden)

        def evaluar(red):
            dur = {a: dur0[a] - red[a] for a in acts}
            dem = {a: base[a] + extra[a] * red[a] for a in acts}
            if any(np.any(dem[a] > m["maxlim"] + 1e-9) for a in acts):
                return None
            try:
                ini, fin, traza, _ = _planificar_paralelo(preds, orden, dur, dem, m["limite"],
                                                          m["horizonte"], etiquetas)
            except ValueError:
                return None
            return dict(T=max(fin.values(), default=0), ini=ini, fin=fin, traza=traza, dur=dur, dem=dem)

        import itertools
        red = {a: 0 for a in acts}
        ev = evaluar(red)
        if ev is None:
            raise ValueError("Error: el proyecto no es factible con los límites de recursos indicados")
        T0 = ev["T"]
        coste_acum, filas, motivo = 0.0, [], "máximo de pasos"
        for paso in range(1, max_pasos + 1):
            T = ev["T"]
            if objetivo is not None and T <= objetivo:
                motivo = "objetivo alcanzado"
                break
            if reduccion is not None and T <= T0 - reduccion:
                motivo = "reducción alcanzada"
                break
            criticas, arcos = _cadena_critica(preds, ev["ini"], ev["fin"], ev["traza"], ev["dem"], T, etiquetas)
            reducible = lambda a: dur0[a] - red[a] > mind[a] and np.isfinite(ucr[a])
            reducibles = [a for a in sorted(criticas) if reducible(a)]
            if not reducibles:
                motivo = "ninguna actividad de la cadena crítica es reducible"
                break
            candidatos = [(a,) for a in reducibles]
            if combinaciones >= 2:
                candidatos += list(itertools.combinations(reducibles, 2))
            fuentes = [a for a in criticas if not arcos[a]]
            sumideros = [a for a in criticas if ev["fin"][a] == T]
            corte = self._corte_minimo(arcos, criticas, fuentes, sumideros, ucr, reducible)
            if corte and all(reducible(a) for a in corte) and tuple(corte) not in candidatos:
                candidatos.append(tuple(corte))
            mejor, evaluados, sin_mejora = None, 0, 0
            for cand in candidatos:
                red2 = dict(red)
                for a in cand:
                    red2[a] += 1
                ev2 = evaluar(red2)
                if ev2 is None:
                    continue
                evaluados += 1
                if ev2["T"] >= T:
                    if len(cand) == 1:
                        sin_mejora += 1
                    continue
                clave = (sum(ucr[a] for a in cand), -(T - ev2["T"]), cand)
                if mejor is None or clave < mejor[0]:
                    mejor = (clave, cand, red2, ev2)
            if mejor is None:
                motivo = "ninguna reducción de una unidad acorta el plazo con estos recursos"
                break
            (coste_paso, _, _), cand, red, ev_nuevo = mejor
            coste_acum += coste_paso
            try:
                durs = pd.Series(ev["dur"])
                _, mej, _, _ = self.ackoff(durs, pd.Series(mind, dtype=float), pd.Series(ucr, dtype=float),
                                           reduction=1, report=False)
                libre = sorted(set(itertools.chain.from_iterable(mej.values())))
            except Exception:
                libre = []
            filas.append(dict(paso=paso, actividades=", ".join(cand), coste_paso=coste_paso,
                              coste_acumulado=coste_acum, duracion_antes=T, duracion_despues=ev_nuevo["T"],
                              cadena_critica=", ".join(sorted(criticas)),
                              ackoff_sin_recursos=", ".join(libre),
                              candidatos_evaluados=evaluados, reducciones_sin_efecto=sin_mejora,
                              _cand=cand))
            ev = ev_nuevo
        pasos = pd.DataFrame(filas)
        if len(pasos):
            pasos = pasos.drop(columns="_cand").set_index("paso")
        else:
            pasos = pd.DataFrame(columns=["actividades", "coste_paso", "coste_acumulado", "duracion_antes",
                                          "duracion_despues", "cadena_critica", "ackoff_sin_recursos",
                                          "candidatos_evaluados", "reducciones_sin_efecto"])
            pasos.index.name = "paso"
        duraciones = pd.Series(ev["dur"]).sort_index()
        demanda = pd.DataFrame(ev["dem"], index=etiquetas).T.sort_index()
        asignacion = None
        if dibujar:
            d2 = data.copy()
            d2[duration_label] = duraciones.reindex(d2.index).fillna(d2[duration_label]).astype(int)
            for rl in etiquetas:
                d2[rl] = demanda[rl].reindex(d2.index).fillna(d2[rl])
            asignacion = self.asignar_multi(d2, duration_label, etiquetas, maximos)
        return CrashingRecursos(pasos, duraciones, pd.Series(red).sort_index(), demanda, T0, ev["T"],
                                coste_acum, motivo, pd.Series(ev["ini"]).sort_index(), asignacion)

    def crashing_recursos_exacto(self, data, duration_label, resource_labels, maximos,
                                 min_durations, costs, objetivo, demanda_extra=None,
                                 time_limit=60.0):
        """Óptimo exacto (MILP indexado en el tiempo, HiGHS) del crashing con recursos.

        Minimiza el coste de reducción tal que existe una planificación factible (precedencias
        y límites de todos los recursos en cada periodo) de duración ≤ ``objetivo``, eligiendo a
        la vez la duración de cada actividad y su inicio.  Pensado para proyectos pequeños (≈10
        actividades): sirve de referencia para medir la calidad de ``crashing_recursos``.
        Requiere duraciones ≥ 1.  Devuelve un dict ``estado`` ('óptimo', 'infactible',
        'límite de tiempo'), ``coste``, ``duraciones`` e ``inicios`` (o None si no hay solución).
        """
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
        m = self._preparar_crashing(data, duration_label, resource_labels, maximos,
                                    min_durations, costs, demanda_extra)
        etiquetas, preds, orden, K = m["etiquetas"], m["preds"], m["orden"], m["K"]
        dur0, mind, ucr, base, extra, limite = m["dur0"], m["mind"], m["ucr"], m["base"], m["extra"], m["limite"]
        if min(dur0.values()) < 1:
            raise ValueError("crashing_recursos_exacto requiere duraciones ≥ 1")
        H = int(objetivo)
        var, por_act = [], {a: [] for a in orden}
        for a in orden:
            opciones = range(max(mind[a], 1), dur0[a] + 1) if np.isfinite(ucr[a]) else [dur0[a]]
            for d in opciones:
                dem = base[a] + extra[a] * (dur0[a] - d)
                if np.any(dem > m["maxlim"] + 1e-9):
                    continue
                for t in range(0, H - d + 1):
                    por_act[a].append(len(var))
                    var.append((a, d, t, dem))
        if any(not v for v in por_act.values()):
            return dict(estado="infactible", coste=None, duraciones=None, inicios=None)
        nv = len(var)
        c = np.array([ucr[a] * (dur0[a] - d) if np.isfinite(ucr[a]) else 0.0 for a, d, t, _ in var])
        asign = lil_matrix((len(orden), nv))
        for i, a in enumerate(orden):
            for j in por_act[a]:
                asign[i, j] = 1.0
        filas_prec = [(p, a) for a in orden for p in preds[a] if p in por_act]
        prec = lil_matrix((max(len(filas_prec), 1), nv))
        for i, (p, a) in enumerate(filas_prec):
            for j in por_act[a]:
                prec[i, j] += var[j][2]
            for j in por_act[p]:
                prec[i, j] -= var[j][2] + var[j][1]
        res_m = lil_matrix((K * H, nv))
        for j, (a, d, t, dem) in enumerate(var):
            for tau in range(t, t + d):
                for k in range(K):
                    if dem[k]:
                        res_m[k * H + tau, j] = dem[k]
        ub = np.array([limite[k, tau] for k in range(K) for tau in range(H)])
        restricciones = [LinearConstraint(asign.tocsr(), 1, 1),
                         LinearConstraint(prec.tocsr(), 0, np.inf)]
        if K * H:
            restricciones.append(LinearConstraint(res_m.tocsr(), -np.inf, ub))
        r = milp(c, constraints=restricciones, integrality=np.ones(nv), bounds=Bounds(0, 1),
                 options=dict(time_limit=time_limit))
        if r.x is None:
            return dict(estado="infactible" if r.status == 2 else "sin solución", coste=None,
                        duraciones=None, inicios=None)
        elegidas = [var[j] for j in range(nv) if r.x[j] > 0.5]
        return dict(estado="óptimo" if r.status == 0 else "límite de tiempo", coste=float(r.fun),
                    duraciones=pd.Series({a: d for a, d, t, _ in elegidas}).sort_index(),
                    inicios=pd.Series({a: t for a, d, t, _ in elegidas}).sort_index())

    # ------------------------------------------------------------------
    # Monte Carlo con recursos limitados
    # ------------------------------------------------------------------

    def monte_carlo_recursos(self, distributions, data, resource_labels, maximos,
                             n_samples=1000, seed=None):
        """Distribución de la duración con y sin límites de recursos (RCPSP estocástico).

        En cada réplica se simulan las duraciones (redondeadas a periodos enteros, mínimo 1), se calcula la
        duración sin recursos (CPM) y se **replanifica** con los límites (planificador en
        paralelo de ``asignar_multi``).  Además de las dos distribuciones mide, por actividad,
        el índice de criticidad sin recursos (holgura nula en el CPM) y con recursos (pertenencia
        a la cadena crítica con recursos), y con qué frecuencia cada recurso bloquea alguna actividad.

        ``data`` aporta solo las columnas de demanda (una por recurso).
        """
        resource_labels = [resource_labels] if isinstance(resource_labels, str) else list(resource_labels)
        if maximos is None:
            raise ValueError("Error: Debe introducir maximos (un límite por recurso)")
        for rl in resource_labels:
            if rl not in data.columns:
                raise ValueError(f"Error: la columna '{rl}' no está en data")
        vectors = self._vectores_limite(resource_labels, maximos)
        preds = self._real_predecessors()
        orden = _orden_topologico(preds)
        faltan = [a for a in orden if a not in data.index]
        if faltan:
            raise ValueError(f"Error: faltan filas en data para {faltan}")
        dem = {a: np.array([float(data.loc[a, rl]) for rl in resource_labels]) for a in orden}
        maxlim = np.array([vectors[rl].max() for rl in resource_labels])
        for a in orden:
            if np.any(dem[a] > maxlim + 1e-9):
                raise ValueError(f"Error: la actividad {a} requiere más recurso que el máximo permitido")
        rng = np.random.default_rng(seed)
        D = np.zeros((n_samples, len(orden)))
        for j, a in enumerate(orden):
            if a not in distributions:
                raise ValueError(f"Falta la distribución de la duración de {a}")
            D[:, j] = _make_sampler(rng, distributions[a])(n_samples)
        D = np.maximum(np.rint(D), 1.0).astype(int)      # el planificador trabaja en periodos enteros
        n0 = max(len(v) for v in vectors.values())
        ancho = n0 + int(D.sum(axis=1).max()) + int(D.max()) + 2
        limite = _matriz_limite([vectors[rl] for rl in resource_labels], ancho)
        tau_libre = np.zeros(n_samples)
        tau_rec = np.zeros(n_samples)
        crit_libre = {a: 0 for a in orden}
        crit_rec = {a: 0 for a in orden}
        bloqueos = {rl: 0 for rl in resource_labels}
        for i in range(n_samples):
            dur = {a: int(D[i, j]) for j, a in enumerate(orden)}
            ES, LS, Tl = _cpm_puro(preds, dur, orden)
            ini, fin, traza, _ = _planificar_paralelo(preds, orden, dur, dem, limite,
                                                      n0 + sum(dur.values()) + 1, resource_labels, ls=LS)
            Tr = max(fin.values(), default=0)
            tau_libre[i], tau_rec[i] = Tl, Tr
            for a in orden:
                crit_libre[a] += ES[a] == LS[a]
            criticas, _ = _cadena_critica(preds, ini, fin, traza, dem, Tr, resource_labels)
            for a in criticas:
                crit_rec[a] += 1
            vistos = set()
            for _, _, decision, recursos in traza:
                if decision == "espera":
                    vistos.update(recursos.split(", "))
            for rl in vistos:
                bloqueos[rl] += 1
        criticidad = pd.DataFrame({"sin recursos": pd.Series(crit_libre) / n_samples,
                                   "con recursos": pd.Series(crit_rec) / n_samples}).sort_index()
        return MonteCarloRecursosResult(tau_libre, tau_rec, criticidad,
                                        pd.Series(bloqueos) / n_samples, n_samples, seed)

    # ------------------------------------------------------------------
    # Combinaciones: VAN, crashing, valor ganado y recursos
    # ------------------------------------------------------------------
    def _calendario_con_recursos(self, data, duration_label, resource_labels, maximos):
        """Inicios (0-based) y duración con recursos de un proyecto determinista."""
        return self.copy().asignar_multi(data.copy(), duration_label, resource_labels, maximos,
                                         report=False, tikz=False, dibujar=False)

    def van_recursos(self, data, duration_label, resource_labels, maximos, bac,
                     ingreso_mensual, n_meses, tasa_anual, ampliaciones=3):
        """VAN de un proyecto de inversión con y sin límite de recursos.

        Con límites el proyecto termina más tarde (τ mayor): los ingresos se retrasan y los costes de
        las actividades que esperan también.  Devuelve un dict con ``escenarios`` (DataFrame: duración,
        VAN de costes, VAN de ingresos, VAN) y ``ampliaciones`` (qué ocurre al ampliar *todos* los
        límites en 0, 1, … unidades: duración y VAN), para decidir si compensa ampliar el recurso.
        """
        resource_labels = [resource_labels] if isinstance(resource_labels, str) else list(resource_labels)
        R_df = self.path_matrix(dummies=False)
        acts = list(R_df.columns)
        bac = pd.Series(bac, dtype=float).reindex(acts)
        D = data[duration_label].reindex(acts).to_numpy(dtype=float)[None, :]
        B = bac.to_numpy()[None, :]
        ES0, EF0 = self._calendarios_simulados(D, acts)

        def van(ES, tau):
            ES = np.asarray(ES, dtype=float)[None, :]
            return _van_cerrado(ES, D, B, np.array([float(tau)]), np.array([float(ingreso_mensual)]),
                                n_meses, tasa_anual)

        def fila(nombre, ES, tau):
            c, i = van(ES, tau)
            return {"escenario": nombre, "duración": float(tau), "VAN costes": float(c[0]),
                    "VAN ingresos": float(i[0]), "VAN": float(i[0] - c[0])}

        filas = [fila("sin límite", ES0[0], EF0.max())]
        limites_base = self._vectores_limite(resource_labels, maximos)
        res = self._calendario_con_recursos(data, duration_label, resource_labels, maximos)
        ES1 = res.inicios.reindex(acts).to_numpy(dtype=float)
        filas.append(fila("con límite", ES1, res.duracion))
        escen = pd.DataFrame(filas).set_index("escenario")
        amp = []
        for u in range(int(ampliaciones) + 1):
            mx = {rl: np.asarray(limites_base[rl], dtype=float) + u for rl in resource_labels}
            r = self._calendario_con_recursos(data, duration_label, resource_labels, mx)
            f = fila(f"+{u}", r.inicios.reindex(acts).to_numpy(dtype=float), r.duracion)
            amp.append({"unidades de más": u, "duración": f["duración"], "VAN": f["VAN"]})
        amp = pd.DataFrame(amp).set_index("unidades de más")
        amp["mejora de VAN"] = amp["VAN"].diff()
        return {"escenarios": escen, "ampliaciones": amp}

    def monte_carlo_van_recursos(self, distributions, data, resource_labels, maximos, bac,
                                 ingreso_mensual, n_meses, tasa_anual, n_samples=1000, seed=None):
        """Monte Carlo del VAN con y sin límite de recursos (réplica a réplica se replanifica)."""
        resource_labels = [resource_labels] if isinstance(resource_labels, str) else list(resource_labels)
        vectors = self._vectores_limite(resource_labels, maximos)
        preds = self._real_predecessors()
        orden = _orden_topologico(preds)
        bac = pd.Series(bac, dtype=float).reindex(orden)
        if bac.isna().any():
            raise ValueError("Faltan BAC para alguna actividad")
        dem = {a: np.array([float(data.loc[a, rl]) for rl in resource_labels]) for a in orden}
        rng = np.random.default_rng(seed)
        D = np.zeros((n_samples, len(orden)))
        for j, a in enumerate(orden):
            D[:, j] = _make_sampler(rng, distributions[a])(n_samples)
        D = np.maximum(np.rint(D), 1.0)
        if isinstance(ingreso_mensual, (int, float, np.integer, np.floating)):
            F = np.full(n_samples, float(ingreso_mensual))
        else:
            F = _make_sampler(rng, ingreso_mensual)(n_samples)
        n0 = max(len(v) for v in vectors.values())
        ancho = n0 + int(D.sum(axis=1).max()) + int(D.max()) + 2
        limite = _matriz_limite([vectors[rl] for rl in resource_labels], ancho)
        ES_lib, EF_lib = self._calendarios_simulados(D, orden)
        ES_rec = np.zeros_like(D)
        tau_rec = np.zeros(n_samples)
        for i in range(n_samples):
            dur = {a: int(D[i, j]) for j, a in enumerate(orden)}
            LS = _cpm_puro(preds, dur, orden)[1]
            ini, fin, _, _ = _planificar_paralelo(preds, orden, dur, dem, limite,
                                                  n0 + sum(dur.values()) + 1, resource_labels, ls=LS)
            ES_rec[i] = [ini[a] for a in orden]
            tau_rec[i] = max(fin.values(), default=0)
        B = np.tile(bac.to_numpy(), (n_samples, 1))
        tau_lib = EF_lib.max(axis=1)
        c0, i0 = _van_cerrado(ES_lib, D, B, tau_lib, F, n_meses, tasa_anual)
        c1, i1 = _van_cerrado(ES_rec, D, B, tau_rec, F, n_meses, tasa_anual)
        return MonteCarloVANRecursosResult(i0 - c0, i1 - c1, tau_lib, tau_rec, n_samples, seed)

    def _ruta_crashing(self, data, duration_label, min_durations, costs, reduccion_max,
                       resource_labels=None, maximos=None):
        """Lista de (r, duraciones, coste acumulado) con r = 0, 1, … periodos recortados."""
        d0 = data[duration_label].astype(float)
        mind = pd.Series(min_durations, dtype=float).reindex(d0.index)
        ucr = pd.Series(costs, dtype=float).reindex(d0.index)
        con_rec = resource_labels is not None
        T0 = (self.copy().asignar_multi(data.copy(), duration_label, resource_labels, maximos,
                                        report=False, tikz=False, dibujar=False).duracion
              if con_rec else self.duration(d0))
        ruta = [(0, d0.copy(), 0.0, float(T0))]
        previo = float(T0)
        for r in range(1, int(reduccion_max) + 1):
            if con_rec:
                cr = self.copy().crashing_recursos(data, duration_label, resource_labels, maximos,
                                                   mind, ucr, reduccion=r)
                dur = cr.duraciones.astype(float)
                coste = float(cr.coste_total)
                T = float(cr.duracion_final)
            else:
                _, _, dur, _ = self.ackoff(d0, mind, ucr, reduction=r, report=False)
                dur = pd.Series(dur, dtype=float).reindex(d0.index)
                coste = float(((d0 - dur) * ucr.replace(np.inf, 0.0)).sum())
                T = float(self.duration(dur))
            if T >= previo - 1e-9:
                break
            ruta.append((r, dur, coste, T))
            previo = T
        return ruta

    def crashing_van(self, data, duration_label, min_durations, costs, bac, ingreso_mensual,
                     n_meses, tasa_anual, resource_labels=None, maximos=None, reduccion_max=20):
        """Duración óptima por VAN: cada semana recortada cuesta (CUR) pero adelanta los ingresos.

        El coste de reducción se suma al BAC de cada actividad y se reparte en su nueva duración.
        Con ``resource_labels`` y ``maximos`` se reduce con ``crashing_recursos`` (límites de recursos);
        sin ellos con Ackoff-Sasieni.  Devuelve un DataFrame indexado por periodos recortados con
        duración, coste de reducción acumulado, VAN y ganancia marginal; la fila de VAN máximo es el óptimo.
        """
        R_df = self.path_matrix(dummies=False)
        acts = list(R_df.columns)
        bac = pd.Series(bac, dtype=float).reindex(acts)
        ucr = pd.Series(costs, dtype=float).reindex(data.index)
        ruta = self._ruta_crashing(data, duration_label, min_durations, costs, reduccion_max,
                                   resource_labels, maximos)
        filas = []
        d0 = data[duration_label].astype(float)
        for r, dur, coste, T in ruta:
            D = dur.reindex(acts).to_numpy(dtype=float)[None, :]
            extra = ((d0 - dur) * ucr.replace(np.inf, 0.0)).reindex(acts).to_numpy()
            B = (bac.to_numpy() + extra)[None, :]
            if resource_labels is not None:
                d_data = data.copy()
                d_data[duration_label] = dur
                res = self._calendario_con_recursos(d_data, duration_label, resource_labels, maximos)
                ES = res.inicios.reindex(acts).to_numpy(dtype=float)[None, :]
                tau = float(res.duracion)
            else:
                ES, EF = self._calendarios_simulados(D, acts)
                tau = float(EF.max())
            c, i = _van_cerrado(ES, D, B, np.array([tau]), np.array([float(ingreso_mensual)]),
                                n_meses, tasa_anual)
            filas.append({"periodos recortados": r, "duración": tau, "coste de reducción": coste,
                          "VAN": float(i[0] - c[0])})
        tabla = pd.DataFrame(filas).set_index("periodos recortados")
        tabla["ganancia marginal"] = tabla["VAN"].diff()
        return tabla

    def monte_carlo_crashing(self, distributions, data, duration_label, min_durations, costs,
                             plazo, reduccion_max=6, n_samples=5000, seed=None):
        """Probabilidad de cumplir el plazo en función de las periodos recortados (y su coste).

        Con números aleatorios comunes: se sortean las duraciones una vez y a cada réplica se le restan
        las reducciones que da Ackoff-Sasieni en cada paso (mínimo 1 y no menos que la duración tope).
        """
        R_df = self.path_matrix(dummies=False)
        acts = list(R_df.columns)
        rng = np.random.default_rng(seed)
        D = np.zeros((n_samples, len(acts)))
        for j, a in enumerate(acts):
            D[:, j] = _make_sampler(rng, distributions[a])(n_samples)
        D = np.maximum(np.rint(D), 1.0)
        d0 = data[duration_label].astype(float).reindex(acts)
        ruta = self._ruta_crashing(data, duration_label, min_durations, costs, reduccion_max)
        filas = []
        for r, dur, coste, T in ruta:
            red = (d0 - dur.reindex(acts)).to_numpy()[None, :]
            Dr = np.maximum(D - red, 1.0)
            _, EF = self._calendarios_simulados(Dr, acts)
            tau = EF.max(axis=1)
            filas.append({"periodos recortados": r, "duración determinista": T, "coste": coste,
                          "τ media": float(tau.mean()), "τ P90": float(np.quantile(tau, 0.9)),
                          "P(τ ≤ plazo)": float((tau <= plazo).mean())})
        return pd.DataFrame(filas).set_index("periodos recortados")

    def monte_carlo_valor_ganado(self, distributions, plan, bac, t_control, n_samples=2000,
                                 seed=None, factor_coste=None):
        """Calidad de las previsiones de valor ganado cuando las duraciones son aleatorias.

        ``plan`` son las duraciones previstas (línea base, calendario más temprano), ``bac`` el
        presupuesto de cada actividad (avance lineal) y ``factor_coste`` la distribución del
        factor multiplicativo del coste real (por actividad o ``None``).  En la fecha de control
        ``t_control`` se calcula, réplica a réplica, SPI = EV/PV y CPI = EV/AC, y las previsiones
        clásicas: duración T_plan/SPI y coste BAC/CPI.  Se comparan con la duración y el coste que de
        verdad resultan.  Devuelve ``MonteCarloValorGanadoResult``.
        """
        R_df = self.path_matrix(dummies=False)
        acts = list(R_df.columns)
        rng = np.random.default_rng(seed)
        plan = pd.Series(plan, dtype=float).reindex(acts)
        bac_s = pd.Series(bac, dtype=float).reindex(acts)
        Dp = plan.to_numpy()[None, :]
        ESp, EFp = self._calendarios_simulados(Dp, acts)
        T_plan = float(EFp.max())
        B = bac_s.to_numpy()[None, :]
        pv = float((B * np.clip((t_control - ESp) / Dp, 0.0, 1.0)).sum())
        D = np.zeros((n_samples, len(acts)))
        for j, a in enumerate(acts):
            D[:, j] = _make_sampler(rng, distributions[a])(n_samples)
        D = np.maximum(np.rint(D), 1.0)
        ES, EF = self._calendarios_simulados(D, acts)
        avance = np.clip((t_control - ES) / D, 0.0, 1.0)
        W = np.tile(bac_s.to_numpy(), (n_samples, 1))
        K = np.ones_like(W)
        if factor_coste:
            for j, a in enumerate(acts):
                if a in factor_coste:
                    K[:, j] = _make_sampler(rng, factor_coste[a])(n_samples)
        ev = (W * avance).sum(axis=1)
        ac = (W * K * avance).sum(axis=1)
        spi = ev / pv if pv > 0 else np.full(n_samples, np.nan)
        cpi = np.where(ac > 0, ev / np.where(ac > 0, ac, 1.0), np.nan)
        tau = EF.max(axis=1)
        coste_final = (W * K).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_pred = T_plan / spi
            eac_pred = float(bac_s.sum()) / cpi
        return MonteCarloValorGanadoResult(t_control, T_plan, float(bac_s.sum()), pv, ev, ac, spi, cpi,
                                           tau, tau_pred, coste_final, eac_pred, n_samples, seed)

    def monte_carlo_valor_ganado_recursos(self, distributions, plan, data, resource_labels, maximos, bac,
                                          t_control, n_samples=1000, seed=None):
        """Previsión de duración por valor ganado con duraciones aleatorias y límite de recursos.

        Se ejecuta el proyecto réplica a réplica (duraciones sorteadas y replanificación con los
        límites) y se evalúa el SPI en ``t_control`` con dos líneas base construidas con las
        duraciones previstas ``plan``: la **ingenua** (calendario más temprano, sin recursos) y la
        **realista** (calendario con el límite de recursos).  La previsión T_plan/SPI se compara con la
        duración real de cada réplica.  Devuelve ``MonteCarloValorGanadoRecursosResult``.
        """
        resource_labels = [resource_labels] if isinstance(resource_labels, str) else list(resource_labels)
        vectors = self._vectores_limite(resource_labels, maximos)
        preds = self._real_predecessors()
        orden = _orden_topologico(preds)
        bac_s = pd.Series(bac, dtype=float).reindex(orden)
        B = bac_s.to_numpy()
        plan = pd.Series(plan, dtype=float).reindex(orden)
        dem = {a: np.array([float(data.loc[a, rl]) for rl in resource_labels]) for a in orden}
        n0 = max(len(v) for v in vectors.values())
        # líneas base
        Dp = plan.to_numpy()[None, :]
        ES_l, EF_l = self._calendarios_simulados(Dp, orden)
        T_libre = float(EF_l.max())
        dp = {a: int(round(plan[a])) for a in orden}
        ancho = n0 + sum(dp.values()) + max(dp.values()) + 2
        limite = _matriz_limite([vectors[rl] for rl in resource_labels], ancho)
        ini_p, fin_p, _, _ = _planificar_paralelo(preds, orden, dp, dem, limite, n0 + sum(dp.values()) + 1, resource_labels)
        ES_r = np.array([ini_p[a] for a in orden], dtype=float)
        T_rec = float(max(fin_p.values()))
        Dpv = np.array([dp[a] for a in orden], dtype=float)
        pv_l = float((B * np.clip((t_control - ES_l[0]) / Dp[0], 0.0, 1.0)).sum())
        pv_r = float((B * np.clip((t_control - ES_r) / Dpv, 0.0, 1.0)).sum())
        # ejecuciones
        rng = np.random.default_rng(seed)
        D = np.zeros((n_samples, len(orden)))
        for j, a in enumerate(orden):
            D[:, j] = _make_sampler(rng, distributions[a])(n_samples)
        D = np.maximum(np.rint(D), 1.0).astype(int)
        ancho = n0 + int(D.sum(axis=1).max()) + int(D.max()) + 2
        limite = _matriz_limite([vectors[rl] for rl in resource_labels], ancho)
        tau = np.zeros(n_samples)
        ev = np.zeros(n_samples)
        for i in range(n_samples):
            dur = {a: int(D[i, j]) for j, a in enumerate(orden)}
            LS = _cpm_puro(preds, dur, orden)[1]
            ini, fin, _, _ = _planificar_paralelo(preds, orden, dur, dem, limite,
                                                  n0 + sum(dur.values()) + 1, resource_labels, ls=LS)
            tau[i] = max(fin.values(), default=0)
            ES = np.array([ini[a] for a in orden], dtype=float)
            ev[i] = float((B * np.clip((t_control - ES) / D[i], 0.0, 1.0)).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            spi_l = ev / pv_l if pv_l > 0 else np.full(n_samples, np.nan)
            spi_r = ev / pv_r if pv_r > 0 else np.full(n_samples, np.nan)
            pred_l = T_libre / spi_l
            pred_r = T_rec / spi_r
        return MonteCarloValorGanadoRecursosResult(t_control, T_libre, T_rec, pv_l, pv_r, ev, spi_l, spi_r,
                                                   tau, pred_l, pred_r, n_samples, seed)

    def valor_ganado_recursos(self, data, duration_label, resource_labels, maximos, bac):
        """Valor ganado cuando la línea base ignora los recursos y la ejecución los respeta.

        Devuelve una tabla por periodo con el valor planificado (calendario más temprano sin recursos),
        el valor ganado (calendario con recursos), el SPI y las previsiones de duración
        T_plan/SPI; ``attrs`` guarda T_plan, T_real y BAC.  Con límite de recursos el SPI cae por debajo
        de 1 desde que se produce la primera espera, aunque nadie trabaje «peor».
        """
        R_df = self.path_matrix(dummies=False)
        acts = list(R_df.columns)
        bac = pd.Series(bac, dtype=float).reindex(acts).to_numpy()
        D = data[duration_label].reindex(acts).to_numpy(dtype=float)
        ESp, EFp = self._calendarios_simulados(D[None, :], acts)
        res = self._calendario_con_recursos(data, duration_label, resource_labels, maximos)
        ESr = res.inicios.reindex(acts).to_numpy(dtype=float)
        T_plan, T_real = float(EFp.max()), float(res.duracion)
        filas = []
        for t in range(0, int(max(T_plan, T_real)) + 1):
            pv = float((bac * np.clip((t - ESp[0]) / D, 0.0, 1.0)).sum())
            ev = float((bac * np.clip((t - ESr) / D, 0.0, 1.0)).sum())
            spi = ev / pv if pv > 0 else np.nan
            filas.append({"periodo": t, "PV": pv, "EV": ev, "SPI": spi,
                          "duración prevista": T_plan / spi if spi and spi > 0 else np.nan})
        tabla = pd.DataFrame(filas).set_index("periodo")
        tabla.attrs.update(T_plan=T_plan, T_real=T_real, BAC=float(bac.sum()))
        return tabla

    def standard_deviation(self, durations, variances, report=False):
        varianza = {key: 0 for key in self.activities}
        varianza.update(variances)
        caminos = self.critical_path(durations=durations)
        varianza_caminos = {
            key: sum(varianza[activity] for activity in path)
            for key, path in caminos.items()
        }
        if report:
            [print("Variance path:", key, ":", round(value,2)) for key, value in varianza_caminos.items()]
        varianza_proyecto = max(varianza_caminos.values())
        #ruta_max = varianza_caminos.values().argmax()

        std_deviation = varianza_proyecto**0.5
        if report:
            print("Project duration variance:", varianza_proyecto)
            print("Project duration standard deviation:", std_deviation)

        fmt = lambda x: f"{x:.2f}".rstrip("0").rstrip(".")
        formulas = {
            key: " + ".join(f"σ²({a})" for a in path) + " = "
                 + " + ".join(fmt(varianza[a]) for a in path) + f" = {fmt(varianza_caminos[key])}"
            for key, path in caminos.items()
        }
        varianza_caminos = pd.Series(varianza_caminos).to_frame(name='Variance').round(2)
        varianza_caminos = (pd.concat([pd.Series(caminos).to_frame(name='Activities').map(lambda x: ", ".join(x)),
                                      varianza_caminos,
                                      pd.Series(formulas).to_frame(name='Formula')], axis=1)
                            .style.apply(highlight_maximum, axis=0, subset=['Variance'])
                            .format('{:.2f}', na_rep='', subset=['Variance'])
                            )
        return std_deviation, varianza_caminos

    def ackoff(self, durations, min_durations, costs, reduction=100, report=False):
        def calculate_reduction_cost(activities, costs):
            return sum([costs[name] for name in set(activities)])

        durations = durations.copy()
        min_durations = min_durations.copy()
        costs = costs.copy()

        if isinstance(durations, dict):
            durations = pd.Series(durations)
        if isinstance(min_durations, dict):
            min_durations = pd.Series(min_durations)
        if isinstance(costs, dict):
            costs = pd.Series(costs)

        periods_available = durations - min_durations
        # print(80*"-")
        # print("Periods available for reduction:")
        # print(periods_available)
        # Data preparation
        path_matrix = self.path_matrix(dummies=False)
        path_names = path_matrix.index
        activity_names = self.actual_activities
        result = path_matrix.copy()

        result.replace(0, np.nan, inplace=True)
        result = result.multiply(costs.astype(float), axis="columns")
        step = 0
        result.loc[:, step] = path_matrix @ durations
        result.loc[step, :] = periods_available
        # print(80*"-")
        # print("Result initial:")
        # print(result)
        best_option = dict()
        while step < reduction:
            # print(80*"-")
            # print(f"Step {step}: Durations:")
            # print(durations)
            # print(f"Step {step}: Periods available:")
            # print(periods_available)
            critical_paths = self.critical_path(durations)
            # print(80*"-")
            # print(f"Step {step}: Critical paths found: {list(critical_paths.keys())}")
            critical_path_matrix = result.loc[critical_paths.keys(), activity_names]
            # print(80*"-")
            # print("Critical path matrix:")
            # print(critical_path_matrix)
            mini_path_matrix = result.loc[
                critical_paths.keys(), self.actual_activities
            ].loc[:, periods_available != 0]
            # print(80*"-")
            # print("Mini path matrix:")
            # print(mini_path_matrix)
            if mini_path_matrix.shape[1] == 0:
                if report:
                    print("No more paths can be reduced.")
                break
            mini_path_matrix_filtered = mini_path_matrix.apply(
                lambda x: mini_path_matrix.columns[x.notna()], axis=1
            )

            mini_path_matrix_filtered_has_empty_rows = any(
                mini_path_matrix_filtered.apply(len) == 0
            )
            if mini_path_matrix_filtered_has_empty_rows:
                if report:
                    print("No more paths can be reduced.")
                break

            costes = {
                key: calculate_reduction_cost(key, costs)
                for key in product(*mini_path_matrix_filtered.values)
            }
            # sorted(): el orden de un set de strings varía entre procesos
            best_option[step] = sorted(set(min(costes, key=costes.get)))
            if report:
                print(f"Step: {step},\t Critical paths: {list(critical_paths.keys())}")
                print(f"\t\t Best option: {best_option[step]}, \t Cost: {calculate_reduction_cost(best_option[step], costs)}")
            for activity in best_option[step]:
                periods_available[activity] -= 1
                durations[activity] -= 1

            result.loc[step + 1, activity_names] = periods_available
            result.loc[path_names, step + 1] = path_matrix @ durations
            step += 1

        idx = pd.IndexSlice
        filas_steps = idx[range(step + 1), :]
        abajo = idx[range(step), activity_names]
        derecha = idx[path_names, range(step + 1)]
        curs = idx[path_names, activity_names]

        result = (
            result.astype(float)
            .style.set_table_styles(
                [
                    dict(
                        selector="th",
                        props=[("text-align", "center"), ("border", "1px")],
                    ),
                    dict(selector="", props=[("border", "1px solid")]),
                ]
            )
            .format(na_rep="", precision=1)
            .map(highlight_cur, subset=curs)
            # .apply(lambda x: ['background-color: cyan' for _ in x], axis=1, subset=abajo )
            # .apply(lambda x: ['background-color: cyan' for _ in x], axis=1, subset=derecha )
            .apply(highlight_maximum, axis=0, subset=derecha)
            # .apply(highlight_minimum, axis=1, subset=filas_steps)
        )
        for index in range(step):
            for activity in best_option[index]:
                recortadas = idx[index, activity]
                result.apply(
                    lambda x: ["background-color: yellow" for _ in x],
                    axis=1,
                    subset=recortadas,
                )
        if report:
            print("\n" + 80 * "-")
        return result, best_option, durations, periods_available

    def ackoff_pasos_html(self, durations, min_durations, costs, reduction=100) -> str:
        """El algoritmo de Ackoff-Sasieni como visor HTML con botones.

        El primer fotograma es la tabla completa que devuelve :meth:`ackoff`; la
        primera pulsación deja solo los datos (costes unitarios de reducción por
        ruta y actividad, duración inicial de las rutas y periodos reducibles de
        cada actividad) y las siguientes van mostrando, iteración a iteración, la
        decisión (rutas críticas y actividades que se acortan, en amarillo) y la
        fila y la columna que resultan (periodos disponibles y duración de las
        rutas). Admite los mismos argumentos que :meth:`ackoff`."""
        import html as _h
        tabla, mejores, _, _ = self.ackoff(durations, min_durations, costs, reduction=reduction)
        df = tabla.data
        costes = pd.Series(costs, dtype=float) if not isinstance(costs, pd.Series) else costs.astype(float)
        actividades = list(self.actual_activities)
        rutas = [i for i in df.index if isinstance(i, str)]
        pasos = [c for c in df.columns if not isinstance(c, str)]
        k = len(pasos) - 1

        def num(v):
            if pd.isna(v):
                return ""
            if np.isinf(v):
                return "∞" if v > 0 else "-∞"
            return f"{int(v)}" if float(v) == int(v) else f"{v:g}"

        def criticas(r):
            col = df.loc[rutas, r]
            return [x for x in rutas if col[x] == col.max()]

        SOMBRA = "background:#fde3cc;"          # celdas ruta crítica × actividad acortada
        APAGADO = "color:#c4c4c4;"              # lo que ya no interviene en la decisión
        linea = "border-bottom:1px solid #ccc;"
        rejilla = "border:1px solid #000;"

        def fotograma(vis, texto, actuan_rutas=(), actuan_acts=(), amarillo_hasta=0, verde=None, apagar=True):
            """``vis``: pasos visibles (columnas y filas 0..vis-1); se sombrean las rutas
            críticas y las actividades que se acortan (``actuan_*``; solo sus celdas de cruce); en verde el máximo de
            cada una de las ``verde`` primeras columnas (por defecto, todas las visibles); en amarillo, las actividades acortadas de las filas
            anteriores a ``amarillo_hasta``."""
            borde = "border-bottom:1px solid #444"
            sep = ";border-right:1px solid #444"
            # pasos anteriores al de la decisión (fila y columna P_j con j < vis-2) y
            # actividades que ya no pueden reducirse (sin periodos en la fila de decisión)
            viejo = lambda j: APAGADO if apagar and j < vis - 2 else ""
            # la fila de decisión es la anterior a la de resultados: una actividad que
            # se agota en el resultado aún no se atenúa, hasta que sea la fila de decisión
            fila_decision = pasos[max(vis - 2, 0)]
            agotada = {a: (APAGADO if apagar and df.loc[fila_decision, a] == 0 else "") for a in actividades}
            cab = ['<th style="' + borde + sep + '">ruta \\ act.</th>']
            cab += [f'<th style="{borde};{agotada[a]}">{_h.escape(str(a))}</th>' for a in actividades]
            cab += [f'<th style="{borde};{viejo(j)}">' + (f"{pasos[j]}" if j < vis else "&nbsp;") + "</th>"
                    for j in range(k + 1)]
            filas = ["<tr>" + "".join(cab) + "</tr>"]
            for ruta in rutas:
                fila_actua = ruta in actuan_rutas
                celdas = [f'<td style="font-weight:bold{sep};{linea}">'
                          f'{_h.escape(str(ruta))}</td>']
                for a in actividades:
                    sombra = SOMBRA if (fila_actua and a in actuan_acts and pd.notna(df.loc[ruta, a])) else ""
                    celdas.append(f'<td style="{rejilla}{sombra}{agotada[a]}">{num(df.loc[ruta, a])}</td>')
                for j in range(k + 1):
                    if j < vis:
                        v = df.loc[ruta, pasos[j]]
                        verde_ok = j < (vis if verde is None else verde) and v == df.loc[rutas, pasos[j]].max()
                        estilo = (("background:#e2f5b5;" if viejo(j) else "background:#adff2f;")
                                  if verde_ok else "") + viejo(j)
                        celdas.append(f'<td style="{linea}{estilo}">{num(v)}</td>')
                    else:
                        celdas.append(f'<td style="{linea}"></td>')
                filas.append("<tr>" + "".join(celdas) + "</tr>")
            for j in range(k + 1):
                celdas = [f'<td style="font-weight:bold{sep};{linea}{viejo(j)}">'
                          + (f"{pasos[j]}" if j < vis else "&nbsp;") + "</td>"]
                for a in actividades:
                    if j < vis:
                        v = df.loc[pasos[j], a]
                        amarillo = j < amarillo_hasta and a in mejores.get(pasos[j], [])
                        estilo = (("background:#ffffb3;" if (viejo(j) or agotada[a]) else "background:#ffff00;")
                                  if amarillo else "") + viejo(j) + agotada[a]
                    else:
                        v, estilo = float("nan"), ""
                    celdas.append(f'<td style="{linea}{estilo}">{num(v)}</td>')
                celdas += [f'<td style="{linea}"></td>'] * (k + 1)
                filas.append("<tr>" + "".join(celdas) + "</tr>")
            return ('<style>.pmule-ackoff td,.pmule-ackoff th{padding:.12em .55em;min-width:1.6em}'
                    # sin el fondo gris alterno del tema (rayado de filas)
                    'table.pmule-ackoff.pmule-ackoff>tbody>tr:nth-of-type(n)>td,'
                    'table.pmule-ackoff.pmule-ackoff>thead>tr>th,'
                    'table.pmule-ackoff.pmule-ackoff>tbody>tr:nth-of-type(n)>th'
                    '{background-color:transparent;box-shadow:none;--bs-table-accent-bg:transparent}</style>'
                    '<table class="pmule-ackoff" style="border-collapse:collapse;text-align:center;margin:0 auto">'
                    + "".join(filas) + "</table>"
                    f'<p style="text-align:center;margin-top:.5em">{texto}</p>')

        fotos = [fotograma(k + 1, "Algoritmo de Ackoff-Sasieni: tabla completa. Pulse «siguiente» para ver cómo se construye.",
                           amarillo_hasta=k, apagar=False)]
        fotos.append(fotograma(1, "Datos: coste unitario de reducción de cada actividad en las rutas que la contienen, "
                                  "duración inicial de cada ruta (columna 0) y periodos que puede reducirse cada actividad (fila 0).",
                               verde=0))
        for r in range(k):
            crit = criticas(pasos[r])
            dur = df.loc[crit[0], pasos[r]]
            cortadas = mejores.get(pasos[r], [])
            coste = float(sum(costes[a] for a in cortadas))
            nueva = df.loc[rutas, pasos[r + 1]].max()
            texto = (f"Iteración {pasos[r]}: rutas críticas {', '.join(map(str, crit))} (duración {num(dur)}); "
                     f"se acorta un periodo la combinación de menor coste: {', '.join(cortadas)} (coste {num(coste)}). "
                     f"Resultado: periodos disponibles (fila {pasos[r + 1]}) y duración de las rutas "
                     f"(columna {pasos[r + 1]}); el proyecto dura {num(nueva)}.")
            if r + 1 == k:
                crit_f = criticas(pasos[k])
                libres = df.loc[pasos[k], actividades]
                reducible = all(any(pd.notna(df.loc[x, a]) and libres[a] > 0 for a in actividades) for x in crit_f)
                texto += (" Se alcanzó el número de reducciones pedido." if reducible
                          else " Alguna ruta crítica ya no puede acortarse más: fin del algoritmo.")
            fotos.append(fotograma(r + 2, texto, actuan_rutas=crit, actuan_acts=cortadas, amarillo_hasta=r + 1,
                                   verde=r + 1))
        return _html_pasos(fotos, inicio=-1)

    def display_ackoff_pasos(self, durations, min_durations, costs, reduction=100) -> None:
        """Muestra en un cuaderno el algoritmo de Ackoff-Sasieni paso a paso (HTML con
        botones, :meth:`ackoff_pasos_html`)."""
        display(HTML(self.ackoff_pasos_html(durations, min_durations, costs, reduction)))

    def ackoff_pasos_tikz_html(self, durations, min_durations, costs, reduction=100,
                               escala: float = 1.0) -> str:
        """Igual que :meth:`ackoff_pasos_html` pero con la tabla dibujada en TikZ (celdas
        coloreadas, rejilla fina y números nítidos); mismos fotogramas y mismas reglas:
        verde = máximo de cada columna de duraciones, amarillo = actividad acortada,
        melocotón = celda de decisión (ruta crítica × actividad acortada), gris claro =
        pasos anteriores y actividades que ya no pueden reducirse."""
        import base64
        tabla, mejores, _, _ = self.ackoff(durations, min_durations, costs, reduction=reduction)
        df = tabla.data
        costes = pd.Series(costs, dtype=float) if not isinstance(costs, pd.Series) else costs.astype(float)
        actividades = list(self.actual_activities)
        rutas = [i for i in df.index if isinstance(i, str)]
        pasos = [c for c in df.columns if not isinstance(c, str)]
        k = len(pasos) - 1
        na, nr = len(actividades), len(rutas)

        def num(v):
            if pd.isna(v):
                return ""
            if np.isinf(v):
                return r"$\infty$" if v > 0 else r"$-\infty$"
            return f"{int(v)}" if float(v) == int(v) else f"{v:g}"

        def tex(t):
            for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
                t = t.replace(a, b)
            return t

        def criticas(r):
            col = df.loc[rutas, r]
            return [x for x in rutas if col[x] == col.max()]

        W, H, X0, GAP = 0.95, 0.62, 2.3, 0.45
        cx_act = lambda i: X0 + (i + 0.5) * W
        cx_paso = lambda j: X0 + na * W + GAP + (j + 0.5) * W
        cy = lambda r: -(r + 0.5) * H
        ancho = X0 + na * W + GAP + (k + 1) * W
        alto = (1 + nr + k + 1) * H
        VERDE, VERDE_V = "green!45!lime!60", "green!15"
        AMARILLO, AMARILLO_V = "yellow!90", "yellow!25"
        DECISION = "orange!28"
        GRIS = "black!28"

        def fotograma(vis, actuan_rutas=(), actuan_acts=(), amarillo_hasta=0, verde=None, apagar=True):
            viejo = lambda j: apagar and j < vis - 2
            fila_decision = pasos[max(vis - 2, 0)]
            agotada = {a: apagar and df.loc[fila_decision, a] == 0 for a in actividades}
            fondo, texto, rayas = [], [], []

            def celda(x, y, txt, color=None, fondo_c=None, bold=False):
                if fondo_c:
                    fondo.append(rf"\fill[{fondo_c}, rounded corners=1.5pt] ({x - W / 2 + .04:.3f},{y - H / 2 + .04:.3f}) "
                                 rf"rectangle ({x + W / 2 - .04:.3f},{y + H / 2 - .04:.3f});")
                if txt != "":
                    estilo = ("text=" + color + "," if color else "") + ("font=\\bfseries," if bold else "")
                    texto.append(rf"\node[{estilo}] at ({x:.3f},{y:.3f}) {{{txt}}};")

            celda(1.0, cy(0), r"\textit{ruta} \textbackslash{} \textit{act.}", bold=False)
            for i, a in enumerate(actividades):
                celda(cx_act(i), cy(0), tex(str(a)), GRIS if agotada[a] else None, bold=True)
            for j in range(k + 1):
                if j < vis:
                    celda(cx_paso(j), cy(0), f"{pasos[j]}", GRIS if viejo(j) else None, bold=True)
            for n, ruta in enumerate(rutas):
                y = cy(1 + n)
                celda(1.0, y, tex(str(ruta)), bold=True)
                for i, a in enumerate(actividades):
                    v = df.loc[ruta, a]
                    cruce = ruta in actuan_rutas and a in actuan_acts and pd.notna(v)
                    celda(cx_act(i), y, num(v), GRIS if agotada[a] else None, DECISION if cruce else None)
                for j in range(min(vis, k + 1)):
                    v = df.loc[ruta, pasos[j]]
                    es_max = j < (vis if verde is None else verde) and v == df.loc[rutas, pasos[j]].max()
                    celda(cx_paso(j), y, num(v), GRIS if viejo(j) else None,
                          (VERDE_V if viejo(j) else VERDE) if es_max else None)
            for j in range(min(vis, k + 1)):
                y = cy(1 + nr + j)
                celda(1.0, y, f"{pasos[j]}", GRIS if viejo(j) else None, bold=True)
                for i, a in enumerate(actividades):
                    amar = j < amarillo_hasta and a in mejores.get(pasos[j], [])
                    celda(cx_act(i), y, num(df.loc[pasos[j], a]), GRIS if (viejo(j) or agotada[a]) else None,
                          (AMARILLO_V if (viejo(j) or agotada[a]) else AMARILLO) if amar else None)
            # rejilla: caja de costes, separadores finos por fila y línea bajo la cabecera
            y_nr = -(1 + nr) * H
            # (``grid`` se ancla a múltiplos del paso desde el origen: las líneas van explícitas)
            for i in range(1, na):
                rayas.append(rf"\draw[black!25, very thin] ({X0 + i * W:.3f},{-H}) -- ({X0 + i * W:.3f},{y_nr});")
            for n in range(1, nr):
                rayas.append(rf"\draw[black!25, very thin] ({X0},{-(1 + n) * H:.3f}) -- ({X0 + na * W},{-(1 + n) * H:.3f});")
            rayas.append(rf"\draw[black!70, thin] ({X0},{-H}) rectangle ({X0 + na * W},{y_nr});")
            rayas.append(rf"\draw[black!70] (0,{-H}) -- ({ancho},{-H});")
            filas_vis = 1 + nr + min(vis, k + 1)          # las filas aún vacías no se dibujan
            rayas.append(rf"\draw[black!70] ({X0 - 0.04},0) -- ({X0 - 0.04},{-filas_vis * H});")
            for r in range(2 + nr, filas_vis):
                rayas.append(rf"\draw[black!12] (0,{-r * H}) -- ({ancho},{-r * H});")
            cuerpo = "\n".join(fondo + rayas + texto)
            return (rf"\begin{{tikzpicture}}[x={escala}cm,y={escala}cm,font=\small]" + "\n"
                    rf"\useasboundingbox (-0.1,0.1) rectangle ({ancho + 0.1:.3f},{-alto - 0.1:.3f});" + "\n"
                    + cuerpo + "\n\\end{tikzpicture}")

        pies, codigos = [], []
        codigos.append(fotograma(k + 1, amarillo_hasta=k, apagar=False))
        pies.append("Algoritmo de Ackoff-Sasieni: tabla completa. Pulse «siguiente» para ver cómo se construye.")
        codigos.append(fotograma(1, verde=0))
        pies.append("Datos: coste unitario de reducción de cada actividad en las rutas que la contienen, "
                    "duración inicial de cada ruta (columna 0) y periodos que puede reducirse cada actividad (fila 0).")
        for r in range(k):
            crit = criticas(pasos[r])
            dur = df.loc[crit[0], pasos[r]]
            cortadas = mejores.get(pasos[r], [])
            coste = float(sum(costes[a] for a in cortadas))
            nueva = df.loc[rutas, pasos[r + 1]].max()
            t = (f"Iteración {pasos[r]}: rutas críticas {', '.join(map(str, crit))} (duración {num(dur)}); "
                 f"se acorta un periodo la combinación de menor coste: {', '.join(cortadas)} (coste {num(coste)}). "
                 f"Resultado: periodos disponibles (fila {pasos[r + 1]}) y duración de las rutas "
                 f"(columna {pasos[r + 1]}); el proyecto dura {num(nueva)}.").replace("$\\infty$", "∞")
            if r + 1 == k:
                crit_f = criticas(pasos[k])
                libres = df.loc[pasos[k], actividades]
                reducible = all(any(pd.notna(df.loc[x, a]) and libres[a] > 0 for a in actividades) for x in crit_f)
                t += (" Se alcanzó el número de reducciones pedido." if reducible
                      else " Alguna ruta crítica ya no puede acortarse más: fin del algoritmo.")
            codigos.append(fotograma(r + 2, actuan_rutas=crit, actuan_acts=cortadas, amarillo_hasta=r + 1, verde=r + 1))
            pies.append(t)
        import html as _h
        fotos = []
        for svg, pie in zip(_svgs_de_tikz(codigos), pies):
            b64 = base64.b64encode(svg.encode("utf8")).decode("ascii")
            fotos.append('<img alt="paso" style="display:block;margin:0 auto;width:100%;max-width:1000px;height:auto" '
                         f'src="data:image/svg+xml;base64,{b64}">'
                         f'<p style="text-align:center;margin-top:.5em">{_h.escape(pie)}</p>')
        return _html_pasos(fotos, inicio=-1)

    def display_ackoff_pasos_tikz(self, durations, min_durations, costs, reduction=100,
                                  escala: float = 1.0) -> None:
        """Muestra el algoritmo de Ackoff-Sasieni paso a paso con la tabla dibujada en TikZ
        (:meth:`ackoff_pasos_tikz_html`); sin LaTeX, cae en :meth:`display_ackoff_pasos`."""
        try:
            display(HTML(self.ackoff_pasos_tikz_html(durations, min_durations, costs, reduction, escala)))
        except (FileNotFoundError, RuntimeError) as e:
            import warnings
            warnings.warn(f"No se pudo generar el visor TikZ ({e}); se usa el visor HTML.")
            self.display_ackoff_pasos(durations, min_durations, costs, reduction)

    def incidence_matrix(self):
        nodos_ordenados = self.nodes  # orden topológico (no lexicográfico)
        aristas_sin_ordenar = self.pert_graph.edges
        nombres_sin_ordenar = [
            self.pert_graph.edges[edge]["activity"] for edge in self.pert_graph.edges
        ]
        aristas_ordenadas = [
            x for _, x in sorted(zip(nombres_sin_ordenar, aristas_sin_ordenar))
        ]
        nombres_ordenados = [
            self.pert_graph.edges[edge]["activity"] for edge in aristas_ordenadas
        ]
        H = pd.DataFrame(
            nx.incidence_matrix(
                self.pert_graph,
                oriented=True,
                nodelist=nodos_ordenados,
                edgelist=aristas_ordenadas,
            )
            .toarray()
            .T,
            index=nombres_ordenados,
            columns=nodos_ordenados,
        ).astype(int)
        return H

    def nullspace(self):
        """Matriz de incidencia sin las columnas del nodo origen y del sumidero."""
        nodos_ordenados = self.nodes
        H = self.incidence_matrix()
        nullspace = H.drop(columns=[nodos_ordenados[0], nodos_ordenados[-1]])
        return nullspace

    # ── Serialización ──────────────────────────────────────────────────────

    def to_graphml(self, path) -> None:
        """Guarda el grafo PERT en un fichero GraphML (portable, legible).

        Los atributos de tipo dict se serializan a JSON string para
        compatibilidad con GraphML. Las claves tupla se convierten a str
        para que json.dumps las acepte.
        """
        import json as _json

        def _make_serializable(obj):
            """Convierte recursivamente claves tupla a str."""
            if isinstance(obj, dict):
                return {str(k) if isinstance(k, tuple) else k: _make_serializable(v)
                        for k, v in obj.items()}
            return obj

        G = self.pert_graph.copy()
        for _, data in G.nodes(data=True):
            for k, v in list(data.items()):
                if isinstance(v, dict):
                    data[k] = _json.dumps(_make_serializable(v))
        for _, _, data in G.edges(data=True):
            for k, v in list(data.items()):
                if isinstance(v, dict):
                    data[k] = _json.dumps(_make_serializable(v))
        nx.write_graphml(G, str(path))

    @classmethod
    def from_graphml(cls, path) -> "ProjectGraph":
        """Reconstruye un ProjectGraph desde un fichero GraphML."""
        G = nx.read_graphml(str(path))
        return cls(G)

    @staticmethod
    def save_route_matrix(R, path) -> None:
        """Guarda la matriz de rutas como sparse .npz + .json con los nombres de actividades."""
        import json as _json, pathlib as _pl
        import scipy.sparse as _sp
        _sp.save_npz(str(path), _sp.csr_matrix(R.values.astype(np.uint8)))
        meta_path = _pl.Path(str(path)).with_suffix(".json")
        meta_path.write_text(_json.dumps({"activities": list(R.columns), "n_paths": len(R)}))

    @staticmethod
    def load_route_matrix(path):
        """Carga la matriz de rutas desde .npz + .json; devuelve un DataFrame."""
        import json as _json, pathlib as _pl
        import scipy.sparse as _sp
        meta = _json.loads(_pl.Path(str(path)).with_suffix(".json").read_text())
        R = _sp.load_npz(str(path)).toarray()
        return pd.DataFrame(R, columns=meta["activities"])

    # ------------------------------------------------------------------
    # Analítica: tau, tau_sensitivity, svd_approximation_error
    # ------------------------------------------------------------------

    def tau(self, durations: dict) -> float:
        """‖R·t‖∞ — completion-time formula (Pasos 1-2 del paper).

        Parameters
        ----------
        durations : dict
            {nombre_actividad: duración} para actividades no-dummy.

        Returns
        -------
        float  τ = max_{rutas p} Σ_{i∈p} t_i
        """
        R = self.path_matrix(dummies=False)   # (n_paths, n_activities)
        t = np.array([durations.get(a, 0.0) for a in R.columns])
        return float(np.max(R.values @ t))

    def tau_sensitivity(self, durations: dict) -> "pd.Series":
        """∂τ/∂t_i — peso de cada actividad en el tiempo de compleción.

        Si hay una sola ruta crítica los pesos son 0/1.
        Con k rutas críticas de igual longitud se distribuyen
        proporcionalmente (media de sus vectores indicadores).

        Returns
        -------
        pd.Series indexada por actividad, orden descendente.
        """
        R = self.path_matrix(dummies=False)
        t = np.array([durations.get(a, 0.0) for a in R.columns])
        lengths = R.values @ t
        tau_val = float(np.max(lengths))
        critical_mask = np.isclose(lengths, tau_val)
        n_critical = int(critical_mask.sum())
        gradient = R.values[critical_mask].sum(axis=0) / n_critical
        return pd.Series(gradient, index=R.columns).sort_values(ascending=False)

    def svd_approximation_error(self, durations: dict) -> dict:
        """Error de la aproximación de rango 1 SVD respecto al τ exacto.

        Con R ≈ σ₁ u₁ v₁ᵀ, la longitud de la ruta i es ≈ σ₁ u₁ᵢ (v₁ᵀ t) y τ es
        el máximo de esas longitudes:

            τ_svd = σ₁ · máx_i(u₁ᵢ) · (v₁ᵀ · t)

        (signo normalizado: u₁ y v₁ se toman con Σu₁ ≥ 0, como corresponde a
        una R no negativa). Hasta octubre de 2026 el método usaba Σu₁ en lugar
        de máx(u₁), lo que sobrestimaba τ en un factor que crece con el número
        de rutas (mediana de 24 a 127 veces τ en PSPLIB j30–j120).

        Returns
        -------
        dict con claves: tau_exact, tau_svd, abs_error, rel_error, sigma_ratio.
        """
        R = self.path_matrix(dummies=False)
        t = np.array([durations.get(a, 0.0) for a in R.columns])
        tau_exact = float(np.max(R.values @ t))
        U, s, Vt = np.linalg.svd(R.values, full_matrices=False)
        u1, v1 = U[:, 0], Vt[0]
        if u1.sum() < 0:
            u1, v1 = -u1, -v1
        tau_svd = float(s[0] * u1.max() * (v1 @ t))
        abs_error = abs(tau_exact - tau_svd)
        rel_error = abs_error / tau_exact if tau_exact != 0.0 else 0.0
        sigma_ratio = float(s[0] / s[1]) if len(s) > 1 and s[1] > 0 else float("inf")
        return {
            "tau_exact": tau_exact,
            "tau_svd": tau_svd,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "sigma_ratio": sigma_ratio,
        }

    # ------------------------------------------------------------------
    # SVD implícita (sin construir R)
    # ------------------------------------------------------------------

    def roy_dag(self) -> "RoyDAG":
        """Grafo de Roy reducido transitivamente (solo actividades reales) como ``RoyDAG``."""
        cierre = self._cierre_en_bits()
        if cierre is None:        # con ciclos: recorrido por actividad (RoyDAG lo rechazara)
            pred = self.distant_predecessor(dummies=False)
            acts = list(pred)
            return RoyDAG(acts, _reduccion_transitiva_desde_ancestros(acts, pred))
        todas, _, _, inmediatas = cierre
        acts = [a for a in todas if a[0] != "@"]
        return RoyDAG(acts, [(todas[i], a) for a in acts for i in _indices_de_bits(inmediatas[a])])

    def svd_implicit(self, k: int = 5, dense_max: int = 400) -> dict:
        """k primeros valores/vectores singulares de R SIN enumerar las rutas.

        Lanczos sobre el producto (RᵀR)x por DP (O(aristas)); válido aunque el nº
        de rutas sea exponencial. Con n ≤ dense_max usa la matriz RᵀR densa
        (también obtenida por DP). Devuelve ``{"sigma": ndarray, "V": DataFrame
        (actividades × k, v₁ ≥ 0), "info": dict}``; coincide con
        ``path_matrix_SVD(dummies=False)`` salvo signo.
        """
        dag = self.roy_dag(); sigma, V, info = implicit_svd_roy(dag, k=k, dense_max=dense_max)
        return {"sigma": sigma, "V": pd.DataFrame(V, index=dag.acts, columns=[f"v{i + 1}" for i in range(V.shape[1])]), "info": info}

    def tau_rank_k(self, durations: dict, k: int = 1) -> float:
        """τ_k = ‖R_k·t‖∞ sin construir R: ruta más larga con duraciones V_k V_kᵀ t."""
        dag = self.roy_dag(); _, V, _ = implicit_svd_roy(dag, k=k); t = np.array([durations.get(a, 0.0) for a in dag.acts], float)
        return tau_rank_k_roy(dag, t, V, ks=[k])[k]

    # ------------------------------------------------------------------
    # Lectura nativa de .sm (PSPLIB)
    # ------------------------------------------------------------------

    @classmethod
    def from_sm(cls, path) -> "ProjectGraph":
        """Construye un ProjectGraph desde un fichero .sm de PSPLIB.

        Sin dependencias externas (solo ``re`` y ``pathlib``).
        Los metadatos horizon, mpm_time, n_jobs y resource_availabilities
        se guardan en ``project.pert_graph.graph``.
        """
        instance = PSPLIBInstance.read(path)
        project = cls.from_dict_of_sets(instance.predecessors, simplify=True)
        project.pert_graph.graph.update({
            "horizon": instance.horizon,
            "mpm_time": instance.mpm_time,
            "n_jobs": instance.n_jobs,
            "resource_availabilities": instance.resource_availabilities,
        })
        return project

    # ------------------------------------------------------------------
    # Visualización TikZ (sin graphviz)
    # ------------------------------------------------------------------

    def _pert_layout(self, durations: dict = None) -> dict:
        """Calcula posiciones (x, y) en cm para TikZ.

        x = tiempo early del evento (CPM, calculado con durations) * x_scale
        y = posición vertical dentro de la capa (centrada)

        Parameters
        ----------
        durations : dict | None
            {nombre_actividad: float}. Las ficticias se ignoran (duración 0).
        """
        from collections import defaultdict
        G = self.pert_graph
        dur_map = durations or {}
        source = self.nodes[0]
        early: dict = {source: 0}
        for node in nx.topological_sort(G):
            for pred in G.predecessors(node):
                act = G.edges[pred, node].get("activity", "")
                is_dummy = (not act) or act.startswith("@")
                d = 0.0 if is_dummy else float(dur_map.get(act, 0))
                early[node] = max(early.get(node, 0), early.get(pred, 0) + d)
        layers: dict = defaultdict(list)
        for node, t in early.items():
            layers[t].append(node)
        x_scale, y_scale = 2.0, 1.5
        pos: dict = {}
        for t, layer_nodes in sorted(layers.items()):
            n = len(layer_nodes)
            for i, node in enumerate(sorted(layer_nodes)):
                pos[node] = (t * x_scale, (i - (n - 1) / 2.0) * y_scale)
        return pos

    def plot_tikz(self, durations: dict = None, svd: bool = False, path=None) -> str:
        """Genera código TikZ del grafo PERT. Sin dependencia de graphviz.

        Layout: coordenada X = tiempo early del evento (CPM); Y = rank
        dentro de la capa. El resultado es semánticamente correcto:
        cada nodo queda en el instante exacto en que ocurre su evento.

        Parameters
        ----------
        durations : dict | None
            {nombre_actividad: float}. Necesario para el layout CPM y las
            etiquetas de arco. Las ficticias reciben duración 0 automáticamente.
        svd : bool
            Si True, el grosor de los arcos es proporcional al peso v₁.
        path : str | pathlib.Path | None
            Si se pasa, compila con jupyter_tikz y guarda el PDF.

        Returns
        -------
        str   Código TikZ listo para usar en LaTeX/jupyter_tikz.
        """
        G = self.pert_graph
        dur_map = durations or {}
        pos = self._pert_layout(dur_map)

        act_weight: dict = {}
        if svd:
            R = self.path_matrix(dummies=False)
            if R.shape[0] > 0 and R.shape[1] > 0:
                _, _s, Vt = np.linalg.svd(R.values, full_matrices=False)
                v1 = np.abs(Vt[0])
                if v1.max() > 0:
                    v1 = v1 / v1.max()
                act_weight = dict(zip(R.columns, v1))

        lines = [r"\begin{tikzpicture}[>=stealth, font=\small]", ""]

        for node, (x, y) in sorted(pos.items(), key=lambda kv: kv[1][0]):
            label = str(node)
            lines.append(
                rf"  \node[draw, circle, inner sep=3pt] ({label})"
                rf" at ({x:.2f}cm, {y:.2f}cm) {{{label}}};"
            )

        lines.append("")
        for u, v, data in G.edges(data=True):
            act = data.get("activity", "")
            is_dummy = (not act) or act.startswith("@")
            lw_str = ""
            if svd and not is_dummy and act in act_weight:
                w = act_weight[act]
                lw_str = f"line width={0.4 + 2.6 * w:.2f}pt, "
            if is_dummy:
                edge_label = ""
                style = f"dashed, {lw_str}"
            else:
                d_val = dur_map.get(act, "")
                edge_label = str(act) + (f" ({d_val})" if d_val != "" else "")
                style = lw_str
            lines.append(
                rf"  \draw[->, {style}] ({u}) -- ({v})"
                rf" node[midway, above, font=\tiny] {{{edge_label}}};"
            )

        lines.append("")
        lines.append(r"\end{tikzpicture}")
        result = "\n".join(lines)

        if path is not None:
            tex = TexFragment(result, tikz_libraries="arrows.meta").run_latex(
                save_pdf=str(path)
            )
            display(tex)

        return result

    # ------------------------------------------------------------------
    # PERT paso a paso (docencia)
    # ------------------------------------------------------------------

    @staticmethod
    def _num_paso(x) -> str:
        """Formatea un número para los pasos: sin decimales si es entero."""
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return str(x)
        if xf.is_integer():
            return str(int(xf))
        return f"{xf:g}"

    def pert_pasos(self, durations, formato: str = "texto", colores=None) -> dict:
        """Cálculo paso a paso de los tiempos tempranos y tardíos de los sucesos.

        Reproduce el procedimiento que se enseña en clase: recorrido hacia
        delante (``t_j = máx{t_i + t_ij}`` sobre las actividades que *llegan*
        al suceso) y recorrido hacia atrás (``t*_i = mín{t*_j - t_ij}`` sobre
        las actividades que *salen*). Los resultados coinciden con
        :meth:`calculate_pert`.

        Parameters
        ----------
        durations : dict | pandas.Series
            Duraciones de las actividades (las ficticias valen 0).
        formato : {"texto", "latex"}
            Formato de la columna ``calculo``: texto legible
            (``t3 = máx{0 + 3, 2 + 7} = 9``) o expresión LaTeX para modo
            matemático (``t_{3} = \\max\\{0+3,\\ 2+7\\} = 9``).
        colores : list[str] | None
            Con ``formato="latex"``, colores TikZ para teñir cada término del cálculo
            (uno por actividad que llega o sale, por orden); en las holguras, todo
            el cálculo con el primero. Las filas llevan además ``arcos``.

        Returns
        -------
        dict
            ``{"tempranos": DataFrame, "tardios": DataFrame}``, una fila por
            paso (índice 1, 2, …) con las columnas ``suceso``, ``actividades``
            (las que llegan o salen), ``terminos`` (lista de pares
            ``(actividad, valor candidato)``), ``calculo`` y ``valor``.
        """
        if formato not in ("texto", "latex"):
            raise ValueError("formato debe ser 'texto' o 'latex'")
        dur = self._normalize_durations(durations, dummies=True).to_dict()
        G = self.pert_graph
        e2a = self.edges_from_nodes
        orden = self.nodes
        f = self._num_paso
        latex = formato == "latex"

        def nombre(tipo, nodo):
            if latex:
                return f"t_{{{nodo}}}" + ("^{*}" if tipo == "tardio" else "")
            return f"t{nodo}" + ("*" if tipo == "tardio" else "")

        def resultado(valor):
            """El resultado en negrita cuando el cálculo va coloreado."""
            return rf"\mathbf{{{f(valor)}}}" if (latex and colores) else f(valor)

        def expresion(tipo, nodo, operaciones, valor, base=None):
            izq = nombre(tipo, nodo)
            if base is not None:
                if base == f(valor):
                    return f"{izq} = {base}"
                return f"{izq} = {base} = {resultado(valor)}"
            if len(operaciones) == 1:
                return f"{izq} = {operaciones[0]} = {resultado(valor)}"
            if latex:
                op = r"\max" if tipo == "temprano" else r"\min"
                cuerpo = r",\ ".join(operaciones)
                return rf"{izq} = {op}\{{{cuerpo}\}} = {resultado(valor)}"
            op = "máx" if tipo == "temprano" else "mín"
            return f"{izq} = {op}{{{', '.join(operaciones)}}} = {f(valor)}"

        sep = "+" if latex else " + "
        sem = "-" if latex else " - "

        def tinta(ops):
            """Cada término del cálculo con el color de su flecha (solo en LaTeX)."""
            if not (latex and colores):
                return ops
            return [rf"\textcolor{{{colores[k % len(colores)]}}}{{\mathbf{{{o}}}}}" for k, o in enumerate(ops)]

        # Recorrido hacia delante
        early, filas = {}, []
        for nodo in orden:
            arcos = sorted(G.in_edges(nodo),
                           key=lambda e: e2a[(f"{e[0]}", f"{e[1]}")])
            if not arcos:
                early[nodo] = 0
                filas.append(dict(suceso=nodo, actividades="", terminos=[], arcos=[],
                                  calculo=expresion("temprano", nodo, [], 0, base="0"),
                                  valor=0))
                continue
            terminos, ops = [], []
            for u, v in arcos:
                act = e2a[(f"{u}", f"{v}")]
                d = dur.get(act, 0)
                terminos.append((act, early[f"{u}"] + d))
                ops.append(f"{f(early[f'{u}'])}{sep}{f(d)}")
            early[nodo] = max(t for _, t in terminos)
            filas.append(dict(suceso=nodo,
                              actividades=", ".join(a for a, _ in terminos),
                              terminos=terminos,
                              arcos=[(f"{u}", f"{v}") for u, v in arcos],
                              calculo=expresion("temprano", nodo, tinta(ops), early[nodo]),
                              valor=early[nodo]))
        tempranos = pd.DataFrame(filas, index=pd.RangeIndex(1, len(filas) + 1, name="paso"))

        # Recorrido hacia atrás
        late, filas = {}, []
        for nodo in orden[::-1]:
            arcos = sorted(G.out_edges(nodo),
                           key=lambda e: e2a[(f"{e[0]}", f"{e[1]}")])
            if not arcos:
                late[nodo] = early[nodo]
                filas.append(dict(suceso=nodo, actividades="", terminos=[], arcos=[],
                                  calculo=expresion("tardio", nodo, [], late[nodo],
                                                    base=nombre("temprano", nodo)),
                                  valor=late[nodo]))
                continue
            terminos, ops = [], []
            for u, v in arcos:
                act = e2a[(f"{u}", f"{v}")]
                d = dur.get(act, 0)
                terminos.append((act, late[f"{v}"] - d))
                ops.append(f"{f(late[f'{v}'])}{sem}{f(d)}")
            late[nodo] = min(t for _, t in terminos)
            filas.append(dict(suceso=nodo,
                              actividades=", ".join(a for a, _ in terminos),
                              terminos=terminos,
                              arcos=[(f"{u}", f"{v}") for u, v in arcos],
                              calculo=expresion("tardio", nodo, tinta(ops), late[nodo]),
                              valor=late[nodo]))
        tardios = pd.DataFrame(filas, index=pd.RangeIndex(1, len(filas) + 1, name="paso"))

        # Holguras totales: H_ij = t*_j - t_i - d_ij
        filas = []
        for nodo in orden:
            for u, v in sorted(G.out_edges(nodo), key=lambda e: e2a[(f"{e[0]}", f"{e[1]}")]):
                act = e2a[(f"{u}", f"{v}")]
                d = dur.get(act, 0)
                h = late[f"{v}"] - early[f"{u}"] - d
                izq = (rf"H_{{\mbox{{{self._tex_texto(act)}}}}}" if latex else f"H({act})")
                if latex:
                    cuerpo = (rf"t_{{{v}}}^{{*}} - t_{{{u}}} - d = "
                              rf"{f(late[f'{v}'])} - {f(early[f'{u}'])} - {f(d)}")
                    if colores:
                        cuerpo = rf"\textcolor{{{colores[0]}}}{{\mathbf{{{cuerpo}}}}}"
                else:
                    cuerpo = f"t{v}* - t{u} - d = {f(late[f'{v}'])} - {f(early[f'{u}'])} - {f(d)}"
                res = rf"\mathbf{{{f(h)}}}" if (latex and colores) else f(h)
                filas.append(dict(actividad=act, origen=u, destino=v,
                                  calculo=f"{izq} = {cuerpo} = {res}", valor=h))
        holguras = pd.DataFrame(filas, index=pd.RangeIndex(1, len(filas) + 1, name="paso"))
        return dict(tempranos=tempranos, tardios=tardios, holguras=holguras)

    # ------------------------------------------------------------------
    # Visor de la construcción del grafo PERT a partir del cuadro de prelaciones
    # ------------------------------------------------------------------

    def _matriz_prelaciones(self, tabla=None) -> pd.DataFrame:
        """Matriz booleana de predecesoras inmediatas (filas: actividad; columnas:
        predecesora). Con ``tabla`` (una :class:`PredecessorTable`) es la suya; sin ella
        se deduce del proyecto."""
        if tabla is not None:
            return tabla.immediate_linkage_matrix
        dist = self.distant_predecessor()
        nombres = sorted(dist)
        M = pd.DataFrame(False, index=nombres, columns=nombres, dtype=bool)
        M.index.name = "activities"
        for a, preds in dist.items():
            for p in preds:
                if not any(p in dist[q] for q in preds if q != p):
                    M.loc[a, p] = True
        return M

    def plot_tikz_construccion(self, escala: float = 1.0, tabla=None, ordenar: bool = False,
                               textos: dict = None) -> list:
        """Fotogramas TikZ (uno por ``tikzpicture``) de la construcción del grafo PERT
        a partir del cuadro de prelaciones: a la izquierda el cuadro y a la derecha el
        grafo con la numeración de nodos y sin tiempos.

        Secuencia: (1) las actividades iniciales (filas sin predecesoras) se señalan en
        el cuadro y en el grafo, y lo que no actúa queda en gris claro; (2) una actividad
        por fotograma (con las ficticias que necesite), que pasa a negro mientras se
        señala su fila; (3) cuando todas las actividades con columnas iguales en el cuadro
        (mismas sucesoras, o ninguna) ya están dibujadas, se resaltan esas columnas y el
        suceso en que se fusionan sus finales; y el grafo completo al final. Con
        ``ordenar=True`` el cuadro va en orden de precedencia. ``textos`` cambia los
        rótulos (``{"completo": ..., "iniciales": ...}``)."""
        rot = {"completo": "Cuadro de prelaciones y grafo PERT con la numeración de nodos",
               "iniciales": "Actividades iniciales (sin predecesoras): {acts}",
               "actividad": "Actividad {act}: predecesoras {preds}",
               "ficticia": "se añade la actividad ficticia {fict}",
               "iguales": "Las columnas {cols} son iguales: sus actividades terminan en el mismo suceso ({nodo})",
               "vacias": "Las columnas {cols} están vacías: sus actividades terminan en el suceso final ({nodo})"}
        rot.update(textos or {})
        tex = self._tex_texto
        G = self.pert_graph
        datos = self._matriz_prelaciones(tabla)
        orden = list(_ordenar_por_precedencia(datos).index)            # orden de dibujo
        datos = _ordenar_por_precedencia(datos) if ordenar else datos
        nombres = [f"{n}" for n in datos.index]
        preds = {f"{a}": [f"{p}" for p in datos.columns if bool(datos.loc[a, p])] for a in datos.index}
        arco = {}                                                         # actividad -> (u, v)
        ficticias = []
        for u, v, d in G.edges(data=True):
            act = d.get("activity", "")
            if (not act) or act.startswith("@"):
                ficticias.append((f"{u}", f"{v}"))
            else:
                arco[act] = (f"{u}", f"{v}")
        paso_de = {a: k for k, a in enumerate(orden)}
        # cada ficticia se dibuja con la primera actividad que sale de su destino
        # (o, si no hay, con la última que llega a su origen)
        ficticia_en = {}
        for (u, v) in ficticias:
            sale = [a for a, (x, _) in arco.items() if x == v]
            llega = [a for a, (_, y) in arco.items() if y == u]
            cand = min(sale, key=paso_de.get) if sale else (max(llega, key=paso_de.get) if llega else orden[-1])
            ficticia_en.setdefault(cand, []).append((u, v))
        # grupos de columnas iguales (mismas sucesoras) y paso en que se completan
        suc = {a: frozenset(r for r in nombres if a in preds[r]) for a in nombres}
        grupos = {}
        for a in nombres:
            grupos.setdefault(suc[a], []).append(a)
        fusion = {}                                                       # actividad final -> grupo
        for s_, miembros in grupos.items():
            if len(miembros) < 2:
                continue
            fin = {arco[a][1] for a in miembros if a in arco}
            if len(fin) != 1:
                continue
            ultimo = max(miembros, key=paso_de.get)
            fusion[ultimo] = (sorted(miembros, key=paso_de.get), fin.pop(), not s_)

        # geometría y diseño del grafo (el de pert(), sin tiempos)
        trazado = self._trazado_pert({}, escala, False)
        pos = trazado["pos"] if trazado else self._pert_posiciones(escala, None)
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        if trazado:
            for x, y in trazado["etiquetas"].values():
                xs.append(x); ys.append(y)
            for r in trazado["rutas"].values():
                for x, y in r:
                    xs.append(x); ys.append(y)
        xmin, xmax, ymin, ymax = min(xs) - 0.7, max(xs) + 0.7, min(ys) - 0.7, max(ys) + 0.7
        _, ancho, margen, alto = _geometria_matriz(datos)
        n = len(nombres)
        Wm, Hm = (n + margen) * ancho, (n + alto) * 0.78
        hueco = 1.0
        yc_g = (ymin + ymax) / 2
        yc_m = (alto - n) / 2 * 0.78
        ancho_total = Wm + hueco + (xmax - xmin)
        y_texto = min(ymin, yc_g - Hm / 2) - 0.4
        f = lambda *a, **k: self._pert_fotograma(pos, {}, {}, escala, {}, {}, trazado=trazado,
                                                 con_duracion=False, **k)

        def lista(xs_):
            xs_ = list(xs_)
            return xs_[0] if len(xs_) == 1 else ", ".join(xs_[:-1]) + " y " + xs_[-1]

        def _previas(actuales, ficticias_actuales):
            """Actividades (y ficticias) previas que dan origen a las que se añaden."""
            if not actuales and not ficticias_actuales:
                return set(), set()
            prev = {p for a in actuales for p in preds.get(a, [])}
            origen = {arco[a][0] for a in actuales if a in arco}
            origen |= {u for (u, _) in ficticias_actuales}
            pf, cambio = set(), True
            while cambio:                    # ficticias existentes que desembocan en el origen
                cambio = False
                for (u, v) in ficticias:
                    if (u, v) not in ficticias_actuales and (u, v) not in pf and v in origen:
                        pf.add((u, v)); origen.add(u); cambio = True
            return prev, pf

        def fotograma(hechas, actuales, ficticias_hechas, ficticias_actuales, texto,
                      filas=None, columnas=(), nodo_resaltado=None, completo=False):
            ea = {}
            previas, previas_f = _previas(actuales, ficticias_actuales)
            for u, v, d in G.edges(data=True):
                clave = (f"{u}", f"{v}")
                act = d.get("activity", "")
                fict = (not act) or act.startswith("@")
                if completo:
                    ea[clave] = "hecho"
                elif (clave in ficticias_actuales) if fict else (act in actuales):
                    ea[clave] = "actual"
                elif (clave in previas_f) if fict else (act in previas):
                    ea[clave] = "previa"
                elif (clave in ficticias_hechas) if fict else (act in hechas):
                    ea[clave] = "hecho"
                else:
                    ea[clave] = "oculto"
            tocados = {x for k, e in ea.items() if e != "oculto" for x in k}
            en = {nd: ("hecho" if nd in tocados else "oculto") for nd in pos}
            if nodo_resaltado is not None:
                en[nodo_resaltado] = "actual"
            cuerpo = self._pert_fotograma(pos, {}, {}, escala, {}, {}, trazado=trazado,
                                          con_duracion=False, estilo_arcos=ea, estilo_nodos=en,
                                          numeros=completo)
            L = [r"\begin{tikzpicture}[>=stealth]", _COLORES_MATRIZ,
                 rf"\begin{{scope}}[xshift={margen * ancho:.2f}cm,yshift={yc_g - yc_m:.2f}cm,"
                 rf"x={ancho:.2f}cm,y=0.78cm,font=\small]"]
            L += _cuerpo_matriz_tikz(datos, filas, columnas)
            L += [r"\end{scope}",
                  rf"\begin{{scope}}[xshift={Wm + hueco - xmin:.2f}cm]", cuerpo, r"\end{scope}",
                  rf"\node[anchor=north, font=\small, align=center, text width={ancho_total:.2f}cm] "
                  rf"at ({ancho_total / 2:.2f}cm, {y_texto:.2f}cm) {{{texto}}};",
                  r"\end{tikzpicture}"]
            return "\n".join(L)

        codigos = []
        hechas, fict_hechas = set(), set()
        iniciales = [a for a in orden if not preds[a]]

        def estado_filas(actuales):
            return {a: ("actual" if a in actuales else None if a in hechas else "gris") for a in nombres}

        # (1) actividades iniciales
        fict_ini = [fc for a in iniciales for fc in ficticia_en.get(a, [])]
        codigos.append(fotograma(hechas, set(iniciales), fict_hechas, set(fict_ini),
                                 tex(rot["iniciales"].format(acts=lista(iniciales))),
                                 filas=estado_filas(set(iniciales))))
        hechas |= set(iniciales)
        fict_hechas |= set(fict_ini)
        for a in iniciales:
            if a in fusion:
                grp, nodo, vacia = fusion[a]
                codigos.append(self._paso_fusion(fotograma, grp, nodo, vacia, hechas, fict_hechas, rot, lista, tex))
        # (2) y (3) el resto, una actividad por fotograma
        for a in orden:
            if a in iniciales:
                continue
            fi = ficticia_en.get(a, [])
            texto = rot["actividad"].format(act=a, preds=lista(preds[a]))
            if fi:
                texto += "; " + rot["ficticia"].format(fict=lista([f"@{u}→{v}" for u, v in fi]))
            codigos.append(fotograma(hechas, {a}, fict_hechas, set(fi), tex(texto),
                                     filas=estado_filas({a})))
            hechas.add(a)
            fict_hechas |= set(fi)
            if a in fusion:
                grp, nodo, vacia = fusion[a]
                codigos.append(self._paso_fusion(fotograma, grp, nodo, vacia, hechas, fict_hechas, rot, lista, tex))
        codigos.append(fotograma(hechas, set(), fict_hechas, set(), tex(rot["completo"]), completo=True))
        return codigos

    @staticmethod
    def _paso_fusion(fotograma, grupo, nodo, vacia, hechas, fict_hechas, rot, lista, tex):
        """Fotograma de las columnas iguales de ``grupo`` y del suceso donde terminan."""
        texto = rot["vacias" if vacia else "iguales"].format(cols=lista(grupo), nodo=nodo)
        return fotograma(hechas, set(), fict_hechas, set(), tex(texto),
                         columnas=grupo, nodo_resaltado=nodo)

    def plot_tikz_construccion_roy_tabla(self, escala: float = 1.0, tabla=None, ordenar: bool = False,
                                         textos: dict = None, etiquetas: dict = None) -> list:
        """Fotogramas TikZ (uno por ``tikzpicture``) de la construcción del grafo Roy, sin
        tiempos, a partir del cuadro de prelaciones: a la izquierda el cuadro y a la derecha
        el grafo.

        Secuencia: (1) el nodo Inicio y las actividades iniciales (filas sin predecesoras);
        (2) una actividad por fotograma, en naranja, con sus arcos desde las predecesoras
        (se señala su fila en el cuadro); (3) el nodo Fin
        tras las actividades que no preceden a ninguna; y el grafo completo. Con
        ``ordenar=True`` el cuadro va en orden de precedencia. ``textos`` cambia los rótulos
        (``{"completo", "iniciales", "actividad", "fin"}``) y ``etiquetas`` los nombres de
        ``start`` y ``finish``."""
        rot = {"completo": "Cuadro de prelaciones y grafo Roy",
               "iniciales": "Nodo {inicio} y actividades iniciales (sin predecesoras): {acts}",
               "actividad": "Actividad {act}: predecesoras {preds}",
               "fin": "Nodo {fin}: lo siguen las actividades que no preceden a ninguna ({acts})"}
        rot.update(textos or {})
        et = {"start": "Inicio", "finish": "Fin"}
        et.update(etiquetas or {})
        tex = self._tex_texto
        G = self._roy_grafo_actual()
        datos = self._matriz_prelaciones(tabla)
        orden = list(_ordenar_por_precedencia(datos).index)
        datos = _ordenar_por_precedencia(datos) if ordenar else datos
        nombres = [f"{n}" for n in datos.index]
        preds = {f"{a}": [f"{p}" for p in G.predecessors(a) if f"{p}" != "start"] for a in orden}
        orden = [f"{a}" for a in orden]
        iniciales = [a for a in orden if not preds[a]]
        finales = [a for a in orden if all(a not in preds[b] for b in orden)]
        paso = {"start": 1}
        paso.update({a: 1 for a in iniciales})
        k = 1
        for a in orden:
            if a not in iniciales:
                k += 1
                paso[a] = k
        paso["finish"] = k + 1
        ultimo = paso["finish"]

        trazado = _disposicion_capas({f"{n}": (self._ROY_ANCHO, self._ROY_ALTO) for n in G.nodes},
                                     [(f"{u}", f"{v}") for u, v in G.edges()],
                                     separacion=0.3 * escala, hueco=0.4 * escala)
        if trazado:
            pos = trazado["pos"]
        else:
            capa = self._roy_capas(G)
            from collections import defaultdict
            por_capa = defaultdict(list)
            for n in sorted(G.nodes, key=lambda n: f"{n}"):
                por_capa[capa[n]].append(n)
            pos = {f"{n}": ((self._ROY_ANCHO + 1.1) * c * escala, ((len(ns) - 1) / 2.0 - i) * (self._ROY_ALTO + 0.6) * escala)
                   for c, ns in por_capa.items() for i, n in enumerate(ns)}
        ident = {f"{n}": f"r{i}" for i, n in enumerate(sorted(G.nodes, key=lambda n: f"{n}"))}
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        if trazado:
            for r in trazado["rutas"].values():
                for x, y in r:
                    xs.append(x); ys.append(y)
        xmin, xmax, ymin, ymax = min(xs) - 1.5, max(xs) + 1.5, min(ys) - 1.2, max(ys) + 1.2
        _, ancho, margen, alto = _geometria_matriz(datos)
        n = len(nombres)
        Wm, Hm = (n + margen) * ancho, (n + alto) * 0.78
        hueco = 1.0
        yc_g = (ymin + ymax) / 2
        yc_m = (alto - n) / 2 * 0.78
        ancho_total = Wm + hueco + (xmax - xmin)
        y_texto = min(ymin, yc_g - Hm / 2) - 0.4

        def lista(xs_):
            xs_ = list(xs_)
            return xs_[0] if len(xs_) == 1 else ", ".join(xs_[:-1]) + " y " + xs_[-1]

        def grafo(kk, actuales, completo):
            L, A = [], []
            for u, v in G.edges():
                u, v = f"{u}", f"{v}"
                if paso[v] > kk:
                    continue
                actual = (paso[v] == kk) and not completo
                estilo = ["->", "thick"] + (["orange!80!black", "ultra thick"] if actual else [])
                if trazado and (u, v) in trazado["rutas"]:
                    A.append(rf"  \draw[{', '.join(estilo)}] "
                             + self._ruta_tikz(u, v, pos, trazado["rutas"][(u, v)],
                                               trazado["curva"][(u, v)], lambda m: ident[m]) + ";")
                    continue
                curva = self._curvatura_tikz(pos, u, v, 0.8 * escala)
                trazo = f"to[{curva}]" if curva else "--"
                A.append(rf"  \draw[{', '.join(estilo)}] ({ident[u]}) {trazo} ({ident[v]});")
            for nodo, (x, y) in pos.items():
                r = ident[nodo]
                if paso[nodo] > kk:
                    L.append(rf"  \coordinate ({r}) at ({x:.2f}cm, {y:.2f}cm);")
                    continue
                estilo = (dict(borde="orange!80!black", grosor="very thick", relleno="orange!20",
                               barra="orange!80!black") if (nodo in actuales and not completo) else {})
                L += self._roy_caja_tikz(r, x, y, tex(f"{et.get(nodo, nodo)}"), None, **estilo)
            return "\n".join(L + A)

        def fotograma(kk, actuales, texto, filas=None, columnas=(), completo=False):
            L = [r"\begin{tikzpicture}[>=stealth]", _COLORES_MATRIZ,
                 rf"\begin{{scope}}[xshift={margen * ancho:.2f}cm,yshift={yc_g - yc_m:.2f}cm,"
                 rf"x={ancho:.2f}cm,y=0.78cm,font=\small]"]
            L += _cuerpo_matriz_tikz(datos, filas, columnas)
            L += [r"\end{scope}",
                  rf"\begin{{scope}}[xshift={Wm + hueco - xmin:.2f}cm]", grafo(kk, actuales, completo), r"\end{scope}",
                  rf"\path (0,{ymin:.2f}) rectangle ({ancho_total:.2f},{ymax:.2f});",
                  rf"\node[anchor=north, font=\small, align=center, text width={ancho_total:.2f}cm] "
                  rf"at ({ancho_total / 2:.2f}cm, {y_texto:.2f}cm) {{{texto}}};",
                  r"\end{tikzpicture}"]
            return "\n".join(L)

        hechas = set()

        def estado_filas(actuales):
            return {a: ("actual" if a in actuales else None if a in hechas else "gris") for a in nombres}

        codigos = []
        codigos.append(fotograma(1, set(iniciales) | {"start"},
                                 tex(rot["iniciales"].format(inicio=et["start"], acts=lista(iniciales))),
                                 filas=estado_filas(set(iniciales))))
        hechas |= set(iniciales)
        for a in orden:
            if a in iniciales:
                continue
            codigos.append(fotograma(paso[a], {a}, tex(rot["actividad"].format(act=a, preds=lista(preds[a]))),
                                     filas=estado_filas({a})))
            hechas.add(a)
        codigos.append(fotograma(ultimo, {"finish"},
                                 tex(rot["fin"].format(fin=et["finish"], acts=lista(finales))),
                                 filas={a: ("actual" if a in finales else None) for a in nombres}))
        codigos.append(fotograma(ultimo, set(), tex(rot["completo"]), completo=True))
        return codigos

    def roy_construccion_html(self, escala: float = 1.0, tabla=None, ordenar: bool = False,
                              textos: dict = None, etiquetas: dict = None) -> str:
        """Visor HTML con botones de la construcción del grafo Roy (véase
        :meth:`plot_tikz_construccion_roy_tabla`). El primer fotograma (paso 0) es el cuadro
        con el grafo completo; a partir del paso 1 se construye desde el principio."""
        codigos = self.plot_tikz_construccion_roy_tabla(escala, tabla, ordenar, textos, etiquetas)
        return _html_fotogramas_tikz(codigos, resultado_primero=True)

    def display_roy_construccion(self, escala: float = 1.0, tabla=None, ordenar: bool = False,
                                 textos: dict = None, etiquetas: dict = None, visor: bool = True) -> None:
        """Muestra en un cuaderno la construcción del grafo Roy a partir del cuadro de
        prelaciones, con botones (``visor=True``). Con ``visor=False``, o si no hay LaTeX,
        se muestra el grafo final con :meth:`roy`."""
        if visor:
            try:
                display(HTML(self.roy_construccion_html(escala, tabla, ordenar, textos, etiquetas)))
                return
            except (FileNotFoundError, RuntimeError) as e:
                import warnings
                warnings.warn(f"No se pudo generar el visor TikZ ({e}); se muestra el grafo final.")
        display(self.roy())

    def plot_tikz_rutas(self, grafo: str = "pert", escala: float = 1.0, etiquetas: dict = None):
        """Imágenes TikZ de la matriz de rutas con el grafo PERT o Roy (sin tiempos) encima:
        la primera con todo el grafo y una más por cada ruta, con sus actividades y nodos
        en naranja (el resto en gris claro) y su fila resaltada en la matriz. Todas tienen la
        misma caja, para poder superponer zonas de clic. Devuelve ``(codigos, matriz, filas)``:
        ``matriz`` = ``path_matrix(dummies=False)`` y ``filas`` = ``[(x, y, ancho, alto)]``
        con la posición de cada fila como fracción de la imagen (origen arriba a la izquierda)."""
        if grafo not in ("pert", "roy"):
            raise ValueError("grafo debe ser 'pert' o 'roy'")
        R = self.path_matrix(dummies=False).astype(bool)
        if grafo == "pert":
            Rd = self.path_matrix(dummies=True).astype(bool)
            G = self.pert_graph
            trazado = self._trazado_pert({}, escala, False)
            pos = trazado["pos"] if trazado else self._pert_posiciones(escala, None)
            xs = [x for x, _ in pos.values()]; ys = [y for _, y in pos.values()]
            if trazado:
                for x, y in trazado["etiquetas"].values():
                    xs.append(x); ys.append(y)
                for r in trazado["rutas"].values():
                    for x, y in r:
                        xs.append(x); ys.append(y)
            gx0, gx1, gy0, gy1 = min(xs) - 0.7, max(xs) + 0.7, min(ys) - 0.7, max(ys) + 0.7
            arco_de = {d.get("activity", ""): (f"{u}", f"{v}") for u, v, d in G.edges(data=True)}
            todos = [(f"{u}", f"{v}") for u, v in G.edges()]

            def grafo_cuerpo(i):
                if i is None:
                    ea = {e: "hecho" for e in todos}
                    en = {n: "hecho" for n in pos}
                else:
                    usados = [c for c in Rd.columns if bool(Rd.iloc[i][c])]
                    act = {arco_de[c] for c in usados if c in arco_de}
                    ea = {e: ("actual" if e in act else "gris") for e in todos}
                    nodos = {x for e in act for x in e}
                    en = {n: ("actual" if n in nodos else "gris") for n in pos}
                return self._pert_fotograma(pos, {}, {}, escala, {}, {}, trazado=trazado,
                                            con_duracion=False, estilo_arcos=ea, estilo_nodos=en,
                                            numeros=True, color_actual="blue")
        else:
            et = {"start": "Inicio", "finish": "Fin"}
            et.update(etiquetas or {})
            G = self._roy_grafo_actual()
            trazado = _disposicion_capas({f"{n}": (self._ROY_ANCHO, self._ROY_ALTO) for n in G.nodes},
                                         [(f"{u}", f"{v}") for u, v in G.edges()],
                                         separacion=0.3 * escala, hueco=0.4 * escala)
            if trazado:
                pos = trazado["pos"]
            else:
                capa = self._roy_capas(G)
                from collections import defaultdict
                por_capa = defaultdict(list)
                for n in sorted(G.nodes, key=lambda n: f"{n}"):
                    por_capa[capa[n]].append(n)
                pos = {f"{n}": ((self._ROY_ANCHO + 1.1) * c * escala, ((len(ns) - 1) / 2.0 - i) * (self._ROY_ALTO + 0.6) * escala)
                       for c, ns in por_capa.items() for i, n in enumerate(ns)}
            ident = {f"{n}": f"r{k}" for k, n in enumerate(sorted(G.nodes, key=lambda n: f"{n}"))}
            capas = self._roy_capas(G)
            xs = [x for x, _ in pos.values()]; ys = [y for _, y in pos.values()]
            if trazado:
                for r in trazado["rutas"].values():
                    for x, y in r:
                        xs.append(x); ys.append(y)
            gx0, gx1, gy0, gy1 = min(xs) - 1.5, max(xs) + 1.5, min(ys) - 1.2, max(ys) + 1.2

            def grafo_cuerpo(i):
                if i is None:
                    nodos = set(pos)
                    arcos = {(f"{u}", f"{v}") for u, v in G.edges()}
                else:
                    ruta = sorted((c for c in R.columns if bool(R.iloc[i][c])), key=lambda c: capas[c])
                    cadena = ["start"] + [f"{c}" for c in ruta] + ["finish"]
                    nodos = set(cadena)
                    arcos = set(zip(cadena[:-1], cadena[1:]))
                L, A = [], []
                for u, v in G.edges():
                    u, v = f"{u}", f"{v}"
                    est = ["->", "thick"]
                    if i is not None:
                        est += (["blue!80!black", "ultra thick"] if (u, v) in arcos else ["gray!45"])
                    if trazado and (u, v) in trazado["rutas"]:
                        A.append(rf"  \draw[{', '.join(est)}] "
                                 + self._ruta_tikz(u, v, pos, trazado["rutas"][(u, v)],
                                                   trazado["curva"][(u, v)], lambda m: ident[m]) + ";")
                        continue
                    curva = self._curvatura_tikz(pos, u, v, 0.8 * escala)
                    A.append(rf"  \draw[{', '.join(est)}] ({ident[u]}) {'to[' + curva + ']' if curva else '--'} ({ident[v]});")
                for nodo, (x, y) in pos.items():
                    r = ident[nodo]
                    if i is None:
                        estilo, gris = {}, ""
                    elif nodo in nodos:
                        estilo, gris = dict(borde="blue!80!black", grosor="very thick", relleno="blue!15",
                                            barra="blue!70!black"), ""
                    else:
                        estilo, gris = dict(borde="gray!45", grosor="thick", relleno="white"), "text=gray!55"
                    L += self._roy_caja_tikz(r, x, y, self._tex_texto(f"{et.get(nodo, nodo)}"), None,
                                             color_texto=gris, **estilo)
                return "\n".join(L + A)
        # ensamblaje: grafo arriba, matriz debajo, todo centrado en una caja común
        nf, nc = R.shape
        _, ancho, margen, alto = _geometria_matriz(R)
        Wm, Hm = (nc + margen) * ancho, (nf + alto) * 0.78
        gw, gh = gx1 - gx0, gy1 - gy0
        BW = max(gw, Wm) + 0.4
        hueco = 0.9
        BH = gh + hueco + Hm + 0.1
        gx_off = (BW - gw) / 2
        mx_off = (BW - Wm) / 2                       # borde izquierdo de las etiquetas
        y_cab = -(gh + hueco)                        # arriba de la cabecera de la matriz
        y_cel = y_cab - alto * 0.78                  # arriba de la primera fila
        nombres = [f"{n}" for n in R.index]

        def ensambla(i):
            filas = {nombres[i]: "actual"} if i is not None else None
            L = [r"\begin{tikzpicture}[>=stealth]", _COLORES_MATRIZ,
                 rf"\path (0,0) rectangle ({BW:.2f},{-BH:.2f});",
                 rf"\begin{{scope}}[xshift={gx_off - gx0:.2f}cm,yshift={-gy1:.2f}cm]", grafo_cuerpo(i), r"\end{scope}",
                 rf"\begin{{scope}}[xshift={mx_off + margen * ancho:.2f}cm,yshift={y_cel:.2f}cm,"
                 rf"x={ancho:.2f}cm,y=0.78cm,font=\small]"]
            L += _cuerpo_matriz_tikz(R, filas, valor="1", valor_cero="0", resalte="blue",
                                     rotulo=r"actividad $\to$\\ ruta $\downarrow$")
            L += [r"\end{scope}", r"\end{tikzpicture}"]
            return "\n".join(L)

        codigos = [ensambla(None)] + [ensambla(i) for i in range(nf)]
        filas_pos = [((mx_off) / BW, -(y_cel - i * 0.78) / BH, Wm / BW, 0.78 / BH) for i in range(nf)]
        return codigos, R, filas_pos

    def rutas_html(self, grafo: str = "pert", escala: float = 1.0, etiquetas: dict = None) -> str:
        """Visor HTML de la matriz de rutas, todo en TikZ: el grafo PERT o Roy encima de la
        matriz y, al hacer clic en una fila de la matriz, se resaltan en el grafo las
        actividades y los nodos de esa ruta (otro clic la deselecciona). Hace falta LaTeX."""
        import base64, hashlib
        codigos, R, filas = self.plot_tikz_rutas(grafo, escala, etiquetas)
        svgs = _svgs_de_tikz(codigos)
        uid = "pmr" + hashlib.md5((grafo + "".join(codigos)).encode("utf-8")).hexdigest()[:10]
        imgs = "".join(
            f'<img alt="grafo y matriz de rutas" class="pmule-rut-img" style="display:{"block" if k == 0 else "none"};'
            f'width:100%;height:auto" '
            f'src="data:image/svg+xml;base64,{base64.b64encode(sv.encode("utf8")).decode("ascii")}">'
            for k, sv in enumerate(svgs))
        zonas = "".join(
            f'<div class="pmule-rut-zona" data-k="{k + 1}" title="Clic para resaltar la ruta" onclick="pmuleRut(this)" '
            f'style="position:absolute;left:{x * 100:.3f}%;top:{y * 100:.3f}%;width:{w * 100:.3f}%;height:{h * 100:.3f}%;'
            f'cursor:pointer"></div>' for k, (x, y, w, h) in enumerate(filas))
        ayuda = ("&#128070; <b>Interactivo:</b> haz clic en una fila de la matriz para resaltar en el grafo "
                 "las actividades y los nodos de esa ruta (otro clic la deselecciona).")
        return (f'<div id="{uid}" class="pmule-rut" style="margin:1em 0">'
                '<style>.pmule-rut-zona:hover{background:rgba(204,102,0,.12)}</style>'
                f'<div style="position:relative;max-width:760px;margin:0 auto">{imgs}{zonas}</div>'
                f'<div style="text-align:center;margin:.6em 0;font-size:.95em;color:#8a4b00">{ayuda}</div></div>'
                '<script>window.pmuleRut = window.pmuleRut || function(z) {'
                'var box = z.closest(".pmule-rut"), imgs = box.querySelectorAll(".pmule-rut-img");'
                'var k = parseInt(z.getAttribute("data-k")), sel = box.getAttribute("data-sel") === String(k);'
                'for (var i = 0; i < imgs.length; i++) imgs[i].style.display = "none";'
                'if (sel) { box.removeAttribute("data-sel"); imgs[0].style.display = "block"; return; }'
                'box.setAttribute("data-sel", String(k)); imgs[k].style.display = "block"; };</script>')

    def display_rutas(self, grafo: str = "pert", escala: float = 1.0, etiquetas: dict = None,
                      max_rutas: int = 40) -> None:
        """Muestra en un cuaderno el visor de la matriz de rutas (:meth:`rutas_html`); sin
        LaTeX, o con más de ``max_rutas`` rutas (una imagen por ruta), se muestra la matriz de rutas
        con :meth:`display_path_matrix`."""
        if len(self.path_matrix(dummies=False)) > max_rutas:       # demasiadas imágenes: matriz estática
            display(self.display_path_matrix(dummies=False))
            return
        try:
            display(HTML(self.rutas_html(grafo, escala, etiquetas)))
        except (FileNotFoundError, RuntimeError) as e:
            import warnings
            warnings.warn(f"No se pudo generar el visor de rutas ({e}); se muestra la matriz.")
            display(self.display_path_matrix(dummies=False))

    def pert_construccion_html(self, escala: float = 1.0, tabla=None, ordenar: bool = False,
                               textos: dict = None) -> str:
        """Visor HTML con botones de la construcción del grafo PERT (véase
        :meth:`plot_tikz_construccion`). El primer fotograma (paso 0) es el cuadro con
        el grafo completo; a partir del paso 1 se construye desde el principio."""
        codigos = self.plot_tikz_construccion(escala, tabla, ordenar, textos)
        return _html_fotogramas_tikz(codigos, resultado_primero=True)

    def display_pert_construccion(self, escala: float = 1.0, tabla=None, ordenar: bool = False,
                                  textos: dict = None, visor: bool = True) -> None:
        """Muestra en un cuaderno la construcción del grafo PERT a partir del cuadro de
        prelaciones, con botones (``visor=True``, :meth:`pert_construccion_html`). Con
        ``visor=False``, o si no hay LaTeX, se muestra el grafo final con :meth:`pert`."""
        if visor:
            try:
                display(HTML(self.pert_construccion_html(escala, tabla, ordenar, textos)))
                return
            except (FileNotFoundError, RuntimeError) as e:
                import warnings
                warnings.warn(f"No se pudo generar el visor TikZ ({e}); se muestra el grafo final.")
        display(self.pert())

    def pert_pasos_html(self, durations, escala: float = 1.0, posiciones: dict = None,
                        textos: dict = None) -> str:
        """El cálculo de los tiempos del PERT como visor HTML con botones
        (anterior / siguiente) cuyos fotogramas son los dibujos TikZ de
        :meth:`plot_tikz_pasos`, compilados a SVG (hace falta ``pdflatex`` y
        ``pdftocairo``). El primer fotograma (paso 0) es el grafo resuelto; a partir
        del paso 1 se calcula desde el principio."""
        codigos = self.plot_tikz_pasos(durations, beamer=False, escala=escala,
                                       posiciones=posiciones, textos=textos)
        return _html_fotogramas_tikz(codigos, resultado_primero=True)

    def display_pert_pasos(self, durations, escala: float = 1.0, posiciones: dict = None,
                           textos: dict = None, visor: bool = True) -> None:
        """Muestra en un cuaderno el cálculo de los tiempos del PERT paso a paso:
        grafo TikZ que se va rellenando, con botones (``visor=True``,
        :meth:`pert_pasos_html`). Con ``visor=False``, o si no hay LaTeX, la
        lista de cálculos como Markdown con LaTeX."""
        if visor:
            try:
                display(HTML(self.pert_pasos_html(durations, escala, posiciones, textos)))
                return
            except (FileNotFoundError, RuntimeError) as e:
                import warnings
                warnings.warn(f"No se pudo generar el visor TikZ ({e}); se muestra la lista de cálculos.")
        pasos = self.pert_pasos(durations, formato="latex")
        for titulo, clave in (("Tiempos tempranos (hacia delante)", "tempranos"),
                              ("Tiempos tardíos (hacia atrás)", "tardios")):
            lineas = [f"**{titulo}**", ""]
            for _, fila in pasos[clave].iterrows():
                act = f" &nbsp; ({fila['actividades']})" if fila["actividades"] else ""
                lineas.append(f"- ${fila['calculo']}${act}")
            display(Markdown("\n".join(lineas)))
        lineas = ["**Holguras totales**", ""]
        for _, fila in pasos["holguras"].iterrows():
            lineas.append(f"- ${fila['calculo']}$")
        display(Markdown("\n".join(lineas)))

    def _pert_layout_capas(self, escala: float = 1.0) -> dict:
        """Posiciones (x, y) en cm por capas topológicas (número de arcos del
        camino más largo desde el origen), compactas para diapositivas."""
        from collections import defaultdict
        G = self.pert_graph
        capa: dict = {}
        for nodo in nx.topological_sort(G):
            capa[nodo] = max((capa[p] + 1 for p in G.predecessors(nodo)), default=0)
        por_capa: dict = defaultdict(list)
        for nodo, c in capa.items():
            por_capa[c].append(nodo)
        pos: dict = {}
        for c, nodos in por_capa.items():
            n = len(nodos)
            for i, nodo in enumerate(sorted(nodos, key=lambda s: (len(f"{s}"), f"{s}"))):
                pos[f"{nodo}"] = (2.4 * c * escala, ((n - 1) / 2.0 - i) * 1.7 * escala)
        return pos

    def plot_tikz_pasos(self, durations, beamer: bool = False,
                        escala: float = 1.0, posiciones: dict = None,
                        textos: dict = None, layout: str = "auto"):
        """Dibujo del grafo PERT paso a paso (código TikZ).

        Secuencia de fotogramas: grafo vacío; un fotograma por suceso en el
        recorrido hacia delante (se rellena su tiempo temprano y se resaltan
        las actividades que llegan); uno por suceso en el recorrido hacia
        atrás (tiempo tardío y actividades que salen); uno por actividad con el cálculo
        de su holgura total (``H = t*_j - t_i - d``; la actividad y sus sucesos en naranja
        y, si la holgura es 0, en rojo desde el fotograma siguiente); y un fotograma final con el camino crítico. Bajo el grafo se escribe el cálculo
        del paso (véase :meth:`pert_pasos`).

        Los nodos siguen el convenio de los apuntes: índice arriba, temprano a
        la izquierda y tardío a la derecha. El código necesita las bibliotecas
        TikZ ``calc`` y ``arrows.meta``
        (``TexFragment(c, tikz_libraries="calc,arrows.meta")``).

        Parameters
        ----------
        durations : dict | pandas.Series
            Duraciones de las actividades.
        layout : {"auto", "capas"}
            ``"auto"`` (por defecto): disposición propia por capas con pocos cruces,
            flechas que esquivan los nodos y nombres colocados donde no pisan nodos,
            flechas ni otros nombres (:func:`_disposicion_capas`); ``"capas"``:
            disposición simple por capas topológicas. Con ``posiciones`` se usa
            siempre la impuesta.
        beamer : bool
            ``False`` (por defecto): lista de cadenas, un ``tikzpicture`` por
            fotograma (útil con ``jupyter_tikz`` o para figuras sueltas).
            ``True``: una sola cadena con un ``tikzpicture`` que usa
            superposiciones ``\\only<k>`` de beamer; basta ponerlo dentro de un
            ``frame``.
        escala : float
            Factor de escala de las distancias entre nodos.
        posiciones : dict | None
            ``{suceso: (x, y)}`` en cm para imponer la disposición del grafo
            (por ejemplo, la de una figura de los apuntes). Por defecto se
            calcula por capas.
        textos : dict | None
            Rótulos de los fotogramas inicial y final:
            ``{"inicio": "Grafo del proyecto",
            "final": "Camino crítico: duración = {duracion}"}``
            (útil para versiones en otro idioma).

        Returns
        -------
        list[str] | str
        """
        dur = self._normalize_durations(durations, dummies=True).to_dict()
        pasos = self.pert_pasos(durations, formato="latex", colores=PALETA_FLECHAS)
        holguras = self.calculate_pert(durations)["activities"]["H_total"]
        trazado = None
        if posiciones is None and layout == "auto":
            trazado = self._trazado_pert(dur, escala)
        pos = trazado["pos"] if trazado else self._pert_posiciones(escala, posiciones)
        rotulos = {"inicio": "Grafo del proyecto",
                   "final": "Camino crítico: duración = {duracion}"}
        rotulos.update(textos or {})
        G = self.pert_graph
        f = self._num_paso

        def fotograma(*args, **kwargs):
            return self._pert_fotograma(pos, dur, holguras, escala, *args, trazado=trazado, **kwargs)

        fotogramas = []
        fotogramas.append(fotograma({}, {}, texto=rotulos["inicio"]))
        tempranos: dict = {}
        for _, fila in pasos["tempranos"].iterrows():
            nodo = fila["suceso"]
            tempranos[nodo] = fila["valor"]
            arcos = {a: PALETA_FLECHAS[k % len(PALETA_FLECHAS)] for k, a in enumerate(fila["arcos"])}
            valores = {(u, "e"): c for (u, _), c in arcos.items()}
            fotogramas.append(fotograma(dict(tempranos), {}, nodo, arcos,
                                        f"${fila['calculo']}$", colores_valores=valores,
                                        negrita=set(valores) | {(nodo, "e")},
                                        colores_region=self._region_dominante(fila, arcos, nodo, "e")))
        tardios: dict = {}
        for _, fila in pasos["tardios"].iterrows():
            nodo = fila["suceso"]
            tardios[nodo] = fila["valor"]
            arcos = {a: PALETA_FLECHAS[k % len(PALETA_FLECHAS)] for k, a in enumerate(fila["arcos"])}
            valores = {(v, "l"): c for (_, v), c in arcos.items()}
            fotogramas.append(fotograma(dict(tempranos), dict(tardios), nodo, arcos,
                                        f"${fila['calculo']}$", colores_valores=valores,
                                        negrita=set(valores) | {(nodo, "l")},
                                        colores_region=self._region_dominante(fila, arcos, nodo, "l")))
        criticas: set = set()
        for _, fila in pasos["holguras"].iterrows():
            act, u, v = fila["actividad"], f"{fila['origen']}", f"{fila['destino']}"
            # la actividad calculada y sus dos sucesos van en naranja; el rojo
            # (si es crítica) aparece en los fotogramas siguientes
            fotogramas.append(fotograma(dict(tempranos), dict(tardios), {u, v}, {(u, v): PALETA_FLECHAS[0]},
                                        f"${fila['calculo']}$"
                                        + (r"\\ \small\textcolor{red}{actividad crítica}" if fila["valor"] == 0 else ""),
                                        criticas=set(criticas),
                                        colores_valores={(u, "e"): PALETA_FLECHAS[0], (v, "l"): PALETA_FLECHAS[0]},
                                        negrita={(u, "e"), (v, "l")}))
            if fila["valor"] == 0:
                criticas.add(act)
        duracion = max(tempranos.values())
        fotogramas.append(fotograma(dict(tempranos), dict(tardios), critico=True,
                                    texto=rotulos["final"].format(duracion=f(duracion))))

        return self._beamer_o_lista(fotogramas, beamer=beamer)


    def plot_tikz_pert(self, durations=False, escala: float = 1.0, posiciones: dict = None,
                       layout: str = "auto") -> str:
        """Grafo PERT como un ``tikzpicture`` (sin graphviz): con ``durations`` lleva los
        tiempos tempranos y tardíos en los nodos, ``actividad (duración)`` en las flechas y
        el camino crítico en rojo (flechas y nodos); sin ellas, el grafo con los nombres
        (ficticias ``@i→j`` punteadas y en verde). Disposición propia por capas que evita
        que se pisen los elementos (véase :meth:`plot_tikz_pasos`). Necesita las
        bibliotecas TikZ ``calc`` y ``arrows.meta``."""
        con = durations is not False and durations is not None
        if con:
            dur = self._normalize_durations(durations, dummies=True).to_dict()
            res = self.calculate_pert(durations)
            tempranos = {f"{k}": v for k, v in res["nodes"]["early"].items()}
            tardios = {f"{k}": v for k, v in res["nodes"]["late"].items()}
            holguras = res["activities"]["H_total"]
        else:
            dur, tempranos, tardios, holguras = {}, {}, {}, {}
        trazado = None
        if posiciones is None and layout == "auto":
            trazado = self._trazado_pert(dur, escala, con)
        pos = trazado["pos"] if trazado else self._pert_posiciones(escala, posiciones)
        cuerpo = self._pert_fotograma(pos, dur, holguras, escala, tempranos, tardios,
                                      critico=con, trazado=trazado, con_duracion=con)
        return "\\begin{tikzpicture}[>=stealth]\n" + cuerpo + "\n\\end{tikzpicture}"

    def pert_tikz(self, filename=None, durations=False, escala: float = 1.0,
                  posiciones: dict = None, layout: str = "auto", **graphviz):
        """Dibuja el grafo PERT con TikZ y devuelve la imagen SVG (se muestra al ser la
        última expresión de la celda). Sustituye a :meth:`pert` en los cuadernos: acepta
        sus mismos argumentos (``durations``, ``filename``…) e ignora los de disposición
        de graphviz (``ranksep``, ``nodesep``, ``rowsep``…). Con ``filename`` guarda
        además el dibujo (``.pdf``, ``.svg`` o ``.png``). Hace falta ``pdflatex``."""
        codigo = self.plot_tikz_pert(durations, escala, posiciones, layout)
        return SVG(data=_svg_ancho_columna(_svgs_de_tikz([codigo], guardar=filename)[0]))

    def plot_tikz_roy(self, durations=False, escala: float = 1.0, posiciones: dict = None,
                      etiquetas: dict = None, layout: str = "auto") -> str:
        """Grafo Roy como un ``tikzpicture`` (sin graphviz): nodos al estilo del PMBOK,
        ``ES | d | EF`` arriba, el nombre en la banda central y ``LS | H | LF`` abajo; con
        ``durations`` el camino crítico en rojo. Sin ellas, los nodos vacíos. Véase
        :meth:`plot_tikz_roy_pasos`."""
        con = durations is not False and durations is not None
        fotos = self.plot_tikz_roy_pasos(durations if con else {}, beamer=False, escala=escala,
                                         posiciones=posiciones, etiquetas=etiquetas,
                                         layout=layout, estatico=True, con_duracion=con)
        return fotos[0]

    def roy_tikz(self, filename=None, durations=False, escala: float = 1.0,
                 posiciones: dict = None, layout: str = "auto", etiquetas: dict = None, **graphviz):
        """Dibuja el grafo Roy con TikZ y devuelve la imagen SVG; sustituye a :meth:`roy`
        en los cuadernos (acepta sus argumentos e ignora los de graphviz)."""
        codigo = self.plot_tikz_roy(durations, escala, posiciones, etiquetas, layout)
        return SVG(data=_svg_ancho_columna(_svgs_de_tikz([codigo], guardar=filename)[0]))

    # ------------------------------------------------------------------
    # Piezas comunes de los dibujos paso a paso
    # ------------------------------------------------------------------

    @staticmethod
    def _tex_texto(texto: str) -> str:
        """Texto de una etiqueta a LaTeX (flecha de las ficticias ``@i→j`` y caracteres especiales)."""
        for a, b in (("\\", r"\textbackslash{}"), ("$", r"\$"), ("_", r"\_"), ("&", r"\&"),
                     ("%", r"\%"), ("#", r"\#"), ("→", r"$\to$"), ("💤", r"\textit{zz}")):
            texto = texto.replace(a, b)
        # lo que pdflatex no sabe escribir (emoji, etc.) no debe romper la compilación
        return "".join(c if ord(c) < 0x2000 else "?" for c in texto)

    @staticmethod
    def _ruta_tikz(a: str, b: str, pos: dict, intermedios: list, curva: bool, nombre) -> str:
        """Camino TikZ de una arista del trazado de :func:`_disposicion_capas`: recta o
        curvas de Bézier con tangente horizontal por los puntos intermedios.
        ``nombre(id)`` da el nombre del nodo TikZ; TikZ recorta en su borde."""
        A, B = f"({nombre(a)})", f"({nombre(b)})"
        if not intermedios and not curva:
            return f"{A} -- {B}"
        pts = [pos[a]] + list(intermedios) + [pos[b]]
        t = A
        for i in range(len(pts) - 1):
            (x1, y1), (x2, y2) = pts[i], pts[i + 1]
            dx = (x2 - x1) * 0.45
            fin = B if i == len(pts) - 2 else f"({x2:.2f}cm,{y2:.2f}cm)"
            t += (f" .. controls ({x1 + dx:.2f}cm,{y1:.2f}cm) and ({x2 - dx:.2f}cm,{y2:.2f}cm) .. {fin}")
        return t

    def _trazado_pert(self, dur: dict, escala: float = 1.0, con_duracion: bool = True):
        """Disposición propia (sin graphviz) del grafo PERT con las etiquetas
        ``actividad (duración)``; las ficticias llevan ``@i→j (0)`` como en ``pert()``."""
        f = self._num_paso
        nodos = {f"{n}": (1.1, 1.1) for n in self.nodes}
        aristas, anchos = [], {}
        for u, v, d in self.pert_graph.edges(data=True):
            clave = (f"{u}", f"{v}")
            aristas.append(clave)
            act = d.get("activity", "")
            anchos[clave] = _ancho_etiqueta(f"{act} ({f(dur.get(act, 0))})" if con_duracion else act)
        tr = _disposicion_capas(nodos, aristas, anchos, separacion=0.6 * escala, hueco=0.8 * escala)
        return tr

    @staticmethod
    def _curvatura_tikz(pos, u, v, umbral):
        """Si el segmento u→v pasa cerca de otro nodo, devuelve la curva TikZ
        que lo esquiva (hacia el lado contrario); si no, cadena vacía."""
        (x1, y1), (x2, y2) = pos[f"{u}"], pos[f"{v}"]
        dx, dy = x2 - x1, y2 - y1
        largo = (dx * dx + dy * dy) ** 0.5 or 1.0
        peor = None
        for nodo, (x, y) in pos.items():
            if nodo in (f"{u}", f"{v}"):
                continue
            t = ((x - x1) * dx + (y - y1) * dy) / (largo * largo)
            if not 0.0 < t < 1.0:
                continue
            cruz = (dx * (y - y1) - dy * (x - x1)) / largo
            if abs(cruz) < umbral and (peor is None or abs(cruz) < abs(peor)):
                peor = cruz
        if peor is None:
            return ""
        return "bend right=25" if peor >= 0 else "bend left=25"

    @staticmethod
    def _region_dominante(fila, arcos, nodo, clave):
        """Color de los radios (o segmentos) del resultado de un cálculo con varios
        términos: el de la flecha que da el máximo (o el mínimo). Con empate, o con un
        solo término, no se fija (queda el color del nodo del resultado)."""
        if len(arcos) < 2:
            return None
        dom = [a for a, (_, t) in zip(list(arcos), fila["terminos"]) if t == fila["valor"]]
        return {(nodo, clave): arcos[dom[0]]} if len(dom) == 1 else None

    def _pert_fotograma(self, pos, dur, holguras, escala, tempranos, tardios,
                        actual=None, arcos_activos=(), texto="", critico=False,
                        trazado=None, con_duracion=True, criticas=None, colores_valores=None,
                        negrita=None, colores_region=None,
                        estilo_arcos=None, estilo_nodos=None, numeros=True, color_actual="orange"):
        """Un fotograma TikZ del grafo PERT (cuerpo, sin ``tikzpicture``).

        ``arcos_activos`` puede ser un ``dict`` ``{(i, j): color}`` (un color por arco) y
        ``colores_valores`` un ``dict`` ``{(suceso, "e" | "l"): color}`` que tiñe el tiempo
        temprano o tardío de ese suceso con el color del arco que lo usa en el cálculo;
        ``negrita`` es un conjunto de esas mismas claves cuyo número va en negrita;
        ``estilo_arcos`` y ``estilo_nodos`` (``{(i, j): estado}`` y ``{suceso: estado}``,
        estados ``"hecho"`` negro, ``"actual"`` naranja y grueso, ``"gris"`` gris claro)
        fijan el aspecto de cada arco y cada suceso (visor de construcción del grafo);
        ``colores_region`` da el color de los radios del quesito resaltado (por defecto el
        de ``colores_valores``). En un
        paso de cálculo los arcos de ``arcos_activos`` se dibujan más gruesos y con la
        etiqueta en negrita, y lo que no participa (ni es crítico) va en gris.

        ``criticas``: conjunto de actividades ya reconocidas como críticas (holgura
        total 0); si se da, sustituye a ``critico`` y los nodos extremos de esas
        actividades se pintan también de rojo.

        ``actual`` es un suceso o un conjunto de sucesos que se resaltan en
        naranja; ``arcos_activos``, los arcos ``(i, j)`` resaltados."""
        G = self.pert_graph
        f = self._num_paso
        if isinstance(actual, (set, frozenset, list, tuple)):
            actuales = {f"{a}" for a in actual}
        else:
            actuales = {f"{actual}"} if actual is not None else set()
        def tinta(nodo, clave, texto):
            if (nodo, clave) in (negrita or ()):
                texto = rf"\textbf{{{texto}}}"
            c = (colores_valores or {}).get((nodo, clave))
            return rf"\textcolor{{{c}}}{{{texto}}}" if c else texto

        GRIS = "gray!55"
        CL = "gray!40"
        COLOR_PREVIA = "blue!70!black"
        atenuar = bool(arcos_activos) or bool(actuales)
        col_nodo = ({x: c for (u, v), c in arcos_activos.items() for x in (u, v)
                     if x not in actuales} if isinstance(arcos_activos, dict) else {})
        implicados = actuales | {x for a in arcos_activos for x in a}

        y_min = min(y for _, y in pos.values())
        if trazado:
            y_min = min([y_min] + [y for _, y in trazado["etiquetas"].values()]) - 0.2
            for r in trazado["rutas"].values():
                y_min = min([y_min] + [y for _, y in r])
        x_mid = (min(x for x, _ in pos.values()) + max(x for x, _ in pos.values())) / 2
        L, A = [], []
        for (u, v, data) in G.edges(data=True):
            act = data.get("activity", "")
            ficticia = (not act) or act.startswith("@")
            estilo = ["->", "thick"]
            if ficticia:
                estilo.append("dashed")
            color = ""
            col_act = (arcos_activos.get((f"{u}", f"{v}")) if isinstance(arcos_activos, dict)
                       else (f"{color_actual}!80!black" if (f"{u}", f"{v}") in arcos_activos else None))
            st = (estilo_arcos or {}).get((f"{u}", f"{v}"))
            if st == "oculto":
                estilo.append("white")
                color = "white"
            elif st == "gris":
                estilo.append(CL)
                color = CL
            elif st == "hecho":
                pass
            elif st == "previa":
                estilo += [COLOR_PREVIA, "ultra thick"]
                color = col_act = COLOR_PREVIA
            elif st == "actual":
                estilo += [f"{color_actual}!80!black", "ultra thick"]
                color = col_act = f"{color_actual}!80!black"
            elif col_act:
                estilo += [col_act, "ultra thick"]
                color = col_act
            elif (act in criticas if criticas is not None
                  else critico and holguras.get(act, 1) == 0):
                estilo += ["red", "very thick"]
                color = "red"
            elif ficticia and not con_duracion:
                estilo.append("green!50!black")
                color = "green!50!black"
            elif atenuar:
                estilo.append(GRIS)
                color = GRIS
            etq = self._tex_texto(act) + (f" ({f(dur.get(act, 0))})" if con_duracion else "")
            if trazado and (f"{u}", f"{v}") in trazado["rutas"]:
                clave = (f"{u}", f"{v}")
                A.append(rf"  \draw[{', '.join(estilo)}] "
                         + self._ruta_tikz(clave[0], clave[1], pos, trazado["rutas"][clave],
                                           trazado["curva"][clave], lambda n: f"n{n}") + ";")
                if clave in trazado["etiquetas"]:
                    lx, ly = trazado["etiquetas"][clave]
                    A.append(rf"  \node[font=\scriptsize{r'\bfseries' if col_act else ''}, inner sep=1pt, fill=white{', text=' + color if color else ''}]"
                             rf" at ({lx:.2f}cm, {ly:.2f}cm) {{{etq}}};")
                continue
            etiqueta = etq
            curva = self._curvatura_tikz(pos, u, v, 0.75 * escala)
            trazo = f"to[{curva}]" if curva else "--"
            lado = "below" if curva.startswith("bend right") else "above"
            A.append(rf"  \draw[{', '.join(estilo)}] (n{u}) {trazo}"
                     rf" node[midway, {lado}, sloped, font=\scriptsize{r'\bfseries' if col_act else ''}{', text=' + color if color else ''}] {{{etiqueta}}} (n{v});")
        for nodo, (x, y) in pos.items():
            if criticas is not None:
                es_critico = any(G.edges[e].get("activity", "") in criticas
                                 for e in list(G.in_edges(nodo)) + list(G.out_edges(nodo)))
            else:
                es_critico = (critico and nodo in tempranos and nodo in tardios
                              and tempranos[nodo] == tardios[nodo])
            est = (estilo_nodos or {}).get(nodo)
            if est == "oculto":
                relleno = "fill=white, draw=white, thick"
            elif est == "gris":
                relleno = f"fill=white, draw={CL}, thick"
            elif est == "hecho":
                relleno = "fill=white, draw, thick"
            elif est == "actual" or nodo in actuales:
                relleno = f"fill={color_actual}!20, draw={color_actual}!80!black, very thick"
            elif es_critico:
                relleno = "fill=white!90!red, draw=red, very thick"
            elif nodo in col_nodo:
                relleno = f"fill=white, draw={col_nodo[nodo]}, ultra thick"
            elif atenuar and nodo not in implicados:
                relleno = f"fill=white, draw={GRIS}, thick"
            else:
                relleno = "fill=white, draw, thick"
            gris = (", text=white" if est == "oculto" else
                    f", text={CL}" if est == "gris" else
                    f", text={GRIS}" if (est is None and atenuar and nodo not in implicados
                                         and not es_critico) else "")
            L.append(rf"  \node[circle, {relleno}, minimum size=1.1cm, inner sep=0pt]"
                     rf" (n{nodo}) at ({x:.2f}cm, {y:.2f}cm) {{}};")
            L.append(rf"  \draw[{'white' if est == 'oculto' else CL if est == 'gris' else 'gray'}] (n{nodo}.center) -- (n{nodo}.45)"
                     rf" (n{nodo}.center) -- (n{nodo}.135) (n{nodo}.center) -- (n{nodo}.270);")
            # «quesito» del valor que interviene: sus dos radios, más gruesos y del color de la flecha
            for clave, (ini, fin) in (("e", (135, 270)), ("l", (270, 45))):
                if (nodo, clave) in (negrita or ()):
                    c = (colores_region or {}).get((nodo, clave)) or (colores_valores or {}).get(
                        (nodo, clave), f"{color_actual}!80!black" if nodo in actuales else "black")
                    L.append(rf"  \draw[{c}, ultra thick] (n{nodo}.{ini}) -- (n{nodo}.center)"
                             rf" -- (n{nodo}.{fin});")
            if numeros:
                L.append(rf"  \node[font=\footnotesize\bfseries{gris}]"
                         rf" at ($(n{nodo}.center)+(90:0.28)$) {{{self._tex_texto(f"{nodo}")}}};")
            if nodo in tempranos:
                L.append(rf"  \node[font=\scriptsize{r'\bfseries' if es_critico else ''}{gris}] at ($(n{nodo}.center)+(210:0.27)$) {{{tinta(nodo, 'e', f(tempranos[nodo]))}}};")
            if nodo in tardios:
                L.append(rf"  \node[font=\scriptsize{r'\bfseries' if es_critico else ''}{gris}] at ($(n{nodo}.center)+(330:0.27)$) {{{tinta(nodo, 'l', f(tardios[nodo]))}}};")
        L += A
        if texto:
            L.append(rf"  \node[anchor=north, font=\small, align=center] at ({x_mid:.2f}cm, {y_min - 1.2:.2f}cm) {{{texto}}};")
        return "\n".join(L)

    @staticmethod
    def _beamer_o_lista(fotogramas, cabecera=r"\begin{tikzpicture}[>=stealth]",
                        beamer=False):
        if not beamer:
            return [f"{cabecera}\n{c}\n\\end{{tikzpicture}}" for c in fotogramas]
        cuerpo = [cabecera]
        for k, c in enumerate(fotogramas, start=1):
            cuerpo.append(rf"\only<{k}>{{" + "\n" + c + "\n}")
        cuerpo.append(r"\end{tikzpicture}")
        return "\n".join(cuerpo)

    def _pert_posiciones(self, escala, posiciones):
        if posiciones is None:
            return self._pert_layout_capas(escala)
        pos = {f"{k}": (float(x) * escala, float(y) * escala)
               for k, (x, y) in posiciones.items()}
        faltan = set(self.nodes) - set(pos)
        if faltan:
            raise ValueError(f"Faltan posiciones para los sucesos {sorted(faltan)}")
        return pos

    # ------------------------------------------------------------------
    # Holguras del PERT paso a paso (docencia)
    # ------------------------------------------------------------------

    def holguras_pasos(self, durations, formato: str = "texto",
                       ficticias: bool = False) -> pd.DataFrame:
        """Cálculo paso a paso de las holguras total, libre e independiente.

        Una fila por actividad (orden alfabético; las ficticias solo si
        ``ficticias=True``) con los sucesos ``i`` y ``j``, la duración
        ``t_ij``, los tiempos ``t_i``, ``t_j``, ``t*_i``, ``t*_j``, las
        holguras ``HT = t*_j - t_i - t_ij``, ``HL = t_j - t_i - t_ij`` e
        ``HI = t_j - t*_i - t_ij`` (que puede ser negativa) y el cálculo
        escrito de cada una (columnas ``calculo_HT``, ``calculo_HL`` y
        ``calculo_HI``; texto o LaTeX según ``formato``).
        """
        if formato not in ("texto", "latex"):
            raise ValueError("formato debe ser 'texto' o 'latex'")
        dur = self._normalize_durations(durations, dummies=True).to_dict()
        nodos = self.calculate_pert(durations)["nodes"]
        e, l = nodos["early"], nodos["late"]
        f = self._num_paso
        latex = formato == "latex"
        largos = any(len(f"{n}") > 1 for n in self.nodes)
        filas = []
        for act, (i, j) in sorted(self.nodes_from_edges.items(), key=lambda x: x[0].lstrip("@")):
            if act.startswith("@") and not ficticias:
                continue
            d = dur.get(act, 0)
            ti, tj, li, lj = e[f"{i}"], e[f"{j}"], l[f"{i}"], l[f"{j}"]
            HT, HL, HI = lj - ti - d, tj - ti - d, tj - li - d
            if latex:
                tij = f"t_{{{i},{j}}}" if largos else f"t_{{{i}{j}}}"
                cT = rf"H^{{T}}=t_{{{j}}}^{{*}}-t_{{{i}}}-{tij}={f(lj)}-{f(ti)}-{f(d)}={f(HT)}"
                cL = rf"H^{{L}}=t_{{{j}}}-t_{{{i}}}-{tij}={f(tj)}-{f(ti)}-{f(d)}={f(HL)}"
                cI = rf"H^{{I}}=t_{{{j}}}-t_{{{i}}}^{{*}}-{tij}={f(tj)}-{f(li)}-{f(d)}={f(HI)}"
            else:
                cT = f"HT = t{j}* - t{i} - t{i}{j} = {f(lj)} - {f(ti)} - {f(d)} = {f(HT)}"
                cL = f"HL = t{j} - t{i} - t{i}{j} = {f(tj)} - {f(ti)} - {f(d)} = {f(HL)}"
                cI = f"HI = t{j} - t{i}* - t{i}{j} = {f(tj)} - {f(li)} - {f(d)} = {f(HI)}"
            filas.append(dict(actividad=act, i=f"{i}", j=f"{j}", duracion=d,
                              t_i=ti, t_j=tj, t_i_tardio=li, t_j_tardio=lj,
                              HT=HT, HL=HL, HI=HI,
                              calculo_HT=cT, calculo_HL=cL, calculo_HI=cI))
        return pd.DataFrame(filas).set_index("actividad")

    def plot_tikz_holguras_pasos(self, durations, beamer: bool = False,
                                 escala: float = 1.0, posiciones: dict = None,
                                 textos: dict = None):
        """Dibujo TikZ del cálculo de las holguras paso a paso.

        Primer fotograma: grafo con los tiempos tempranos y tardíos. Después,
        un fotograma por actividad real con su arco y sus dos sucesos en
        naranja y, debajo, las tres holguras calculadas (véase
        :meth:`holguras_pasos`). Último: actividades críticas en rojo.
        Mismos convenios y parámetros que :meth:`plot_tikz_pasos`;
        ``textos`` admite las claves ``inicio``, ``final`` y ``actividad``.
        """
        dur = self._normalize_durations(durations, dummies=True).to_dict()
        res = self.calculate_pert(durations)
        holguras = res["activities"]["H_total"]
        tempranos = {f"{k}": v for k, v in res["nodes"]["early"].items()}
        tardios = {f"{k}": v for k, v in res["nodes"]["late"].items()}
        pos = self._pert_posiciones(escala, posiciones)
        rotulos = {"inicio": "Grafo con los tiempos tempranos y tardíos",
                   "final": r"Actividades críticas: $H^{T}=0$",
                   "actividad": "Actividad"}
        rotulos.update(textos or {})
        pasos = self.holguras_pasos(durations, formato="latex")

        def foto(**k):
            return self._pert_fotograma(pos, dur, holguras, escala,
                                        tempranos, tardios, **k)

        fotogramas = [foto(texto=rotulos["inicio"])]
        for act, fila in pasos.iterrows():
            texto = (rf"{rotulos['actividad']} {act} ({fila['i']}--{fila['j']})\\"
                     rf"${fila['calculo_HT']}$\quad ${fila['calculo_HL']}$\quad"
                     rf" ${fila['calculo_HI']}$")
            fotogramas.append(foto(actual={fila["i"], fila["j"]},
                                   arcos_activos={(fila["i"], fila["j"])},
                                   texto=texto))
        fotogramas.append(foto(critico=True, texto=rotulos["final"]))
        return self._beamer_o_lista(fotogramas, beamer=beamer)

    # ------------------------------------------------------------------
    # Algoritmo de Zaderenko paso a paso (docencia)
    # ------------------------------------------------------------------

    def _zaderenko_fotogramas(self, durations, formato: str = "latex") -> list:
        """Estado de la matriz de Zaderenko en cada paso (base de las salidas
        LaTeX y HTML). Cada fotograma es un dict con ``filas`` (lista de
        ``(t_i, i, celdas)``), ``ultima`` (celdas de la fila de tardíos) y
        ``calculo`` (``(tipo, suceso, texto)`` o ``None``); cada celda es una
        terna ``(texto, color, sombreada)`` con color ``None``, ``'azul'`` o
        ``'rojo'``."""
        dur = self._normalize_durations(durations, dummies=True).to_dict()
        res = self.calculate_pert(durations)
        early = {f"{k}": v for k, v in res["nodes"]["early"].items()}
        late = {f"{k}": v for k, v in res["nodes"]["late"].items()}
        H = res["activities"]["H_total"]
        pasos = self.pert_pasos(durations, formato=formato)
        N = [f"{n}" for n in sorted(self.nodes, key=int)]
        celda = {(f"{i}", f"{j}"): (act, dur.get(act, 0))
                 for act, (i, j) in self.nodes_from_edges.items()}
        f = self._num_paso

        def foto(te, tl, modo=None, nodo=None, calculo=None):
            filas = []
            for i in N:
                ti, col = (f(te[i]) if i in te else ""), None
                if ti and modo == "del" and i == nodo:
                    col = "rojo"
                elif ti and modo == "del" and (i, nodo) in celda:
                    col = "azul"
                elif ti and modo == "fin" and early[i] == late[i]:
                    col = "rojo"
                celdas = []
                for j in N:
                    sombra = (modo == "del" and j == nodo) or (modo == "atr" and i == nodo)
                    c, cc = "", None
                    if i == j:
                        c = "×"
                    elif (i, j) in celda:
                        act, d = celda[(i, j)]
                        c = f(d)
                        if sombra:
                            cc = "azul"
                        elif modo == "fin" and H.get(act, 1) == 0:
                            cc = "rojo"
                    celdas.append((c, cc, sombra))
                filas.append(((ti, col, False), i, celdas))
            ultima = []
            for j in N:
                c, cc = (f(tl[j]) if j in tl else ""), None
                if c and modo == "atr" and j == nodo:
                    cc = "rojo"
                elif c and modo == "atr" and (nodo, j) in celda:
                    cc = "azul"
                elif c and modo == "fin" and early[j] == late[j]:
                    cc = "rojo"
                ultima.append((c, cc, False))
            return dict(nodos=N, filas=filas, ultima=ultima, calculo=calculo, modo=modo)

        fotos = [foto({}, {}, calculo=("inicio", None, ""))]
        te, tl = {}, {}
        del_ = {f"{r['suceso']}": r for _, r in pasos["tempranos"].iterrows()}
        atr_ = {f"{r['suceso']}": r for _, r in pasos["tardios"].iterrows()}
        # Por columnas (filas) en orden numérico: con los sucesos numerados de
        # menor a mayor es un orden topológico válido.
        for j in N:
            te[j] = del_[j]["valor"]
            fotos.append(foto(dict(te), {}, "del", j, ("columna", j, del_[j]["calculo"])))
        for i in reversed(N):
            tl[i] = atr_[i]["valor"]
            fotos.append(foto(dict(te), dict(tl), "atr", i, ("fila", i, atr_[i]["calculo"])))
        fotos.append(foto(dict(te), dict(tl), "fin", calculo=("final", None, "")))
        return fotos

    @staticmethod
    def _rotulos_zaderenko(textos):
        rot = {"inicio": r"Matriz de duraciones $t_{ij}$", "final": "Camino crítico",
               "columna": "Columna", "fila": "Fila"}
        rot.update(textos or {})
        return rot

    def zaderenko_pasos(self, durations, textos: dict = None) -> list:
        """Matriz de Zaderenko paso a paso, como cuadros LaTeX coloreados.

        Fotogramas: matriz con las duraciones; uno por suceso hacia delante
        (columna ``j`` sombreada, en azul los tiempos ``t_i`` de las filas
        con duración en esa columna y esas duraciones, en rojo el ``t_j``
        resultante); uno por suceso hacia atrás (fila ``i`` sombreada, en
        azul las duraciones y los ``t*_j`` de sus columnas, en rojo el
        ``t*_i``); y uno final con las actividades y sucesos críticos en
        rojo. Bajo cada cuadro, el cálculo del paso.

        Necesita los paquetes ``colortbl`` y ``xcolor``. Devuelve la lista de
        fotogramas; :meth:`zaderenko_pasos_beamer` los une con ``\\only<k>``
        y :meth:`display_zaderenko_pasos` los muestra en un cuaderno.
        ``textos`` admite ``inicio``, ``final``, ``columna`` y ``fila``.
        """
        rot = self._rotulos_zaderenko(textos)
        color = {"azul": "blue!70!black", "rojo": "red!75!black"}

        def tex(c):
            texto, col, sombra = c
            if texto == "×":
                texto = r"$\times$"
            if col and texto:
                texto = rf"\textbf{{\textcolor{{{color[col]}}}{{{texto}}}}}"
            return (r"\cellcolor{orange!18}" if sombra else "") + texto

        salida = []
        for fo in self._zaderenko_fotogramas(durations, formato="latex"):
            N = fo["nodos"]
            L = [r"\begin{tabular}{c|c|" + "c" * len(N) + "}",
                 r"$t_i\!\downarrow$ & $i\backslash j$ & " + " & ".join(N) + r"\\ \hline"]
            for ti, i, celdas in fo["filas"]:
                L.append(" & ".join([tex(ti), i] + [tex(c) for c in celdas]) + r"\\")
            L.append(r"\hline")
            L.append(r" & $t_j^*\!\leftarrow$ & " + " & ".join(tex(c) for c in fo["ultima"]) + r"\\")
            L.append(r"\end{tabular}")
            tipo, nodo, calc = fo["calculo"]
            texto = {"inicio": rot["inicio"], "final": rot["final"]}.get(
                tipo, f"{rot.get(tipo, tipo)} {nodo}: ${calc}$")
            if texto:
                L.append(r"\par\smallskip " + texto)
            salida.append("\n".join(L))
        return salida

    def zaderenko_pasos_html(self, durations, textos: dict = None) -> str:
        """La matriz de Zaderenko paso a paso como HTML autocontenido: los
        mismos fotogramas que :meth:`zaderenko_pasos` (mismos colores) con
        botones para avanzar y retroceder. El primer fotograma (paso 0) es el
        resultado final; a partir del paso 1 se construye desde el principio."""
        rot = self._rotulos_zaderenko(textos)
        rot["inicio"] = rot["inicio"].replace(r"$t_{ij}$", "t<sub>ij</sub>")
        css = {"azul": "color:#1f3c99;font-weight:bold", "rojo": "color:#b01c1c;font-weight:bold"}

        def td(c, extra=""):
            texto, col, sombra = c
            estilo = [css[col]] if (col and texto) else []
            if sombra:
                estilo.append("background:#fde3cc")
            return f'<td style="{";".join(estilo)}{extra}">{texto}</td>'

        fotos_html = []
        fotogramas = self._zaderenko_fotogramas(durations, formato="texto")
        for fo in [fotogramas[-1]] + fotogramas:        # primero el resultado final (paso 0)
            N = fo["nodos"]
            sep = ";border-right:1px solid #444"
            filas = ['<tr><th style="border-bottom:1px solid #444' + sep + '">t<sub>i</sub>↓</th>'
                     '<th style="border-bottom:1px solid #444' + sep + '">i \\ j</th>'
                     + "".join(f'<th style="border-bottom:1px solid #444">{j}</th>' for j in N) + "</tr>"]
            for ti, i, celdas in fo["filas"]:
                filas.append("<tr>" + td(ti, sep) + f'<td style="font-weight:bold{sep}">{i}</td>'
                             + "".join(td(c) for c in celdas) + "</tr>")
            filas.append('<tr style="border-top:1px solid #444"><td style="border-top:1px solid #444' + sep + '"></td>'
                         '<td style="border-top:1px solid #444' + sep + '">t*<sub>j</sub>←</td>'
                         + "".join(td(c, ";border-top:1px solid #444") for c in fo["ultima"]) + "</tr>")
            tipo, nodo, calc = fo["calculo"]
            texto = {"inicio": rot["inicio"], "final": rot["final"]}.get(
                tipo, f"{rot.get(tipo, tipo)} {nodo}: {calc}")
            fotos_html.append('<table class="pmule-zad" style="border-collapse:collapse;text-align:center;'
                              'margin:0 auto">' + "".join(filas) + "</table>"
                              f'<p style="text-align:center;margin-top:.5em">{texto}</p>')
        return _html_pasos(fotos_html, inicio=0)

    def display_zaderenko_pasos(self, durations, textos: dict = None) -> None:
        """Muestra en un cuaderno la matriz de Zaderenko paso a paso (HTML con botones)."""
        display(HTML(self.zaderenko_pasos_html(durations, textos)))

    def zaderenko_pasos_beamer(self, durations, textos: dict = None) -> str:
        """Fotogramas de :meth:`zaderenko_pasos` unidos con ``\\only<k>``."""
        fotos = self.zaderenko_pasos(durations, textos=textos)
        return "\n".join(rf"\only<{k}>{{%" + "\n" + c + "\n}%"
                         for k, c in enumerate(fotos, start=1))

    # ------------------------------------------------------------------
    # Zaderenko con TikZ: PERT con tiempos arriba y matriz debajo, a la vez
    # ------------------------------------------------------------------

    def _cuerpo_zaderenko_tikz(self, N, celdas, early, late, te, tl, modo=None, nodo=None,
                               arcos=None, res_color=None, holguras=None):
        """Matriz de Zaderenko como código TikZ (sin ``tikzpicture``), con la columna
        ``j`` o la fila ``i`` del paso sombreadas y, con los colores de las flechas del
        PERT, las duraciones y tiempos que intervienen. Origen en la esquina superior
        izquierda de la celda de cabecera; unidades ``x`` e ``y`` de la escala vigente."""
        f = self._num_paso
        n = len(N)
        idx = {j: k + 1 for k, j in enumerate(N)}
        arcos = arcos or {}
        NARANJA = "orange!80!black"
        ROJO = "red!75!black"
        L = []
        if modo == "del":
            c = idx[nodo]
            L.append(rf"\fill[orange!18] ({c - .5},.5) rectangle ({c + .5},{-(n + 1) - .5});")
        elif modo == "atr":
            r = idx[nodo]
            L.append(rf"\fill[orange!18] (-1.5,{-r + .5}) rectangle ({n + .5},{-r - .5});")
        L.append(rf"\draw[gray!70] (-1.5,{-.5}) -- ({n + .5},{-.5});")
        L.append(rf"\draw[gray!70] (-1.5,{-n - .5}) -- ({n + .5},{-n - .5});")
        L.append(rf"\draw[gray!70] (-.5,.5) -- (-.5,{-(n + 1) - .5}) (.5,.5) -- (.5,{-(n + 1) - .5});")

        def txt(x, y, t, col=None, neg=False, caja=False):
            if col:
                t = rf"\textcolor{{{col}}}{{{t}}}"
            if neg or col:
                t = rf"\textbf{{{t}}}"
            marco = rf"[draw={col}, rounded corners=1.5pt, inner sep=2pt, line width=.7pt]" if (caja and col) else ""
            L.append(rf"\node{marco} at ({x},{-y}) {{{t}}};")

        txt(-1, 0, r"$t_i\!\downarrow$")
        txt(0, 0, r"$i\backslash j$")
        for j in N:
            txt(idx[j], 0, j)
        txt(n + 1.55, n + 1, r"$\leftarrow t_j^*$")
        for i in N:
            ri = idx[i]
            txt(0, ri, i, neg=True)
            if i in te:
                col, caja_i = None, False
                if modo == "del" and i == nodo:
                    col = res_color or NARANJA
                elif modo == "del" and (i, nodo) in arcos:
                    col, caja_i = arcos[(i, nodo)], True
                txt(-1, ri, f(te[i]), col, caja=caja_i)
            for j in N:
                if i == j:
                    txt(idx[j], ri, r"$\times$")
                elif (i, j) in celdas:
                    act, d = celdas[(i, j)]
                    col = arcos.get((i, j))
                    txt(idx[j], ri, f(d), col, caja=col is not None)
        for j in N:
            if j in tl:
                col, caja_j = None, False
                if modo == "atr" and j == nodo:
                    col = res_color or NARANJA
                elif modo == "atr" and (nodo, j) in arcos:
                    col, caja_j = arcos[(nodo, j)], True
                txt(idx[j], n + 1, f(tl[j]), col, caja=caja_j)
        return L

    def plot_tikz_zaderenko(self, durations, escala: float = 1.0, textos: dict = None) -> list:
        """Fotogramas TikZ (uno por ``tikzpicture``) del cálculo de tiempos con el
        algoritmo de Zaderenko: arriba el grafo PERT, que va rellenando los tiempos
        tempranos y tardíos como en :meth:`plot_tikz_pasos`, y debajo la matriz de
        Zaderenko dibujada con TikZ, que lo hace a la vez. Las duraciones y los tiempos
        que intervienen en cada cálculo llevan en la matriz el color de la flecha
        correspondiente del grafo; la columna (hacia delante) o la fila (hacia atrás)
        del suceso se sombrea y el resultado va en el color de la flecha dominante (o
        naranja). Fotogramas: grafo y matriz con las duraciones; uno por suceso hacia
        delante; uno por suceso hacia atrás; y el resultado final, sin resaltes (no se
        marcan en rojo las actividades críticas).
        ``textos`` admite ``inicio`` y ``final`` (con ``{duracion}``)."""
        dur = self._normalize_durations(durations, dummies=True).to_dict()
        res = self.calculate_pert(durations)
        early = {f"{k}": v for k, v in res["nodes"]["early"].items()}
        late = {f"{k}": v for k, v in res["nodes"]["late"].items()}
        H = res["activities"]["H_total"]
        pasos = self.pert_pasos(durations, formato="latex", colores=PALETA_FLECHAS)
        trazado = self._trazado_pert(dur, escala)
        pos = trazado["pos"] if trazado else self._pert_posiciones(escala, None)
        rotulos = {"inicio": "Grafo del proyecto y matriz de duraciones $t_{ij}$",
                   "final": "Duración del proyecto = {duracion}"}
        rotulos.update(textos or {})
        N = [f"{n}" for n in sorted(self.nodes, key=int)]
        celdas = {(f"{i}", f"{j}"): (act, dur.get(act, 0))
                  for act, (i, j) in self.nodes_from_edges.items()}
        f = self._num_paso

        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        if trazado:
            for x, y in trazado["etiquetas"].values():
                xs.append(x); ys.append(y)
            for r in trazado["rutas"].values():
                for x, y in r:
                    xs.append(x); ys.append(y)
        xmin, xmax, ymin = min(xs) - 0.7, max(xs) + 0.7, min(ys) - 0.9
        n = len(N)
        ux, uy = 0.8, 0.6
        Wm, Hm = (n + 3.6) * ux, (n + 3) * uy
        x0 = (xmin + xmax) / 2 - Wm / 2 + 1.5 * ux          # centro de la columna -1 en x0 - ux
        y0 = ymin - 0.6
        y_texto = y0 - Hm - 0.3

        def fotograma(te, tl, texto, modo=None, nodo=None, arcos=None, pert_kwargs=None, res_color=None):
            cuerpo = self._pert_fotograma(pos, dur, H, escala, te, tl, trazado=trazado,
                                          **(pert_kwargs or {}))
            M = self._cuerpo_zaderenko_tikz(N, celdas, early, late, te, tl, modo, nodo, arcos,
                                            res_color, H)
            L = [r"\begin{tikzpicture}[>=stealth]", cuerpo,
                 rf"\begin{{scope}}[xshift={x0:.2f}cm,yshift={y0:.2f}cm,x={ux}cm,y={uy}cm,font=\footnotesize]"]
            L += M
            L += [r"\end{scope}",
                  rf"\node[anchor=north, font=\small, align=center] at ({(xmin + xmax) / 2:.2f}cm, {y_texto:.2f}cm) {{{texto}}};",
                  r"\end{tikzpicture}"]
            return "\n".join(L)

        fotogramas = [fotograma({}, {}, rotulos["inicio"])]
        te: dict = {}
        for _, fila in pasos["tempranos"].iterrows():
            nodo = fila["suceso"]
            te[nodo] = fila["valor"]
            arcos = {a: PALETA_FLECHAS[k % len(PALETA_FLECHAS)] for k, a in enumerate(fila["arcos"])}
            valores = {(u, "e"): c for (u, _), c in arcos.items()}
            reg = self._region_dominante(fila, arcos, nodo, "e")
            fotogramas.append(fotograma(
                dict(te), {}, f"Columna {nodo}: ${fila['calculo']}$", "del", nodo, arcos,
                pert_kwargs=dict(actual=nodo, arcos_activos=arcos, texto="", colores_valores=valores,
                                 negrita=set(valores) | {(nodo, "e")}, colores_region=reg),
                res_color=(reg or {}).get((nodo, "e"))))
        tl: dict = {}
        for _, fila in pasos["tardios"].iterrows():
            nodo = fila["suceso"]
            tl[nodo] = fila["valor"]
            arcos = {a: PALETA_FLECHAS[k % len(PALETA_FLECHAS)] for k, a in enumerate(fila["arcos"])}
            valores = {(v, "l"): c for (_, v), c in arcos.items()}
            reg = self._region_dominante(fila, arcos, nodo, "l")
            fotogramas.append(fotograma(
                dict(te), dict(tl), f"Fila {nodo}: ${fila['calculo']}$", "atr", nodo, arcos,
                pert_kwargs=dict(actual=nodo, arcos_activos=arcos, texto="", colores_valores=valores,
                                 negrita=set(valores) | {(nodo, "l")}, colores_region=reg),
                res_color=(reg or {}).get((nodo, "l"))))
        duracion = max(te.values())
        fotogramas.append(fotograma(dict(te), dict(tl), rotulos["final"].format(duracion=f(duracion)),
                                    "fin"))
        return fotogramas

    def zaderenko_tikz_html(self, durations, escala: float = 1.0, textos: dict = None) -> str:
        """Visor HTML con botones del cálculo de Zaderenko con PERT y matriz en TikZ
        (véase :meth:`plot_tikz_zaderenko`); el paso 0 es el resultado final."""
        return _html_fotogramas_tikz(self.plot_tikz_zaderenko(durations, escala, textos),
                                     resultado_primero=True)

    def display_zaderenko_tikz(self, durations, escala: float = 1.0, textos: dict = None) -> None:
        """Muestra en un cuaderno el cálculo de Zaderenko con PERT y matriz en TikZ
        (:meth:`zaderenko_tikz_html`); sin LaTeX, cae en :meth:`display_zaderenko_pasos`."""
        try:
            display(HTML(self.zaderenko_tikz_html(durations, escala, textos)))
        except (FileNotFoundError, RuntimeError) as e:
            import warnings
            warnings.warn(f"No se pudo generar el visor TikZ ({e}); se usa el visor HTML.")
            self.display_zaderenko_pasos(durations)

    # ------------------------------------------------------------------
    # Calendario del proyecto interactivo: clic en cada número
    # ------------------------------------------------------------------

    _CAL_COLUMNAS = [("dur", "Duración"), ("ES", "Inicio más temprano"), ("EF", "Fin más temprano"),
                     ("LS", "Inicio más tardío"), ("LF", "Fin más tardío"),
                     ("HT", "H total"), ("HL", "H libre"), ("HI", "H independiente")]
    # Paleta propia del calendario: tonos bien separados (azul, verde oscuro, rojo oscuro,
    # magenta, marrón, cian); el naranja queda reservado para la celda elegida.
    _CAL_PALETA = ["blue!80!black", "green!50!black", "red!75!black",
                   "magenta!85!black", "brown!80!black", "cyan!60!black"]
    _CAL_HEX = {"blue!80!black": "#0000cc", "green!50!black": "#008000", "red!75!black": "#bf0000",
                "magenta!85!black": "#d900d9", "brown!80!black": "#996633", "cyan!60!black": "#00a3a3"}

    def _calendario_datos(self, durations):
        """Por actividad real: extremos ``(u, v)``, duración y tiempos de sus sucesos."""
        dur = self._normalize_durations(durations, dummies=True).to_dict()
        res = self.calculate_pert(durations)
        te = {f"{k}": v for k, v in res["nodes"]["early"].items()}
        tl = {f"{k}": v for k, v in res["nodes"]["late"].items()}
        orden = (list(durations.index) if isinstance(durations, pd.Series)
                 else list(self.calendar(durations).index))
        filas = {}
        for act, (u, v) in self.nodes_from_edges.items():
            if act.startswith("@"):
                continue
            u, v, d = f"{u}", f"{v}", dur.get(act, 0)
            suc, pre = self._vecinas_de_actividad(act)
            ns = sorted({self.nodes_from_edges[b][0] for b in suc}, key=str)
            np_ = sorted({self.nodes_from_edges[b][1] for b in pre}, key=str)
            fin = max(te.values())
            ms = min((te[n] for n in ns), default=fin)
            mp = max((tl[n] for n in np_), default=0)
            filas[act] = dict(u=u, v=v, d=d, eu=te[u], ev=te[v], lu=tl[u], lv=tl[v],
                              ES=te[u], EF=te[u] + d, LF=tl[v], LS=tl[v] - d,
                              HT=tl[v] - te[u] - d, HL=ms - te[u] - d,
                              HI=max(0, ms - mp - d), dur=d,
                              ns=ns, np=np_, ms=ms, mp=mp)
        orden = [a for a in orden if a in filas] + [a for a in filas if a not in orden]
        return dur, te, tl, {a: filas[a] for a in orden}

    def _calendario_celda(self, a, c, fila, grafo="pert", roy=None):
        """Qué interviene en el cálculo de la celda ``(a, c)`` del calendario.

        Devuelve ``dict(sim, num, res, graf, celdas, arco)``: la expresión simbólica y la
        numérica (HTML con cada término de su color), el resultado, los valores del
        grafo a resaltar ``[(nodo, clave, k)]`` (``k`` es el índice de color; claves
        ``"e"``/``"l"`` en el PERT y ``"ES"``/``"EF"``/``"LS"``/``"LF"``/``"dur"`` en el Roy), las celdas
        de la misma fila de la tabla implicadas ``[(columna, k)]`` y el índice de color
        del arco de la actividad (PERT) o ``None``. Con ``grafo="roy"`` hace falta
        ``roy = (mn, mx, G, etiquetas, dur)``."""
        from html import escape as _html_escape
        f = self._num_paso
        S = lambda k, t: f'<span class="pmule-t" data-k="{k}"><b>{t}</b></span>'
        nombre = dict(self._CAL_COLUMNAS)[c]
        d = fila["d"]
        if grafo == "pert":
            u, v = fila["u"], fila["v"]
            eu, ev, lu, lv = (f"t<sub>{u}</sub>", f"t<sub>{v}</sub>", f"t*<sub>{u}</sub>", f"t*<sub>{v}</sub>")
            if c == "dur":
                return dict(sim="d", num=None, res=d, graf=[], celdas=[], arco=0)
            if c == "ES":
                return dict(sim=S(0, eu), num=None, res=fila[c], graf=[(u, "e", 0)], celdas=[], arco=None)
            if c == "EF":
                return dict(sim=S(0, eu) + " + " + S(1, "d"), num=S(0, f(fila["eu"])) + " + " + S(1, f(d)),
                            res=fila[c], graf=[(u, "e", 0)], celdas=[("ES", 0), ("dur", 1)], arco=1)
            if c == "LF":
                return dict(sim=S(0, lv), num=None, res=fila[c], graf=[(v, "l", 0)], celdas=[], arco=None)
            if c == "LS":
                return dict(sim=S(0, lv) + " − " + S(1, "d"), num=S(0, f(fila["lv"])) + " − " + S(1, f(d)),
                            res=fila[c], graf=[(v, "l", 0)], celdas=[("LF", 0), ("dur", 1)], arco=1)
            if c == "HT":
                return dict(sim=S(0, lv) + " − " + S(1, eu) + " − " + S(2, "d"),
                            num=S(0, f(fila["lv"])) + " − " + S(1, f(fila["eu"])) + " − " + S(2, f(d)),
                            res=fila[c], graf=[(v, "l", 0), (u, "e", 1)],
                            celdas=[("LF", 0), ("ES", 1), ("dur", 2)], arco=2)
            ns, np_ = fila["ns"], fila["np"]
            lista = lambda nodos, pre_: ", ".join(f"{pre_}<sub>{n}</sub>" for n in nodos)
            ms_txt = (lista(ns, "t") if ns else "T")
            mn_ = lambda txt: f"mín({txt})" if len(ns) > 1 else txt
            ms_num = ", ".join(f(roy[0][n]) for n in ns) if ns else f(fila["ms"])
            if c == "HL":
                return dict(sim=mn_(S(0, ms_txt)) + " − " + S(1, eu) + " − " + S(2, "d"),
                            num=mn_(S(0, ms_num)) + " − " + S(1, f(fila["eu"])) + " − " + S(2, f(d)),
                            res=fila[c], graf=[(n, "e", 0) for n in ns] + [(u, "e", 1)],
                            celdas=[("ES", 1), ("dur", 2)], arco=2)
            mx_ = lambda txt: f"máx({txt})" if len(np_) > 1 else txt
            mp_txt = lista(np_, "t*") if np_ else "0"
            mp_num = ", ".join(f(roy[1][n]) for n in np_) if np_ else "0"
            return dict(sim="máx(0, " + mn_(S(0, ms_txt)) + " − " + mx_(S(1, mp_txt)) + " − " + S(2, "d") + ")",
                        num="máx(0, " + mn_(S(0, ms_num)) + " − " + mx_(S(1, mp_num)) + " − " + S(2, f(d)) + ")",
                        res=fila[c], graf=[(n, "e", 0) for n in ns] + [(n, "l", 1) for n in np_],
                        celdas=[("dur", 2)], arco=2)
        # grafo Roy: en cada nodo, ES | d | EF arriba y LS | H | LF abajo (mn = ES, mx = LS)
        mn, mx, G, et, dr = roy
        lab = lambda n: _html_escape(f"{et.get(n, n)}")
        ES_, EF_, LS_, LF_ = (lambda n, k=k: f"{k}<sub>{lab(n)}</sub>" for k in ("ES", "EF", "LS", "LF"))
        suc = sorted(f"{x}" for x in G.successors(a))
        pre = sorted(f"{x}" for x in G.predecessors(a))
        if c == "dur":
            return dict(sim="d", num=None, res=d, graf=[(a, "dur", 0)], celdas=[], arco=None)
        if c == "ES":
            return dict(sim=S(0, ES_(a)), num=None, res=fila[c], graf=[(a, "ES", 0)], celdas=[], arco=None)
        if c == "EF":
            return dict(sim=S(0, ES_(a)) + " + " + S(1, "d"), num=S(0, f(mn[a])) + " + " + S(1, f(d)),
                        res=fila[c], graf=[(a, "ES", 0), (a, "dur", 1)], celdas=[("ES", 0), ("dur", 1)], arco=None)
        if c == "LS":
            return dict(sim=S(0, LS_(a)), num=None, res=fila[c], graf=[(a, "LS", 0)], celdas=[], arco=None)
        if c == "LF":
            return dict(sim=S(0, LS_(a)) + " + " + S(1, "d"), num=S(0, f(mx[a])) + " + " + S(1, f(d)),
                        res=fila[c], graf=[(a, "LS", 0), (a, "dur", 1)], celdas=[("LS", 0), ("dur", 1)], arco=None)
        if c == "HT":
            return dict(sim=S(0, LS_(a)) + " − " + S(1, ES_(a)), num=S(0, f(mx[a])) + " − " + S(1, f(mn[a])),
                        res=fila[c], graf=[(a, "LS", 0), (a, "ES", 1)], celdas=[("LS", 0), ("ES", 1)], arco=None)
        ms = ", ".join(ES_(s_) for s_ in suc)
        mv = ", ".join(f(mn[s_]) for s_ in suc)
        graf_s = [(s_, "ES", 0) for s_ in suc]
        val_s = min(mn[s_] for s_ in suc)
        if c == "HL":
            val = val_s - mn[a] - d
            return dict(sim="mín(" + S(0, ms) + ") − " + S(1, EF_(a)),
                        num="mín(" + S(0, mv) + ") − " + S(1, f(mn[a] + d)),
                        res=val, graf=graf_s + [(a, "EF", 1)],
                        celdas=[("EF", 1)], arco=None)
        ps = ", ".join(LF_(p_) for p_ in pre)
        pv = ", ".join(f(mx[p_] + dr[p_]) for p_ in pre)
        graf_p = [(p_, "LF", 1) for p_ in pre]
        val = max(0, val_s - max(mx[p_] + dr[p_] for p_ in pre) - d)
        return dict(res=val, sim="máx(0, mín(" + S(0, ms) + ") − máx(" + S(1, ps) + ") − " + S(2, "d") + ")",
                    num="máx(0, mín(" + S(0, mv) + ") − máx(" + S(1, pv) + ") − " + S(2, f(d)) + ")",
                    graf=graf_s + graf_p + [(a, "dur", 2)], celdas=[("dur", 2)], arco=None)

    def _calendario_graficos(self, durations, escala, grafo):
        """Devuelve ``(dibuja, filas, extra)``: ``dibuja(valores, arcos, actual)`` da el
        ``tikzpicture`` del grafo con lo indicado resaltado, ``filas`` el calendario de
        :meth:`_calendario_datos` y ``extra`` los datos que necesita :meth:`_calendario_celda`."""
        dur, te, tl, filas = self._calendario_datos(durations)
        if grafo == "pert":
            H = self.calculate_pert(durations)["activities"]["H_total"]
            trazado = self._trazado_pert(dur, escala)
            pos = trazado["pos"] if trazado else self._pert_posiciones(escala, None)

            def dibuja(valores=None, arcos=None, actual=None):
                cuerpo = self._pert_fotograma(
                    pos, dur, H, escala, dict(te), dict(tl), arcos_activos=arcos or (), texto="",
                    trazado=trazado, colores_valores=valores, negrita=set(valores or {}))
                return "\n".join([r"\begin{tikzpicture}[>=stealth]", cuerpo, r"\end{tikzpicture}"])
            return dibuja, filas, (te, tl)
        if grafo != "roy":
            raise ValueError("grafo debe ser 'pert' o 'roy'")
        fotograma, mn, mx, G, et, completo = self.plot_tikz_roy_pasos(durations, escala=escala, _fotograma=True)
        dr = self._normalize_durations(durations, dummies=False).to_dict()
        dr["start"] = dr["finish"] = 0

        def dibuja(valores=None, arcos=None, actual=None):
            cuerpo = fotograma(completo, actual, arcos or (), "",
                               colores_valores=valores, negrita=set(valores or {}))
            return "\n".join([r"\begin{tikzpicture}[>=stealth]", cuerpo, r"\end{tikzpicture}"])
        return dibuja, filas, (mn, mx, G, et, dr)

    def plot_tikz_calendario(self, durations, escala: float = 1.0, grafo: str = "pert"):
        """Imágenes TikZ (una neutra y una por celda del calendario) del grafo PERT o Roy
        con los datos que intervienen en el cálculo de cada número resaltados con los
        colores de :attr:`_CAL_PALETA`. Devuelve ``(codigos, indice, datos)``: la lista
        de ``tikzpicture``, ``{(actividad, columna): posición en la lista}`` y
        ``(filas, extra, celdas)`` con el calendario, los datos del Roy y el detalle de
        cada celda (véase :meth:`_calendario_celda`)."""
        dibuja, filas, extra = self._calendario_graficos(durations, escala, grafo)
        paleta = lambda k: self._CAL_PALETA[k % len(self._CAL_PALETA)]
        codigos, vistos, indice, celdas = [dibuja()], {}, {}, {}
        vistos[codigos[0]] = 0
        for a, fila in filas.items():
            for c, _ in self._CAL_COLUMNAS:
                info = self._calendario_celda(a, c, fila, grafo, extra)
                celdas[(a, c)] = info
                valores = {(n, cl): paleta(k) for n, cl, k in info["graf"]}
                if grafo == "pert":
                    col_arco = paleta(info["arco"]) if info["arco"] is not None else "orange!80!black"
                    arcos = {(fila["u"], fila["v"]): col_arco}
                    cod = dibuja(valores, arcos)
                else:
                    G = extra[2]
                    arcos = {}
                    for n, cl, k in info["graf"]:
                        if n != a and cl == "ES" and (a, n) in G.edges:
                            arcos[(a, n)] = paleta(k)
                        elif n != a and (n, a) in G.edges:
                            arcos[(n, a)] = paleta(k)
                    cod = dibuja(valores, arcos, a)
                if cod not in vistos:
                    vistos[cod] = len(codigos)
                    codigos.append(cod)
                indice[(a, c)] = vistos[cod]
        return codigos, indice, (filas, extra, celdas)

    def calendario_html(self, durations, escala: float = 1.0, grafo: str = "pert") -> str:
        """Calendario del proyecto como tabla HTML interactiva junto al grafo: al hacer clic
        en un número se iluminan en el grafo y en la tabla los datos que intervienen en su
        cálculo (con los colores de las flechas) y debajo aparece la fórmula con los
        valores; otro clic en la misma celda lo quita. ``grafo``: ``"pert"`` (por defecto)
        o ``"roy"``."""
        import base64, hashlib, html as _html, json
        codigos, indice, (filas, extra, celdas) = self.plot_tikz_calendario(durations, escala, grafo)
        svgs = _svgs_de_tikz(codigos)
        f = self._num_paso
        uid = "pmc" + hashlib.md5((grafo + "".join(codigos)).encode("utf-8")).hexdigest()[:10]
        imgs = "".join(
            f'<img alt="grafo" class="pmule-cal-img" style="display:{"block" if k == 0 else "none"};'
            f'margin:0 auto;width:100%;max-width:1100px;height:auto" '
            f'src="data:image/svg+xml;base64,{base64.b64encode(s.encode("utf8")).decode("ascii")}">'
            for k, s in enumerate(svgs))
        hexcol = lambda k: self._CAL_HEX.get(self._CAL_PALETA[k % len(self._CAL_PALETA)], "#000")
        import re
        pinta = lambda t: re.sub(r'<span class="pmule-t" data-k="(\d+)">',
                                 lambda m: f'<span style="color:{hexcol(int(m.group(1)))}">', t)
        nombres = dict(self._CAL_COLUMNAS)
        datos = {}
        for (a, c), info in celdas.items():
            cab = f"{nombres[c]}({_html.escape(a)}) = "
            if info["num"] is None:
                texto = cab + pinta(info["sim"]) + (f" = <b>{f(info['res'])}</b>" if c != "dur" else "")
                if c == "dur":
                    texto = cab + f"<b>{f(info['res'])}</b>"
            else:
                texto = cab + pinta(info["sim"]) + " = " + pinta(info["num"]) + f" = <b>{f(info['res'])}</b>"
            datos[f"{a}|{c}"] = {"i": indice[(a, c)], "f": texto,
                                 "t": [[a, col, hexcol(k)] for col, k in info["celdas"]]}
        celda = "padding:.25em .7em;text-align:center;cursor:pointer;border:1px solid #ddd"
        cab = "".join(f'<th style="padding:.3em .6em;border-bottom:2px solid #444">{n}</th>'
                      for n in ["Actividad"] + [n for _, n in self._CAL_COLUMNAS])
        filas_html = []
        for a, fila in filas.items():
            tds = "".join(f'<td class="pmule-cal-td" data-a="{_html.escape(a)}" data-c="{c}" '
                          f'style="{celda}" title="Clic para ver el c&aacute;lculo" onclick="pmuleCal(this)">{f(fila[c])}</td>'
                          for c, _ in self._CAL_COLUMNAS)
            filas_html.append(f'<tr><td style="padding:.25em .7em;font-weight:bold;text-align:center">{_html.escape(a)}</td>{tds}</tr>')
        ayuda = "Haz clic en un número de la tabla para ver cómo se calcula."
        return (
            f'<div id="{uid}" class="pmule-cal" style="margin:1em 0">'
            f'<div>{imgs}</div>'
            f'<div class="pmule-cal-formula" data-ayuda="{ayuda}" style="text-align:center;min-height:2.4em;'
            f'margin:.6em 0;font-size:1.05em">{ayuda}</div>'
            f'<div style="text-align:center;margin:.8em 0 .4em;font-size:.95em;color:#8a4b00">'
            f'&#128070; <b>Interactivo:</b> haz clic en cualquier celda de la tabla para ver c&oacute;mo se calcula '
            f'(otro clic la deselecciona).</div>'
            f'<table style="border-collapse:collapse;margin:0 auto"><tr>{cab}</tr>{"".join(filas_html)}</table></div>'
            f'<script>window.PMULE_CAL = window.PMULE_CAL || {{}}; PMULE_CAL["{uid}"] = {json.dumps(datos, ensure_ascii=False)};'
            'window.pmuleCal = window.pmuleCal || function(td) {'
            'var box = td.closest(".pmule-cal"), D = window.PMULE_CAL[box.id], clave = td.getAttribute("data-a") + "|" + td.getAttribute("data-c");'
            'var imgs = box.querySelectorAll(".pmule-cal-img"), F = box.querySelector(".pmule-cal-formula"), cs = box.querySelectorAll(".pmule-cal-td");'
            'for (var i = 0; i < cs.length; i++) { cs[i].style.background = ""; cs[i].style.outline = ""; cs[i].style.color = ""; cs[i].style.fontWeight = ""; }'
            'var mostrar = function(k) { for (var i = 0; i < imgs.length; i++) imgs[i].style.display = (i === k ? "block" : "none"); };'
            'if (box.getAttribute("data-sel") === clave) { box.removeAttribute("data-sel"); mostrar(0); F.innerHTML = F.getAttribute("data-ayuda"); return; }'
            'box.setAttribute("data-sel", clave); var e = D[clave]; mostrar(e.i); F.innerHTML = e.f;'
            'td.style.background = "#fde3cc"; td.style.outline = "2px solid #cc6600";'
            'for (var j = 0; j < e.t.length; j++) { var o = box.querySelector(\'.pmule-cal-td[data-a="\' + e.t[j][0] + \'"][data-c="\' + e.t[j][1] + \'"]\');'
            'if (o) { o.style.outline = "2px solid " + e.t[j][2]; o.style.color = e.t[j][2]; o.style.fontWeight = "bold"; } } };</script>')

    def display_calendario(self, durations, escala: float = 1.0, grafo: str = "pert") -> None:
        """Muestra en un cuaderno el calendario interactivo (:meth:`calendario_html`) con
        el grafo ``"pert"`` o ``"roy"``; sin LaTeX, cae en el DataFrame de :meth:`calendar`."""
        try:
            display(HTML(self.calendario_html(durations, escala, grafo)))
        except (FileNotFoundError, RuntimeError) as e:
            import warnings
            warnings.warn(f"No se pudo generar el calendario interactivo ({e}); se muestra la tabla.")
            display(self.calendar(durations))

    # ------------------------------------------------------------------
    # Grafo Roy paso a paso (docencia)
    # ------------------------------------------------------------------

    def _roy_grafo_actual(self) -> nx.DiGraph:
        df = PredecessorTable.from_project(self).immediate_linkage_matrix
        filas = ~df.index.str.startswith("@")
        columnas = ~df.columns.str.startswith("@")
        return make_Roy(df.loc[filas, columnas])

    @staticmethod
    def _roy_capas(G) -> dict:
        capa = {}
        for nodo in nx.topological_sort(G):
            capa[nodo] = max((capa[p] + 1 for p in G.predecessors(nodo)), default=0)
        return capa

    def roy_pasos(self, durations, formato: str = "texto",
                  etiquetas: dict = None, colores=None) -> dict:
        """Cálculo paso a paso de los tiempos del grafo Roy, con los nodos al estilo del PMBOK.

        Cada nodo tiene seis casillas: inicio más temprano ``ES``, duración ``d``, fin más
        temprano ``EF``, inicio más tardío ``LS``, holgura total ``H`` y fin más tardío
        ``LF`` (origen de tiempos en 0). Relaciones fin-comienzo: hacia delante,
        ``ES_J = máx{EF_I}`` sobre las precedentes de J y ``EF_J = ES_J + d_J``; hacia atrás,
        ``LF_I = mín{LS_J}`` sobre sus sucesoras y ``LS_I = LF_I - d_I``; la holgura es
        ``H = LS - ES``. Los nodos ficticios ``start`` y ``finish`` se rotulan con
        ``etiquetas`` (por defecto «Inicio» y «Fin»).

        Returns
        -------
        dict
            ``{"minimos": DataFrame, "maximos": DataFrame}`` con una fila por paso y las
            columnas ``nodo``, ``rotulo``, ``actividades``, ``terminos`` (rótulo y valor de
            cada vecino que interviene: su ``EF`` hacia delante, su ``LS`` hacia atrás),
            ``vecinos`` (nodos de los términos, en orden), ``calculo`` y ``valor``. En
            ``minimos`` el ``valor`` es ``ES`` y se añaden ``calculo_fin`` y ``fin`` (``EF``);
            en ``maximos`` el ``valor`` es ``LF`` y se añaden ``calculo_inicio`` e
            ``inicio`` (``LS``). Con ``formato="latex"`` y ``colores`` (lista de colores
            TikZ), cada término del cálculo lleva el color de su arco.
        """
        if formato not in ("texto", "latex"):
            raise ValueError("formato debe ser 'texto' o 'latex'")
        et = {"start": "Inicio", "finish": "Fin"}
        et.update(etiquetas or {})
        dur = self._normalize_durations(durations, dummies=False).to_dict()
        dur["start"] = dur["finish"] = 0
        G = self._roy_grafo_actual()
        capa = self._roy_capas(G)
        orden = sorted(G.nodes, key=lambda n: (capa[n], f"{n}"))
        f = self._num_paso
        latex = formato == "latex"

        def rot(n):
            return et.get(n, f"{n}")

        def V(clave, n):
            """Símbolo de una casilla del nodo ``n``: ``ES_A``, ``EF_A``, ``LS_A``, ``LF_A``, ``d_A``."""
            r = rot(n)
            if latex:
                r = rf"\mathrm{{{r}}}" if len(r) > 1 else r
                return f"{clave}_{{{r}}}"
            return f"{clave}_{r}"

        def res(valor):
            return rf"\mathbf{{{f(valor)}}}" if (latex and colores) else f(valor)

        sep, sem = ("+", "-") if latex else (" + ", " - ")

        def tinta(ops):
            """Cada término del cálculo con el color de su arco (solo en LaTeX)."""
            if not (latex and colores):
                return ops
            return [rf"\textcolor{{{colores[k % len(colores)]}}}{{\mathbf{{{o}}}}}" for k, o in enumerate(ops)]

        def extremo(izq, simbolos, numeros, valor, maximo):
            """``izq = máx{símbolos} = máx{números} = valor`` (sin ``máx`` con un solo término)."""
            if len(simbolos) == 1:
                return f"{izq} = {simbolos[0]} = {res(valor)}"
            if latex:
                op = r"\min" if maximo else r"\max"
                return (rf"{izq} = {op}\{{{r',\ '.join(simbolos)}\}}"
                        rf" = {op}\{{{r',\ '.join(numeros)}\}} = {res(valor)}")
            op = "mín" if maximo else "máx"
            return f"{izq} = {op}{{{', '.join(simbolos)}}} = {op}{{{', '.join(numeros)}}} = {f(valor)}"

        ES, EF, filas = {}, {}, []
        for nodo in orden:
            prev = sorted(G.predecessors(nodo), key=lambda n: (capa[n], f"{n}"))
            if prev:
                terminos = [(rot(p), EF[p]) for p in prev]
                ES[nodo] = max(t for _, t in terminos)
                calculo = extremo(V("ES", nodo), tinta([V("EF", p) for p in prev]),
                                  tinta([f(EF[p]) for p in prev]), ES[nodo], False)
            else:
                terminos, ES[nodo] = [], 0
                calculo = f"{V('ES', nodo)} = 0"
            EF[nodo] = ES[nodo] + dur[nodo]
            fin = (f"{V('EF', nodo)} = {V('ES', nodo)} + {V('d', nodo)} = "
                   f"{f(ES[nodo])}{sep}{f(dur[nodo])} = {res(EF[nodo])}")
            filas.append(dict(nodo=nodo, rotulo=rot(nodo),
                              actividades=", ".join(a for a, _ in terminos),
                              terminos=terminos, vecinos=list(prev), calculo=calculo,
                              valor=ES[nodo], calculo_fin=fin, fin=EF[nodo]))
        minimos = pd.DataFrame(filas, index=pd.RangeIndex(1, len(filas) + 1, name="paso"))

        LF, LS, filas = {}, {}, []
        for nodo in orden[::-1]:
            sig = sorted(G.successors(nodo), key=lambda n: (capa[n], f"{n}"))
            if sig:
                terminos = [(rot(s), LS[s]) for s in sig]
                LF[nodo] = min(t for _, t in terminos)
                calculo = extremo(V("LF", nodo), tinta([V("LS", s) for s in sig]),
                                  tinta([f(LS[s]) for s in sig]), LF[nodo], True)
            else:
                terminos, LF[nodo] = [], EF[nodo]
                calculo = f"{V('LF', nodo)} = {V('EF', nodo)} = {f(LF[nodo])}"
            LS[nodo] = LF[nodo] - dur[nodo]
            inicio = (f"{V('LS', nodo)} = {V('LF', nodo)} - {V('d', nodo)} = "
                      f"{f(LF[nodo])}{sem}{f(dur[nodo])} = {res(LS[nodo])}")
            filas.append(dict(nodo=nodo, rotulo=rot(nodo),
                              actividades=", ".join(a for a, _ in terminos),
                              terminos=terminos, vecinos=list(sig), calculo=calculo,
                              valor=LF[nodo], calculo_inicio=inicio, inicio=LS[nodo]))
        maximos = pd.DataFrame(filas, index=pd.RangeIndex(1, len(filas) + 1, name="paso"))
        return dict(minimos=minimos, maximos=maximos)

    # Nodo del Roy al estilo del PMBOK: tres filas por tres columnas,
    #     ES | d | EF
    #      nombre
    #     LS | H | LF
    # (inicio más temprano, duración, fin más temprano; inicio más tardío, holgura total,
    # fin más tardío), con esquinas rectas. En estado neutro es solo blanco y negro (sin
    # fondo); los colores (rojo, naranja, azul, el de cada flecha) aparecen únicamente
    # cuando el nodo o una casilla se resaltan por algún motivo.
    _ROY_ANCHO, _ROY_ALTO = 2.25, 1.3                  # cm
    _ROY_ALTO_FILA, _ROY_ALTO_NOMBRE = 0.42, 0.46      # cm: filas de tiempos y del nombre
    _ROY_CELDAS = {"ES": (0, 0), "dur": (1, 0), "EF": (2, 0),
                   "LS": (0, 2), "H": (1, 2), "LF": (2, 2)}      # (columna, fila)

    @classmethod
    def _roy_caja_tikz(cls, r, x, y, nombre, celdas=None, borde=None, grosor="thick",
                       relleno=None, barra=None, color_texto="", negrita=False,
                       resaltadas=None) -> list:
        """Líneas TikZ de un nodo del Roy (véase ``_ROY_CELDAS``) centrado en ``(x, y)`` cm.

        ``celdas``: ``{casilla: código TikZ del contenido}`` con las casillas ``ES``,
        ``dur``, ``EF``, ``LS``, ``H`` y ``LF`` (las que falten quedan vacías);
        ``resaltadas``: ``{casilla: (color, grosor)}`` redibuja el borde de esas casillas;
        ``color_texto``: opción ``text=...`` de los números; ``negrita``: números en negrita;
        ``barra``: color de la banda del nombre (por defecto sin relleno y nombre en negro; con
        color, nombre en blanco).
        El nodo TikZ ``r`` es el rectángulo exterior (las flechas se recortan en su borde)."""
        W, H = cls._ROY_ANCHO, cls._ROY_ALTO
        h, hb = cls._ROY_ALTO_FILA, cls._ROY_ALTO_NOMBRE
        borde = borde or "black"
        relleno = relleno or "white"
        x0, x1, c = x - W / 2, x + W / 2, W / 3
        yb0, yb1, y0, y1 = y - hb / 2, y + hb / 2, y - H / 2, y + H / 2
        P = lambda px, py: f"({px:.3f}cm,{py:.3f}cm)"
        L = [rf"  \node[rectangle, fill={relleno}, draw={borde}, {grosor}, minimum width={W}cm,"
             rf" minimum height={H}cm, inner sep=0pt] ({r}) at ({x:.2f}cm, {y:.2f}cm) {{}};",
             rf"  \draw[draw={borde}] {P(x0, yb0)} -- {P(x1, yb0)} {P(x0, yb1)} -- {P(x1, yb1)}"
             rf" {P(x0 + c, yb1)} -- {P(x0 + c, y1)} {P(x0 + 2 * c, yb1)} -- {P(x0 + 2 * c, y1)}"
             rf" {P(x0 + c, yb0)} -- {P(x0 + c, y0)} {P(x0 + 2 * c, yb0)} -- {P(x0 + 2 * c, y0)};"]

        def limites(clave):
            col, fila = cls._ROY_CELDAS[clave]
            ya, yb = (yb1, y1) if fila == 0 else (y0, yb0)
            return x0 + col * c, ya, x0 + (col + 1) * c, yb
        for clave, (color, gr) in (resaltadas or {}).items():
            xa, ya, xb, yb = limites(clave)
            L.append(rf"  \draw[draw={color}, {gr}] {P(xa, ya)} rectangle {P(xb, yb)};")
        fuente = r"\scriptsize" + (r"\bfseries" if negrita else "")
        for clave, contenido in (celdas or {}).items():
            xa, ya, xb, yb = limites(clave)
            L.append(rf"  \node[font={fuente}{', ' + color_texto if color_texto else ''}]"
                     rf" at {P((xa + xb) / 2, (ya + yb) / 2)} {{{contenido}}};")
        if barra:
            L.insert(1, rf"  \path[fill={barra}] {P(x0, yb0)} rectangle {P(x1, yb1)};")
        L.append(rf"  \node[font=\footnotesize\bfseries, {'text=white' if barra else (color_texto or 'text=black')}]"
                 rf" at ({x:.2f}cm, {y:.2f}cm) {{{nombre}}};")
        return L

    def plot_tikz_roy_pasos(self, durations, beamer: bool = False,
                            escala: float = 1.0, posiciones: dict = None,
                            textos: dict = None, etiquetas: dict = None,
                            layout: str = "auto", estatico: bool = False,
                            con_duracion: bool = True, _fotograma: bool = False):
        """Dibujo del grafo Roy paso a paso (código TikZ).

        Como :meth:`plot_tikz_pasos`, pero con las actividades en los nodos, al estilo del
        PMBOK: tres filas por tres columnas, ``ES | d | EF`` arriba, el nombre en la banda
        central y ``LS | H | LF`` abajo (origen de tiempos en 0). Fotogramas: grafo con las
        duraciones; uno por nodo hacia delante (se rellenan ``ES`` y ``EF``; se resaltan las
        casillas ``EF`` de las precedentes que intervienen y la ``ES`` calculada, con el
        color de sus flechas); uno por nodo hacia atrás (``LF`` y ``LS``, con las casillas
        ``LS`` de las sucesoras); y el camino crítico en rojo con las holguras ``H``. Bajo
        el grafo, el cálculo del paso (véase :meth:`roy_pasos`).

        ``posiciones`` es ``{nodo: (x, y)}`` en cm con los nombres internos
        de pmule (``start``, ``finish`` y las actividades). Necesita las
        bibliotecas TikZ ``calc`` y ``arrows.meta``.
        """
        et = {"start": "Inicio", "finish": "Fin"}
        et.update(etiquetas or {})
        dur = self._normalize_durations(durations, dummies=False).to_dict()
        dur["start"] = dur["finish"] = 0
        pasos = self.roy_pasos(durations, formato="latex", etiquetas=et, colores=PALETA_FLECHAS)
        G = self._roy_grafo_actual()
        capa = self._roy_capas(G)
        ancho, alto = self._ROY_ANCHO, self._ROY_ALTO
        trazado = None
        if posiciones is None and layout == "auto":
            trazado = _disposicion_capas({f"{n}": (ancho, alto) for n in G.nodes},
                                         [(f"{u}", f"{v}") for u, v in G.edges()],
                                         separacion=0.3 * escala, hueco=0.4 * escala)
        if trazado:
            pos = trazado["pos"]
        elif posiciones is None:
            from collections import defaultdict
            por_capa = defaultdict(list)
            for n in sorted(G.nodes, key=lambda n: f"{n}"):
                por_capa[capa[n]].append(n)
            pos = {}
            for c, nodos in por_capa.items():
                for i, n in enumerate(nodos):
                    pos[f"{n}"] = ((ancho + 1.1) * c * escala,
                                   ((len(nodos) - 1) / 2.0 - i) * (alto + 0.6) * escala)
        else:
            pos = {f"{k}": (float(x) * escala, float(y) * escala)
                   for k, (x, y) in posiciones.items()}
            faltan = {f"{n}" for n in G.nodes} - set(pos)
            if faltan:
                raise ValueError(f"Faltan posiciones para los nodos {sorted(faltan)}")
        ident = {f"{n}": f"r{k}" for k, n in enumerate(sorted(G.nodes, key=lambda n: f"{n}"))}
        rotulos = {"inicio": "Grafo Roy del proyecto",
                   "final": "Camino crítico: duración = {duracion}"}
        rotulos.update(textos or {})
        f = self._num_paso
        y_min = min(y for _, y in pos.values()) - alto / 2
        x_mid = (min(x for x, _ in pos.values()) + max(x for x, _ in pos.values())) / 2

        # todos los tiempos (el visor los va mostrando) y la holgura total H = LS - ES
        ES = {f"{r['nodo']}": r["valor"] for _, r in pasos["minimos"].iterrows()}
        EF = {f"{r['nodo']}": r["fin"] for _, r in pasos["minimos"].iterrows()}
        LF = {f"{r['nodo']}": r["valor"] for _, r in pasos["maximos"].iterrows()}
        LS = {f"{r['nodo']}": r["inicio"] for _, r in pasos["maximos"].iterrows()}
        completo = {n: dict(ES=ES[n], EF=EF[n], LS=LS[n], LF=LF[n], H=LS[n] - ES[n]) for n in ES}

        def fotograma(visibles, actual=None, arcos_activos=(), texto="", criticos=(),
                      colores_valores=None, negrita=None, colores_region=None, derivadas=()):
            """``visibles``: ``{nodo: {casilla: valor}}`` con lo ya calculado (casillas ``ES``,
            ``EF``, ``LS``, ``LF``, ``H``; la duración ``dur`` sale siempre);
            ``arcos_activos``: ``{(u, v): color}`` (o un conjunto, en naranja), más gruesos;
            ``colores_valores``: ``{(nodo, casilla): color}``; ``negrita``: conjunto de esas
            claves cuya casilla se resalta (borde grueso y número en negrita), con el color de
            ``colores_region`` si lo hay; ``derivadas``: las que se obtienen del resultado
            del paso (borde menos grueso). En un paso de cálculo lo que no participa (ni es
            crítico) va en gris."""
            def tinta(nodo, clave, texto):
                if (nodo, clave) in (negrita or ()):
                    texto = rf"\textbf{{{texto}}}"
                c = (colores_valores or {}).get((nodo, clave))
                return rf"\textcolor{{{c}}}{{{texto}}}" if c else texto

            GRIS = "gray!55"
            atenuar = bool(arcos_activos) or actual is not None
            col_nodo = ({x: c for (u, v), c in arcos_activos.items() for x in (u, v)
                         if x != actual} if isinstance(arcos_activos, dict) else {})
            implicados = ({actual} if actual is not None else set()) | {x for a in arcos_activos for x in a}

            L, A = [], []
            for u, v in G.edges():
                u, v = f"{u}", f"{v}"
                estilo = ["->", "thick"]
                col_act = (arcos_activos.get((u, v)) if isinstance(arcos_activos, dict)
                           else ("orange!80!black" if (u, v) in arcos_activos else None))
                if col_act:
                    estilo += [col_act, "ultra thick"]
                elif u in criticos and v in criticos:
                    estilo += ["red", "very thick"]
                elif atenuar:
                    estilo.append(GRIS)
                if trazado and (u, v) in trazado["rutas"]:
                    A.append(rf"  \draw[{', '.join(estilo)}] "
                             + self._ruta_tikz(u, v, pos, trazado["rutas"][(u, v)],
                                               trazado["curva"][(u, v)], lambda n: ident[n]) + ";")
                    continue
                curva = self._curvatura_tikz(pos, u, v, 0.8 * escala)
                trazo = f"to[{curva}]" if curva else "--"
                A.append(rf"  \draw[{', '.join(estilo)}] ({ident[u]}) {trazo} ({ident[v]});")
            for nodo, (x, y) in pos.items():
                if nodo == actual:
                    estilo = dict(borde="orange!80!black", grosor="very thick",
                                  relleno="orange!20", barra="orange!80!black")
                elif nodo in criticos:
                    estilo = dict(borde="red", grosor="very thick", relleno="white!90!red", barra="red")
                elif nodo in col_nodo:
                    estilo = dict(borde=col_nodo[nodo], grosor="ultra thick")
                elif atenuar and nodo not in implicados:
                    estilo = dict(borde=GRIS, grosor="thick", relleno="white")
                else:
                    estilo = {}
                gris = atenuar and nodo not in implicados and nodo not in criticos and nodo != actual
                celdas = {}
                if con_duracion:
                    celdas["dur"] = tinta(nodo, "dur", f(dur[nodo]))
                for clave, valor in visibles.get(nodo, {}).items():
                    celdas[clave] = tinta(nodo, clave, f(valor))
                resaltadas = {}
                # un conjunto no tiene orden: se ordena para que el TikZ (y la caché) sea el mismo en cada proceso
                for n_, clave in (sorted(negrita) if isinstance(negrita, (set, frozenset)) else (negrita or ())):
                    if n_ == nodo:
                        color = (colores_region or {}).get((nodo, clave)) or (colores_valores or {}).get(
                            (nodo, clave), "orange!80!black" if nodo == actual else "black")
                        resaltadas[clave] = (color, "ultra thick" if (nodo == actual and
                                                                       (nodo, clave) not in derivadas)
                                             else "very thick")
                L += self._roy_caja_tikz(ident[nodo], x, y, self._tex_texto(f"{et.get(nodo, nodo)}"),
                                         celdas, negrita=nodo in criticos,
                                         color_texto=f"text={GRIS}" if gris else "",
                                         resaltadas=resaltadas, **estilo)
            L += A
            if texto:
                L.append(rf"  \node[anchor=north, font=\small, align=center] at ({x_mid:.2f}cm, {y_min - 0.6:.2f}cm) {{{texto}}};")
            return "\n".join(L)

        if _fotograma:                     # uso interno: la función que dibuja un fotograma
            return fotograma, ES, LS, G, et, completo
        if estatico:                       # un solo dibujo, sin texto (grafo Roy del resultado)
            if con_duracion:
                fotos = [fotograma(completo, criticos={n for n in ES if ES[n] == LS[n]})]
            else:
                fotos = [fotograma({})]
            return self._beamer_o_lista(fotos, beamer=False)
        copia = lambda v: {n: dict(c) for n, c in v.items()}
        fotos = [fotograma({}, texto=rotulos["inicio"])]
        visibles = {}
        for _, fila in pasos["minimos"].iterrows():
            n = f"{fila['nodo']}"
            visibles[n] = {"ES": fila["valor"], "EF": fila["fin"]}
            arcos = {(f"{p}", n): PALETA_FLECHAS[k % len(PALETA_FLECHAS)]
                     for k, p in enumerate(fila["vecinos"])}
            valores = {(p, "EF"): col for (p, _), col in arcos.items()}
            fotos.append(fotograma(copia(visibles), n, arcos,
                                   rf"${fila['calculo']}$" + r"\\" + rf"${fila['calculo_fin']}$",
                                   colores_valores=valores,
                                   negrita=set(valores) | {(n, "ES"), (n, "EF")},
                                   colores_region=self._region_dominante(fila, arcos, n, "ES"),
                                   derivadas={(n, "EF")}))
        for _, fila in pasos["maximos"].iterrows():
            n = f"{fila['nodo']}"
            visibles[n] = dict(visibles[n], LF=fila["valor"], LS=fila["inicio"])
            arcos = {(n, f"{q}"): PALETA_FLECHAS[k % len(PALETA_FLECHAS)]
                     for k, q in enumerate(fila["vecinos"])}
            valores = {(q, "LS"): col for (_, q), col in arcos.items()}
            fotos.append(fotograma(copia(visibles), n, arcos,
                                   rf"${fila['calculo']}$" + r"\\" + rf"${fila['calculo_inicio']}$",
                                   colores_valores=valores,
                                   negrita=set(valores) | {(n, "LF"), (n, "LS")},
                                   colores_region=self._region_dominante(fila, arcos, n, "LF"),
                                   derivadas={(n, "LS")}))
        criticos = {n for n in ES if ES[n] == LS[n]}
        fotos.append(fotograma(completo, criticos=criticos,
                               texto=rotulos["final"].format(duracion=f(ES["finish"]))))
        return self._beamer_o_lista(fotos, beamer=beamer)


    def roy_pasos_html(self, durations, escala: float = 1.0, posiciones: dict = None,
                       textos: dict = None, etiquetas: dict = None) -> str:
        """El cálculo de los tiempos del Roy como visor HTML con botones cuyos
        fotogramas son los dibujos TikZ de :meth:`plot_tikz_roy_pasos`,
        compilados a SVG (hace falta ``pdflatex`` y ``pdftocairo``). El primer
        fotograma (paso 0) es el grafo resuelto; a partir del paso 1 se calcula
        desde el principio."""
        codigos = self.plot_tikz_roy_pasos(durations, beamer=False, escala=escala,
                                           posiciones=posiciones, textos=textos,
                                           etiquetas=etiquetas)
        return _html_fotogramas_tikz(codigos, resultado_primero=True)

    def display_roy_pasos(self, durations, etiquetas: dict = None, escala: float = 1.0,
                          posiciones: dict = None, textos: dict = None,
                          visor: bool = True) -> None:
        """Muestra en un cuaderno el cálculo de los tiempos del Roy paso a paso:
        grafo TikZ que se va rellenando, con botones (``visor=True``). Con
        ``visor=False``, o si no hay LaTeX, la lista de cálculos como Markdown."""
        if visor:
            try:
                display(HTML(self.roy_pasos_html(durations, escala, posiciones, textos, etiquetas)))
                return
            except (FileNotFoundError, RuntimeError) as e:
                import warnings
                warnings.warn(f"No se pudo generar el visor TikZ ({e}); se muestra la lista de cálculos.")
        pasos = self.roy_pasos(durations, formato="latex", etiquetas=etiquetas)
        for titulo, clave, segundo in (("Hacia delante: inicios y fines más tempranos", "minimos", "calculo_fin"),
                                       ("Hacia atrás: fines e inicios más tardíos", "maximos", "calculo_inicio")):
            lineas = [f"**{titulo}**", ""]
            for _, fila in pasos[clave].iterrows():
                act = f" &nbsp; ({fila['actividades']})" if fila["actividades"] else ""
                lineas.append(f"- ${fila['calculo']}${act}; ${fila[segundo]}$")
            display(Markdown("\n".join(lineas)))

    # ------------------------------------------------------------------
    # Construcción del grafo a partir de la matriz de prelaciones (docencia)
    # ------------------------------------------------------------------

    def _prelaciones_reales(self) -> dict:
        """``{actividad: [precedentes inmediatas]}`` sin ficticias."""
        df = PredecessorTable.from_project(self).immediate_linkage_matrix
        reales = [a for a in df.index if not f"{a}".startswith("@")]
        return {a: sorted(p for p in reales if bool(df.loc[a, p])) for a in reales}

    def orden_construccion(self) -> list:
        """Orden en que se añaden las actividades al construir el grafo: por
        niveles (las que no tienen precedentes, después las que solo dependen
        de ellas, …) y alfabético dentro de cada nivel."""
        G = self._roy_grafo_actual()
        capa = self._roy_capas(G)
        return sorted(self._prelaciones_reales(), key=lambda a: (capa[a], f"{a}"))

    def matriz_prelaciones_beamer(self, primero: int = 2, simbolo: str = r"$\otimes$") -> str:
        """Matriz de prelaciones (filas: actividad; columnas: sus precedentes)
        como ``tabular`` de beamer, con la fila de cada actividad resaltada en
        el fotograma en que se añade al grafo (los de
        :meth:`plot_tikz_construccion_pert` y :meth:`plot_tikz_construccion_roy`:
        la actividad k-ésima de :meth:`orden_construccion` en el fotograma
        ``primero + k - 1``). Necesita ``colortbl``."""
        pre = self._prelaciones_reales()
        acts = sorted(pre)
        paso = {a: primero + k for k, a in enumerate(self.orden_construccion())}
        # tabcolsep pequeño en vez de @{\,}: con @{} colortbl tapa el contenido
        L = [r"{\setlength{\tabcolsep}{1.5pt}\begin{tabular}{c|" + "c" * len(acts) + "}",
             " & " + " & ".join(acts) + r"\\ \hline"]
        for a in acts:
            marca = rf"\only<{paso[a]}>{{\cellcolor{{orange!30}}}}"
            celdas = [marca + rf"\textbf{{{a}}}"]
            celdas += [marca + (simbolo if p in pre[a] else "") for p in acts]
            L.append(" & ".join(celdas) + r"\\")
        L.append(r"\end{tabular}}")
        return "\n".join(L)

    @staticmethod
    def _rotulos_construccion(textos):
        rot = {"inicio": "Se parte del suceso inicial",
               "inicio_roy": "Se parte del nodo Inicio",
               "precedentes": "precedentes", "sin": "sin precedentes",
               "ficticia": "con una ficticia", "ficticias": "con {n} ficticias",
               "final": "Grafo terminado: se numeran los sucesos ($i<j$)",
               "final_roy": "Se añade Fin tras las actividades finales"}
        rot.update(textos or {})
        return rot

    def plot_tikz_construccion_pert(self, beamer: bool = False, escala: float = 1.0,
                                    posiciones: dict = None, textos: dict = None):
        """Construcción del grafo PERT paso a paso a partir de la matriz de
        prelaciones (código TikZ).

        Fotogramas: el suceso inicial; uno por actividad, en el orden de
        :meth:`orden_construccion`, que añade su arco (en naranja, con las
        ficticias que necesita) y escribe debajo sus precedentes; y uno final
        con los sucesos numerados. Disposición como en :meth:`plot_tikz_pasos`
        (por capas o ``posiciones``). Combínese con
        :meth:`matriz_prelaciones_beamer` para resaltar la fila de la matriz.
        """
        rot = self._rotulos_construccion(textos)
        pos = self._pert_posiciones(escala, posiciones)
        G = self.pert_graph
        orden = self.orden_construccion()
        pre = self._prelaciones_reales()
        paso = {a: k for k, a in enumerate(orden, start=1)}
        ultimo = len(orden) + 1
        arcos = {}
        for u, v, data in G.edges(data=True):
            arcos[(f"{u}", f"{v}")] = data.get("activity", "")

        def aparece(nodo, visto=()):
            """Paso en que hace falta el suceso ``nodo`` como origen."""
            cands = []
            for (u, v), a in arcos.items():
                if u != nodo:
                    continue
                if a and not a.startswith("@"):
                    cands.append(paso[a])
                elif v not in visto:
                    cands.append(aparece(v, visto + (v,)))
            return min(cands) if cands else ultimo

        paso_arco = {}
        for (u, v), a in arcos.items():
            paso_arco[(u, v)] = paso[a] if (a and not a.startswith("@")) else aparece(v)
        inicial = [n for n in pos if G.in_degree(n) == 0]
        f = self._num_paso
        y_min = min(y for _, y in pos.values())
        x_mid = (min(x for x, _ in pos.values()) + max(x for x, _ in pos.values())) / 2

        def fotograma(k, texto, numeros=False):
            visibles = {e for e, s in paso_arco.items() if s <= k}
            nodos = set(inicial) | {n for e in visibles for n in e}
            L, A = [], []
            for (u, v) in sorted(visibles):
                a = arcos[(u, v)]
                ficticia = (not a) or a.startswith("@")
                estilo = ["->", "thick"] + (["dashed"] if ficticia else [])
                if paso_arco[(u, v)] == k and k < ultimo:
                    estilo += ["orange!80!black", "very thick"]
                etiqueta = "" if ficticia else a
                curva = self._curvatura_tikz(pos, u, v, 0.75 * escala)
                trazo = f"to[{curva}]" if curva else "--"
                lado = "below" if curva.startswith("bend right") else "above"
                A.append(rf"  \draw[{', '.join(estilo)}] (n{u}) {trazo}"
                         rf" node[midway, {lado}, sloped, font=\scriptsize] {{{etiqueta}}} (n{v});")
            actuales = {u for (u, v), s in paso_arco.items() if s == k} if k < ultimo else set()
            for nodo in sorted(nodos):
                x, y = pos[nodo]
                estilo = ("fill=orange!20, draw=orange!80!black, very thick" if nodo in actuales
                          else "fill=white, draw, thick")
                L.append(rf"  \node[circle, {estilo}, minimum size=1.1cm, inner sep=0pt]"
                         rf" (n{nodo}) at ({x:.2f}cm, {y:.2f}cm) {{}};")
                L.append(rf"  \draw[gray] (n{nodo}.center) -- (n{nodo}.45)"
                         rf" (n{nodo}.center) -- (n{nodo}.135) (n{nodo}.center) -- (n{nodo}.270);")
                if numeros:
                    L.append(rf"  \node[font=\footnotesize\bfseries] at ($(n{nodo}.center)+(90:0.28)$) {{{nodo}}};")
            # nodos aún no visibles: coordenadas para que las curvas no cambien
            for nodo in sorted(set(pos) - nodos):
                x, y = pos[nodo]
                L.append(rf"  \coordinate (n{nodo}) at ({x:.2f}cm, {y:.2f}cm);")
            L += A
            L.append(rf"  \path ({min(x for x, _ in pos.values()) - 0.6:.2f}cm, {max(y for _, y in pos.values()) + 0.6:.2f}cm)"
                     rf" rectangle ({max(x for x, _ in pos.values()) + 0.6:.2f}cm, {y_min - 0.6:.2f}cm);")
            L.append(rf"  \node[anchor=north, font=\Large] at ({x_mid:.2f}cm, {y_min - 1.2:.2f}cm) {{{texto}}};")
            return "\n".join(L)

        fotos = [fotograma(0, rot["inicio"])]
        for k, a in enumerate(orden, start=1):
            texto = f"{a}: " + (f"{rot['precedentes']} {', '.join(pre[a])}" if pre[a] else rot["sin"])
            nf = sum(1 for e, s in paso_arco.items() if s == k and arcos[e].startswith("@"))
            if nf:
                texto += " (" + (rot["ficticia"] if nf == 1 else rot["ficticias"].format(n=nf)) + ")"
            fotos.append(fotograma(k, texto))
        fotos.append(fotograma(ultimo, rot["final"], numeros=True))
        return self._beamer_o_lista(fotos, beamer=beamer)

    def plot_tikz_construccion_roy(self, beamer: bool = False, escala: float = 1.0,
                                   posiciones: dict = None, textos: dict = None,
                                   etiquetas: dict = None):
        """Construcción del grafo Roy paso a paso a partir de la matriz de
        prelaciones (código TikZ): el nodo Inicio; uno por actividad (en el
        orden de :meth:`orden_construccion`) con su nodo y los arcos desde sus
        precedentes en naranja; y Fin tras las actividades finales. Mismos
        parámetros que :meth:`plot_tikz_roy_pasos`."""
        rot = self._rotulos_construccion(textos)
        et = {"start": "Inicio", "finish": "Fin"}
        et.update(etiquetas or {})
        rot["inicio_roy"] = rot["inicio_roy"].replace("Inicio", et["start"])
        rot["final_roy"] = rot["final_roy"].replace("Fin", et["finish"])
        G = self._roy_grafo_actual()
        capa = self._roy_capas(G)
        pre = self._prelaciones_reales()
        orden = self.orden_construccion()
        paso = {"start": 0, "finish": len(orden) + 1}
        paso.update({a: k for k, a in enumerate(orden, start=1)})
        if posiciones is None:
            from collections import defaultdict
            por_capa = defaultdict(list)
            for n in sorted(G.nodes, key=lambda n: f"{n}"):
                por_capa[capa[n]].append(n)
            pos = {f"{n}": ((self._ROY_ANCHO + 1.1) * c * escala, ((len(ns) - 1) / 2.0 - i) * (self._ROY_ALTO + 0.6) * escala)
                   for c, ns in por_capa.items() for i, n in enumerate(ns)}
        else:
            pos = {f"{k}": (float(x) * escala, float(y) * escala) for k, (x, y) in posiciones.items()}
        ident = {f"{n}": f"r{k}" for k, n in enumerate(sorted(G.nodes, key=lambda n: f"{n}"))}
        y_min = min(y for _, y in pos.values()) - self._ROY_ALTO / 2
        x_mid = (min(x for x, _ in pos.values()) + max(x for x, _ in pos.values())) / 2

        def fotograma(k, texto):
            L, A = [], []
            for u, v in G.edges():
                u, v = f"{u}", f"{v}"
                if paso[v] > k:
                    continue
                estilo = ["->", "thick"] + (["orange!80!black", "very thick"] if paso[v] == k else [])
                curva = self._curvatura_tikz(pos, u, v, 0.8 * escala)
                trazo = f"to[{curva}]" if curva else "--"
                A.append(rf"  \draw[{', '.join(estilo)}] ({ident[u]}) {trazo} ({ident[v]});")
            for nodo, (x, y) in pos.items():
                r = ident[nodo]
                if paso[nodo] > k:
                    L.append(rf"  \coordinate ({r}) at ({x:.2f}cm, {y:.2f}cm);")
                    continue
                estilo = (dict(borde="orange!80!black", grosor="very thick", relleno="orange!20",
                               barra="orange!80!black") if paso[nodo] == k else {})
                L += self._roy_caja_tikz(r, x, y, self._tex_texto(f"{et.get(nodo, nodo)}"), None, **estilo)
            L += A
            xs = [x for x, _ in pos.values()]; ys = [y for _, y in pos.values()]
            L.append(rf"  \path ({min(xs) - 1.5:.2f}cm, {max(ys) + 1.2:.2f}cm) rectangle ({max(xs) + 1.5:.2f}cm, {y_min - 0.6:.2f}cm);")
            L.append(rf"  \node[anchor=north, font=\Large] at ({x_mid:.2f}cm, {y_min - 1.0:.2f}cm) {{{texto}}};")
            return "\n".join(L)

        fotos = [fotograma(0, rot["inicio_roy"])]
        for k, a in enumerate(orden, start=1):
            texto = f"{a}: " + (f"{rot['precedentes']} {', '.join(pre[a])}" if pre[a] else rot["sin"])
            fotos.append(fotograma(k, texto))
        fotos.append(fotograma(len(orden) + 1, rot["final_roy"]))
        return self._beamer_o_lista(fotos, beamer=beamer)


    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        distributions: dict,
        n_samples: int = 10_000,
        seed: int = None,
    ) -> "MonteCarloResult":
        """Simulación Monte Carlo sobre las duraciones del proyecto.

        Parameters
        ----------
        distributions : dict
            {nombre_actividad: callable | (str, *params)}
            Acepta:
            - Callable ``f() -> float`` (o ``f(size=N) -> array``).
            - Tupla ``("normal", mu, sigma)``, ``("uniform", a, b)``,
              ``("triangular", low, mode, high)``, ``("lognormal", mu, sigma)``,
              ``("pert", a, m, b)`` — equivalente PERT beta aproximado.
            Las actividades ficticias (``"@…"``) se ignoran automáticamente.
        n_samples : int
            Número de réplicas. Por defecto 10 000.
        seed : int | None
            Semilla para reproducibilidad.

        Returns
        -------
        MonteCarloResult
        """
        rng = np.random.default_rng(seed)

        # ── Simular (vectorizado) ──────────────────────────────────
        # T: (n_samples × n_act) muestras; R: (n_paths × n_act) rutas.
        # Longitudes de ruta P = T @ Rᵀ → (n_samples × n_paths); τ = max fila.
        R_df = self.path_matrix(dummies=False)
        R = R_df.values.astype(float)
        acts = list(R_df.columns)
        unknown = sorted(set(distributions) - set(acts) - set(self.dummies))
        if unknown:
            import warnings
            warnings.warn(f"Distribuciones para actividades desconocidas: {unknown}. "
                          "Se ignoran.", UserWarning, stacklevel=2)
        T = np.zeros((n_samples, len(acts)))
        for j, a in enumerate(acts):
            if a in distributions:
                T[:, j] = _make_sampler(rng, distributions[a])(n_samples)

        P = T @ R.T
        tau_samples = P.max(axis=1)
        # Sensibilidad empírica: media sobre réplicas de la fracción de rutas
        # críticas que contienen cada actividad (misma definición que
        # tau_sensitivity, calculada en bloque).
        critical = np.isclose(P, tau_samples[:, None])          # (n_samples × n_paths)
        n_crit = critical.sum(axis=1, keepdims=True)             # ≥ 1 siempre
        grad = (critical.astype(float) @ R) / n_crit             # (n_samples × n_act)
        sens_mean = pd.Series(grad.mean(axis=0), index=acts,
                              name="empirical_sensitivity").sort_values(ascending=False)

        return MonteCarloResult(
            tau_samples=tau_samples,
            sensitivity_mean=sens_mean,
            n_samples=n_samples,
            seed=seed,
        )


    def _calendarios_simulados(self, D, acts):
        """Inicios y fines más tempranos de cada réplica (vectorizado).

        ``D`` tiene forma (n_réplicas × n_actividades) con las columnas en el orden
        ``acts``.  Devuelve (ES, EF) con la misma forma.
        """
        preds = self._real_predecessors()
        col = {a: j for j, a in enumerate(acts)}
        pendientes = {a: set(preds.get(a, ())) & set(acts) for a in acts}
        orden = []
        while pendientes:
            listas = sorted(a for a, p in pendientes.items() if not p)
            if not listas:
                raise ValueError("El grafo tiene un ciclo")
            for a in listas:
                orden.append(a)
                del pendientes[a]
            for p in pendientes.values():
                p.difference_update(listas)
        ES = np.zeros_like(D)
        EF = np.zeros_like(D)
        for a in orden:
            j = col[a]
            ps = [col[x] for x in preds.get(a, ()) if x in col]
            ES[:, j] = EF[:, ps].max(axis=1) if ps else 0.0
            EF[:, j] = ES[:, j] + D[:, j]
        return ES, EF

    def monte_carlo_van(
        self,
        distributions: dict,
        bac,
        ingreso_mensual,
        n_meses: int,
        tasa_anual: float,
        n_samples: int = 10_000,
        seed: int = None,
        bac_distributions: dict = None,
        plazo=None,
        penalizacion_mensual: float = 0.0,
        redondear: bool = True,
    ) -> "MonteCarloVANResult":
        """Distribución del VAN de un proyecto de inversión con duraciones aleatorias.

        Modelo (el mismo del caso «vínculo con el proyecto» de la colección 17): el
        coste de cada actividad (``bac``) se desembolsa uniformemente a lo largo de su
        duración, empezando en su inicio más temprano; cuando el proyecto termina
        (τ) empieza un flujo de ``ingreso_mensual`` durante ``n_meses`` meses. Los
        meses se descuentan a la tasa mensual equivalente (1 + k_m)¹² = 1 + tasa_anual.
        Cada réplica simula las duraciones, calcula su calendario y su VAN en forma
        cerrada (series geométricas), todo vectorizado.

        Parameters
        ----------
        distributions : {actividad: especificación} duraciones (ver ``monte_carlo``).
        bac : dict | Series  presupuesto de cada actividad.
        ingreso_mensual : número o especificación de distribución (incertidumbre de
            mercado, un valor por réplica).
        n_meses : meses de explotación.
        tasa_anual : tasa de descuento anual.
        bac_distributions : opcional, {actividad: especificación} del **factor
            multiplicativo** del BAC (p. ej. ``("triangular", 0.9, 1.0, 1.4)``).
        plazo, penalizacion_mensual : penalización por mes de retraso sobre el plazo,
            pagada en el mes τ.
        redondear : si True (por defecto) las duraciones simuladas se redondean a
            meses enteros (mínimo 1), coherente con la hipótesis de duraciones enteras;
            entonces, con duraciones constantes, el VAN coincide con el del flujo mensual.

        Returns
        -------
        MonteCarloVANResult
        """
        rng = np.random.default_rng(seed)
        R_df = self.path_matrix(dummies=False)
        acts = list(R_df.columns)
        bac = pd.Series(bac, dtype=float)
        falta = [a for a in acts if a not in bac.index]
        if falta:
            raise ValueError(f"Faltan BAC para las actividades {falta}")
        D = np.zeros((n_samples, len(acts)))
        for j, a in enumerate(acts):
            if a not in distributions:
                raise ValueError(f"Falta la distribución de la duración de {a}")
            D[:, j] = _make_sampler(rng, distributions[a])(n_samples)
        if redondear:
            D = np.maximum(np.rint(D), 1.0)
        B = np.tile(bac.reindex(acts).to_numpy(), (n_samples, 1))
        if bac_distributions:
            for j, a in enumerate(acts):
                if a in bac_distributions:
                    B[:, j] *= _make_sampler(rng, bac_distributions[a])(n_samples)
        if isinstance(ingreso_mensual, (int, float, np.integer, np.floating)):
            F = np.full(n_samples, float(ingreso_mensual))
        else:
            F = _make_sampler(rng, ingreso_mensual)(n_samples)

        ES, EF = self._calendarios_simulados(D, acts)
        tau = EF.max(axis=1)
        km = (1.0 + tasa_anual) ** (1.0 / 12.0) - 1.0
        v = 1.0 / (1.0 + km)
        serie = lambda inicio, n: v ** (inicio + 1.0) * (1.0 - v ** n) / (1.0 - v)  # Σ_{m=inicio+1}^{inicio+n} v^m
        coste = (B / D * serie(ES, D)).sum(axis=1)
        ingreso = F * serie(tau, float(n_meses))
        penal = 0.0
        if plazo is not None and penalizacion_mensual:
            penal = penalizacion_mensual * np.maximum(tau - plazo, 0.0) * v ** tau
        van = ingreso - coste - penal
        return MonteCarloVANResult(van, tau, D, B, F, acts, n_samples, seed)


def _make_sampler(rng, spec):
    """Sampler ``f(n) -> array`` a partir de una especificación de distribución.

    ``spec``: callable ``f() -> float`` (o ``f(size=N) -> array``) o tupla
    ``("normal", mu, sigma)``, ``("uniform", a, b)``, ``("triangular", low, mode, high)``,
    ``("lognormal", mu, sigma)``, ``("pert", a, m, b)`` (normal PERT recortada a [a, b]).
    """
    if callable(spec):
        import inspect
        sig = inspect.signature(spec)
        if "size" in sig.parameters:
            return lambda n, _spec=spec: _spec(size=n)
        return lambda n, _spec=spec: np.array([_spec() for _ in range(n)])
    kind, *args = spec
    kind = kind.lower()
    if kind == "normal":
        mu, sigma = args
        return lambda n: rng.normal(mu, sigma, n)
    elif kind == "uniform":
        a, b = args
        return lambda n: rng.uniform(a, b, n)
    elif kind == "triangular":
        low, mode, high = args
        if low == high:                      # distribución degenerada: constante
            return lambda n: np.full(n, float(low))
        return lambda n: rng.triangular(low, mode, high, n)
    elif kind == "lognormal":
        mu, sigma = args
        return lambda n: rng.lognormal(mu, sigma, n)
    elif kind == "pert":
        # PERT beta: mean = (a + 4m + b) / 6, std = (b - a) / 6
        a, m, b = args
        mu_pert = (a + 4 * m + b) / 6.0
        sigma_pert = (b - a) / 6.0
        return lambda n: np.clip(rng.normal(mu_pert, sigma_pert, n), a, b)
    raise ValueError(f"Distribución desconocida: {kind!r}")



def _orden_topologico(preds):
    """Orden topológico (determinista) de un dict {actividad: {predecesoras}}."""
    pend = {a: set(p) & set(preds) for a, p in preds.items()}
    orden = []
    while pend:
        lista = sorted(a for a, p in pend.items() if not p)
        if not lista:
            raise ValueError("El grafo tiene un ciclo")
        for a in lista:
            orden.append(a)
            del pend[a]
        for p in pend.values():
            p.difference_update(lista)
    return orden


def _cpm_puro(preds, dur, orden):
    """(ES, LS, T) del CPM sin recursos para duraciones ``dur`` (dict)."""
    ES, EF = {}, {}
    for a in orden:
        ES[a] = max((EF[x] for x in preds[a] if x in EF), default=0)
        EF[a] = ES[a] + dur[a]
    T = max(EF.values(), default=0)
    succ = {a: [] for a in orden}
    for a in orden:
        for x in preds[a]:
            if x in succ:
                succ[x].append(a)
    LS = {}
    for a in reversed(orden):
        LS[a] = min((LS[x] for x in succ[a]), default=T) - dur[a]
    return ES, LS, T


def _matriz_limite(vectores, ancho):
    """Límites por recurso y periodo (K × ancho); el último valor de cada vector se prolonga."""
    lim = np.empty((len(vectores), ancho))
    for k, v in enumerate(vectores):
        v = np.asarray(v, dtype=float)[:ancho]
        lim[k, :len(v)] = v
        lim[k, len(v):] = v[-1]
    return lim


def _planificar_paralelo(preds, orden, dur, dem, limite, horizonte, etiquetas, ls=None):
    """Planificador en paralelo, sin interrupciones, con varios recursos.

    ``dur``: {actividad: int}; ``dem``: {actividad: array K}; ``limite``: K × ancho
    (ancho ≥ horizonte + max duración + 1).  Prioridad: menor ``ls`` (por defecto el
    del CPM sin recursos), menor duración, nombre.  Devuelve (inicio, fin, traza, carga)
    con ``traza`` = lista de (periodo, actividad, 'empieza'|'espera', recursos que bloquean).
    """
    K = limite.shape[0]
    if ls is None:
        ls = _cpm_puro(preds, dur, orden)[1]
    prior = sorted(orden, key=lambda a: (ls[a], dur[a], a))
    carga = np.zeros_like(limite)
    inicio, fin, traza = {}, {}, []
    pendientes = set(orden)
    t = 0
    while pendientes:
        if t > horizonte:
            raise ValueError("Error: el proyecto no es factible con los límites de recursos indicados")
        elegibles = [a for a in prior if a in pendientes
                     and all(x in fin and fin[x] <= t for x in preds[a])]
        for a in elegibles:
            d = dur[a]
            if d == 0:
                inicio[a], fin[a] = t, t
                pendientes.discard(a)
                continue
            exceso = (carga[:, t:t + d] + dem[a][:, None] > limite[:, t:t + d] + 1e-9)
            bloqueo = [etiquetas[k] for k in range(K) if exceso[k].any()]
            if not bloqueo:
                carga[:, t:t + d] += dem[a][:, None]
                inicio[a], fin[a] = t, t + d
                pendientes.discard(a)
                traza.append((t + 1, a, "empieza", ""))
            else:
                traza.append((t + 1, a, "espera", ", ".join(bloqueo)))
        t += 1
    return inicio, fin, traza, carga


def _cadena_critica(preds, inicio, fin, traza, dem, T, etiquetas):
    """Actividades de la cadena crítica con recursos y sus arcos «ajustados».

    a → b es un arco ajustado si ``a`` termina justo cuando empieza ``b`` y, o bien ``a``
    es predecesora de ``b``, o bien ``b`` esperaba por un recurso que ``a`` usa y que
    se liberó al terminar ``a``.  La cadena crítica son las actividades desde las que se
    llega al final del proyecto (τ) por arcos ajustados.
    """
    bloqueo_previo = {}
    for periodo, a, decision, recursos in traza:
        if decision == "espera" and periodo == inicio[a]:       # semana justo anterior al inicio
            bloqueo_previo[a] = set(recursos.split(", "))
    idx = {r: k for k, r in enumerate(etiquetas)}
    acts = list(inicio)
    listo = {a: max((fin[x] for x in preds[a] if x in fin), default=0) for a in acts}
    arcos = {a: set() for a in acts}              # arcos[b] = {a : a → b ajustado}
    for b in acts:
        for a in acts:
            if a == b or fin[a] != inicio[b] or fin[a] == inicio[a]:
                continue
            if a in preds[b]:
                arcos[b].add(a)
            elif inicio[b] > listo[b] and any(dem[a][idx[r]] > 0 for r in bloqueo_previo.get(b, ())):
                arcos[b].add(a)
    criticas = {a for a in acts if fin[a] == T and fin[a] > inicio[a]}
    pila = list(criticas)
    while pila:
        b = pila.pop()
        for a in arcos[b]:
            if a not in criticas:
                criticas.add(a)
                pila.append(a)
    return criticas, {b: arcos[b] & criticas for b in criticas}

# ---------------------------------------------------------------------------
# AsignacionMulti
# ---------------------------------------------------------------------------

class AsignacionMulti:
    """Resultado de ``ProjectGraph.asignar_multi`` (asignación con varios recursos).

    Atributos: ``data`` (duraciones con filas de espera 💤X), ``inicios`` y
    ``fines`` (instante, desde 0), ``esperas`` (periodos que espera cada
    actividad desde que quedó lista), ``duracion`` (del proyecto con recursos),
    ``cargas`` y ``limites`` (DataFrame periodo × recurso), ``traza``
    (periodo, actividad, decisión y recurso que la bloquea), ``gantts`` y
    ``dibujos`` (dict por recurso), ``proyecto`` (copia del ProjectGraph con
    las esperas 💤X insertadas, coherente con ``data``).
    """

    def __init__(self, data, inicios, fines, esperas, duracion, cargas, limites,
                 traza, gantts, dibujos, proyecto=None):
        self.proyecto = proyecto
        self.data = data
        self.inicios = inicios
        self.fines = fines
        self.esperas = esperas
        self.duracion = duracion
        self.cargas = cargas
        self.limites = limites
        self.traza = traza
        self.gantts = gantts
        self.dibujos = dibujos

    def __iter__(self):
        # permite `data, gantts, dibujos = proyecto.asignar_multi(...)`
        return iter((self.data, self.gantts, self.dibujos))

    def __repr__(self):
        return (f"AsignacionMulti(duracion={self.duracion}, "
                f"recursos={list(self.cargas.columns)}, esperas={self.esperas.to_dict()})")


# ---------------------------------------------------------------------------
# Cadena crítica (CCPM)
# ---------------------------------------------------------------------------

def _recursos_unitarios(valor):
    """Recursos que usa una actividad a partir de una celda («X», «X, Y», '', '---', NaN)."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return ()
    texto = str(valor).strip()
    if texto in ("", "---", "-", "—", "nan", "None"):
        return ()
    return tuple(sorted({t.strip() for t in texto.split(",") if t.strip()}))


Z90 = 1.2815515655446004      # percentil 90 de la normal tipificada


class PlanCadenaCritica:
    """Resultado de ``ProjectGraph.plan_cadena_critica`` (método de la cadena crítica, CCPM).

    Atributos: ``cadena`` (lista ordenada), ``enlaces_cadena`` (de, a, motivo
    'precedencia' | 'recurso', recurso), ``enlaces_recurso`` (enlaces que la nivelación añade
    a la red), ``predecesoras`` (red nivelada), ``duracion_cadena`` (con las estimaciones
    agresivas), ``margen`` (segura − agresiva), ``colchon_proyecto`` (CP adoptado) y
    ``calculo_colchon_proyecto`` (valor sin redondear con los dos métodos),
    ``alimentacion`` (un colchón de alimentación por fila: desde, antes_de, cadena de
    alimentación, márgenes, colchón bruto, requerido, holgura disponible, adoptado, inicio y
    fin), ``colchones_recursos`` (avisos), ``inicios`` / ``fines`` (plan: cadena en su
    inicio más temprano, alimentación lo más tarde posible), ``compromiso`` (cadena + CP),
    ``duracion_tradicional`` (red nivelada con las estimaciones seguras), ``cadenas_empatadas``
    (rutas críticas de la red nivelada) y ``traza`` (planificador de recursos).

    Métodos: ``tabla()``, ``red_nivelada()``, ``gantt_tikz()``, ``seguimiento()`` y ``simular()``.
    """

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.fines = (self.inicios + self.agresiva.reindex(self.inicios.index)).rename("fin").astype(int)
        self.compromiso = int(self.duracion_cadena + self.colchon_proyecto)

    def __repr__(self):
        return (f"PlanCadenaCritica(cadena={'-'.join(self.cadena)}, duracion_cadena={self.duracion_cadena}, "
                f"CP={self.colchon_proyecto}, compromiso={self.compromiso}, "
                f"CA={list(self.alimentacion['adoptado']) if len(self.alimentacion) else []})")

    # ------------------------------------------------------------------
    def tabla(self):
        """Una fila por actividad: estimaciones, margen, recursos, si es de la cadena e inicio/fin del plan."""
        acts = list(self.agresiva.index)
        return pd.DataFrame({
            "agresiva": self.agresiva, "segura": self.segura, "margen": self.margen,
            "recurso": pd.Series({a: ", ".join(self.recursos[a]) for a in acts}),
            "cadena": pd.Series({a: ("sí" if a in self.cadena else "") for a in acts}),
            "inicio": self.inicios, "fin": self.fines}).reindex(acts)

    def red_nivelada(self):
        """Cuadro de prelaciones de la red nivelada (prelaciones + enlaces de recurso), con las
        dos estimaciones; sirve para ``PredecessorTable.from_dataframe_of_strings``."""
        acts = list(self.agresiva.index)
        pred = {a: (",".join(sorted(self.predecesoras[a])) if self.predecesoras[a] else "---") for a in acts}
        df = pd.DataFrame({"predecessor": pd.Series(pred), "agresiva": self.agresiva,
                           "segura": self.segura}).reindex(acts)
        df.index.name = "activity"
        return df

    # ------------------------------------------------------------------
    def gantt_tikz(self, params=None, mostrar_recursos=True):
        """Diagrama de Gantt del plan (TikZ): cadena crítica en rojo, alimentación en verde
        (programada lo más tarde posible), colchones en naranja (CA delante de la cadena, CP al
        final). Los enlaces de recurso se dibujan con flechas discontinuas."""
        acts = list(self.agresiva.index)
        alim = self.alimentacion[self.alimentacion["adoptado"] > 0] if len(self.alimentacion) else self.alimentacion
        filas = acts + list(alim.index) + (["CP"] if self.colchon_proyecto > 0 else [])
        gd = pd.DataFrame(index=filas, columns=["start", "duration", "resource", "Htotal"], dtype=object)
        for a in acts:
            gd.loc[a] = [int(self.inicios[a]), int(self.agresiva[a]),
                         ", ".join(self.recursos[a]) if mostrar_recursos else "", 0 if a in self.cadena else 1]
        for nombre, f in alim.iterrows():
            gd.loc[nombre] = [int(f["inicio"]), int(f["adoptado"]), "", 1]
        if self.colchon_proyecto > 0:
            gd.loc["CP"] = [int(self.duracion_cadena), int(self.colchon_proyecto), "", 0]
        gd["start"] = gd["start"].astype(int)
        gd["duration"] = gd["duration"].astype(int)
        gd["Htotal"] = gd["Htotal"].astype(int)
        gd["resource"] = gd["resource"].astype(str)
        ca_de = {(f["desde"], f["antes_de"]): nombre for nombre, f in alim.iterrows()}
        prec = []
        for a in acts:
            for p in sorted(self.predecesoras[a]):
                if (p, a) in ca_de:
                    prec += [(p, ca_de[(p, a)]), (ca_de[(p, a)], a)]
                else:
                    prec.append((p, a))
        for (p, c), nombre in ca_de.items():
            if c == "fin":
                prec.append((p, nombre))
                if self.colchon_proyecto > 0:
                    prec.append((nombre, "CP"))
        if self.colchon_proyecto > 0 and self.cadena:
            prec.append((self.cadena[-1], "CP"))
        colores = {b: "orange!45" for b in list(alim.index) + ["CP"]}
        estilos = {(x, y): "dashed" for x, y in zip(self.enlaces_recurso["de"], self.enlaces_recurso["a"])}
        p = dict(inner_text="resource", fill_colors=colores, arrow_styles=estilos)
        p.update(params or {})
        return make_gantt_tikz(gantt_data=gd, params=p, precedences=prec)

    # ------------------------------------------------------------------
    def seguimiento(self, dia, terminadas=None, en_curso=None, verde=(10, 70), roja=(30, 90)):
        """Informe de avance con la gestión de colchones.

        ``dia``: fecha del informe (instante, contado desde 0); ``terminadas``: ``{actividad: fin
        real}``; ``en_curso``: ``{actividad: duración restante estimada}``; las demás no han
        empezado. Fin previsto: recorrido hacia delante de la red nivelada (las terminadas en su
        fin real; las en curso en ``dia + restante``; las no empezadas de la cadena en cuanto
        terminan sus predecesoras —corredor de relevos— y las de alimentación no antes de su
        inicio del plan) con las duraciones agresivas.

        - colchón consumido = fin previsto − duración de la cadena (en días y en % del CP);
        - cadena completada = duración de la cadena − duración restante de la cadena, es decir,
          la duración agresiva de las terminadas más ``máx(0, agresiva − restante)`` de las que
          están en curso (en días y en % de la cadena);
        - BPI = % de colchón restante / % de cadena restante;
        - zona del gráfico de fiebre: verde si el consumo es menor que la recta que va de
          ``verde[0]`` (avance 0 %) a ``verde[1]`` (avance 100 %); roja si es mayor o igual que la
          recta de ``roja``; amarilla entre ambas.

        Devuelve un dict con esas cifras, los fines previstos y el consumo de cada colchón de
        alimentación.
        """
        terminadas = dict(terminadas or {})
        en_curso = dict(en_curso or {})
        d = self.agresiva
        fin = {}
        for a in self.orden:
            if a in terminadas:
                fin[a] = float(terminadas[a])
            elif a in en_curso:
                fin[a] = float(dia + en_curso[a])
            else:
                st = max([float(dia)] + [fin[p] for p in self.predecesoras[a]])
                if a not in self.cadena:
                    st = max(st, float(self.inicios[a]))
                fin[a] = st + float(d[a])
        fin_previsto = max(fin.values()) if fin else 0.0
        consumo = fin_previsto - self.duracion_cadena
        pct_consumo = 100.0 * consumo / self.colchon_proyecto if self.colchon_proyecto else float("inf")
        filas = {}
        hecho = 0.0
        for c in self.cadena:
            if c in terminadas:
                estado, h, rest = "terminada", float(d[c]), 0.0
            elif c in en_curso:
                estado, rest = "en curso", float(en_curso[c])
                h = max(0.0, float(d[c]) - rest)
            else:
                estado, h, rest = "pendiente", 0.0, float(d[c])
            hecho += h
            filas[c] = {"estado": estado, "agresiva": int(d[c]), "completado": h, "restante": rest,
                        "fin previsto": fin[c]}
        pct_avance = 100.0 * hecho / self.duracion_cadena if self.duracion_cadena else 100.0
        lim_verde = verde[0] + (verde[1] - verde[0]) * pct_avance / 100.0
        lim_roja = roja[0] + (roja[1] - roja[0]) * pct_avance / 100.0
        zona = "verde" if pct_consumo < lim_verde - 1e-9 else ("roja" if pct_consumo >= lim_roja - 1e-9 else "amarilla")
        bpi = ((100.0 - pct_consumo) / (100.0 - pct_avance)) if pct_avance < 100.0 - 1e-9 else float("inf")
        cons_ca = []
        for nombre, f in (self.alimentacion.iterrows() if len(self.alimentacion) else []):
            entrada = self.inicios[f["antes_de"]] if f["antes_de"] != "fin" else self.duracion_cadena
            usado = max(0.0, fin[f["desde"]] - (entrada - f["adoptado"]))
            cons_ca.append(dict(colchon=nombre, desde=f["desde"], antes_de=f["antes_de"],
                                fin_previsto=fin[f["desde"]], adoptado=int(f["adoptado"]), consumido=usado,
                                pct=(100.0 * usado / f["adoptado"]) if f["adoptado"] else (0.0 if usado == 0 else float("inf"))))
        return dict(dia=dia, fin_previsto=fin_previsto, consumo=consumo, pct_consumo=pct_consumo,
                    avance=hecho, pct_avance=pct_avance, bpi=bpi, zona=zona,
                    limite_verde=lim_verde, limite_roja=lim_roja,
                    detalle=pd.DataFrame(filas).T, fines=pd.Series(fin).reindex(sorted(fin)),
                    alimentacion=pd.DataFrame(cons_ca, columns=["colchon", "desde", "antes_de", "fin_previsto",
                                                                 "adoptado", "consumido", "pct"]))

    # ------------------------------------------------------------------
    def simular(self, n=10000, semilla=None, regimen="relevos"):
        """Duraciones simuladas del proyecto (array de ``n`` réplicas).

        La duración de cada actividad es lognormal con mediana = estimación agresiva y
        percentil 90 = estimación segura (constante si coinciden). Se respeta la red nivelada
        (el orden de los recursos no cambia). Regímenes de ejecución:

        - ``"relevos"`` (cadena crítica): las actividades de la cadena empiezan en cuanto
          terminan sus predecesoras; las de alimentación, no antes de su inicio del plan.
        - ``"tradicional"``: cronograma con las estimaciones seguras; cada actividad empieza en
          su fecha del cronograma (o más tarde si sus predecesoras se retrasan) y, por la ley de
          Parkinson, nunca termina antes de su estimación segura: dura ``máx(simulada, segura)``.
        """
        if regimen not in ("relevos", "tradicional"):
            raise ValueError("Error: regimen debe ser 'relevos' o 'tradicional'")
        rng = np.random.default_rng(semilla)
        acts = sorted(self.predecesoras_originales)
        D = {}
        for a in acts:
            med, p90 = float(self.agresiva[a]), float(self.segura[a])
            sigma = math.log(p90 / med) / Z90 if (p90 > med and med > 0) else 0.0
            D[a] = med * np.exp(sigma * rng.standard_normal(n))
        fin = {}
        for a in self.orden:
            st = np.zeros(n)
            for p in self.predecesoras[a]:
                st = np.maximum(st, fin[p])
            if regimen == "relevos":
                if a not in self.cadena:
                    st = np.maximum(st, float(self.inicios[a]))
                fin[a] = st + D[a]
            else:
                st = np.maximum(st, float(self.inicio_tradicional[a]))
                fin[a] = st + np.maximum(D[a], float(self.segura[a]))
        return np.max(np.vstack([fin[a] for a in self.orden]), axis=0) if self.orden else np.zeros(n)


def grafico_fiebre_tikz(puntos, verde=(10, 70), roja=(30, 90), ancho=9.0, alto=6.0, unir=True):
    """Gráfico de fiebre de la gestión por cadena crítica (TikZ).

    ``puntos``: lista de tuplas ``(% de cadena completada, % de colchón consumido, etiqueta)``
    o de dicts devueltos por ``PlanCadenaCritica.seguimiento`` (se rotulan «día N»). Zonas:
    verde por debajo de la recta que va de ``verde[0]`` (0 % de avance) a ``verde[1]`` (100 %),
    roja por encima de la recta de ``roja``, amarilla entre ambas.
    """
    pts = []
    for p in puntos:
        if isinstance(p, dict):
            pts.append((float(p["pct_avance"]), float(p["pct_consumo"]), f"día {p['dia']}"))
        else:
            pts.append((float(p[0]), float(p[1]), str(p[2]) if len(p) > 2 else ""))
    ys = [y for _, y, _ in pts] or [0.0]
    ymax = max(100.0, math.ceil(max(ys) / 10.0) * 10.0)
    ymin = min(0.0, math.floor(min(ys) / 10.0) * 10.0)
    sx, sy = ancho / 100.0, alto / (ymax - ymin)
    t = [rf"\begin{{tikzpicture}}[x={sx:.4f}cm, y={sy:.4f}cm, font=\small]"]
    v0, v1, r0, r1 = verde[0], verde[1], roja[0], roja[1]
    t.append(rf"\fill[green!25] (0,{ymin}) -- (100,{ymin}) -- (100,{v1}) -- (0,{v0}) -- cycle;")
    t.append(rf"\fill[yellow!45] (0,{v0}) -- (100,{v1}) -- (100,{r1}) -- (0,{r0}) -- cycle;")
    t.append(rf"\fill[red!30] (0,{r0}) -- (100,{r1}) -- (100,{ymax}) -- (0,{ymax}) -- cycle;")
    t.append(rf"\draw[->] (0,{ymin}) -- (106,{ymin});")
    t.append(rf"\node[below=14pt] at (50,{ymin}) {{Cadena crítica completada (\%)}};")
    t.append(rf"\draw[->] (0,{ymin}) -- (0,{ymax + 8});")
    t.append(rf"\node[rotate=90] at (-13,{(ymin + ymax) / 2:.1f}) {{Colchón consumido (\%)}};")
    if ymin < 0:
        t.append(r"\draw[gray] (0,0) -- (100,0);")
    for x in (0, 25, 50, 75, 100):
        t.append(rf"\draw ({x},{ymin}) -- ++(0,-1.5pt) node[below, font=\scriptsize] {{{x}}};")
    paso = 25 if (ymax - ymin) <= 150 else 50
    y = ymin
    while y <= ymax + 1e-9:
        t.append(rf"\draw (0,{y:g}) -- ++(-1.5pt,0) node[left, font=\scriptsize] {{{y:g}}};")
        y += paso

    def recta(lim, x):
        return lim[0] + (lim[1] - lim[0]) * x / 100.0
    t.append(rf"\node[font=\scriptsize, text=black!70] at (85,{(ymin + recta(verde, 85)) / 2:.1f}) {{verde}};")
    t.append(rf"\node[font=\scriptsize, text=black!70] at (20,{(recta(verde, 20) + recta(roja, 20)) / 2:.1f}) {{amarilla}};")
    t.append(rf"\node[font=\scriptsize, text=black!70] at (12,{(recta(roja, 12) + ymax) / 2:.1f}) {{roja}};")
    if unir and len(pts) > 1:
        t.append(r"\draw[thick, gray] " + " -- ".join(f"({x:.2f},{y:.2f})" for x, y, _ in pts) + ";")
    for x, y, e in pts:
        t.append(rf"\fill[blue] ({x:.2f},{y:.2f}) circle (2pt);")
        if e:
            t.append(rf"\node[above left, font=\scriptsize] at ({x:.2f},{y:.2f}) {{{e}}};")
    t.append(r"\end{tikzpicture}")
    return "\n".join(t)


class MonteCarloVANResult:
    """Resultado de ``ProjectGraph.monte_carlo_van``.

    Atributos: ``van_samples``, ``tau_samples`` (duración del proyecto de cada
    réplica), ``duraciones`` y ``bac`` (réplicas × actividades), ``ingreso_samples``,
    ``actividades``, ``n_samples``, ``seed``.
    """

    def __init__(self, van_samples, tau_samples, duraciones, bac, ingreso_samples,
                 actividades, n_samples, seed):
        self.van_samples = np.asarray(van_samples, dtype=float)
        self.tau_samples = np.asarray(tau_samples, dtype=float)
        self.duraciones = duraciones
        self.bac = bac
        self.ingreso_samples = ingreso_samples
        self.actividades = list(actividades)
        self.n_samples = n_samples
        self.seed = seed

    @property
    def van_mean(self) -> float:
        return float(self.van_samples.mean())

    @property
    def van_std(self) -> float:
        return float(self.van_samples.std(ddof=1))

    @property
    def prob_negativo(self) -> float:
        """P(VAN < 0)."""
        return float((self.van_samples < 0).mean())

    def cuantil(self, p: float) -> float:
        return float(np.quantile(self.van_samples, p))

    def cvar(self, alpha: float = 0.05) -> float:
        """Media del VAN en el ``alpha`` peor de los casos (cola inferior)."""
        corte = np.quantile(self.van_samples, alpha)
        return float(self.van_samples[self.van_samples <= corte].mean())

    def summary(self) -> "pd.DataFrame":
        d = {"media": self.van_mean, "desv. típica": self.van_std,
             "q05": self.cuantil(0.05), "q50": self.cuantil(0.50), "q95": self.cuantil(0.95),
             "P(VAN<0)": self.prob_negativo, "CVaR 5%": self.cvar(0.05)}
        return pd.DataFrame(d, index=["VAN"])

    def tornado(self) -> "pd.DataFrame":
        """Correlación de Spearman del VAN con la duración de cada actividad (y con el
        ingreso y los BAC si son aleatorios), ordenada por valor absoluto."""
        cols = {f"duración {a}": self.duraciones[:, j] for j, a in enumerate(self.actividades)}
        if np.ptp(self.ingreso_samples) > 0:
            cols["ingreso mensual"] = self.ingreso_samples
        for j, a in enumerate(self.actividades):
            if np.ptp(self.bac[:, j]) > 0:
                cols[f"coste {a}"] = self.bac[:, j]
        df = pd.DataFrame(cols)
        df = df.loc[:, df.std() > 0]
        rho = df.rank().corrwith(pd.Series(self.van_samples).rank())
        out = rho.to_frame("Spearman con el VAN")
        return out.reindex(out["Spearman con el VAN"].abs().sort_values(ascending=False).index)

    def __repr__(self) -> str:
        return (f"MonteCarloVANResult(n={self.n_samples}, VAN̄={self.van_mean:.1f}, "
                f"σ={self.van_std:.1f}, P(VAN<0)={self.prob_negativo:.1%})")


class CrashingRecursos:
    """Resultado de ``ProjectGraph.crashing_recursos``.

    ``pasos`` (DataFrame por paso: actividades reducidas, coste, duración antes y después, cadena
    crítica con recursos, lo que habría elegido Ackoff sin recursos y cuántas reducciones
    individuales no tuvieron efecto), ``duraciones`` y ``reducciones`` finales, ``demanda`` final,
    ``duracion_inicial``/``duracion_final`` (con recursos), ``coste_total``, ``motivo_parada``,
    ``inicios`` y, si se pidió, ``asignacion`` (AsignacionMulti del plan final).
    """

    def __init__(self, pasos, duraciones, reducciones, demanda, duracion_inicial, duracion_final,
                 coste_total, motivo_parada, inicios, asignacion=None):
        self.pasos = pasos
        self.duraciones = duraciones
        self.reducciones = reducciones
        self.demanda = demanda
        self.duracion_inicial = duracion_inicial
        self.duracion_final = duracion_final
        self.coste_total = coste_total
        self.motivo_parada = motivo_parada
        self.inicios = inicios
        self.asignacion = asignacion

    def __repr__(self):
        return (f"CrashingRecursos({self.duracion_inicial}→{self.duracion_final}, coste={self.coste_total:g}, "
                f"pasos={len(self.pasos)}, {self.motivo_parada})")


class MonteCarloRecursosResult:
    """Resultado de ``ProjectGraph.monte_carlo_recursos``.

    ``tau_libre_samples`` (CPM) y ``tau_recursos_samples`` (replanificada con los límites),
    ``criticidad`` (DataFrame actividades × {sin recursos, con recursos}), ``bloqueos`` (fracción de
    réplicas en que cada recurso hace esperar a alguna actividad).
    """

    def __init__(self, tau_libre, tau_recursos, criticidad, bloqueos, n_samples, seed):
        self.tau_libre_samples = np.asarray(tau_libre, dtype=float)
        self.tau_recursos_samples = np.asarray(tau_recursos, dtype=float)
        self.criticidad = criticidad
        self.bloqueos = bloqueos
        self.n_samples = n_samples
        self.seed = seed

    @property
    def retraso_samples(self):
        return self.tau_recursos_samples - self.tau_libre_samples

    def prob_superar(self, plazo, con_recursos=True):
        x = self.tau_recursos_samples if con_recursos else self.tau_libre_samples
        return float((x > plazo).mean())

    def summary(self):
        def fila(x):
            return {"media": x.mean(), "desv. típica": x.std(ddof=1), "q05": np.quantile(x, 0.05),
                    "q50": np.quantile(x, 0.5), "q95": np.quantile(x, 0.95)}
        return pd.DataFrame({"sin recursos": fila(self.tau_libre_samples),
                             "con recursos": fila(self.tau_recursos_samples),
                             "retraso por recursos": fila(self.retraso_samples)}).T

    def __repr__(self):
        return (f"MonteCarloRecursosResult(n={self.n_samples}, τ libre={self.tau_libre_samples.mean():.2f}, "
                f"τ con recursos={self.tau_recursos_samples.mean():.2f})")


def _van_cerrado(ES, D, B, tau, F, n_meses, tasa_anual):
    """(VAN de costes, VAN de ingresos) en forma cerrada, vectorizado por réplicas.

    Coste de cada actividad repartido uniformemente en su duración desde su inicio ES; ingresos F
    por mes durante ``n_meses`` desde τ; tasa mensual equivalente a la anual.
    """
    km = (1.0 + tasa_anual) ** (1.0 / 12.0) - 1.0
    v = 1.0 / (1.0 + km)
    serie = lambda inicio, n: v ** (inicio + 1.0) * (1.0 - v ** n) / (1.0 - v)
    coste = (B / D * serie(ES, D)).sum(axis=1)
    ingreso = F * serie(tau, float(n_meses))
    return coste, ingreso


class MonteCarloVANRecursosResult:
    """VAN simulado con y sin límite de recursos."""

    def __init__(self, van_libre, van_recursos, tau_libre, tau_recursos, n_samples, seed):
        self.van_libre_samples = np.asarray(van_libre, dtype=float)
        self.van_recursos_samples = np.asarray(van_recursos, dtype=float)
        self.tau_libre_samples = np.asarray(tau_libre, dtype=float)
        self.tau_recursos_samples = np.asarray(tau_recursos, dtype=float)
        self.n_samples = n_samples
        self.seed = seed

    @property
    def perdida_samples(self):
        return self.van_libre_samples - self.van_recursos_samples

    def prob_van_negativo(self, con_recursos=True):
        x = self.van_recursos_samples if con_recursos else self.van_libre_samples
        return float((x < 0).mean())

    def summary(self):
        def fila(x):
            return {"media": x.mean(), "desv. típica": x.std(ddof=1), "P10": np.quantile(x, .1),
                    "P50": np.quantile(x, .5), "P90": np.quantile(x, .9)}
        return pd.DataFrame({"VAN sin recursos": fila(self.van_libre_samples),
                             "VAN con recursos": fila(self.van_recursos_samples),
                             "pérdida por recursos": fila(self.perdida_samples)}).T

    def __repr__(self):
        return (f"MonteCarloVANRecursosResult(n={self.n_samples}, VAN libre={self.van_libre_samples.mean():.2f}, "
                f"VAN con recursos={self.van_recursos_samples.mean():.2f})")


class MonteCarloValorGanadoResult:
    """Previsiones de valor ganado frente a lo que de verdad ocurre (réplicas)."""

    def __init__(self, t_control, T_plan, BAC, pv, ev, ac, spi, cpi, tau, tau_pred, coste, eac_pred,
                 n_samples, seed):
        self.t_control, self.T_plan, self.BAC, self.pv = t_control, T_plan, BAC, pv
        self.ev, self.ac, self.spi, self.cpi = ev, ac, spi, cpi
        self.tau_samples, self.tau_pred_samples = tau, tau_pred
        self.coste_samples, self.eac_pred_samples = coste, eac_pred
        self.n_samples, self.seed = n_samples, seed

    def summary(self):
        def fila(x):
            x = x[np.isfinite(x)]
            return {"media": x.mean(), "desv. típica": x.std(ddof=1), "P10": np.quantile(x, .1),
                    "P90": np.quantile(x, .9)}
        return pd.DataFrame({"SPI": fila(self.spi), "CPI": fila(self.cpi),
                             "duración real": fila(self.tau_samples),
                             "duración prevista (T/SPI)": fila(self.tau_pred_samples),
                             "coste real": fila(self.coste_samples),
                             "coste previsto (BAC/CPI)": fila(self.eac_pred_samples)}).T

    def calidad_prevision(self):
        """Sesgo, error absoluto medio y correlación de las previsiones de duración y de coste."""
        def medidas(real, pred):
            m = np.isfinite(pred)
            e = pred[m] - real[m]
            c = np.corrcoef(real[m], pred[m])[0, 1] if m.sum() > 2 and pred[m].std() > 0 and real[m].std() > 0 else np.nan
            return {"sesgo": float(e.mean()), "error absoluto medio": float(np.abs(e).mean()),
                    "correlación": float(c)}
        return pd.DataFrame({"duración": medidas(self.tau_samples, self.tau_pred_samples),
                             "coste": medidas(self.coste_samples, self.eac_pred_samples)}).T

    def __repr__(self):
        return (f"MonteCarloValorGanadoResult(t={self.t_control}, n={self.n_samples}, "
                f"SPI medio={np.nanmean(self.spi):.3f})")


class MonteCarloValorGanadoRecursosResult:
    """SPI y previsión de duración con dos líneas base (sin y con recursos) y ejecución con límites."""

    def __init__(self, t_control, T_libre, T_recursos, pv_libre, pv_recursos, ev, spi_libre, spi_recursos,
                 tau, pred_libre, pred_recursos, n_samples, seed):
        self.t_control, self.T_libre, self.T_recursos = t_control, T_libre, T_recursos
        self.pv_libre, self.pv_recursos, self.ev = pv_libre, pv_recursos, ev
        self.spi_libre, self.spi_recursos = spi_libre, spi_recursos
        self.tau_samples = tau
        self.pred_libre_samples, self.pred_recursos_samples = pred_libre, pred_recursos
        self.n_samples, self.seed = n_samples, seed

    def calidad_prevision(self):
        """Sesgo, error absoluto medio y correlación de T/SPI con cada línea base."""
        def medidas(pred):
            m = np.isfinite(pred)
            e = pred[m] - self.tau_samples[m]
            c = np.corrcoef(self.tau_samples[m], pred[m])[0, 1] if pred[m].std() > 0 else np.nan
            return {"sesgo": float(e.mean()), "error absoluto medio": float(np.abs(e).mean()),
                    "correlación": float(c)}
        return pd.DataFrame({"línea base sin recursos": medidas(self.pred_libre_samples),
                             "línea base con recursos": medidas(self.pred_recursos_samples)}).T

    def summary(self):
        def fila(x):
            x = x[np.isfinite(x)]
            return {"media": x.mean(), "desv. típica": x.std(ddof=1), "P10": np.quantile(x, .1), "P90": np.quantile(x, .9)}
        return pd.DataFrame({"SPI sin recursos": fila(self.spi_libre), "SPI con recursos": fila(self.spi_recursos),
                             "duración real": fila(self.tau_samples),
                             "T/SPI sin recursos": fila(self.pred_libre_samples),
                             "T/SPI con recursos": fila(self.pred_recursos_samples)}).T

    def __repr__(self):
        return (f"MonteCarloValorGanadoRecursosResult(t={self.t_control}, n={self.n_samples}, "
                f"SPI sin recursos={np.nanmean(self.spi_libre):.3f}, con recursos={np.nanmean(self.spi_recursos):.3f})")


# ---------------------------------------------------------------------------
# MonteCarloResult
# ---------------------------------------------------------------------------

class MonteCarloResult:
    """Resultado de una simulación Monte Carlo sobre un ProjectGraph.

    Atributos
    ---------
    tau_samples : np.ndarray  shape (n_samples,)
        Valores de τ simulados.
    sensitivity_mean : pd.Series
        Sensibilidad empírica media (∂τ/∂tᵢ promediado sobre réplicas).
    n_samples : int
    seed : int | None

    Propiedades convenientes
    ------------------------
    tau_mean, tau_std, tau_q05, tau_q50, tau_q95 : float
    summary() : pd.DataFrame  con estadísticos de τ
    """

    def __init__(
        self,
        tau_samples: "np.ndarray",
        sensitivity_mean: "pd.Series",
        n_samples: int,
        seed,
    ):
        self.tau_samples = np.asarray(tau_samples, dtype=float)
        self.sensitivity_mean = sensitivity_mean
        self.n_samples = n_samples
        self.seed = seed

    # ── Estadísticos de τ ───────────────────────────────────────────

    @property
    def tau_mean(self) -> float:
        return float(np.mean(self.tau_samples))

    @property
    def tau_std(self) -> float:
        return float(np.std(self.tau_samples, ddof=1))

    @property
    def tau_q05(self) -> float:
        return float(np.quantile(self.tau_samples, 0.05))

    @property
    def tau_q50(self) -> float:
        return float(np.quantile(self.tau_samples, 0.50))

    @property
    def tau_q95(self) -> float:
        return float(np.quantile(self.tau_samples, 0.95))

    def summary(self) -> "pd.DataFrame":
        """DataFrame con estadísticos básicos de τ."""
        data = {
            "mean": self.tau_mean,
            "std":  self.tau_std,
            "q05":  self.tau_q05,
            "q50":  self.tau_q50,
            "q95":  self.tau_q95,
            "min":  float(self.tau_samples.min()),
            "max":  float(self.tau_samples.max()),
        }
        return pd.DataFrame(data, index=["τ"])

    def __repr__(self) -> str:
        return (
            f"MonteCarloResult(n={self.n_samples}, "
            f"τ̄={self.tau_mean:.3f}, σ={self.tau_std:.3f}, "
            f"[{self.tau_q05:.3f}, {self.tau_q95:.3f}])"
        )


class PSPLIBInstance:
    """Instancia PSPLIB leída desde un fichero .sm.

    Atributos
    ---------
    n_jobs : int           incluyendo source (1) y sink (n_jobs)
    horizon : int          cota superior del makespan (≠ duración proyecto)
    mpm_time : int         τ CPM (campo MPM-Time del fichero .sm)
    predecessors : dict    {str(jobnr): set of str(jobnr)}
    durations : dict       {str(jobnr): int}
    resources : dict       {str(jobnr): list[int]}
    resource_availabilities : list[int]
    """

    def __init__(self, n_jobs, horizon, mpm_time, predecessors,
                 durations, resources, resource_availabilities):
        self.n_jobs = n_jobs
        self.horizon = horizon
        self.mpm_time = mpm_time
        self.predecessors = predecessors
        self.durations = durations
        self.resources = resources
        self.resource_availabilities = resource_availabilities

    @classmethod
    def read(cls, path) -> "PSPLIBInstance":
        """Parsea un fichero .sm de PSPLIB sin dependencias externas."""
        import re as _re, pathlib as _pl
        text = _pl.Path(path).read_text(encoding="utf-8", errors="replace")
        sections = [s.strip() for s in _re.split(r"\*{3,}", text) if s.strip()]

        # Buscar secciones por nombre (robusto a variaciones de índice)
        def _find_section(keyword):
            for s in sections:
                if keyword.lower() in s.lower()[:80]:
                    return s
            return ""

        sec_header   = _find_section("jobs (incl")          # n_jobs, horizon
        sec_projinfo = _find_section("project information")  # MPM-Time
        sec_prec     = _find_section("precedence relations")
        sec_req      = _find_section("requests/durations")
        sec_res_av   = _find_section("resourceavailabilities")

        # n_jobs, horizon
        n_jobs = int(_re.search(
            r"jobs\s*\(incl[^)]+\)\s*:\s*(\d+)", sec_header).group(1))
        horizon = int(_re.search(r"horizon\s*:\s*(\d+)", sec_header).group(1))

        # mpm_time (último campo de la fila de datos, omitiendo la cabecera "pronr")
        proj_data = [l for l in sec_projinfo.splitlines()
                     if l.strip() and not l.strip().lower().startswith("pronr")
                     and not l.strip().lower().startswith("project")]
        mpm_time = int(proj_data[0].split()[-1]) if proj_data else 0

        # PRECEDENCE RELATIONS
        successors: dict = {}
        reading = False
        for line in sec_prec.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("jobnr"):
                reading = True
                continue
            if not reading:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                jobnr = int(parts[0])
            except ValueError:
                continue
            n_succ = int(parts[2])
            successors[jobnr] = [int(x) for x in parts[3: 3 + n_succ]]

        predecessors: dict = {str(j): set() for j in successors}
        for j, succs in successors.items():
            for s_job in succs:
                predecessors.setdefault(str(s_job), set()).add(str(j))

        # REQUESTS/DURATIONS
        durations: dict = {}
        resources: dict = {}
        reading = False
        for line in sec_req.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("-") or line.lower().startswith("jobnr"):
                reading = True
                continue
            if not reading:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                jobnr = int(parts[0])
            except ValueError:
                continue
            durations[str(jobnr)] = int(parts[2])
            resources[str(jobnr)] = [int(x) for x in parts[3:]]

        # RESOURCEAVAILABILITIES
        resource_availabilities: list = []
        for line in sec_res_av.splitlines():
            line = line.strip()
            if _re.match(r"^[\d\s]+$", line) and line:
                resource_availabilities = [int(x) for x in line.split()]
                break

        return cls(n_jobs=n_jobs, horizon=horizon, mpm_time=mpm_time,
                   predecessors=predecessors, durations=durations,
                   resources=resources,
                   resource_availabilities=resource_availabilities)


class EarnedValue:
    def __init__(self, pert):
        self.pert = pert

    def calcula_gantts(
        self,
        data,
        planned_durations_label,
        actual_durations_label,
        PV_label,
        AC_label,
        percentage_complete_label,
        tikz=False,
        params=None,
    ):
        my_data = data.copy()

        if any(my_data.loc[:, percentage_complete_label] > 1):
            my_data = my_data.astype({percentage_complete_label: "float"})
            my_data.loc[:, percentage_complete_label] = (
                my_data.loc[:, percentage_complete_label] / 100
            )

        my_data.loc[:, "PV_per_period"] = (
            my_data.loc[:, PV_label] / my_data.loc[:, planned_durations_label]
        )

        gantt_PV, dibujo_PV = self.pert.gantt(
            my_data,
            planned_durations_label,
            "PV_per_period",
            total="ambas",
            acumulado=True,
            tikz=True,
            params=params,
        )

        my_data.loc[:, "AC_per_period"] = (
            my_data.loc[:, AC_label] / my_data.loc[:, actual_durations_label]
        )

        if params is None:
            params = dict()
        params["critical_background_color"] = params.get(
            "regular_background_color", "white!90!green"
        )
        params["critical_text_color"] = params.get("regular_text_color", "black")
        gantt_AC, dibujo_AC = self.pert.gantt(
            my_data,
            actual_durations_label,
            "AC_per_period",
            total="ambas",
            acumulado=True,
            tikz=True,
            params=params,
        )

        my_data.loc[:, "EV"] = (
            my_data.loc[:, PV_label] * my_data.loc[:, percentage_complete_label]
        )
        my_data.loc[:, "EV_per_period"] = (
            my_data.loc[:, "EV"] / my_data.loc[:, actual_durations_label]
        )

        gantt_EV, dibujo_EV = self.pert.gantt(
            my_data,
            actual_durations_label,
            "EV_per_period",
            total="ambas",
            acumulado=True,
            tikz=True,
            params=params,
        )

        # Curvas S acumuladas. El horizonte es el de la ejecución real (las
        # duraciones reales describen el trabajo hasta la fecha de estado), de
        # modo que la última fila de `acumulados` es el punto de control y
        # PV/EV/AC en ella son los que se usan para CPI y SPI.  Si la ejecución
        # real sobrepasa el fin planificado, PV se prolonga plana en el BAC
        # (antes aparecía NaN).
        def _cumulative(gantt):
            total = gantt.data.loc["Total", :].drop("Total", errors="ignore")
            total = pd.to_numeric(total, errors="coerce").fillna(0.0)
            total.index = total.index.astype(int)
            return total.cumsum()

        pv, ev, ac = _cumulative(gantt_PV), _cumulative(gantt_EV), _cumulative(gantt_AC)
        horizon = ev.index                      # periodos hasta la fecha de estado
        acumulados = pd.DataFrame(dict(
            PV=pv.reindex(horizon).ffill(),
            EV=ev,
            AC=ac.reindex(horizon),
        ))

        return dict(
            Gantt_PV=gantt_PV,
            Gantt_AC=gantt_AC,
            Gantt_EV=gantt_EV,
            acumulados=acumulados,
            gantt_PV=dibujo_PV,
            gantt_AC=dibujo_AC,
            gantt_EV=dibujo_EV,
        )


def make_Roy(linkage_matrix):
    graph = nx.DiGraph()
    graph.add_nodes_from(["start", "finish"])

    enlaces_iniciales = [
        ("start", actividad)
        for actividad in linkage_matrix.index
        if not any(linkage_matrix.loc[actividad, :])
    ]

    graph.add_edges_from(enlaces_iniciales)

    enlaces_finales = [
        (actividad, "finish")
        for actividad in linkage_matrix.index
        if not any(linkage_matrix.loc[:, actividad])
    ]

    graph.add_edges_from(enlaces_finales)

    # Recorrido explícito: con pandas 3, ``df[df].stack()`` ya no descarta los
    # NaN y devolvía TODOS los pares (grafo completo con bucles).
    resto_de_enlaces = [
        (predecesora, actividad)
        for actividad in linkage_matrix.index
        for predecesora in linkage_matrix.columns
        if bool(linkage_matrix.loc[actividad, predecesora])
    ]
    graph.add_edges_from(resto_de_enlaces)
    return graph


# ----------------------------------------------------------------------
# SVD implícita de la matriz de rutas R (sin construir R) — octubre 2026
# R^T R = Σ_p 1_p 1_p^T se aplica a un vector con dos barridos de DP (O(aristas)),
# de modo que Lanczos da los σ_i, v_i dominantes aunque el nº de rutas sea
# exponencial; y τ_k = ‖R_k t‖∞ es el CPM con duraciones proyectadas V_k V_kᵀ t.
# Copia del módulo experiments/cluster_roy/implicit_svd.py (mantener ambos sincronizados).
# ----------------------------------------------------------------------
import scipy.sparse as _sp_impl
from scipy.sparse.linalg import splu as _splu, LinearOperator as _LinearOperator, eigsh as _eigsh


def _reduccion_transitiva_desde_ancestros(acts, ancestros):
    """Aristas (p, a) de la reduccion transitiva de un DAG dado por el conjunto de ANCESTROS de cada nodo.

    p es predecesora inmediata de a si es ancestro de a y no lo es de ningun otro ancestro de a.
    Con conjuntos de bits (enteros de Python) cuesta O(n * ancestros * n/64) en lugar de
    construir el cierre como grafo y reducirlo con networkx (varios segundos con n ~ 1 000).
    Las aristas salen ordenadas por la posicion de ``acts``.
    """
    ix = {a: i for i, a in enumerate(acts)}
    bit = {a: 1 << i for a, i in ix.items()}
    masc = {}
    for a in acts:
        m = 0
        for p in ancestros[a]:
            if p in bit: m |= bit[p]
        masc[a] = m
    aristas = []
    for a in acts:
        anc = [p for p in ancestros[a] if p in bit]
        if not anc: continue
        cubiertos = 0
        for q in anc: cubiertos |= masc[q]
        inmediatas = masc[a] & ~cubiertos
        aristas.extend((p, a) for p in sorted(anc, key=ix.__getitem__) if inmediatas & bit[p])
    return aristas


def _bloque_columnas(n, m, objetivo=1_500_000):
    """Numero de columnas por bloque para los barridos de DP: ~objetivo/n columnas (potencia de 2 entre 512 y 8192), de modo que
    EF, LS y T de un bloque quepan en la cache. Medido en AMD EPYC y en Apple M: con N = 1e5 réplicas acelera 6-8x
    frente a procesar la matriz entera y acota la memoria; con bloques de 256 columnas manda el overhead de Python."""
    b = max(512, min(8192, objetivo // max(n, 1))); b = 1 << int(round(np.log2(b)))
    return int(min(m, b))


def _indices_de_bits(mask: int) -> list:
    """Posiciones (de menor a mayor) de los bits activos de ``mask``, un entero de Python usado como conjunto de bits."""
    if not mask:
        return []
    octetos = np.frombuffer(mask.to_bytes((mask.bit_length() + 7) // 8, "little"), dtype=np.uint8)
    return np.flatnonzero(np.unpackbits(octetos, bitorder="little")).tolist()


def _bits_a_conjunto(mask: int, nombres) -> set:
    """Conjunto de ``nombres[i]`` para cada bit ``i`` activo en ``mask``."""
    return {nombres[i] for i in _indices_de_bits(mask)}


def _ef_inplace(pred, T, EF, row):
    """Pasada hacia delante EF[i] = max_{j en pred[i]} EF[j] + T[i] sin crear temporales (resultado identico bit a bit
    a EF[p].max(axis=0) + T[i]). T y EF (n, w); row (w,) es un auxiliar."""
    for i in range(len(pred)):
        p = pred[i]
        if p.size == 0: EF[i] = T[i]
        elif p.size == 1: np.add(EF[p[0]], T[i], out=EF[i])
        else:
            np.maximum(EF[p[0]], EF[p[1]], out=row)
            for q in p[2:]: np.maximum(row, EF[q], out=row)
            np.add(row, T[i], out=EF[i])


def _ls_inplace(succ, T, tau, LS, row):
    """Pasada hacia atras LS[i] = min_{j en succ[i]} LS[j] - T[i] (sumideros: tau - T[i]), sin temporales."""
    for i in range(len(succ) - 1, -1, -1):
        s = succ[i]
        if not s: np.subtract(tau, T[i], out=LS[i])
        elif len(s) == 1: np.subtract(LS[s[0]], T[i], out=LS[i])
        else:
            np.minimum(LS[s[0]], LS[s[1]], out=row)
            for q in s[2:]: np.minimum(row, LS[q], out=row)
            np.subtract(row, T[i], out=LS[i])


class RoyDAG:
    """DAG de Roy reducido con s y f implicitos. Las actividades se reordenan topologicamente (self.acts)."""

    def __init__(self, acts, edges):
        acts = list(acts); ix = {a: i for i, a in enumerate(acts)}; n = len(acts)
        succ = [[] for _ in range(n)]; indeg = np.zeros(n, int)
        for u, v in edges: succ[ix[u]].append(ix[v]); indeg[ix[v]] += 1
        order, stack, d = [], [i for i in range(n) if indeg[i] == 0], indeg.copy()
        while stack:
            u = stack.pop(); order.append(u)
            for v in succ[u]:
                d[v] -= 1
                if d[v] == 0: stack.append(v)
        if len(order) != n: raise ValueError("el grafo tiene ciclos")
        pos = {old: new for new, old in enumerate(order)}
        self.acts = [acts[o] for o in order]; self.perm = np.array(order); self.n = n
        rows = [pos[ix[u]] for u, v in edges]; cols = [pos[ix[v]] for u, v in edges]
        self.A = _sp_impl.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))        # A[u,v]=1 si u->v (triangular superior)
        self.pred = [[] for _ in range(n)]
        for r, c in zip(rows, cols): self.pred[c].append(r)
        self.pred = [np.array(p, int) for p in self.pred]
        self.src = (np.asarray(self.A.sum(axis=0)).ravel() == 0).astype(float)           # sin predecesoras: unidos a s
        self.snk = (np.asarray(self.A.sum(axis=1)).ravel() == 0).astype(float)           # sin sucesoras: unidos a f
        I = _sp_impl.identity(n, format="csc")
        self._LuF = _splu((I - self.A.T).tocsc(), permc_spec="NATURAL")                    # resuelve (I-A^T) y = r (hacia delante)
        self._LuB = _splu((I - self.A).tocsc(), permc_spec="NATURAL")                      # (I-A) y = r (hacia atras)
        self.a = self._LuF.solve(self.src); self.b = self._LuB.solve(self.snk)
        self.pathcount = self.a * self.b                                                    # diag(R^T R)
        self.n_paths = float(self.a[self.snk > 0].sum())

    def to_topo(self, x):  # vector en el orden original de `acts` -> orden topologico interno
        return np.asarray(x)[self.perm]

    def gram_matvec(self, X):
        """(R^T R) X para X de forma (n,) o (n,m)."""
        X = np.asarray(X, float); one = X.ndim == 1
        if one: X = X[:, None]
        a, b = self.a[:, None], self.b[:, None]
        F = self._LuF.solve(np.ascontiguousarray(X * a)); B = self._LuB.solve(np.ascontiguousarray(self.A @ (X * b)))
        Y = F * b + a * B
        return Y[:, 0] if one else Y

    def gram_dense(self):
        return self.gram_matvec(np.eye(self.n))

    def longest_path(self, w, return_path=False):
        """Ruta s->f mas larga con pesos de nodo w (pueden ser negativos). Devuelve la longitud (y la ruta)."""
        w = np.asarray(w, float); L = np.empty(self.n); arg = np.full(self.n, -1)
        for i in range(self.n):
            p = self.pred[i]
            if p.size: j = p[np.argmax(L[p])]; L[i] = L[j] + w[i]; arg[i] = j
            else: L[i] = w[i]
        ends = np.flatnonzero(self.snk); e = ends[np.argmax(L[ends])]
        if not return_path: return float(L[e])
        path = [e]
        while arg[path[-1]] >= 0: path.append(arg[path[-1]])
        return float(L[e]), path[::-1]

    @property
    def succ(self):
        """Sucesoras de cada actividad (listas de enteros, orden topologico); se calcula una vez."""
        if getattr(self, "_succ_cache", None) is None: self._succ_cache = [list(map(int, r)) for r in self.A.tolil().rows]
        return self._succ_cache

    def longest_path_batch(self, W, block=None):
        """Longitud de la ruta s->f mas larga para cada columna de W (n, m) (pesos de nodo, pueden ser negativos), en una sola
        pasada por columnas. Identico bit a bit a [longest_path(W[:, j]) for j in range(m)] y mucho mas rapido."""
        W = np.asarray(W, float); n, m = W.shape; out = np.empty(m); snk = np.flatnonzero(self.snk)
        blk = block or _bloque_columnas(n, m)
        for a in range(0, m, blk):
            b = min(m, a + blk); L = np.empty((n, b - a)); _ef_inplace(self.pred, W[:, a:b], L, np.empty(b - a)); out[a:b] = L[snk].max(axis=0)
        return out

    def critical_mask(self, T, block=None):
        """Criticidad por actividad para matrices de duraciones T (n, m) en orden topologico: holgura total ~ 0.
        Barridos in situ y por bloques de columnas (resultado identico bit a bit al calculo directo, con memoria acotada)."""
        T = np.asarray(T, float); n, m = T.shape; mask = np.empty((n, m), bool); tau = np.empty(m); snk = np.flatnonzero(self.snk)
        blk = block or _bloque_columnas(n, m); succ = self.succ
        for a in range(0, m, blk):
            b = min(m, a + blk); w = b - a; Tb = T[:, a:b]; row = np.empty(w); EF = np.empty((n, w)); _ef_inplace(self.pred, Tb, EF, row)
            tb = EF[snk].max(axis=0); LS = np.empty((n, w)); _ls_inplace(succ, Tb, tb, LS, row)
            np.add(LS, Tb, out=LS); np.subtract(LS, EF, out=LS)
            mask[:, a:b] = LS < 1e-9 * np.maximum(tb, 1.0); tau[a:b] = tb
        return mask, tau


def _eigsh_robust(op, n, kk, tol, rng):
    """eigsh 'LA'; si ARPACK falla (p. ej. 'no shifts could be applied', habitual con espectros casi nulos o muy degenerados)
    reintenta con mas vectores de Krylov y, en ultimo caso, diagonaliza el operador denso (n x n)."""
    for ncv in (None, min(n - 1, max(4 * kk + 20, 40))):
        try:
            return _eigsh(_LinearOperator((n, n), matvec=op, matmat=op, dtype=float), k=kk, which="LA", tol=tol, ncv=ncv, v0=rng.normal(size=n))
        except Exception:
            pass
    M = op(np.eye(n)); lam, V = np.linalg.eigh((M + M.T) / 2); return lam[-kk:], V[:, -kk:]


def _top_eigs_deflated(mv, n, k, tol=1e-10, max_pass=12, seed=0):
    """k autopares dominantes de un operador simetrico semidefinido positivo por Lanczos con DEFLACION.
    Lanczos de un solo vector ve una sola copia de cada autovalor repetido (frecuente en grafos simetricos);
    tras cada pasada se proyecta fuera lo ya hallado y se repite hasta que no aparece nada mayor que el k-esimo.
    Devuelve (lam, V, matvecs)."""
    rng = np.random.default_rng(seed); cnt = [0]; Vf = np.zeros((n, 0)); lf = np.zeros(0)
    def op(x):
        cnt[0] += 1; x = np.asarray(x, float)
        if Vf.shape[1]: x = x - Vf @ (Vf.T @ x)
        y = mv(x)
        return y - Vf @ (Vf.T @ y) if Vf.shape[1] else y
    for _ in range(max_pass):
        kk = min(k, n - 1 - Vf.shape[1])
        if kk < 1: break
        lam, V = _eigsh_robust(op, n, kk, tol, rng)
        keep = lam > 1e-12 * max(1.0, lam.max(), lf.max() if lf.size else 0.0)
        lam, V = lam[keep], V[:, keep]
        if lam.size == 0: break
        V = V - Vf @ (Vf.T @ V); V, _ = np.linalg.qr(V)                  # la deflacion es aproximada: reortogonalizar
        allL = np.concatenate([lf, lam]); allV = np.hstack([Vf, V]); o = np.argsort(-allL, kind="stable")[:k]
        changed = bool((o >= lf.size).any())                              # ¿entro algun autopar nuevo en los k primeros?
        Vf, lf = allV[:, o], allL[o]
        if not changed: break
    return lf[:k], Vf[:, :k], cnt[0]


def implicit_svd_roy(dag: RoyDAG, k=10, dense_max=400, tol=1e-10):
    """Primeros k valores singulares sigma y vectores V (n x k, orden topologico) de R.
    n <= dense_max: eigh de R^T R denso (se obtiene por DP, NO por R) -> todo el espectro; si no, Lanczos implicito con deflacion
    (recupera tambien los autovalores repetidos). v1 se devuelve con signo >= 0 (Perron-Frobenius). Devuelve (sigma, V, info)."""
    n = dag.n; info = {}
    if n <= dense_max or k >= n - 1:
        lam, V = np.linalg.eigh(dag.gram_dense()); o = np.argsort(-lam); lam, V = lam[o], V[:, o]; info["method"] = "dense"
        info["rank"] = int((lam > 1e-10 * lam[0]).sum()); lam, V = lam[:k], V[:, :k]
    else:
        lam, V, cnt = _top_eigs_deflated(dag.gram_matvec, n, k, tol=tol); info["method"] = "lanczos"; info["matvecs"] = cnt
    if V[:, 0].sum() < 0: V[:, 0] = -V[:, 0]
    return np.sqrt(np.clip(lam, 0, None)), V, info


def tau_rank_k_roy(dag: RoyDAG, t, V, ks=None):
    """tau_k = ||R_k t||_inf = ruta mas larga con duraciones proyectadas V_k V_k^T t (t en orden topologico).
    Devuelve {k: tau_k}. Para k=1 equivale a sigma1*max(u1)*(v1.t)."""
    t = np.asarray(t, float); ks = ks or range(1, V.shape[1] + 1); c = V.T @ t
    ks = list(ks); W = np.column_stack([V[:, :k] @ c[:k] for k in ks])      # todas las proyecciones, una sola pasada de DP
    return {k: float(v) for k, v in zip(ks, dag.longest_path_batch(W))}


class LatexArray(np.ndarray):
    # chatgpt dixit
    """
    A subclass of numpy.ndarray with a custom _repr_mimebundle_ method
    for LaTeX rendering.
    """

    def __new__(cls, input_array):
        # Create an instance of LatexArray
        obj = np.asarray(input_array).view(cls)
        return obj

    def _repr_mimebundle_(self, include=None, exclude=None):
        """
        Custom MIME bundle representation for Jupyter Notebook.
        Returns a LaTeX representation if the array is 2D.
        """
        if self.ndim != 2:
            return {}, None  # Fallback for non-2D arrays

        # Convert the array to LaTeX bmatrix
        rows = [" & ".join(map(str, row)) for row in self]
        latex_str = "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"

        return {"text/latex": f"${latex_str}$"}, None


def SVD(rutas):
    U, S, VT = np.linalg.svd(rutas, full_matrices=True)

    # Create a diagonal matrix from S (singular values)
    S = np.diag(S)

    # Pad the diagonal matrix to match the original dimensions of rutas
    m, n = rutas.shape
    if m > n:
        # If m > n, pad with zeros to make S_full m x m
        S = np.pad(S, ((0, m - n), (0, 0)), mode="constant")
    elif m < n:
        # If m < n, pad with zeros to make S_full n x n
        S = np.pad(S, ((0, 0), (0, n - m)), mode="constant")
    return {"U": U, "S": S, "VT": VT}


def pretty(x, dec=0, latex=False, **kwds):
    if latex:
        pretty_function = to_ltx
    else:
        pretty_function = to_jup

    if dec == 0:
        try:
            result = pretty_function(x, fmt="{:d}", **kwds)
        except:
            result = pretty_function(x, fmt=f"{{:.{dec}f}}", **kwds)
    else:
        result = pretty_function(x, fmt=f"{{:.{dec}f}}", **kwds)

    return result


def pretty_latex(x, dec=0, **kwds):
    return pretty(x, dec=dec, latex=True, **kwds)


def beautify(*args):
    def converter(x):
        return (
            pretty_latex(x)
            if isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame)
            else str(x)
        )

    return Math(" ".join([converter(arg) for arg in args]))


def tanto_por_uno(x):
    return np.abs(x) / np.abs(x).sum()


# ── Serialización de instancias (funciones de módulo) ──────────────────────

def build_and_save(instance_name: str, predecessors_dict: dict, out_dir) -> "ProjectGraph":
    """Construye el grafo PERT y guarda GraphML + path matrix (parquet).

    Parameters
    ----------
    instance_name : str
        Nombre base del fichero (sin extensión).
    predecessors_dict : dict
        Diccionario {actividad: set_de_predecesoras} o {actividad: str_predecesoras}.
    out_dir : str | Path
        Directorio de salida.

    Returns
    -------
    ProjectGraph con el grafo ya construido.
    """
    import pathlib as _pl
    out_dir = _pl.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Acepta dict de sets, listas o strings como valores de predecesores
    first_val = next(iter(predecessors_dict.values()), None)
    if isinstance(first_val, str):
        project = ProjectGraph.from_dict_of_strings(predecessors_dict)
    else:
        # Normalizar a sets por si llegan como listas
        preds_sets = {k: set(v) for k, v in predecessors_dict.items()}
        project = PredecessorTable.from_dict_of_sets(preds_sets).create_project()
    project.to_graphml(out_dir / f"{instance_name}.graphml")
    project.path_matrix().to_parquet(out_dir / f"{instance_name}_paths.parquet")
    return project


def load_instance(instance_name: str, out_dir) -> tuple:
    """Carga un ProjectGraph y su path matrix desde disco.

    Returns
    -------
    (ProjectGraph, pd.DataFrame con la path matrix)
    """
    import pathlib as _pl
    out_dir = _pl.Path(out_dir)
    graph = ProjectGraph(nx.read_graphml(out_dir / f"{instance_name}.graphml"))
    paths = pd.read_parquet(out_dir / f"{instance_name}_paths.parquet")
    return graph, paths


def ordena_rutas(rutas, importancia_actividades, importancia_rutas):
    def highlight_positive(val):
        color = "background-color: yellow" if val > 0 else ""
        return color

    rutas = rutas * importancia_actividades
    sorted_indices = np.flip(np.argsort(importancia_actividades))
    rutas = rutas.iloc[:, sorted_indices]

    rutas["Importancia"] = importancia_rutas
    rutas = rutas.sort_values("Importancia", ascending=False)

    rutas = rutas.style.map(
        lambda x: "" if pd.Series(x).name == "Importancia" else highlight_positive(x),
        subset=rutas.columns.drop("Importancia"),
    )

    return rutas


def highlight_blue_red(df_or_array):  # chatgpt dixit
    """
    Takes a DataFrame or a NumPy array and returns a styled DataFrame
    with:
    - A blue background for cells where the value is 1 or True.
    - A red background for cells where the value is -1 or False.

    Parameters:
            df_or_array (pd.DataFrame or np.ndarray): The input data.

    Returns:
            pd.io.formats.style.Styler: A styled DataFrame.
    """
    # Ensure the input is a DataFrame
    if isinstance(df_or_array, np.ndarray):
        df = pd.DataFrame(df_or_array)
    elif isinstance(df_or_array, pd.DataFrame):
        df = df_or_array
    else:
        raise ValueError("Input must be a pandas DataFrame or a NumPy array.")

    # Define the style function
    def style_cell(val):
        if val == 1 or val is True:
            return "background-color: blue; color: white;"
        elif val == -1 or val is False:
            return "background-color: red; color: white;"
        return ""

    # Apply the style to the DataFrame
    return df.style.map(style_cell)


def make_gantt_tikz(
    gantt_data,
    extra_rows=None,
    extra_cols=None,
    params=None,
    precedences=None,
):
    """Código TikZ de un diagrama de Gantt.

    precedences : lista de pares (predecesora, sucesora) entre actividades de
        ``gantt_data``. Se dibujan por defecto; ``params["arrows"] = False`` las omite.
        Parámetros de estilo: ``arrow_width`` (1pt), ``arrow_color``
        (black!60), ``critical_arrow_color`` (red!70!black, para pares críticos
        sin hueco), ``arrow_corner`` (radio de las esquinas, 1.5pt).
        ``fill_colors`` ({actividad: color}) cambia el relleno de barras concretas (p. ej., los
        colchones de la cadena crítica) y ``arrow_styles`` ({(predecesora, sucesora): estilo})
        añade un estilo TikZ a flechas concretas (p. ej., ``"dashed"``).
    """
    if params is None:
        params = dict()
    gantt_data["start"] = gantt_data["start"].astype(int)
    gantt_data["duration"] = gantt_data["duration"].astype(int)

    background_horizontal_line_color = params.get(
        "background_horizontal_line_color", "white!80!blue"
    )
    background_vertical_bars_color = params.get(
        "background_vertical_bars_color", "white!90!cyan"
    )
    row_height = params.get("row_height", 0.8)
    activity_relative_height = params.get("activity_relative_height", 0.7)
    period_width = params.get("period_width", 1)
    names_width = params.get("names_width", 2)
    extra_cols_width = params.get("extra_cols_width", 2)
    number_of_periods = params.get(
        "number_of_periods", (gantt_data["start"] + gantt_data["duration"]).max()
    )
    number_of_periods = int(number_of_periods)
    regular_background_color = params.get("regular_background_color", "white!90!green")
    critical_background_color = params.get("critical_background_color", "white!90!red")
    regular_text_color = params.get("regular_text_color", "black")
    critical_text_color = params.get("critical_text_color", "black")
    activity_inner_text_style = params.get(
        "activity_inner_text_style", r"\bfseries\large"
    )
    arrow_width = params.get("arrow_width", "1pt")
    arrows = params.get("arrows", True)
    arrow_color = params.get("arrow_color", "black!60")
    critical_arrow_color = params.get("critical_arrow_color", "red!70!black")
    arrow_corner = params.get("arrow_corner", "1.5pt")
    row_totals = params.get("row_totals", False)
    nan_string = params.get("nan_string", "---")
    inner_text = params.get("inner_text", None)
    inner_text_digits = params.get("inner_text_digits", 0)
    fill_colors = params.get("fill_colors", None) or {}      # {actividad: color de relleno}
    arrow_styles = params.get("arrow_styles", None) or {}    # {(predecesora, sucesora): estilo TikZ extra}

    activity_list = list(
        activity for activity in gantt_data.index if not activity.startswith("💤")
    )
    slide_activity = list(
        activity for activity in gantt_data.index if activity.startswith("💤")
    )
    slide_activity_dict = {
        activity: activity_list.index(activity.removeprefix("💤"))
        for activity in slide_activity
    }
    number_of_activities = len(activity_list)
    number_of_extra_rows = 0 if extra_rows is None else extra_rows.shape[0]
    number_of_extra_columns = 0 if extra_cols is None else extra_cols.shape[1]
    activity_box_height = number_of_activities * row_height
    activity_box_width = number_of_periods * period_width

    text = ""
    text += "\n" + r"\begin{tikzpicture}[>= latex, scale=1]"
    text += "\n" + r"% Cuadrícula"
    text += "\n" + r"%Barras verticales fondo"
    for xpos in range(1, number_of_periods, 2):
        text += (
            "\n"
            + f"\\fill[color={background_vertical_bars_color}] ( {xpos * period_width}cm, {row_height}cm) rectangle ++({period_width}cm,-{row_height * (1 + number_of_activities)}cm);"
        )

    text += "\n" + r"%Recuadro exterior"
    text += (
        "\n" + rf"\draw (0,0) rectangle ++({activity_box_width}cm,  {row_height}cm);"
    )
    text += (
        "\n"
        + rf"\draw (0,0) rectangle ++({activity_box_width}cm, -{activity_box_height}cm);"
    )

    text += "\n" + rf"%Columna de actividades"
    text += (
        "\n"
        + rf"\draw ({-names_width},0) rectangle ++({names_width},{row_height} )   node[pos=0.5, anchor=center, transform shape] {{Actividad}};"
    )
    for y, actividad in enumerate(activity_list):
        text += (
            "\n"
            + rf"\draw ({-names_width},{-(y + 1) * row_height}cm) rectangle ++({names_width},{row_height})   node[pos=0.5, anchor=center, transform shape] {{"
            + actividad
            + "};"
        )
        text += (
            "\n"
            + rf"  \draw[{background_horizontal_line_color},very thin] (0, {-(y + 1) * row_height}) -- ({activity_box_width}, {-(y + 1) * row_height});"
        )

    text += "\n" + r"%Columnas extra"
    for column_number in range(0, number_of_extra_columns):
        col_name = extra_cols.columns[column_number]
        text += (
            "\n"
            + rf"\draw ({activity_box_width + column_number * extra_cols_width},0) rectangle ++({extra_cols_width},{row_height} )   node[pos=0.5, anchor=center, transform shape] {{ {col_name} }};"
        )
        for y, activity in enumerate(activity_list):
            text += (
                "\n"
                + rf"\draw ({activity_box_width + column_number * extra_cols_width},{-(y + 1) * row_height}cm) rectangle ++({extra_cols_width},{row_height})   node[pos=0.5, anchor=center, transform shape] {{"
                + str(
                    truncate_and_pad(
                        extra_cols.loc[activity, col_name], inner_text_digits
                    )
                )
                + "};"
            )

    text += "\n" + r"%Fila de periodos"
    for x in range(1, number_of_periods + 1):
        text += (
            "\n"
            + rf"\draw ({(x - 0.5) * period_width},{0.5 * row_height}) node[transform shape, minimum width={period_width}, minimum height={row_height}] { {x} }; "
        )
    text += "\n" + r"% Fin de la cuadrícula principal"

    text += "\n" + r"%Filas extra"
    text += "\n" + r"%Barras verticales fondo"
    for xpos in range(1, number_of_periods, 2):
        text += (
            "\n"
            + rf"\fill[color={background_vertical_bars_color}] ( {xpos * period_width}cm, {-activity_box_height}cm ) rectangle ++({period_width}cm, {-row_height * number_of_extra_rows}cm);"
        )

    for row_number in range(0, number_of_extra_rows):
        row_name = extra_rows.index[row_number]
        text += "\n" + r"%Casilla del nombre de la fila"
        text += (
            "\n"
            + rf"\draw ({-names_width},-{(number_of_activities + (row_number + 1)) * row_height}) rectangle ++({names_width},{row_height})   node[pos=0.5, anchor=center, transform shape] {{ {str(row_name)} }};"
        )

        for x in range(number_of_periods):
            if x >= extra_rows.shape[1]:
                x_data = nan_string
            else:
                x_data = str(
                    truncate_and_pad(extra_rows.iloc[row_number, x], inner_text_digits)
                )
            text += (
                "\n"
                + rf"\draw ({x * period_width},{-activity_box_height - (row_number + 1) * row_height}) rectangle ++({period_width},{row_height}) node[pos=0.5, transform shape, minimum width={period_width}cm, minimum height={row_height}cm, anchor=center]  {{"
                + x_data
                + "};"
            )

        if row_totals:
            x_data = str(
                truncate_and_pad(extra_rows.iloc[row_number,].sum(), inner_text_digits)
            )
            text += (
                "\n"
                + rf"\draw ({activity_box_width},{-activity_box_height - (row_number + 1) * row_height}) rectangle ++({extra_cols_width},{row_height}) node[pos=0.5, transform shape, minimum width={period_width}cm, minimum height={row_height}cm, anchor=center]  {{"
                + x_data
                + "};"
            )

    text += "\n" + r"%Fin retícula adicional inferior"

    if arrows and precedences:
        text += "\n" + r"%Flechas de prelación (debajo de las barras)"
        row_of = {a: i for i, a in enumerate(activity_list)}
        row_of.update(slide_activity_dict)          # 💤X comparte fila con X
        half_bar = row_height * activity_relative_height / 2
        delta = 0.15 * period_width                # entrada vertical: desplazamiento desde la esquina
        for pred, succ in precedences:
            if pred not in row_of or succ not in row_of or row_of[pred] == row_of[succ]:
                continue
            sp, dp = gantt_data.loc[pred, "start"], gantt_data.loc[pred, "duration"]
            ss = gantt_data.loc[succ, "start"]
            if any(pd.isna(v) for v in (sp, dp, ss)):
                continue
            x_end = (sp + dp) * period_width
            x_start = ss * period_width
            y_p = (-row_of[pred] - 0.5) * row_height
            y_s = (-row_of[succ] - 0.5) * row_height
            gap = x_start - x_end
            is_critical = (gantt_data.loc[pred, "Htotal"] == 0
                           and gantt_data.loc[succ, "Htotal"] == 0 and gap <= 1e-9)
            color = critical_arrow_color if is_critical else arrow_color
            style = (f"->, {color}, line width={arrow_width}, "
                     f"rounded corners={arrow_corner}, shorten >=1.5pt")
            if (pred, succ) in arrow_styles:
                style += ", " + arrow_styles[(pred, succ)]
            if gap > 1e-9:
                # codo en el hueco: sale por la derecha, baja/sube, entra por la izquierda
                x_mid = x_end + min(gap, period_width) / 2
                text += ("\n" + rf"\draw[{style}] ({x_end:.3f},{y_p:.3f}) -- "
                         rf"({x_mid:.3f},{y_p:.3f}) -- ({x_mid:.3f},{y_s:.3f}) -- ({x_start:.3f},{y_s:.3f});")
            else:
                # sin hueco (o solape): entra por el borde superior/inferior de la sucesora
                x_in = x_start + delta
                y_edge = y_s + half_bar if y_s < y_p else y_s - half_bar
                text += ("\n" + rf"\draw[{style}] ({x_end:.3f},{y_p:.3f}) -- "
                         rf"({x_in:.3f},{y_p:.3f}) -- ({x_in:.3f},{y_edge:.3f});")

    text += "%Dibujo de las actividades"
    for y, actividad in enumerate(activity_list):
        background_color = (
            critical_background_color
            if gantt_data.loc[actividad, "Htotal"] == 0
            else regular_background_color
        )
        background_color = fill_colors.get(actividad, background_color)
        text_color = (
            critical_text_color
            if gantt_data.loc[actividad, "Htotal"] == 0
            else regular_text_color
        )
        start = gantt_data.loc[actividad, "start"]
        duration = gantt_data.loc[actividad, "duration"]
        name = actividad
        if math.isnan(start) or math.isnan(duration):
            continue

        text += (
            "\n"
            + rf"""\node[draw=black, fill={background_color}, inner sep=0pt, outer sep=0pt, anchor=south west, minimum width={duration * period_width}cm, minimum height={row_height * activity_relative_height}cm, font="""
            + activity_inner_text_style
            + rf""", transform shape, text= {text_color} ] ( {name} ) at ( {start * period_width}, {(-y - 1 + (1 - activity_relative_height) / 2) * row_height} ) {{  \strut }};"""
        )

        if inner_text is not None:
            for idx, x in enumerate(range(start, start + duration)):
                texto = r"\strut"
                if isinstance(inner_text, str):
                    texto = gantt_data.loc[actividad, inner_text]
                    if pd.isna(texto):
                        texto = nan_string
                    elif isinstance(texto, (int, float)):
                        texto = float(texto)
                        texto = truncate_and_pad(texto, inner_text_digits)
                else:
                    if duration == len(inner_text.loc[name, "data"]):
                        texto = truncate_and_pad(
                            inner_text.loc[name, "data"][idx], inner_text_digits
                        )

                text += (
                    "\n"
                    + rf"""\node[anchor=center, inner sep=0pt, outer sep=0pt, minimum width={period_width}cm, minimum height={row_height * activity_relative_height}cm, font="""
                    + activity_inner_text_style
                    + rf""", transform shape, text= {text_color} ] 
				at ( {(x + 0.5) * period_width}, {(-y - 0.5) * row_height} ) {{ {texto} }};"""
                )
    text += "%Dibujo de los desplazamientos"
    for actividad, y in slide_activity_dict.items():
        # print(actividad, y)
        background_color = (
            critical_background_color
            if gantt_data.loc[actividad, "Htotal"] == 0
            else regular_background_color
        )
        text_color = (
            critical_text_color
            if gantt_data.loc[actividad, "Htotal"] == 0
            else regular_text_color
        )
        start = gantt_data.loc[actividad, "start"]
        duration = gantt_data.loc[actividad, "duration"]
        name = actividad
        if math.isnan(start) or math.isnan(duration):
            print("NaN found")
            continue

        text += (
            "\n"
            + rf"""\node[draw=black, fill={background_color}, pattern=north east lines,inner sep=0pt, outer sep=0pt, anchor=south west, minimum width={duration * period_width}cm, minimum height={row_height * activity_relative_height}cm, font="""
            + activity_inner_text_style
            + rf""", transform shape, text= {text_color} ] ( {name} ) at ( {start * period_width}, {(-y - 1 + (1 - activity_relative_height) / 2) * row_height} ) {{  \strut }};"""
        )

    text += "\n" + r"\end{tikzpicture}"
    return text



def _ancho_etiqueta(texto: str) -> float:
    """Ancho aproximado en cm de una etiqueta en ``\\scriptsize``."""
    ancho = {"@": 0.17, "→": 0.17, " ": 0.07, "(": 0.07, ")": 0.07}
    return sum(ancho.get(c, 0.115) for c in texto) + 0.15


def _muestras_ruta(pts: list, curva: bool, n: int = 24) -> list:
    """Puntos de la ruta de una arista: recta entre extremos o curvas de Bézier
    con tangente horizontal en cada punto (como las que dibuja ``_ruta_tikz``)."""
    if not curva:
        return [pts[0], pts[-1]]
    salida = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        dx = (x2 - x1) * 0.45
        for k in range(n + 1):
            t = k / n
            a, b, c, d = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3
            salida.append((a * x1 + b * (x1 + dx) + c * (x2 - dx) + d * x2, a * y1 + b * y1 + c * y2 + d * y2))
    return salida


def _disposicion_capas(nodos: dict, aristas: list, anchos_etiqueta: dict = None,
                       separacion: float = 0.35, hueco: float = 0.5) -> dict:
    """Disposición de un grafo acíclico por capas, de izquierda a derecha, pensada
    para dibujarlo con TikZ sin que se pisen sus elementos.

    1. Capas: la de cada nodo es la longitud del camino más largo que llega a él.
    2. Orden dentro de cada capa: las aristas que saltan capas se parten con nodos
       virtuales y se minimiza el número de cruces (baricentros alternando
       sentidos y trasposiciones de vecinos), conservando el mejor orden.
    3. Alturas: mínimos cuadrados respecto a la media de los vecinos con la
       separación mínima entre nodos contiguos (regresión isótona), de modo que
       las aristas largas quedan casi rectas y esquivan los nodos de las capas
       intermedias por su nodo virtual.
    4. Rutas: rectas entre capas contiguas y curvas por los nodos virtuales en las
       largas; si una recta atraviesa un nodo se cambia por una curva.
    5. Nombres: sobre su flecha, en el punto de la ruta (probando varios) que
       menos pisa nodos, otros nombres y otras flechas.

    ``nodos``: ``{id: (ancho_cm, alto_cm)}``; ``aristas``: ``[(u, v)]``;
    ``anchos_etiqueta``: ``{(u, v): ancho_cm}`` (opcional). Devuelve ``{"pos", "rutas"
    (puntos intermedios), "curva", "etiquetas"}`` con coordenadas en cm.
    """
    import itertools
    anchos_etiqueta = anchos_etiqueta or {}
    ids = list(nodos)
    preds = {n: [] for n in ids}
    succs = {n: [] for n in ids}
    for u, v in aristas:
        succs[u].append(v)
        preds[v].append(u)
    # 1. capas
    grado = {n: len(preds[n]) for n in ids}
    cola, orden = [n for n in ids if grado[n] == 0], []
    while cola:
        n = cola.pop(0)
        orden.append(n)
        for m in succs[n]:
            grado[m] -= 1
            if grado[m] == 0:
                cola.append(m)
    if len(orden) != len(ids):
        raise ValueError("el grafo tiene ciclos")
    capa = {}
    for n in orden:
        capa[n] = max((capa[q] + 1 for q in preds[n]), default=0)
    # 2. nodos virtuales y orden
    L, tam = dict(capa), dict(nodos)
    ep = {n: [] for n in ids}
    es = {n: [] for n in ids}
    cadenas = {}
    for u, v in aristas:
        cur, virt = u, []
        for j in range(1, capa[v] - capa[u]):
            w = ("v", u, v, j)
            L[w], tam[w] = capa[u] + j, (0.0, 0.25)
            ep[w], es[w] = [], []
            es[cur].append(w)
            ep[w].append(cur)
            cur = w
            virt.append(w)
        es[cur].append(v)
        ep[v].append(cur)
        cadenas[(u, v)] = virt
    ncapas = max(L.values()) + 1
    por_capa = [[] for _ in range(ncapas)]
    for n in L:
        por_capa[L[n]].append(n)
    idx = {}

    def reindexa(c):
        for i, n in enumerate(por_capa[c]):
            idx[n] = i

    for c in range(ncapas):
        reindexa(c)

    def cruces_entre(c):
        ar = [(idx[a], idx[b]) for a in por_capa[c] for b in es[a]]
        return sum(1 for (a1, b1), (a2, b2) in itertools.combinations(ar, 2) if (a1 - a2) * (b1 - b2) < 0)

    def cruces():
        return sum(cruces_entre(c) for c in range(ncapas - 1))

    def barrido(rango, vecinos):
        for c in rango:
            clave = {n: (sum(idx[q] for q in vecinos[n]) / len(vecinos[n]) if vecinos[n] else idx[n])
                     for n in por_capa[c]}
            por_capa[c].sort(key=lambda n: (clave[n], idx[n]))
            reindexa(c)

    mejor = (cruces(), [list(c) for c in por_capa])
    for _ in range(24):
        barrido(range(1, ncapas), ep)
        barrido(range(ncapas - 2, -1, -1), es)
        mejora = True
        while mejora:                                   # trasposición de vecinos
            mejora = False
            for c in range(ncapas):
                for i in range(len(por_capa[c]) - 1):
                    antes = cruces()
                    por_capa[c][i], por_capa[c][i + 1] = por_capa[c][i + 1], por_capa[c][i]
                    reindexa(c)
                    if cruces() < antes:
                        mejora = True
                    else:
                        por_capa[c][i], por_capa[c][i + 1] = por_capa[c][i + 1], por_capa[c][i]
                        reindexa(c)
        total = cruces()
        if total < mejor[0]:
            mejor = (total, [list(c) for c in por_capa])
        if total == 0:
            break
    por_capa = [list(c) for c in mejor[1]]
    for c in range(ncapas):
        reindexa(c)

    # 3. alturas (la fila 0 arriba: d crece hacia abajo)
    def separar(c, deseado):
        nodos_c = por_capa[c]
        off, acum = [0.0], 0.0
        for a, b in zip(nodos_c, nodos_c[1:]):
            acum += (tam[a][1] + tam[b][1]) / 2 + separacion
            off.append(acum)
        bloques = []                                    # PAV sobre z = d - offset
        for i, n in enumerate(nodos_c):
            bloques.append([deseado[n] - off[i], 1, i, i])
            while len(bloques) > 1 and bloques[-2][0] > bloques[-1][0]:
                b2, b1 = bloques.pop(), bloques.pop()
                bloques.append([(b1[0] * b1[1] + b2[0] * b2[1]) / (b1[1] + b2[1]), b1[1] + b2[1], b1[2], b2[3]])
        d = {}
        for z, _, i0, i1 in bloques:
            for i in range(i0, i1 + 1):
                d[nodos_c[i]] = z + off[i]
        return d

    d = {}
    for c in range(ncapas):
        ini = {n: float(i) for i, n in enumerate(por_capa[c])}
        d.update(separar(c, ini))
    for _ in range(10):
        for c in range(1, ncapas):
            d.update(separar(c, {n: (sum(d[q] for q in ep[n]) / len(ep[n]) if ep[n] else d[n])
                                 for n in por_capa[c]}))
        for c in range(ncapas - 2, -1, -1):
            d.update(separar(c, {n: (sum(d[q] for q in es[n]) / len(es[n]) if es[n] else d[n])
                                 for n in por_capa[c]}))
    # 3b. posiciones x
    ancho_capa = [max([tam[n][0] for n in por_capa[c] if not isinstance(n, tuple)] or [0.0])
                  for c in range(ncapas)]
    x, xs = 0.0, []
    for c in range(ncapas):
        if c:
            ancho_h = max([anchos_etiqueta.get((u, v), 0.0) for (u, v) in aristas
                           if capa[u] == c - 1 and capa[v] == c] or [0.0])
            x += ancho_capa[c - 1] / 2 + ancho_capa[c] / 2 + max(0.9, ancho_h + hueco)
        xs.append(x)
    pos_ext = {n: (xs[L[n]], -d[n]) for n in L}
    pos = {n: pos_ext[n] for n in ids}

    # 4. rutas
    rects = {n: (pos[n][0], pos[n][1], nodos[n][0] / 2 + 0.06, nodos[n][1] / 2 + 0.06) for n in ids}

    def dentro(p, r):
        return abs(p[0] - r[0]) < r[2] and abs(p[1] - r[1]) < r[3]

    def choca(clave, curva):
        u, v = clave
        pts = [pos[u]] + [pos_ext[w] for w in cadenas[clave]] + [pos[v]]
        for q in _muestras_ruta(pts, curva, 12) if curva else \
                [(pts[0][0] + (pts[-1][0] - pts[0][0]) * k / 40, pts[0][1] + (pts[-1][1] - pts[0][1]) * k / 40)
                 for k in range(41)]:
            for n, r in rects.items():
                if n not in clave and dentro(q, r):
                    return True
        return False

    curva = {}
    rutas = {}
    for clave in aristas:
        virt = cadenas[clave]
        rutas[clave] = [pos_ext[w] for w in virt]
        curva[clave] = bool(virt) or choca(clave, False)

    # 5. nombres
    muestras = {}
    for clave in aristas:
        u, v = clave
        muestras[clave] = _muestras_ruta([pos[u]] + rutas[clave] + [pos[v]], curva[clave])
    poli = {}
    for clave, pts in muestras.items():                # polilíneas densas (misma ruta)
        densa = []
        for a, b in zip(pts, pts[1:]):
            m = max(1, int(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 / 0.08))
            densa += [(a[0] + (b[0] - a[0]) * k / m, a[1] + (b[1] - a[1]) * k / m) for k in range(m)]
        densa.append(pts[-1])
        poli[clave] = densa
    etiquetas, ocupadas = {}, []
    for clave in sorted(anchos_etiqueta, key=lambda c: -len(poli.get(c, []))):
        if clave not in poli:
            continue
        w, h = anchos_etiqueta[clave] / 2 + 0.03, 0.17
        u, v = clave
        densa = [q for q in poli[clave] if not dentro(q, rects[u]) and not dentro(q, rects[v])] or poli[clave]
        mejor_c = None
        candidatos = [(0.5 + sg * 0.04 * k, dy) for k in range(0, 12) for sg in ((1,) if k == 0 else (1, -1))
                      for dy in (0.0, 0.2, -0.2, 0.35, -0.35)]
        for t, dy in candidatos:
            px, py = densa[min(len(densa) - 1, max(0, int(t * (len(densa) - 1))))]
            py += dy
            pen = abs(t - 0.5) + (0.5 if dy else 0.0)
            for n, r in rects.items():
                if abs(px - r[0]) < w + r[2] + 0.15 and abs(py - r[1]) < h + r[3] + 0.05:
                    pen += 100
            for (qx, qy, qw, qh) in ocupadas:
                if abs(px - qx) < w + qw and abs(py - qy) < h + qh:
                    pen += 50
            for otra, dens in poli.items():
                if otra != clave:
                    pen += 6 * sum(1 for (ax, ay) in dens[::2] if abs(px - ax) < w and abs(py - ay) < h)
            if mejor_c is None or pen < mejor_c[0]:
                mejor_c = (pen, px, py)
        etiquetas[clave] = (mejor_c[1], mejor_c[2])
        ocupadas.append((mejor_c[1], mejor_c[2], w, h))
    return dict(pos=pos, rutas=rutas, curva=curva, etiquetas=etiquetas)


# Colores de las flechas que intervienen en un cálculo (uno por flecha); el naranja queda
# para el suceso o nodo que se calcula y el rojo para lo crítico.
PALETA_FLECHAS = ["blue!80!black", "violet", "teal", "magenta!85!black", "brown", "cyan!70!black"]


def _html_pasos(fotos: list, inicio: int = 1) -> str:
    """Une fotogramas HTML en un visor autocontenido con botones
    (anterior / siguiente) para mostrar un cálculo paso a paso en un cuaderno
    o en el HTML de Quarto. El identificador se deriva del contenido para que
    el resultado sea determinista. ``inicio`` es el número que lleva el primer
    fotograma en el contador (por defecto 1; el último muestra ``n + inicio - 1``)."""
    import hashlib
    uid = "pmp" + hashlib.md5("".join(fotos).encode("utf-8")).hexdigest()[:10]
    n = len(fotos)
    divs = "".join(
        f'<div class="pmule-paso" style="display:{"block" if k == 0 else "none"}">{c}</div>'
        for k, c in enumerate(fotos))
    boton = ('style="margin:0 .2em;padding:.1em .7em;border:1px solid #888;'
             'border-radius:3px;background:#f4f4f4;cursor:pointer"')
    return (
        f'<div id="{uid}" class="pmule-pasos" style="margin:1em 0">'
        f'<div style="text-align:center;margin-bottom:.5em">'
        f'<button {boton} onclick="pmulePaso(\'{uid}\',-{n})">«</button>'
        f'<button {boton} onclick="pmulePaso(\'{uid}\',-1)">‹ anterior</button>'
        f'<span class="pmule-contador" data-inicio="{inicio}" title="También con las flechas ← → del teclado" style="margin:0 .8em">paso {inicio} de {n + inicio - 1}</span>'
        f'<button {boton} onclick="pmulePaso(\'{uid}\',1)">siguiente ›</button>'
        f'<button {boton} onclick="pmulePaso(\'{uid}\',{n})">»</button>'
        f'<button {boton} title="Ampliar a pantalla completa (Esc para cerrar)" '
        f'onclick="pmulePantalla(\'{uid}\')">⛶</button>'
        f'</div>{divs}</div>'
        '<script>window.pmulePaso = window.pmulePaso || function(id, d) {'
        'var c = document.getElementById(id), p = c.querySelectorAll(".pmule-paso"), k = 0;'
        'for (var i = 0; i < p.length; i++) { if (p[i].style.display !== "none") k = i; p[i].style.display = "none"; }'
        'k = Math.max(0, Math.min(p.length - 1, k + d)); p[k].style.display = "block";'
        'var e = c.querySelector(".pmule-contador"), o = parseInt(e.getAttribute("data-inicio") || "1", 10); e.textContent = "paso " + (k + o) + " de " + (p.length + o - 1); };'
        'window.pmulePantalla = window.pmulePantalla || function(id) {'
        'var c = document.getElementById(id), abierto = c.getAttribute("data-pantalla") === "1";'
        'var imgs = c.querySelectorAll("img");'
        'if (abierto) { c.removeAttribute("data-pantalla"); c.style.cssText = c.getAttribute("data-estilo");'
        'for (var i = 0; i < imgs.length; i++) imgs[i].style.cssText = imgs[i].getAttribute("data-estilo");'
        'document.body.style.overflow = c.getAttribute("data-overflow") || ""; return; }'
        'c.setAttribute("data-pantalla", "1"); c.setAttribute("data-estilo", c.style.cssText);'
        'c.setAttribute("data-overflow", document.body.style.overflow);'
        'c.style.cssText = "position:fixed;top:0;left:0;right:0;bottom:0;z-index:99999;background:#fff;margin:0;padding:.8em;overflow:auto";'
        'document.body.style.overflow = "hidden";'
        'for (var i = 0; i < imgs.length; i++) { imgs[i].setAttribute("data-estilo", imgs[i].style.cssText);'
        'imgs[i].style.cssText = "display:block;margin:0 auto;width:100%;height:calc(100vh - 5em);object-fit:contain"; } };'
        'if (!window.pmuleTeclas) { window.pmuleTeclas = true; window.pmuleActivo = null;'
        'var marca = function(e) { var c = e.target.closest ? e.target.closest(".pmule-pasos") : null; if (c) window.pmuleActivo = c.id; };'
        'document.addEventListener("mouseover", marca); document.addEventListener("click", marca); document.addEventListener("focusin", marca);'
        'var visible = function(c) { var r = c.getBoundingClientRect(), h = window.innerHeight || document.documentElement.clientHeight;'
        'return Math.max(0, Math.min(r.bottom, h) - Math.max(r.top, 0)); };'
        'document.addEventListener("keydown", function(e) {'
        'if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;'
        'var t = e.target; if (t && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable)) return;'
        'var c = document.querySelector(".pmule-pasos[data-pantalla=\'1\']");'
        'if (c && e.key === "Escape") { window.pmulePantalla(c.id); return; }'
        'if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;'
        'if (!c) { c = window.pmuleActivo ? document.getElementById(window.pmuleActivo) : null;'
        'if (!c || visible(c) <= 0) { c = null; var mejor = 0, todos = document.querySelectorAll(".pmule-pasos");'
        'for (var i = 0; i < todos.length; i++) { var v = visible(todos[i]); if (v > mejor) { mejor = v; c = todos[i]; } } } }'
        'if (!c) return; e.preventDefault(); window.pmulePaso(c.id, e.key === "ArrowRight" ? 1 : -1); }); }</script>')

def _svg_ancho_columna(svg):
    """Quita ``width``/``height`` fijos de la raíz del SVG (conserva ``viewBox``)
    para que el navegador lo escale al ancho disponible de la columna."""
    import re
    def raiz(m):
        t = re.sub(r'\s(?:width|height)="[^"]*"', '', m.group(0))
        return t
    return re.sub(r'<svg\b[^>]*>', raiz, svg, count=1)


def _preambulo_tikz(librerias: str) -> str:
    return ("\\documentclass[border=4pt,multi=tikzpicture]{standalone}\n"
            "\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{amsmath}\n"
            "\\usepackage{tikz}\n" + f"\\usetikzlibrary{{{librerias}}}\n")


def _jobs_tikz() -> int:
    """Procesos de LaTeX en paralelo: ``PMULE_TIKZ_JOBS`` o las CPU disponibles (como mucho 8)."""
    import os
    try:
        return max(1, int(os.environ["PMULE_TIKZ_JOBS"]))
    except (KeyError, ValueError):
        pass
    try:
        n = len(os.sched_getaffinity(0))          # respeta las CPU asignadas por SLURM o el contenedor
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, 8))


def _cache_tikz():
    """Carpeta de la caché de fotogramas compilados (``PMULE_TIKZ_CACHE``; ``0`` o vacío la desactiva).
    Por defecto ``~/.cache/pmule/tikz``. Se puede borrar sin más."""
    import os
    from pathlib import Path
    d = os.environ.get("PMULE_TIKZ_CACHE")
    if d is not None and d.strip() in ("", "0"):
        return None
    if d:
        return Path(d)
    try:
        return Path.home() / ".cache" / "pmule" / "tikz"
    except RuntimeError:                           # sin HOME (algunos contenedores): sin caché
        return None


@lru_cache(maxsize=None)
def _version_latex() -> str:
    """Versiones de pdflatex y pdftocairo: parte de la clave de la caché, para que un cambio de LaTeX la invalide."""
    import subprocess
    v = []
    for cmd in (["pdflatex", "--version"], ["pdftocairo", "-v"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        v.append((r.stdout + r.stderr).strip().splitlines()[0] if (r.stdout + r.stderr).strip() else "?")
    return " | ".join(v)


def _svgs_de_tikz(codigos: list, librerias: str = "calc,arrows.meta", guardar: str = None) -> list:
    """Compila una lista de ``tikzpicture`` (un fotograma cada uno) y devuelve un
    SVG por fotograma, recortado al dibujo. Necesita ``pdflatex`` y ``pdftocairo``.

    El resultado de cada fotograma no depende de los demás (``standalone`` con ``multi=tikzpicture``
    recorta cada uno por separado), así que se puede acelerar sin cambiar ni un byte de la salida:
    - caché en disco por hash (preámbulo + versión de LaTeX + código): un fotograma ya compilado no se
      vuelve a compilar (``PMULE_TIKZ_CACHE``, ver ``_cache_tikz``);
    - los que faltan se reparten en trozos de al menos 6 fotogramas que se compilan en paralelo
      (``PMULE_TIKZ_JOBS``, por defecto las CPU disponibles, como mucho 8).
    Con ``guardar`` se compila todo de una vez, como siempre (hace falta el PDF entero)."""
    import gzip, hashlib, os, shutil
    from concurrent.futures import ThreadPoolExecutor
    for exe in ("pdflatex", "pdftocairo"):
        if shutil.which(exe) is None:
            raise FileNotFoundError(f"no se encuentra {exe}: hace falta LaTeX para los fotogramas TikZ")
    if guardar:
        return _compila_tikz(codigos, librerias, guardar)
    cache = _cache_tikz()
    base = (_preambulo_tikz(librerias) + _version_latex()).encode("utf-8")
    claves = [hashlib.sha256(base + c.encode("utf-8")).hexdigest() for c in codigos]
    svg_de = {}
    if cache is not None:
        for k in set(claves):
            try:
                svg_de[k] = gzip.decompress((cache / f"{k}.svg.gz").read_bytes()).decode("utf-8")
            except (OSError, EOFError, ValueError):
                pass
    faltan = {}                                   # clave -> código, sin repetidos y en orden
    for k, c in zip(claves, codigos):
        if k not in svg_de:
            faltan.setdefault(k, c)
    if faltan:
        ks = list(faltan)
        trozos = max(1, min(_jobs_tikz(), len(ks) // 6))
        tam = -(-len(ks) // trozos)
        partes = [ks[i:i + tam] for i in range(0, len(ks), tam)]
        def compila(parte):
            return _compila_tikz([faltan[k] for k in parte], librerias)
        if len(partes) == 1:
            resultados = [compila(partes[0])]
        else:
            with ThreadPoolExecutor(len(partes)) as ex:
                resultados = list(ex.map(compila, partes))
        for parte, svgs in zip(partes, resultados):
            for k, svg in zip(parte, svgs):
                svg_de[k] = svg
                if cache is not None:
                    try:                          # escritura atómica: varios procesos pueden compartir la caché
                        cache.mkdir(parents=True, exist_ok=True)
                        tmp = cache / f".{k}.{os.getpid()}.tmp"
                        tmp.write_bytes(gzip.compress(svg.encode("utf-8"), 6))
                        os.replace(tmp, cache / f"{k}.svg.gz")
                    except OSError:
                        pass
    return [svg_de[k] for k in claves]


def _compila_tikz(codigos: list, librerias: str = "calc,arrows.meta", guardar: str = None) -> list:
    """Un solo ``pdflatex`` para todos los fotogramas y un ``pdftocairo`` por página. Con ``guardar``
    copia además la primera figura (``.pdf``, ``.svg`` o ``.png``)."""
    import shutil, subprocess, tempfile
    from pathlib import Path
    paginas = "\n".join(codigos)
    n = len(codigos)
    doc = _preambulo_tikz(librerias) + "\\begin{document}\n" + paginas + "\n\\end{document}\n"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "f.tex").write_text(doc, encoding="utf8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"],
                           cwd=d, capture_output=True, text=True)
        if r.returncode != 0 or not (d / "f.pdf").exists():
            raise RuntimeError("pdflatex falló al compilar los fotogramas:\n" + r.stdout[-1500:])
        svgs = []
        for k in range(1, n + 1):
            subprocess.run(["pdftocairo", "-svg", "-f", str(k), "-l", str(k), "f.pdf", f"p{k}.svg"],
                           cwd=d, check=True, capture_output=True)
            svgs.append((d / f"p{k}.svg").read_text(encoding="utf8"))
        if guardar:                                   # copia de la primera figura
            destino = Path(guardar)
            if destino.parent and not destino.parent.exists():
                destino.parent.mkdir(parents=True, exist_ok=True)
            ext = destino.suffix.lower()
            if ext == ".pdf":
                shutil.copy(d / "f.pdf", destino)
            elif ext == ".svg":
                destino.write_text(svgs[0], encoding="utf8")
            elif ext == ".png":
                subprocess.run(["pdftocairo", "-png", "-r", "200", "-f", "1", "-l", "1", "-singlefile",
                                "f.pdf", str(destino.with_suffix(""))], cwd=d, check=True,
                               capture_output=True)
    return svgs


def prev_acum(paso, actividad, cand):
    """Carga acumulada justo antes de considerar ``actividad`` en un paso de :meth:`asignar_pasos`."""
    acum = paso["acumulada"]
    idx = list(cand.index).index(actividad)
    return acum[cand.index[idx - 1]] if idx > 0 else 0


def _ordenar_por_precedencia(data: pd.DataFrame) -> pd.DataFrame:
    """Filas y columnas de una matriz de enlace en orden de precedencia: primero las
    actividades sin predecesoras, luego las que solo dependen de ellas, etc. (dentro de
    cada nivel, alfabético). La matriz queda triangular inferior. Si las dependencias
    tuvieran un ciclo se conserva el orden original."""
    pred = {a: [b for b in data.columns if bool(data.loc[a, b])] for a in data.index}
    nivel = {}
    def calcula(a, visita=()):
        if a in nivel:
            return nivel[a]
        if a in visita:
            raise ValueError("ciclo")
        nivel[a] = 1 + max([calcula(b, visita + (a,)) for b in pred.get(a, [])], default=-1)
        return nivel[a]
    try:
        for a in data.index:
            calcula(a)
    except ValueError:
        return data
    orden = sorted(data.index, key=lambda a: (nivel[a], f"{a}"))
    return data.loc[orden, orden]


def _geometria_matriz(data: pd.DataFrame):
    """Geometría de la figura de una matriz de enlace, en unidades de celda:
    ``(modo, ancho_celda_cm, margen, alto_cabecera)``. Con nombres cortos (hasta 3
    caracteres) cada celda lleva la letra de la predecesora; con nombres largos las
    celdas son estrechas con una marca ✓ y los nombres de las columnas van girados."""
    largo = max([len(f"{n}") for n in list(data.index) + list(data.columns)] + [1])
    if largo <= 3:
        return "letra", max(0.78, 0.2 * largo + 0.3), 3.3, 1.1
    largo_c = max([len(f"{n}") for n in data.columns] + [1])
    if largo_c <= 3:             # columnas cortas y nombres de fila largos (p. ej. rutas)
        largo_f = max([len(f"{n}") for n in data.index] + [1])
        ancho = max(0.78, 0.2 * largo_c + 0.3)
        return "letra", ancho, max(3.3, (0.17 * largo_f + 0.9) / ancho), 1.1
    margen = max(3.3, (0.17 * largo + 0.6) / 0.78)
    return "marca", 0.78, margen, (0.16 * largo + 0.6) / 0.78


_COLORES_MATRIZ = (r"\definecolor{narosc}{RGB}{204,102,0}\definecolor{narcl}{RGB}{255,229,204}" "\n"
                   r"\definecolor{gris}{RGB}{213,219,227}\definecolor{grisosc}{RGB}{91,101,115}")


def _cuerpo_matriz_tikz(data: pd.DataFrame, filas: dict = None, columnas=(), valor: str = None,
                        valor_cero: str = None, resalte: str = None, rotulo: str = r"precedente $\to$\\ actividad $\downarrow$") -> list:
    """Líneas TikZ de una matriz booleana de enlace (sin ``tikzpicture``; con
    ``x`` = ancho de celda e ``y`` = 0,78 cm). Cuadrícula fina, celdas redondeadas en
    naranja claro (con la letra de la predecesora o una marca) y cabeceras en gris.
    ``filas`` da el estado de cada fila: ``"actual"`` (resaltada) o ``"gris"``
    (atenuada); ``columnas``, las columnas resaltadas. La matriz puede ser rectangular.
    ``valor`` escribe ese texto (p. ej. ``"1"``) en las celdas marcadas, en vez de la letra
    de la columna o la marca ✓; ``valor_cero``, si se da, se escribe en gris claro en las
    celdas vacías; ``resalte`` es el color (p. ej. ``"blue"``) con que se resalta la fila
    ``"actual"`` en vez del naranja; ``rotulo`` es el texto de la esquina."""
    filas = filas or {}
    columnas = {f"{c}" for c in columnas}
    nombres = [f"{n}" for n in data.index]
    n = nf = len(nombres)
    nc = len(data.columns)
    tex = ProjectGraph._tex_texto
    modo, ancho, margen, alto = _geometria_matriz(data)
    L = [rf"\fill[gris!25,rounded corners=3pt] ({-margen:.2f},0) rectangle ({nc},{alto:.2f});"]
    for i, nom in enumerate(nombres):
        if filas.get(nom) == "actual":
            L.append(rf"\fill[{(resalte + '!14') if resalte else 'narcl!60'},rounded corners=2pt] "
                     rf"({-margen:.2f},{-i}) rectangle ({nc},{-i - 1});")
    for j, col in enumerate(data.columns):
        if f"{col}" in columnas:
            L.append(rf"\fill[narcl!60,rounded corners=2pt] ({j},{alto:.2f}) rectangle ({j + 1},{-nf});")
    for k in range(nf + 1):
        L.append(rf"\draw[gris,line width=0.4pt] (0,{-k}) -- ({nc},{-k});")
    for k in range(nc + 1):
        L.append(rf"\draw[gris,line width=0.4pt] ({k},0) -- ({k},{-nf});")
    for k, nom in enumerate(nombres):
        est = filas.get(nom)
        c = (f"{resalte}!70!black" if resalte else "narosc") if est == "actual" else "grisosc!35" if est == "gris" else "grisosc"
        L.append(rf"\node[font=\bfseries,{c},anchor=east] at (-0.35,{-k - 0.5}) {{{tex(nom)}}};")
    for k, nom in enumerate(data.columns):
        c = "narosc" if f"{nom}" in columnas else "grisosc"
        if modo == "letra":
            L.append(rf"\node[font=\bfseries,{c}] at ({k + 0.5},{alto / 2:.2f}) {{{tex(f'{nom}')}}};")
        else:
            L.append(rf"\node[font=\bfseries,{c},rotate=90,anchor=west,inner sep=0pt] "
                     rf"at ({k + 0.5},0.25) {{{tex(f'{nom}')}}};")
    L.append(rf"\node[font=\footnotesize\itshape,grisosc,anchor=east,align=right] at (-0.25,{alto / 2:.2f}) "
             rf"{{{rotulo}}};")
    for i in range(nf):
        gris = filas.get(nombres[i]) == "gris"
        for j, col in enumerate(data.columns):
            if not bool(data.iloc[i, j]) and valor_cero is not None:
                L.append(rf"\node[grisosc!35,font=\bfseries] at ({j + 0.5},{-i - 0.5}) {{{valor_cero}}};")
            if bool(data.iloc[i, j]):
                cx, cy = j + 0.5, -i - 0.5
                fila_res = bool(resalte) and filas.get(nombres[i]) == "actual"
                relleno = "gris!25" if gris else (f"{resalte}!28" if fila_res else "narcl")
                L.append(rf"\fill[{relleno},rounded corners=2pt] ({j + 0.08},{cy - 0.42}) rectangle ({j + 0.92},{cy + 0.42});")
                tinta = "grisosc!35" if gris else (f"{resalte}!65!black" if fila_res else "narosc")
                if valor is not None:
                    L.append(rf"\node[{tinta},font=\bfseries] at ({cx},{cy}) {{{valor}}};")
                elif modo == "letra":
                    L.append(rf"\node[{tinta},font=\bfseries] at ({cx},{cy}) {{{tex(f'{col}')}}};")
                else:
                    L.append(rf"\draw[{tinta},line width=1pt,line cap=round,line join=round] "
                             rf"({cx - 0.19},{cy - 0.01}) -- ({cx - 0.05},{cy - 0.14}) -- ({cx + 0.2},{cy + 0.15});")
    return L


def _codigo_matriz_tikz(data: pd.DataFrame, valor: str = None, valor_cero: str = None, **kw) -> str:
    """``tikzpicture`` de una matriz booleana de enlace (véase :func:`_cuerpo_matriz_tikz`)."""
    _, ancho, _, _ = _geometria_matriz(data)
    return "\n".join([rf"\begin{{tikzpicture}}[x={ancho:.2f}cm,y=0.78cm,font=\small]",
                      _COLORES_MATRIZ] + _cuerpo_matriz_tikz(data, valor=valor, valor_cero=valor_cero, **kw) + [r"\end{tikzpicture}"])


class MatrizEnlace:
    """Figura (SVG incrustado, vía ``_repr_html_``) de una matriz booleana de enlace entre
    actividades; el ``DataFrame`` original está en ``.data``."""

    def __init__(self, data: pd.DataFrame, ordenar: bool = False, valor: str = None,
                 valor_cero: str = None, **kw):
        if ordenar:
            data = _ordenar_por_precedencia(data)
        self.data = data
        self._svg = _svgs_de_tikz([_codigo_matriz_tikz(data, valor=valor, valor_cero=valor_cero, **kw)], "calc")[0]

    #: píxeles por celda: el ancho ideal es (n + 3,3) celdas; se encoge hasta el mínimo
    #: y por debajo aparece una barra de desplazamiento dentro de la figura
    PX_CELDA, PX_CELDA_MIN = 46, 32

    def _repr_html_(self):
        import base64
        _, ancho, margen, _ = _geometria_matriz(self.data)
        n = (len(self.data.columns) + margen) * ancho / 0.78
        ideal, minimo = round(n * self.PX_CELDA), round(n * self.PX_CELDA_MIN)
        b64 = base64.b64encode(self._svg.encode("utf8")).decode("ascii")
        return (f'<div style="overflow-x:auto"><img alt="matriz de enlace" style="display:block;'
                f'margin:0 auto;width:100%;min-width:{minimo}px;max-width:{ideal}px;height:auto" '
                f'src="data:image/svg+xml;base64,{b64}"></div>')

    def __repr__(self):
        return f"MatrizEnlace({len(self.data)} actividades)"


def producto_rutas_html(path_matrix, tiempos, resultado=None) -> str:
    """Producto ``R × T = R·T`` (matriz de rutas por tiempos de las actividades) como visor HTML.

    Al pasar el ratón por una fila de ``R`` (o al tocarla; un segundo toque la suelta) se resalta
    esa fila y se recuadran, en cada columna de ``T``, los valores que se suman (los que
    corresponden a un 1 de la ruta) y los resultados de esa ruta: así se ve que el producto es la
    suma de los tiempos de las actividades de la ruta.

    ``path_matrix``: ``DataFrame`` rutas × actividades de 0 y 1. ``tiempos``: ``Series`` o
    ``DataFrame`` con una fila por actividad (columnas de ``R``) y una o más columnas.
    ``resultado``: ``R @ T`` (se calcula si no se da).
    """
    import html as _h, uuid
    R = pd.DataFrame(path_matrix).astype(int)
    T = tiempos.to_frame() if isinstance(tiempos, pd.Series) else pd.DataFrame(tiempos)
    T = T.loc[R.columns]
    RT = (R @ T) if resultado is None else pd.DataFrame(resultado).reindex(index=R.index)
    RT.columns = T.columns

    def fmt(v):
        v = float(v)
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")

    uid = "pmpr" + uuid.uuid4().hex[:8]
    esc = lambda x: _h.escape(str(x))
    nr, na, nc = len(R), len(R.columns), len(T.columns)

    def tabla(cabecera, filas, clase):
        return (f'<table class="{uid}-m {clase}"><thead><tr>{"".join(f"<th>{esc(c)}</th>" for c in cabecera)}'
                f'</tr></thead><tbody>{"".join(filas)}</tbody></table>')

    filas_r = [f'<tr data-i="{i}"><th class="{uid}-rot">{esc(R.index[i])}</th>' + "".join(
        f'<td class="{"u" if R.iat[i, a] else "z"}" data-a="{a}">{R.iat[i, a]}</td>' for a in range(na)) + "</tr>"
        for i in range(nr)]
    filas_t = [f'<tr><th class="{uid}-rot">{esc(R.columns[a])}</th>' + "".join(
        f'<td data-a="{a}" data-j="{j}">{fmt(T.iat[a, j])}</td>' for j in range(nc)) + "</tr>" for a in range(na)]
    filas_x = [f'<tr data-i="{i}">' + "".join(
        f'<td data-i="{i}" data-j="{j}">{fmt(RT.iat[i, j])}</td>' for j in range(nc)) + "</tr>" for i in range(nr)]
    css = (f'<style>#{uid}{{margin:1em 0;overflow-x:auto}}'
           f'#{uid} .box{{display:flex;align-items:flex-start;gap:.7em;justify-content:center;width:max-content;margin:0 auto}}'
           f'#{uid} .op{{align-self:center;font-size:1.3em;padding-top:1.6em}}'
           f'#{uid} table{{border-collapse:separate;border-spacing:2px;font-variant-numeric:tabular-nums}}'
           f'#{uid} .{uid}-m tbody{{border-left:2px solid #444;border-right:2px solid #444}}'
           f'#{uid} th{{font-weight:normal;color:#555;font-size:.85em;padding:0 .3em;text-align:center}}'
           f'#{uid} th.{uid}-rot{{text-align:right;white-space:nowrap}}'
           f'#{uid} td{{text-align:center;min-width:1.7em;padding:.15em .35em;border:2px solid transparent;border-radius:3px}}'
           f'#{uid} td.z{{color:#c4c4c4}}'
           f'#{uid} .rr tbody tr{{cursor:pointer}}'
           f'#{uid} tr.sel td{{background:#dbeafe}}'
           f'#{uid} td.cj{{border-color:#1d4ed8;background:#dbeafe;font-weight:bold;color:#0b2a7a}}'
           f'#{uid} .nota{{font-size:.85em;color:#555;margin:.3em 0 0;text-align:center}}</style>')
    js = ("<script>(function(){var r=document.getElementById('%s'),fijo=null;"
          "function mostrar(i){r.querySelectorAll('.sel').forEach(function(e){e.classList.remove('sel')});"
          "r.querySelectorAll('.cj').forEach(function(e){e.classList.remove('cj')});if(i===null)return;"
          "var fr=r.querySelector('.rr tr[data-i=\"'+i+'\"]');if(!fr)return;fr.classList.add('sel');"
          "fr.querySelectorAll('td.u').forEach(function(c){c.classList.add('cj');var a=c.dataset.a;"
          "r.querySelectorAll('.tt td[data-a=\"'+a+'\"]').forEach(function(t){t.classList.add('cj')})});"
          "r.querySelectorAll('.xx tr[data-i=\"'+i+'\"]').forEach(function(x){x.classList.add('sel');"
          "x.querySelectorAll('td').forEach(function(c){c.classList.add('cj')})})}"
          "r.querySelectorAll('.rr tbody tr').forEach(function(f){var i=f.dataset.i;"
          "f.addEventListener('mouseenter',function(){mostrar(i)});"
          "f.addEventListener('click',function(){fijo=(fijo===i)?null:i;mostrar(fijo)})});"
          "r.querySelector('.rr').addEventListener('mouseleave',function(){mostrar(fijo)})})();</script>") % uid
    return (f'{css}<div id="{uid}"><div class="box">'
            + tabla([""] + list(R.columns), filas_r, "rr") + '<div class="op">×</div>'
            + tabla([""] + [str(c) for c in T.columns], filas_t, "tt")
            + '<div class="op">=</div>' + tabla([str(c) for c in RT.columns], filas_x, "xx")
            + '</div><p class="nota">👆 Interactivo: pasa el ratón (o toca) una fila de la matriz de rutas: '
              'se recuadran los valores que se suman y el resultado.</p></div>' + js)


def display_producto_rutas(path_matrix, tiempos, resultado=None) -> None:
    """Muestra :func:`producto_rutas_html` en el cuaderno."""
    from IPython.display import display, HTML
    display(HTML(producto_rutas_html(path_matrix, tiempos, resultado)))


def muestra_test(preguntas_test):
    """Test de respuesta múltiple legible: pregunta, opciones (la correcta —la primera de cada
    lista— marcada con ✅) y retroalimentación. ``preguntas_test`` es una lista de
    ``(pregunta, [correcta, distractores…], retroalimentación)``."""
    from IPython.display import Markdown
    partes = []
    for i, (q, ops, fb) in enumerate(preguntas_test, 1):
        lineas = [f"**{i}. {q}**", ""]
        lineas += [f"- {'✅' if j == 0 else '◻️'} {o}" for j, o in enumerate(ops)]
        lineas += ["", f"*{fb}*"]
        partes.append("\n".join(lineas))
    return Markdown("\n\n".join(partes))


def matriz_caminos_tikz(path_matrix: pd.DataFrame):
    """Matriz de caminos (rutas × actividades) con el estilo TikZ de la matriz de prelaciones
    extendida: un 1 en las casillas marcadas y un 0 gris claro en las vacías. Si no hay LaTeX
    devuelve el ``DataFrame``."""
    try:
        return MatrizEnlace(path_matrix.astype(bool), valor="1", valor_cero="0",
                            rotulo=r"actividad $\to$\\ ruta $\downarrow$")
    except (FileNotFoundError, RuntimeError):
        return path_matrix


def _html_fotogramas_tikz(codigos: list, librerias: str = "calc,arrows.meta",
                          resultado_primero: bool = False) -> str:
    """Visor HTML con botones (:func:`_html_pasos`) de fotogramas TikZ compilados a
    SVG. Cada SVG va como imagen ``data:`` para que los identificadores de
    glifos de unos no se mezclen con los de otros. Con ``resultado_primero`` el
    último fotograma (el resultado) se repite como primero (paso 0) y el contador
    empieza en 0."""
    import base64
    fotos = []
    svgs = _svgs_de_tikz(codigos, librerias)
    if resultado_primero:
        svgs = [svgs[-1]] + list(svgs)
    for svg in svgs:
        b64 = base64.b64encode(svg.encode("utf8")).decode("ascii")
        fotos.append('<img alt="paso" style="display:block;margin:0 auto;width:100%;max-width:1100px;height:auto" '
                     f'src="data:image/svg+xml;base64,{b64}">')
    return _html_pasos(fotos, inicio=0 if resultado_primero else 1)


def plot_tikz_normal(media, desviacion, hasta=None, desde=None, entre=None, percentil=None,
                     variable: str = "T", decimales: int = 4, escala: float = 1.0) -> str:
    """``tikzpicture`` con la curva de la distribución normal de media ``media`` y desviación
    típica ``desviacion`` y, sombreada, el área bajo la curva de las probabilidades pedidas:

    * ``hasta=x`` (o lista): ``P(V ≤ x)``;
    * ``desde=x`` (o lista): ``P(V ≥ x)``;
    * ``entre=(a, b)`` (o lista de pares): ``P(a ≤ V ≤ b)``;
    * ``percentil=p`` (o lista, con ``0 < p < 1``): área ``p`` a la izquierda del percentil
      correspondiente, cuyo valor se marca en el eje.

    El eje muestra ``media ± k·desviación``; cada valor límite se marca en el eje y las
    probabilidades se escriben (``variable`` es el nombre de la variable aleatoria)."""
    from scipy.stats import norm
    lista = lambda v: [] if v is None else (list(v) if isinstance(v, (list, tuple)) and not (
        len(v) == 2 and isinstance(v[0], (int, float)) and False) else [v])
    huecos = []                                   # (a, b, texto): a y b pueden ser ±inf
    for x in lista(hasta):
        huecos.append((-np.inf, x, rf"P({variable}\le {x:g})"))
    for x in lista(desde):
        huecos.append((x, np.inf, rf"P({variable}\ge {x:g})"))
    pares = [entre] if (entre is not None and np.ndim(entre) == 1) else (list(entre) if entre is not None else [])
    for a, b in pares:
        huecos.append((a, b, rf"P({a:g}\le {variable}\le {b:g})"))
    for q in lista(percentil):
        x = norm.ppf(q, loc=media, scale=desviacion)
        huecos.append((-np.inf, x, rf"P({variable}\le {x:.2f})"))
    if not huecos:
        raise ValueError("Indique al menos una de las opciones: hasta, desde, entre o percentil")

    ancho_sigma, alto = 1.45 * escala, 3.3 * escala
    z = lambda x: (x - media) / desviacion
    # el eje siempre abarca todos los límites marcados, con un margen
    zmax = max([3.6] + [abs(z(v)) + 0.7 for h in huecos for v in h[:2] if np.isfinite(v)])
    X = lambda x: z(min(max(x, media - zmax * desviacion), media + zmax * desviacion)) * ancho_sigma
    Y = lambda zz: np.exp(-zz * zz / 2) * alto
    curva = lambda z0, z1, n=None: " ".join(
        f"({zz * ancho_sigma:.3f},{Y(zz):.3f})" for zz in np.linspace(z0, z1, n or max(4, int(abs(z1 - z0) * 24)) + 1))
    colores = ["orange!50", "blue!28", "green!32", "magenta!28", "cyan!35"]
    # primero los intervalos más anchos, para que los menores queden encima
    orden = sorted(range(len(huecos)), key=lambda i: -(min(huecos[i][1], media + zmax * desviacion) -
                                                       max(huecos[i][0], media - zmax * desviacion)))
    L = []
    for rango, i in enumerate(orden):
        a, b, _ = huecos[i]
        za, zb = max(z(a), -zmax), min(z(b), zmax)
        L.append(rf"\fill[{colores[i % len(colores)]}] ({za * ancho_sigma:.3f},0) -- plot coordinates {{{curva(za, zb)}}} "
                 rf"-- ({zb * ancho_sigma:.3f},0) -- cycle;")
    L.append(rf"\draw[thick, blue!60!black] plot coordinates {{{curva(-zmax, zmax, 144)}}};")
    L.append(rf"\draw[->, black!70] ({-zmax * ancho_sigma - 0.4:.3f},0) -- ({zmax * ancho_sigma + 0.5:.3f},0);")
    L.append(rf"\draw[dashed, black!55] (0,0) -- (0,{alto + 0.15:.3f}) node[above, black!70] {{$\mu$}};")
    fmt = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
    for k in range(-3, 4):
        L.append(rf"\draw[black!70] ({k * ancho_sigma:.3f},0.07) -- ({k * ancho_sigma:.3f},-0.07);")
        L.append(rf"\node[below, font=\footnotesize, black!70] at ({k * ancho_sigma:.3f},-0.1) {{{fmt(media + k * desviacion)}}};")
    for i, (a, b, _) in enumerate(huecos):
        for lim in (a, b):
            if np.isfinite(lim):
                c = colores[i % len(colores)].split("!")[0] + "!70!black"
                x = X(lim)
                L.append(rf"\draw[{c}, thick, dashed] ({x:.3f},-0.58) -- ({x:.3f},{Y(z(lim)) + 0.15:.3f});")
                L.append(rf"\node[below, font=\small\bfseries, {c}] at ({x:.3f},-0.62) {{{fmt(lim)}}};")
    # leyenda con las probabilidades (arriba a la izquierda), cada una con su color
    for i, (a, b, txt) in enumerate(huecos):
        p = norm.cdf(b, loc=media, scale=desviacion) - norm.cdf(a, loc=media, scale=desviacion)
        c = colores[i % len(colores)]
        y = alto * 0.98 - i * 0.62 * escala
        L.append(rf"\fill[{c}] ({-zmax * ancho_sigma:.3f},{y - 0.17:.3f}) rectangle ++(0.34,0.34);")
        L.append(rf"\node[right, font=\normalsize] at ({-zmax * ancho_sigma + 0.42:.3f},{y:.3f}) {{${txt} = {p:.{decimales}f}$}};")
    return "\n".join([r"\begin{tikzpicture}[>=stealth]"] + L + [r"\end{tikzpicture}"])


def display_normal_tikz(media, desviacion, hasta=None, desde=None, entre=None, percentil=None,
                        variable: str = "T", decimales: int = 4, escala: float = 1.0,
                        ancho_max: int = 640) -> None:
    """Muestra en un cuaderno la gráfica de :func:`plot_tikz_normal` (TikZ compilado a SVG).
    Si no hay LaTeX, avisa y no dibuja nada."""
    import base64
    codigo = plot_tikz_normal(media, desviacion, hasta, desde, entre, percentil, variable, decimales, escala)
    try:
        svg = _svgs_de_tikz([codigo])[0]
    except (FileNotFoundError, RuntimeError) as e:
        import warnings
        warnings.warn(f"No se pudo dibujar la distribución normal en TikZ ({e}).")
        return
    b64 = base64.b64encode(svg.encode("utf8")).decode("ascii")
    display(HTML(f'<img alt="distribución normal" style="display:block;margin:0 auto;width:100%;max-width:{ancho_max}px;height:auto" '
                 f'src="data:image/svg+xml;base64,{b64}">'))


def _marcas_nicas(lo, hi, n=6):
    """Valores «redondos» para las marcas de un eje entre ``lo`` y ``hi``."""
    from matplotlib.ticker import MaxNLocator
    return [v for v in MaxNLocator(n).tick_values(lo, hi) if lo - 1e-9 <= v <= hi + 1e-9]


def _svg_a_html(codigo: str, alt: str, ancho_max: int = 640):
    """``<img>`` con el SVG de un ``tikzpicture`` compilado, o ``None`` si no hay LaTeX."""
    import base64
    try:
        svg = _svgs_de_tikz([codigo])[0]
    except (FileNotFoundError, RuntimeError) as e:
        import warnings
        warnings.warn(f"No se pudo dibujar «{alt}» en TikZ ({e}).")
        return None
    b64 = base64.b64encode(svg.encode("utf8")).decode("ascii")
    return HTML(f'<img alt="{alt}" style="display:block;margin:0 auto;width:100%;max-width:{ancho_max}px;height:auto" '
                f'src="data:image/svg+xml;base64,{b64}">')


def plot_tikz_histograma(muestras, media=None, desviacion=None, hasta=None, percentil=None,
                         variable: str = "T", decimales: int = 3, bins: int = 40,
                         escala: float = 1.0) -> str:
    """``tikzpicture`` con el histograma (densidad) de una simulación de Montecarlo y la curva
    de la distribución normal que la aproxima (``media`` y ``desviacion``; por defecto, las de
    la muestra). Con ``hasta=x`` se resaltan en naranja las barras hasta ``x`` y se marca ``x``;
    con ``percentil=p`` se hace lo mismo hasta el percentil ``p`` de la muestra. La leyenda da la
    probabilidad según Montecarlo (fracción de muestras) y según la normal."""
    from scipy.stats import norm
    x = np.asarray(pd.Series(np.asarray(muestras, dtype=float).ravel()).dropna())
    media = float(x.mean()) if media is None else float(media)
    desviacion = float(x.std(ddof=1)) if desviacion is None else float(desviacion)
    if hasta is not None and percentil is not None:
        raise ValueError("Indique hasta o percentil, no ambos")
    corte = None
    if percentil is not None:
        corte = float(np.quantile(x, percentil, method="higher"))
    elif hasta is not None:
        corte = float(hasta)
    dens, bordes = np.histogram(x, bins=bins, density=True)
    lo = min(bordes[0], media - 3.2 * desviacion)
    hi = max(bordes[-1], media + 3.2 * desviacion)
    if corte is not None:   # el valor marcado debe quedar siempre dentro del eje
        margen = 0.08 * (hi - lo)
        lo, hi = min(lo, corte - margen), max(hi, corte + margen)
    pico = max(dens.max(), norm.pdf(media, media, desviacion))
    W, H = 11.0 * escala, 3.4 * escala
    X = lambda v: (v - lo) / (hi - lo) * W
    Y = lambda d: d / pico * H
    L = []
    for d, a, b in zip(dens, bordes[:-1], bordes[1:]):
        dentro = corte is not None and (a + b) / 2 <= corte
        L.append(rf"\fill[{'orange!55' if dentro else 'blue!18'}, draw=white, thin] ({X(a):.3f},0) rectangle ({X(b):.3f},{Y(d):.3f});")
    xs = np.linspace(lo, hi, 160)
    puntos = " ".join(f"({X(v):.3f},{Y(norm.pdf(v, media, desviacion)):.3f})" for v in xs)
    L.append(rf"\draw[thick, blue!60!black] plot coordinates {{{puntos}}};")
    L.append(rf"\draw[->, black!70] (-0.2,0) -- ({W + 0.4:.3f},0);")
    fmt = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
    for v in _marcas_nicas(lo, hi):
        L.append(rf"\draw[black!70] ({X(v):.3f},0.07) -- ({X(v):.3f},-0.07);")
        L.append(rf"\node[below, font=\footnotesize, black!70] at ({X(v):.3f},-0.1) {{{fmt(v)}}};")
    if corte is not None:
        c = "orange!70!black"
        L.append(rf"\draw[{c}, thick, dashed] ({X(corte):.3f},-0.58) -- ({X(corte):.3f},{Y(norm.pdf(corte, media, desviacion)) + 0.45:.3f});")
        L.append(rf"\node[below, font=\small\bfseries, {c}] at ({X(corte):.3f},-0.5) {{{fmt(corte)}}};")
        p_mc = float((x <= corte).mean())
        p_n = float(norm.cdf(corte, media, desviacion))
        yy = H + 1.25
        L.append(rf"\fill[orange!55] (0,{yy - 0.17:.3f}) rectangle ++(0.34,0.34);")
        L.append(rf"\node[right] at (0.42,{yy:.3f}) {{Montecarlo: $P({variable}\le {fmt(corte)}) = {p_mc:.{decimales}f}$}};")
        L.append(rf"\draw[thick, blue!60!black] (0,{yy - 0.62:.3f}) -- ++(0.34,0);")
        L.append(rf"\node[right] at (0.42,{yy - 0.62:.3f}) {{Normal: $P({variable}\le {fmt(corte)}) = {p_n:.{decimales}f}$}};")
    else:
        L.append(rf"\fill[blue!18] (0,{H + 1.08:.3f}) rectangle ++(0.34,0.34);")
        L.append(rf"\node[right] at (0.42,{H + 1.25:.3f}) {{Montecarlo ({len(x)} iteraciones)}};")
        L.append(rf"\draw[thick, blue!60!black] (0,{H + 0.63:.3f}) -- ++(0.34,0);")
        L.append(rf"\node[right] at (0.42,{H + 0.63:.3f}) {{Normal $(\mu={fmt(media)},\ \sigma={fmt(desviacion)})$}};")
    return "\n".join([r"\begin{tikzpicture}[>=stealth]"] + L + [r"\end{tikzpicture}"])


def display_histograma_tikz(muestras, media=None, desviacion=None, hasta=None, percentil=None,
                            variable: str = "T", decimales: int = 3, bins: int = 40,
                            escala: float = 1.0, ancho_max: int = 700) -> None:
    """Muestra en un cuaderno el histograma de Montecarlo con la normal superpuesta
    (:func:`plot_tikz_histograma`). Sin LaTeX, avisa y no dibuja nada."""
    h = _svg_a_html(plot_tikz_histograma(muestras, media, desviacion, hasta, percentil, variable,
                                         decimales, bins, escala), "histograma de Montecarlo", ancho_max)
    if h is not None:
        display(h)


def plot_tikz_beta_pert(a, m, b, actividad: str = "", escala: float = 1.0) -> str:
    """``tikzpicture`` con la densidad de la distribución Beta-PERT de parámetros
    ``(a, m, b)`` (optimista, más probable, pesimista), es decir ``X = a + (b-a)·Y`` con
    ``Y ~ Beta(1+4(m-a)/(b-a), 1+4(b-m)/(b-a))``, con la moda ``m`` y la media ``(a+4m+b)/6``."""
    from scipy.stats import beta as beta_dist
    a, m, b = float(a), float(m), float(b)
    if not a < b:
        raise ValueError("Se requiere a < b")
    al, be = 1 + 4 * (m - a) / (b - a), 1 + 4 * (b - m) / (b - a)
    dens = lambda v: beta_dist.pdf((v - a) / (b - a), al, be) / (b - a)
    media = (a + 4 * m + b) / 6
    W, H = 10.0 * escala, 3.2 * escala
    pico = max(dens(v) for v in np.linspace(a, b, 400)[1:-1])
    X = lambda v: (v - a) / (b - a) * W
    Y = lambda v: min(dens(v) / pico, 1.05) * H
    xs = np.linspace(a, b, 200)
    xs = xs[1:-1] if (al < 1 or be < 1) else xs
    puntos = " ".join(f"({X(v):.3f},{Y(v):.3f})" for v in xs)
    fmt = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
    L = [rf"\fill[blue!18] ({X(a):.3f},0) -- plot coordinates {{{puntos}}} -- ({X(b):.3f},0) -- cycle;",
         rf"\draw[thick, blue!60!black] plot coordinates {{{puntos}}};",
         rf"\draw[->, black!70] (-0.3,0) -- ({W + 0.4:.3f},0);"]
    for v, nom in ((a, "a"), (m, "m"), (b, "b")):
        L.append(rf"\draw[black!70] ({X(v):.3f},0.07) -- ({X(v):.3f},-0.07);")
        L.append(rf"\node[below, font=\footnotesize, black!70] at ({X(v):.3f},-0.1) {{{fmt(v)}}};")
    L.append(rf"\draw[orange!70!black, thick, dashed] ({X(m):.3f},-0.58) -- ({X(m):.3f},{Y(m) + 0.2:.3f});")
    L.append(rf"\node[below, font=\small\bfseries, orange!70!black] at ({X(m):.3f},-0.5) {{$m$}};")
    L.append(rf"\draw[green!45!black, thick, dotted] ({X(media):.3f},0) -- ({X(media):.3f},{Y(media) + 0.2:.3f});")
    L.append(rf"\node[above right, font=\small, green!45!black, inner sep=1pt] at ({X(media):.3f},{Y(media) + 0.1:.3f}) {{media {fmt(media)}}};")
    L.append(rf"\node[below, font=\small\bfseries, black!70] at ({X(a):.3f},-0.5) {{$a$}};")
    L.append(rf"\node[below, font=\small\bfseries, black!70] at ({X(b):.3f},-0.5) {{$b$}};")
    titulo = (f"Actividad {actividad}: " if actividad else "") + rf"Beta-PERT$(a={fmt(a)},\ m={fmt(m)},\ b={fmt(b)})$"
    L.append(rf"\node[anchor=north west, font=\normalsize] at (0,{H + 0.9:.3f}) {{{titulo}}};")
    return "\n".join([r"\begin{tikzpicture}[>=stealth]"] + L + [r"\end{tikzpicture}"])


def display_beta_pert_tikz(a, m, b, actividad: str = "", escala: float = 1.0, ancho_max: int = 620) -> None:
    """Muestra en un cuaderno la densidad Beta-PERT de ``(a, m, b)`` (:func:`plot_tikz_beta_pert`)."""
    h = _svg_a_html(plot_tikz_beta_pert(a, m, b, actividad, escala), "densidad Beta-PERT", ancho_max)
    if h is not None:
        display(h)


def pinta_tikz(dibujo, filename=None):
    if filename is not None:
        display(TexFragment(dibujo, tikz_libraries='patterns').run_latex(save_pdf=filename+'.pdf', save_svg=filename+'.svg'))
    else:
        display(TexFragment(dibujo, tikz_libraries='patterns').run_latex())

def plot_cargas(gantt_df, limite_maximo_recursos=None):
    from matplotlib import pyplot as plt
    import numpy as np

    # 1. Prepare the data
    series_data = gantt_df.data.loc['Total'].drop('H_total')
    num_periods = len(series_data)
    
    # Define bar width
    bar_width = 0.8 
    ax = series_data.plot(kind='bar', width=bar_width, color='skyblue', alpha=0.8)

    # 2. Logic for the limit line (only executes if not None)
    if limite_maximo_recursos is not None:
        if isinstance(limite_maximo_recursos, (list, np.ndarray)):
            limits = list(limite_maximo_recursos)
            
            # Padding logic: extend with the last value if list is short
            if len(limits) < num_periods:
                limits.extend([limits[-1]] * (num_periods - len(limits)))
            
            # Ensure it's not longer than the data
            limits = limits[:num_periods]

            # Coordinate calculation for step line starting at left margin
            x_steps = np.arange(num_periods) - (bar_width / 2)
            
            # Add extra point to cover the full width of the last bar
            x_steps = np.append(x_steps, x_steps[-1] + bar_width)
            limits = np.append(limits, limits[-1])

            plt.step(x_steps, limits, where='post', color='red', 
                     linestyle='--', linewidth=2, label='Límite variable')
            
        else:
            # Single value logic (float/int)
            plt.axhline(y=float(limite_maximo_recursos), color='red', linestyle='--', 
                        alpha=0.7, label=f'Límite máximo = {limite_maximo_recursos}')
        
        # Only show legend if a limit line was actually drawn
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_curvas_valor_ganado(gantts):
    from matplotlib import pyplot as plt
    acumulados = gantts['acumulados']
    acumulados.plot(
        color=['blue', 'green', 'red'],
        title='Curvas S del Valor Ganado (Acumuladas)',
        ylabel='Costo',
        xlabel='Período de Tiempo'
    )
    plt.show()
    