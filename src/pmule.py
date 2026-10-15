from jupyter_tikz import TexFragment
from numpyarray_to_latex.jupyter import to_jup
from numpyarray_to_latex import to_ltx
from IPython.display import display, Image, SVG, Latex, Markdown, Math 
from functools import reduce
from itertools import chain, product
import math
from math import ceil
from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

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
        nx.contracted_nodes(G, v, u, self_loops=False, copy=False)
        A.update(new)
        del A[u]
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
            if try_contract(u, v) or try_remove(u, v):
                reindex()
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
        self.distant_predecessor = {
            key: self.calculate_distant_predecessor_of(key)
            for key in self.immediate_predecessor.keys()
        }
        self.distant_predecessor = dict(sorted(self.distant_predecessor.items()))
        if simplify:
            self.immediate_predecessor = {
                key: self.calculate_immediate_predecessor_of(key)
                for key in self.immediate_predecessor.keys()
            }
            self.immediate_predecessor = dict(
                sorted(self.immediate_predecessor.items())
            )

    def _check_acyclic(self) -> None:
        """Lanza ValueError si la tabla de precedencias contiene un ciclo
        (incluida una actividad predecesora de sí misma)."""
        g = nx.DiGraph()
        g.add_nodes_from(self.immediate_predecessor.keys())
        for activity, predecessors in self.immediate_predecessor.items():
            for ancestor in predecessors:
                g.add_edge(ancestor, activity)
        try:
            cycle = nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            return
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

    def calculate_distant_predecessor_of(self, name: str) -> Set[str]:
        predecessors = self.immediate_predecessor[name].copy()
        result = predecessors.copy()
        for ancestor in predecessors:
            result.update(self.calculate_distant_predecessor_of(ancestor))
        return result

    def calculate_immediate_predecessor_of(self, name: str) -> Dict[str, str]:
        predecessors = self.distant_predecessor[name].copy()
        for ancestor in predecessors:
            predecessors = predecessors - self.distant_predecessor[ancestor]
        return predecessors

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
        return self.display_distant_linkage_matrix()

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

    def display_distant_linkage_matrix(self):
        def highlight_true(value):
            if value:
                return "background-color: cyan;  border: 1px solid black;"
            else:
                return "background-color: white;  border:1px solid black;"

        data = self.distant_linkage_dataframe
        return data.replace(False, "").style.map(highlight_true)

    def display_immediate_linkage_matrix(self):
        def highlight_true(value):
            if value:
                return "background-color: cyan;  border: 1px solid black;"
            else:
                return "background-color: white;  border:1px solid black;"

        data = self.immediate_linkage_matrix
        return data.replace(False, "").style.map(highlight_true)


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
        list_of_dicts = list(
            {attributes["activity"]: (f"{source}", f"{target}")}
            for (source, target, attributes) in self.pert_graph.edges(data=True)
        )
        combined_dict = reduce(lambda x, y: {**x, **y}, list_of_dicts, {})
        return combined_dict

    @property
    def edges_from_nodes(self):
        list_of_dicts = list(
            {(f"{source}", f"{target}"): attributes["activity"]}
            for (source, target, attributes) in self.pert_graph.edges(data=True)
        )
        combined_dict = reduce(lambda x, y: {**x, **y}, list_of_dicts, {})
        return combined_dict

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

    def calculate_distant_predecessor_of(self, activity, dummies=False):
        source, target = self.nodes_from_edges[activity]
        edges_from_nodes_dict = self.edges_from_nodes
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

    def distant_predecessor(self, dummies=False, format=None):
        activities = self.activities if dummies else self.actual_activities
        resultado = {
            key: self.calculate_distant_predecessor_of(key, dummies=dummies)
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
            # las predecesoras acaben lo más tarde posible.
            d = duraciones.get(activity_name)
            calendario.loc[activity_name, "H_libre"] = (
                tempranos[nodo_final] - tempranos[nodo_inicial] - d
            )
            calendario.loc[activity_name, "H_independiente"] = max(
                0, tempranos[nodo_final] - tardios[nodo_inicial] - d
            )

        return calendario

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
        **kwargs,
    ):
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
        gantt = pd.DataFrame("", index=actividades_con_duracion, columns=periodos)

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
        **kwargs,
    ):
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

        dot_graph.node_attr["shape"] = "Mrecord"
        dot_graph.add_edges_from(self.roy_graph.edges)

        for nodo in dot_graph.nodes():
            current_node = dot_graph.get_node(nodo)
            if durations is not False:
                current_node.attr["label"] = (
                    f"{nodo} | {{ "
                    f"<min> {inicio_mas_temprano[str(nodo)]} |"
                    f"<dur> {durations[str(nodo)]}          | "
                    f"<max> {inicio_mas_tardio[str(nodo)]}   }}"
                )
                if str(nodo) in critical_nodes:
                    current_node.attr["color"] = "red:red"
                    current_node.attr["fontcolor"] = "red"
                    current_node.attr["style"] = "bold"
            else:
                current_node.attr["label"] = f"{nodo} | {{ <min>  | <dur>  | <max>   }}"

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

    def asignar_multi(self, data, duration_label="duration", resource_labels=None,
                      maximos=None, report=False, tikz=False, params=None):
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

        # límites por recurso
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

        # prioridad: LS del CPM sin recursos
        cal = self.calendar(pd.Series(dur))
        ls = {a: cal.loc[a, "inicio_mas_tardio"] for a in acts}
        orden = sorted(acts, key=lambda a: (ls[a], dur[a], a))

        n0 = max(len(v) for v in vectors.values())
        horizonte = n0 + sum(dur.values()) + 1
        maxd = max(dur.values(), default=0)
        ancho = horizonte + maxd + 1
        limite = np.empty((K, ancho))
        for k, rl in enumerate(resource_labels):
            v = vectors[rl]
            limite[k, :len(v)] = v[:ancho]
            limite[k, len(v):] = v[-1]
        carga = np.zeros((K, ancho))

        inicio, fin, traza = {}, {}, []
        t = 0
        pendientes = set(acts)
        while pendientes:
            if t > horizonte:
                raise ValueError(
                    "Error: el proyecto no es factible con los límites de recursos indicados")
            elegibles = [a for a in orden if a in pendientes
                         and all(p in fin and fin[p] <= t for p in preds[a])]
            for a in elegibles:
                d = dur[a]
                if d == 0:
                    inicio[a], fin[a] = t, t
                    pendientes.discard(a)
                    continue
                exceso = (carga[:, t:t + d] + dem[a][:, None] > limite[:, t:t + d] + 1e-9)
                bloqueo = [resource_labels[k] for k in range(K) if exceso[k].any()]
                if not bloqueo:
                    carga[:, t:t + d] += dem[a][:, None]
                    inicio[a], fin[a] = t, t + d
                    pendientes.discard(a)
                    traza.append((t + 1, a, "empieza", ""))
                else:
                    traza.append((t + 1, a, "espera", ", ".join(bloqueo)))
            t += 1

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

        # Gantt con las esperas, sobre una copia (desplazar muta el grafo)
        proyecto = self.copy()
        my_data = data.copy()
        if len(esperas):
            # orden cronológico, y alfabético dentro de cada periodo (como asignar)
            shifts = {a: int(esperas[a]) for a in sorted(esperas.index, key=lambda x: (inicio[x], x))}
            my_data, _, _ = proyecto.desplazar(
                my_data, duration_label, resource_labels[0], report=False,
                cuadrados=False, **shifts)
        gantts, dibujos = {}, {}
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

        varianza_caminos = pd.Series(varianza_caminos).to_frame(name='Variance').round(2)
        varianza_caminos = (pd.concat([pd.Series(caminos).to_frame(name='Activities').map(lambda x: ", ".join(x)),
                                      varianza_caminos], axis=1)
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
        result = result.multiply(costs, axis="columns")
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

        Aproximación: τ_svd = σ₁ · (Σ u₁ᵢ) · (v₁ᵀ · t)

        Returns
        -------
        dict con claves: tau_exact, tau_svd, abs_error, rel_error, sigma_ratio.
        """
        R = self.path_matrix(dummies=False)
        t = np.array([durations.get(a, 0.0) for a in R.columns])
        tau_exact = float(np.max(R.values @ t))
        U, s, Vt = np.linalg.svd(R.values, full_matrices=False)
        tau_svd = float(s[0] * U[:, 0].sum() * (Vt[0] @ t))
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

    resto_de_enlaces = [
        (b, a) for a, b in list(linkage_matrix[linkage_matrix].stack().index)
    ]
    graph.add_edges_from(resto_de_enlaces)
    return graph


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
    