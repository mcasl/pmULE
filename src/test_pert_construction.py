"""
test_pert_construction.py — Comprehensive test suite for pmULE PERT construction
=================================================================================

Background — Activity-on-Arrow (AoA) PERT networks
----------------------------------------------------
In an AoA network every *activity* is a directed arc (edge) and every
*event* is a node.  A *dummy activity* (name starts with "@") is a
zero-duration arc added purely to encode precedence that cannot be
expressed by a direct real arc.

Two construction algorithms are implemented:

Roy + simplify (create_project_roy  — also the old create_project)
    1. One dummy per predecessor link plus SOURCE/SINK connectors → raw graph
    2. Repeated _simplify_sweep passes → remove / contract safe dummies

Fondahl / Elmaghraby direct construction (create_project_fondahl)
    Process activities in topological order.  Two rules keep dummy count
    minimal from the start:

    Rule 1 – direct attachment
        If an activity has exactly one predecessor and no other activity
        has already been given that predecessor's head node as its head,
        place the new activity directly at that node (no dummy needed).

    Rule 2 – convergence-node reuse
        If several activities share the *exact same* predecessor set, one
        shared "convergence node" is created for them all (and a single
        set of dummies runs into that shared node), instead of giving each
        its own convergence node and then having to link them with dummies.

    Node layout in raw Fondahl output:
        Node 0 – global SOURCE (all start activities originate here)
        Node 1 – global SINK
        Nodes 2, 3, … – fresh head node per real activity

    Because every end activity's head must connect to the global SINK, the
    raw Fondahl graph always has exactly one dummy per end activity (a
    "terminating dummy").  A single-activity project has 1 terminating
    dummy; a single-path chain of N activities has 1 terminating dummy
    (only the last activity is an end activity).

Core correctness invariant
--------------------------
A PERT graph is correct iff it encodes the *same* precedence as the
original table, ignoring dummy activities:

    project.distant_predecessor(dummies=False) == table.distant_predecessor

Every construction method must satisfy this invariant for every valid input.

calculate_pert API
------------------
project.calculate_pert(durations) returns a dict with two keys:
    'nodes'      – pandas DataFrame with columns 'early' and 'late'
                   (early/late event times), indexed by node ID.
    'activities' – pandas DataFrame with column 'H_total' (total float,
                   "Holgura total"), indexed by activity name (real AND
                   dummy activities are both included).

Key derived quantities:
    Project duration  = result['nodes']['early'].max()
    Float of act X    = result['activities'].loc[X, 'H_total']
    Critical path     = activities with H_total == 0 (real activities only)

path_matrix API
---------------
project.path_matrix(dummies=True) returns a pandas DataFrame where:
    rows    – one row per complete path through the network (Route_1, …)
    columns – activity names (real + optionally dummies)
    values  – 1 if that activity lies on that path, 0 otherwise

It is NOT a square node×node reachability matrix.

Test organisation
-----------------
Class                        What is tested
---------------------------------------------------------------------------
TestPredecessorTable         Logical properties of PredecessorTable itself
TestFondahlConstruct         White-box tests on raw fondahl_construct output
TestProjectGraphStructure    API / structural properties of ProjectGraph
TestAssessDummy              assess_dummy: contract/remove/reject decisions
TestSimplificationSweep      _simplify_sweep: type, count, invariant, halt
TestPertConstructionRoy      Roy+simplify end-to-end via create_project_roy (known bug documented)
TestPertConstructionFondahl  Fondahl end-to-end (all cases must pass)
TestCrossMethod              Both methods agree on shared invariant
TestCalculatePert            PERT schedule on small networks with known answers
"""

import sys
import os
import unittest
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmule import (
    PredecessorTable,
    ProjectGraph,
    fondahl_construct,
    _simplify_sweep,
)


# ============================================================
# Shared helpers
# ============================================================

def make_table(data: dict) -> PredecessorTable:
    """Build a PredecessorTable from {activity: [list-of-preds]}.

    Converts list values to sets, which is what PredecessorTable expects.
    Example::

        make_table({"A": [], "B": ["A"], "C": ["A", "B"]})
    """
    return PredecessorTable({k: set(v) for k, v in data.items()})


def check_invariant(table: PredecessorTable, project: ProjectGraph) -> bool:
    """Return True iff the PERT graph encodes the same precedence as *table*.

    Compares project.distant_predecessor(dummies=False) with
    table.distant_predecessor, which is the canonical correctness invariant
    for any AoA PERT network.
    """
    return project.distant_predecessor(dummies=False) == table.distant_predecessor


# ============================================================
# Shared test-case catalogue
# ============================================================

CASES = {
    # Simplest possible network: one activity, no predecessors.
    "single_activity": {
        "A": [],
    },

    # Linear chain: A → B → C → D.  No branching; no predecessor-dummies needed.
    "chain_four": {
        "A": [], "B": ["A"], "C": ["B"], "D": ["C"],
    },

    # Two independent starts merge into one end.
    # Requires at least one dummy to distinguish the two paths.
    "two_starts_one_end": {
        "A": [], "B": [], "C": ["A", "B"],
    },

    # Classic diamond: A fans out to B,C; both merge into D.
    "diamond": {
        "A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"],
    },

    # C and D share the *exact same* non-empty predecessor set {A, B}.
    # KNOWN BUG for Roy+simplify: simplify() creates a directed cycle.
    # Fondahl handles this correctly via convergence-node reuse.
    "shared_predecessor_set": {
        "A": [], "B": [], "C": ["A", "B"], "D": ["A", "B"],
    },

    # Classic six-activity Fondahl textbook example.
    # A,B start; C<-A; D<-A,B; E<-B; F<-C,D,E.
    "classic_fondahl_six": {
        "A": [], "B": [], "C": ["A"], "D": ["A", "B"], "E": ["B"],
        "F": ["C", "D", "E"],
    },

    # shared_predecessor_set plus a downstream activity — same Roy bug.
    "shared_preds_with_sink": {
        "A": [], "B": [], "C": ["A", "B"], "D": ["A", "B"], "E": ["C", "D"],
    },

    # Ten-activity textbook network with multiple merges and splits.
    "textbook_ten": {
        "A": [], "B": [],
        "C": ["A"], "D": ["A", "B"], "E": ["B"],
        "F": ["C"], "G": ["C", "D"], "H": ["D", "E"],
        "I": ["F", "G"], "J": ["G", "H"],
    },

    # A,B→C; B,C→D — two merge points sharing activity B.
    "two_merges": {
        "A": [], "B": [], "C": ["A", "B"], "D": ["B", "C"],
    },

    # Wide fan-in: five independent activities all lead to F.
    "wide_fan_in": {
        "A": [], "B": [], "C": [], "D": [], "E": [],
        "F": ["A", "B", "C", "D", "E"],
    },
}

# Cases where Roy+simplify has a known cycle-creation bug.
ROY_BROKEN_CASES = {"shared_predecessor_set", "shared_preds_with_sink"}

# Working cases for Roy (excludes the broken ones).
ROY_WORKING = {k: v for k, v in CASES.items() if k not in ROY_BROKEN_CASES}


# ============================================================
# TestPredecessorTable
# ============================================================

class TestPredecessorTable(unittest.TestCase):
    """Unit tests for PredecessorTable's own properties and derived fields.

    PredecessorTable is the input layer: it stores *immediate* predecessors
    and derives *distant* (transitive) predecessors via the transitive closure.
    These tests check the logical guarantees before any graph is built.
    """

    # ------------------------------------------------------------------
    # activity_names
    # ------------------------------------------------------------------

    def test_activity_names_complete(self):
        """activity_names contains every key supplied to the constructor."""
        data = {"A": [], "B": ["A"], "C": ["B"]}
        table = make_table(data)
        self.assertEqual(set(table.activity_names), set(data.keys()))

    def test_activity_names_no_extras(self):
        """activity_names contains no activities that were not supplied."""
        data = {"X": [], "Y": ["X"]}
        table = make_table(data)
        self.assertEqual(len(table.activity_names), 2)

    # ------------------------------------------------------------------
    # start and end activities
    # ------------------------------------------------------------------

    def test_start_activities_have_no_predecessors(self):
        """Every start activity has an empty immediate predecessor set."""
        table = make_table(CASES["classic_fondahl_six"])
        for act in table.start_activities:
            self.assertEqual(table.immediate_predecessor[act], set())

    def test_end_activities_are_not_predecessors_of_anything(self):
        """End activities do not appear in any other activity's predecessor list."""
        for name, data in CASES.items():
            with self.subTest(case=name):
                table = make_table(data)
                all_preds = set().union(*table.immediate_predecessor.values())
                for act in table.end_activities:
                    self.assertNotIn(
                        act, all_preds,
                        f"[{name}] End activity {act!r} appears as a predecessor"
                    )

    def test_single_activity_is_both_start_and_end(self):
        """A single-activity project has that activity as both start and end."""
        table = make_table({"A": []})
        self.assertIn("A", table.start_activities)
        self.assertIn("A", table.end_activities)

    # ------------------------------------------------------------------
    # immediate_predecessor
    # ------------------------------------------------------------------

    def test_immediate_predecessor_subset_of_distant(self):
        """Immediate predecessors are always a subset of distant predecessors."""
        for name, data in CASES.items():
            with self.subTest(case=name):
                table = make_table(data)
                for act in table.activity_names:
                    self.assertTrue(
                        table.immediate_predecessor[act].issubset(
                            table.distant_predecessor[act]
                        ),
                        f"[{name}] immediate_predecessor[{act!r}] not subset of "
                        f"distant_predecessor[{act!r}]"
                    )

    # ------------------------------------------------------------------
    # distant_predecessor — transitivity and anti-reflexivity
    # ------------------------------------------------------------------

    def test_distant_predecessor_transitive(self):
        """If A precedes B and B precedes C, then A precedes C (transitivity)."""
        table = make_table({"A": [], "B": ["A"], "C": ["B"], "D": ["C"]})
        dp = table.distant_predecessor
        # A → B → C → D, so A is a distant predecessor of C and D.
        self.assertIn("A", dp["C"])
        self.assertIn("A", dp["D"])
        self.assertIn("B", dp["D"])

    def test_distant_predecessor_antireflexive(self):
        """No activity is a distant predecessor of itself."""
        for name, data in CASES.items():
            with self.subTest(case=name):
                table = make_table(data)
                dp = table.distant_predecessor
                for act in table.activity_names:
                    self.assertNotIn(
                        act, dp[act],
                        f"[{name}] {act!r} is a distant predecessor of itself"
                    )

    def test_chain_distant_predecessor_full(self):
        """A four-activity chain gives the expected full transitive closure."""
        table = make_table(CASES["chain_four"])
        dp = table.distant_predecessor
        self.assertEqual(dp["A"], set())
        self.assertEqual(dp["B"], {"A"})
        self.assertEqual(dp["C"], {"A", "B"})
        self.assertEqual(dp["D"], {"A", "B", "C"})

    def test_diamond_distant_predecessor(self):
        """Diamond: D has {A, B, C} as its distant predecessors."""
        table = make_table(CASES["diamond"])
        dp = table.distant_predecessor
        self.assertEqual(dp["D"], {"A", "B", "C"})
        # A is common to both B and C
        self.assertIn("A", dp["B"])
        self.assertIn("A", dp["C"])

    def test_no_predecessors_for_starts(self):
        """Start activities always have an empty distant predecessor set."""
        for name, data in CASES.items():
            with self.subTest(case=name):
                table = make_table(data)
                dp = table.distant_predecessor
                for act in table.start_activities:
                    self.assertEqual(dp[act], set(),
                                     f"[{name}] Start activity {act!r} has predecessors")


# ============================================================
# TestFondahlConstruct
# ============================================================

class TestFondahlConstruct(unittest.TestCase):
    """White-box unit tests for the raw fondahl_construct() function.

    fondahl_construct(immediate_predecessors) returns a triple
    (nodes, real_arcs, dummy_arcs) where:
        nodes      – set of integer node IDs
        real_arcs  – list of (tail, head, activity_name) triples
        dummy_arcs – list of (tail, head) pairs

    Node layout:
        0 – global SOURCE (tail of every start activity)
        1 – global SINK
        2, 3, … – fresh head node allocated for each real activity

    Terminating dummies:
        Every end activity's head node is connected to the SINK via a dummy.
        So a single activity has exactly 1 dummy; a chain of N activities
        also has exactly 1 dummy (only the last activity is an end activity).

    These tests inspect the raw output without involving ProjectGraph or
    _simplify_sweep, so they can pinpoint algorithm bugs independently.
    """

    def _run(self, data: dict):
        table = make_table(data)
        return fondahl_construct(table.immediate_predecessor)

    # ------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------

    def test_returns_three_tuple(self):
        """fondahl_construct returns a 3-tuple (nodes, real_arcs, dummy_arcs)."""
        result = self._run(CASES["chain_four"])
        self.assertEqual(len(result), 3)

    def test_real_arcs_cover_all_activities(self):
        """Every activity appears exactly once in real_arcs."""
        for name, data in CASES.items():
            with self.subTest(case=name):
                nodes, real_arcs, dummy_arcs = self._run(data)
                arc_names = [arc[2] for arc in real_arcs]
                self.assertEqual(
                    sorted(arc_names), sorted(data.keys()),
                    f"[{name}] real_arcs activity names don't match input"
                )

    def test_nodes_cover_arc_endpoints(self):
        """All tail/head nodes of real and dummy arcs appear in the nodes set."""
        for name, data in CASES.items():
            with self.subTest(case=name):
                nodes, real_arcs, dummy_arcs = self._run(data)
                for tail, head, _ in real_arcs:
                    self.assertIn(tail, nodes)
                    self.assertIn(head, nodes)
                for tail, head in dummy_arcs:
                    self.assertIn(tail, nodes)
                    self.assertIn(head, nodes)

    # ------------------------------------------------------------------
    # No parallel real activities (same tail AND same head)
    # ------------------------------------------------------------------

    def test_no_parallel_real_activities(self):
        """Two real activities never share both tail and head.

        Parallel activities (same tail, same head) are forbidden in a DiGraph
        because the second edge overwrites the first.  Fondahl avoids this
        by giving every real activity a unique fresh head node.
        """
        for name, data in CASES.items():
            with self.subTest(case=name):
                nodes, real_arcs, dummy_arcs = self._run(data)
                arc_pairs = [(tail, head) for tail, head, _ in real_arcs]
                self.assertEqual(
                    len(arc_pairs), len(set(arc_pairs)),
                    f"[{name}] Duplicate (tail,head) pair in real_arcs"
                )

    # ------------------------------------------------------------------
    # Unique head nodes for real activities
    # ------------------------------------------------------------------

    def test_real_activity_heads_unique(self):
        """Every real activity gets its own fresh head node.

        This is Rule 0 of Fondahl: the convergence-node reuse applies only
        to the *tail* (source) side; each activity's own head is always fresh.
        """
        for name, data in CASES.items():
            with self.subTest(case=name):
                nodes, real_arcs, dummy_arcs = self._run(data)
                heads = [head for _, head, _ in real_arcs]
                self.assertEqual(
                    len(heads), len(set(heads)),
                    f"[{name}] Two real activities share a head node"
                )

    # ------------------------------------------------------------------
    # Convergence-node reuse (Rule 2)
    # ------------------------------------------------------------------

    def test_convergence_node_reuse_shared_predecessor_set(self):
        """C and D share preds {A,B} → Fondahl gives them the same tail node.

        This verifies Rule 2: activities with an identical non-empty predecessor
        set are routed through a single shared convergence node, avoiding the
        cycle bug that arises in Roy+simplify.
        """
        nodes, real_arcs, dummy_arcs = self._run(CASES["shared_predecessor_set"])
        arc_map = {name: (tail, head) for tail, head, name in real_arcs}
        # C and D must share the same tail node.
        self.assertEqual(
            arc_map["C"][0], arc_map["D"][0],
            "C and D should share the same convergence tail node"
        )

    def test_convergence_node_for_classic_fondahl_six(self):
        """In classic_fondahl_six, every arc endpoint is in the nodes set."""
        nodes, real_arcs, dummy_arcs = self._run(CASES["classic_fondahl_six"])
        arc_map = {name: (tail, head) for tail, head, name in real_arcs}
        for name, (tail, head) in arc_map.items():
            self.assertIn(tail, nodes)
            self.assertIn(head, nodes)

    # ------------------------------------------------------------------
    # Direct attachment rule (Rule 1) and terminating dummies
    # ------------------------------------------------------------------

    def test_direct_attachment_chain(self):
        """In a chain A→B→C→D, direct attachment (Rule 1) avoids predecessor dummies.

        B's tail = A's head (direct); C's tail = B's head (direct);
        D's tail = C's head (direct).  Only one dummy is needed: the
        *terminating* dummy that connects D's head to the global SINK node.
        """
        nodes, real_arcs, dummy_arcs = self._run(CASES["chain_four"])
        self.assertEqual(
            len(dummy_arcs), 1,
            f"chain_four should have exactly 1 dummy (terminating SINK connector), "
            f"got {len(dummy_arcs)}: {dummy_arcs}"
        )

    def test_single_activity_one_sink_dummy(self):
        """A project with one activity has exactly one dummy: the SINK connector.

        Node layout: SOURCE(0) → A → head_A → dummy → SINK(1).
        The dummy from head_A to SINK is the sole dummy activity.
        """
        nodes, real_arcs, dummy_arcs = self._run(CASES["single_activity"])
        self.assertEqual(len(real_arcs), 1)
        self.assertEqual(
            len(dummy_arcs), 1,
            f"single_activity should have exactly 1 dummy (SINK connector), "
            f"got {len(dummy_arcs)}: {dummy_arcs}"
        )


# ============================================================
# TestProjectGraphStructure
# ============================================================

class TestProjectGraphStructure(unittest.TestCase):
    """Structural / API tests for ProjectGraph.

    These tests check property consistency, inverse relationships between
    nodes_from_edges / edges_from_nodes, and the semantics of path_matrix.
    They use the Fondahl method to avoid the Roy cycle bug.
    """

    def _project(self, case_name, simplify=True):
        table = make_table(CASES[case_name])
        return table.create_project_fondahl(simplify=simplify)

    # ------------------------------------------------------------------
    # activities / dummies / actual_activities partition
    # ------------------------------------------------------------------

    def test_activities_partition(self):
        """activities = actual_activities ∪ dummies (disjoint)."""
        for name in CASES:
            with self.subTest(case=name):
                p = self._project(name)
                self.assertEqual(
                    set(p.activities),
                    set(p.actual_activities) | set(p.dummies)
                )
                self.assertTrue(
                    set(p.actual_activities).isdisjoint(set(p.dummies)),
                    f"[{name}] actual_activities and dummies overlap"
                )

    def test_dummies_start_with_at(self):
        """All dummy activity names start with '@'."""
        for name in CASES:
            with self.subTest(case=name):
                p = self._project(name)
                for dummy in p.dummies:
                    self.assertTrue(
                        dummy.startswith("@"),
                        f"[{name}] Dummy {dummy!r} does not start with '@'"
                    )

    def test_actual_activities_match_input(self):
        """actual_activities contains exactly the activities supplied."""
        for name, data in CASES.items():
            with self.subTest(case=name):
                table = make_table(data)
                p = table.create_project_fondahl(simplify=True)
                self.assertEqual(
                    set(p.actual_activities), set(data.keys()),
                    f"[{name}] actual_activities don't match input keys"
                )

    # ------------------------------------------------------------------
    # nodes_from_edges / edges_from_nodes
    # ------------------------------------------------------------------

    def test_nodes_from_edges_keys_are_activities(self):
        """nodes_from_edges keys are exactly the project's activities."""
        p = self._project("diamond")
        self.assertEqual(set(p.nodes_from_edges.keys()), set(p.activities))

    def test_edges_from_nodes_inverse(self):
        """edges_from_nodes[(tail,head)] == activity_name is consistent with nodes_from_edges.

        edges_from_nodes is keyed by (tail, head) string-tuple — not a nested
        dict.  nodes_from_edges maps each activity name to its (tail, head)
        tuple.  The two dicts must be perfectly inverse of each other.
        """
        for name in CASES:
            with self.subTest(case=name):
                p = self._project(name)
                nfe = p.nodes_from_edges   # {act: (tail_str, head_str)}
                efn = p.edges_from_nodes   # {(tail_str, head_str): act}
                for act, (tail, head) in nfe.items():
                    self.assertEqual(
                        efn[tail, head], act,
                        f"[{name}] edges_from_nodes[({tail!r},{head!r})] != {act!r}"
                    )

    # ------------------------------------------------------------------
    # path_matrix — semantics
    # ------------------------------------------------------------------

    def test_path_matrix_is_dataframe(self):
        """path_matrix() returns a pandas DataFrame (rows=paths, cols=activities)."""
        import pandas as pd
        for name in CASES:
            with self.subTest(case=name):
                p = self._project(name)
                pm = p.path_matrix()
                self.assertIsInstance(pm, pd.DataFrame,
                                      f"[{name}] path_matrix() is not a DataFrame")

    def test_path_matrix_columns_are_activities(self):
        """path_matrix columns are a subset of the project's activity names."""
        for name in CASES:
            with self.subTest(case=name):
                p = self._project(name)
                pm = p.path_matrix()
                extra = set(pm.columns) - set(p.activities)
                self.assertEqual(extra, set(),
                                 f"[{name}] path_matrix has unexpected columns: {extra}")

    def test_path_matrix_chain_single_route(self):
        """A linear chain has exactly one complete path covering all activities."""
        p = self._project("chain_four")
        pm = p.path_matrix()
        # A chain has exactly one path (Route_1).
        self.assertEqual(pm.shape[0], 1,
                         f"chain_four should have 1 path, got {pm.shape[0]}")
        # Every real activity (A, B, C, D) must lie on that path.
        for act in ("A", "B", "C", "D"):
            self.assertEqual(pm.iloc[0][act], 1,
                             f"Activity {act!r} should be on the single chain path")

    def test_path_matrix_diamond_two_routes(self):
        """A diamond (A→B→D and A→C→D) has exactly two paths."""
        p = self._project("diamond")
        pm = p.path_matrix()
        self.assertEqual(pm.shape[0], 2,
                         f"diamond should have 2 paths, got {pm.shape[0]}")

    def test_path_matrix_values_binary(self):
        """path_matrix values are 0 or 1."""
        for name in CASES:
            with self.subTest(case=name):
                p = self._project(name)
                pm = p.path_matrix()
                unique_vals = set(pm.values.flatten())
                self.assertTrue(
                    unique_vals.issubset({0, 1}),
                    f"[{name}] path_matrix has non-binary values: {unique_vals}"
                )

    # ------------------------------------------------------------------
    # is_dag
    # ------------------------------------------------------------------

    def test_project_is_dag(self):
        """Every Fondahl project is a directed acyclic graph."""
        for name in CASES:
            with self.subTest(case=name):
                p = self._project(name)
                self.assertTrue(
                    nx.is_directed_acyclic_graph(p.pert_graph),
                    f"[{name}] ProjectGraph is not a DAG"
                )


# ============================================================
# TestAssessDummy
# ============================================================

class TestAssessDummy(unittest.TestCase):
    """Unit tests for ProjectGraph.assess_dummy (static method).

    assess_dummy(name, project) decides whether a dummy can be removed or
    contracted without violating the core invariant.  It returns:
        ("contract", new_project)  – the dummy's tail was merged into its head
        ("remove",   new_project)  – the dummy was simply deleted
        ("keep",     project)      – neither operation is safe; return original

    All results must preserve the invariant when the operation is "contract"
    or "remove".
    """

    def _build_with_dummy(self, data: dict):
        """Return (table, project_unsimplified) via Roy, so dummies are visible."""
        table = make_table(data)
        project = table.create_project_roy(simplify=False)
        return table, project

    def test_assess_dummy_returns_tuple(self):
        """assess_dummy always returns a 2-tuple."""
        table, project = self._build_with_dummy(CASES["two_starts_one_end"])
        if not project.dummies:
            self.skipTest("No dummies in this case")
        dummy = next(iter(project.dummies))
        result = ProjectGraph.assess_dummy(dummy, project)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_assess_dummy_valid_decision(self):
        """assess_dummy's decision is a bool: True=simplified, False=kept.

        The actual API returns (bool, ProjectGraph) where True means the dummy
        was safely eliminated (by contract or remove) and False means neither
        operation preserved the invariant so the original project is returned.
        """
        for name, data in CASES.items():
            if name in ROY_BROKEN_CASES:
                continue
            with self.subTest(case=name):
                table, project = self._build_with_dummy(data)
                for dummy in list(project.dummies):
                    decision, _ = ProjectGraph.assess_dummy(dummy, project)
                    self.assertIsInstance(
                        decision, bool,
                        f"[{name}] assess_dummy decision is not a bool: {decision!r} for dummy {dummy!r}"
                    )

    def test_assess_dummy_simplified_preserves_invariant(self):
        """When assess_dummy returns True (dummy eliminated), the result satisfies the invariant.

        assess_dummy tries contract first, then remove; True means one of them
        succeeded.  The returned project must still encode the same precedence
        as the original table (the core invariant).
        """
        for name, data in CASES.items():
            if name in ROY_BROKEN_CASES:
                continue
            with self.subTest(case=name):
                table, project = self._build_with_dummy(data)
                for dummy in list(project.dummies):
                    decision, new_project = ProjectGraph.assess_dummy(dummy, project)
                    if decision is True:
                        self.assertTrue(
                            check_invariant(table, new_project),
                            f"[{name}] Invariant violated after simplifying {dummy!r}"
                        )

    def test_assess_dummy_kept_returns_original_activities(self):
        """When assess_dummy returns False (dummy kept), the project is unchanged.

        If neither contract nor remove is safe (decision is False), assess_dummy
        returns a copy of the original project; its activity set must match.
        """
        for name, data in CASES.items():
            if name in ROY_BROKEN_CASES:
                continue
            with self.subTest(case=name):
                table, project = self._build_with_dummy(data)
                for dummy in list(project.dummies):
                    decision, returned = ProjectGraph.assess_dummy(dummy, project)
                    if decision is False:
                        self.assertEqual(
                            set(returned.activities), set(project.activities),
                            f"[{name}] False decision changed the activity set for dummy {dummy!r}"
                        )

    def test_assess_dummy_result_is_project(self):
        """The second element of assess_dummy's return is always a ProjectGraph."""
        for name, data in CASES.items():
            if name in ROY_BROKEN_CASES:
                continue
            with self.subTest(case=name):
                table, project = self._build_with_dummy(data)
                for dummy in list(project.dummies):
                    _, returned = ProjectGraph.assess_dummy(dummy, project)
                    self.assertIsInstance(
                        returned, ProjectGraph,
                        f"[{name}] second return value is not a ProjectGraph for dummy {dummy!r}"
                    )


# ============================================================
# TestSimplificationSweep
# ============================================================

class TestSimplificationSweep(unittest.TestCase):
    """Unit tests for _simplify_sweep(project).

    _simplify_sweep does a single non-resetting pass over all dummies,
    trying assess_dummy on each, and returns (updated_project, n_removed)
    where n_removed is the count of dummies eliminated in this pass.

    Roy's raw graph includes:
        - One dummy per predecessor link (e.g. B←A → @∇A⤑ΔB)
        - One SOURCE-connector dummy per start activity  (@∇StartOfProject⤑ΔX)
        - One SINK-connector dummy per end activity      (@∇X⤑ΔEndOfProject)
    So chain_four (3 predecessor links, 1 start, 1 end) has 5 raw dummies;
    after one sweep all 5 are contractable/removable, giving n_removed=5.

    The function terminates in O(D·n²) time and must:
    - Return a 2-tuple (ProjectGraph, int).
    - Never increase the number of dummies.
    - Preserve the core invariant after each pass.
    - Return n_removed == 0 when no further simplification is possible.
    """

    def _build_raw(self, data: dict):
        table = make_table(data)
        return table, table.create_project_roy(simplify=False)

    def test_returns_two_tuple(self):
        """_simplify_sweep returns a 2-tuple (ProjectGraph, int)."""
        _, project = self._build_raw(CASES["two_starts_one_end"])
        result = _simplify_sweep(project)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], ProjectGraph)
        self.assertIsInstance(result[1], int)

    def test_n_removed_nonnegative(self):
        """n_removed is always >= 0."""
        for name, data in CASES.items():
            if name in ROY_BROKEN_CASES:
                continue
            with self.subTest(case=name):
                _, project = self._build_raw(data)
                _, n = _simplify_sweep(project)
                self.assertGreaterEqual(n, 0)

    def test_never_increases_dummies(self):
        """A sweep never produces more dummies than it started with."""
        for name, data in CASES.items():
            if name in ROY_BROKEN_CASES:
                continue
            with self.subTest(case=name):
                table, project = self._build_raw(data)
                before = len(project.dummies)
                new_project, n = _simplify_sweep(project)
                after = len(new_project.dummies)
                self.assertLessEqual(
                    after, before,
                    f"[{name}] Dummies increased from {before} to {after}"
                )

    def test_preserves_invariant(self):
        """The invariant holds after each _simplify_sweep pass."""
        for name, data in CASES.items():
            if name in ROY_BROKEN_CASES:
                continue
            with self.subTest(case=name):
                table, project = self._build_raw(data)
                new_project, _ = _simplify_sweep(project)
                self.assertTrue(
                    check_invariant(table, new_project),
                    f"[{name}] Invariant violated after one sweep"
                )

    def test_converges_to_fixed_point(self):
        """Repeated sweeps eventually produce n_removed == 0 (fixed point)."""
        for name, data in CASES.items():
            if name in ROY_BROKEN_CASES:
                continue
            with self.subTest(case=name):
                table, project = self._build_raw(data)
                max_iters = len(project.dummies) + 2
                for _ in range(max_iters):
                    project, n = _simplify_sweep(project)
                    if n == 0:
                        break
                else:
                    self.fail(f"[{name}] _simplify_sweep did not reach fixed point")
                # After fixed point, another sweep should also give 0.
                _, n_extra = _simplify_sweep(project)
                self.assertEqual(n_extra, 0,
                                 f"[{name}] Another sweep after fixed point removed dummies")

    def test_chain_sweep_removes_all_dummies(self):
        """Roy raw chain_four has 5 dummies; a single sweep removes all of them.

        Roy creates dummies for each predecessor link (A←B, B←C, C←D),
        plus a SOURCE connector for A and a SINK connector for D, totalling 5.
        In a simple chain every dummy is either safely contractable or
        removable, so one _simplify_sweep pass returns n_removed == 5.
        """
        table, project = self._build_raw(CASES["chain_four"])
        initial_count = len(project.dummies)
        self.assertGreater(initial_count, 0,
                           "Roy raw chain_four should have at least one dummy")
        _, n = _simplify_sweep(project)
        self.assertEqual(n, initial_count,
                         f"Expected all {initial_count} dummies removed in one sweep")


# ============================================================
# TestPertConstructionRoy
# ============================================================

class TestPertConstructionRoy(unittest.TestCase):
    """End-to-end tests for PredecessorTable.create_project_roy (Roy + simplify).

    The Roy method creates one dummy per predecessor link (plus SOURCE/SINK
    connectors), then iterates _simplify_sweep until no more dummies can be
    removed.  It has a known bug for the two cases in ROY_BROKEN_CASES.

    Structure of the class:
        * Individual correctness tests (one per CASE)
        * @expectedFailure tests documenting the known bug
        * Structural property tests (DAG, dummy count, etc.)
    """

    def _verify(self, case_name):
        data = CASES[case_name]
        table = make_table(data)
        project = table.create_project_roy(simplify=True)
        self.assertTrue(
            check_invariant(table, project),
            msg=(f"[{case_name}] Invariant failed.\n"
                 f"  table   ={table.distant_predecessor}\n"
                 f"  project ={project.distant_predecessor(dummies=False)}")
        )
        return table, project

    # ------------------------------------------------------------------
    # Working cases
    # ------------------------------------------------------------------

    def test_single_activity(self):
        """One activity: trivial two-node graph, invariant satisfied."""
        self._verify("single_activity")

    def test_chain_four(self):
        """Linear chain A→B→C→D: invariant satisfied."""
        self._verify("chain_four")

    def test_two_starts_one_end(self):
        """Two independent starts merge via at least one dummy."""
        self._verify("two_starts_one_end")

    def test_diamond(self):
        """Diamond A→{B,C}→D: Roy correctly handles the merge."""
        self._verify("diamond")

    def test_classic_fondahl_six(self):
        """Six-activity Fondahl textbook example passes under Roy."""
        self._verify("classic_fondahl_six")

    def test_textbook_ten(self):
        """Ten-activity textbook network satisfies the invariant under Roy."""
        self._verify("textbook_ten")

    def test_two_merges(self):
        """Two merge points: A,B→C and B,C→D."""
        self._verify("two_merges")

    def test_wide_fan_in(self):
        """Five-to-one fan-in network satisfies the invariant under Roy."""
        self._verify("wide_fan_in")

    # ------------------------------------------------------------------
    # Known-bug cases (documented as expected failures)
    # ------------------------------------------------------------------

    @unittest.expectedFailure
    def test_shared_predecessor_set_bug(self):
        """KNOWN BUG: Roy+simplify creates a directed cycle for shared pred sets.

        When C and D both have exactly the same predecessor set {A, B},
        simplify() contracts one convergence dummy, which redirects the other
        dummy's tail, creating a cycle.  nx.topological_sort raises
        NetworkXUnfeasible and the test fails — which is *expected*.
        """
        self._verify("shared_predecessor_set")

    @unittest.expectedFailure
    def test_shared_preds_with_sink_bug(self):
        """KNOWN BUG: same cycle issue with a downstream sink activity."""
        self._verify("shared_preds_with_sink")

    # ------------------------------------------------------------------
    # Structural properties (working cases only)
    # ------------------------------------------------------------------

    def test_chain_needs_no_dummies(self):
        """A linear chain requires zero dummy activities after simplification."""
        _, project = self._verify("chain_four")
        self.assertEqual(len(project.dummies), 0,
                         "chain_four should have zero dummies after simplification")

    def test_simplify_false_still_correct(self):
        """The raw un-simplified Roy graph satisfies the invariant for all cases."""
        for name, data in CASES.items():
            with self.subTest(case=name):
                table = make_table(data)
                project_raw = table.create_project_roy(simplify=False)
                self.assertTrue(
                    check_invariant(table, project_raw),
                    f"[{name}] Un-simplified Roy graph invariant failed"
                )

    def test_simplified_has_fewer_or_equal_dummies(self):
        """Simplification never increases dummy count (working cases only)."""
        for name, data in ROY_WORKING.items():
            with self.subTest(case=name):
                table = make_table(data)
                raw = table.create_project_roy(simplify=False)
                simplified = table.create_project_roy(simplify=True)
                self.assertLessEqual(
                    len(simplified.dummies), len(raw.dummies),
                    f"[{name}] Simplification increased dummies"
                )

    def test_project_is_dag(self):
        """Roy+simplify result is a DAG (working cases only)."""
        for name, data in ROY_WORKING.items():
            with self.subTest(case=name):
                table = make_table(data)
                project = table.create_project_roy(simplify=True)
                self.assertTrue(
                    nx.is_directed_acyclic_graph(project.pert_graph),
                    f"[{name}] Roy+simplify graph is not a DAG"
                )

    def test_actual_activities_preserved(self):
        """create_project preserves all original activity names (working cases)."""
        for name, data in ROY_WORKING.items():
            with self.subTest(case=name):
                table = make_table(data)
                project = table.create_project_roy(simplify=True)
                self.assertEqual(
                    set(project.actual_activities), set(data.keys()),
                    f"[{name}] actual_activities don't match input"
                )

    def test_distant_predecessors_correct_chain(self):
        """Full transitive-closure check on chain_four."""
        table, project = self._verify("chain_four")
        dp = project.distant_predecessor(dummies=False)
        self.assertEqual(dp["A"], set())
        self.assertEqual(dp["B"], {"A"})
        self.assertEqual(dp["C"], {"A", "B"})
        self.assertEqual(dp["D"], {"A", "B", "C"})


# ============================================================
# TestPertConstructionFondahl
# ============================================================

class TestPertConstructionFondahl(unittest.TestCase):
    """End-to-end tests for PredecessorTable.create_project_fondahl.

    ALL ten CASES must pass, including the two that break Roy+simplify.
    Fondahl's direct construction avoids the cycle-creation bug by reusing
    convergence nodes for identical predecessor sets (Rule 2) instead of
    creating separate dummies that later collide.
    """

    def _verify(self, case_name, simplify=True):
        data = CASES[case_name]
        table = make_table(data)
        project = table.create_project_fondahl(simplify=simplify)
        self.assertTrue(
            check_invariant(table, project),
            msg=(f"[{case_name}] Invariant failed (simplify={simplify}).\n"
                 f"  table   ={table.distant_predecessor}\n"
                 f"  project ={project.distant_predecessor(dummies=False)}")
        )
        return table, project

    # ------------------------------------------------------------------
    # All ten cases (including the two broken under Roy)
    # ------------------------------------------------------------------

    def test_single_activity(self):
        self._verify("single_activity")

    def test_chain_four(self):
        self._verify("chain_four")

    def test_two_starts_one_end(self):
        self._verify("two_starts_one_end")

    def test_diamond(self):
        self._verify("diamond")

    def test_shared_predecessor_set(self):
        """Must pass: Fondahl handles shared predecessor sets without cycles."""
        self._verify("shared_predecessor_set")

    def test_classic_fondahl_six(self):
        self._verify("classic_fondahl_six")

    def test_shared_preds_with_sink(self):
        """Must pass: Fondahl handles this variant without the Roy cycle bug."""
        self._verify("shared_preds_with_sink")

    def test_textbook_ten(self):
        self._verify("textbook_ten")

    def test_two_merges(self):
        self._verify("two_merges")

    def test_wide_fan_in(self):
        self._verify("wide_fan_in")

    # ------------------------------------------------------------------
    # simplify=False variant (should also pass for all cases)
    # ------------------------------------------------------------------

    def test_simplify_false_all_cases(self):
        """Raw Fondahl output (simplify=False) satisfies the invariant for all cases."""
        for name in CASES:
            with self.subTest(case=name):
                self._verify(name, simplify=False)

    # ------------------------------------------------------------------
    # Structural properties
    # ------------------------------------------------------------------

    def test_result_is_dag(self):
        """Fondahl output is always a valid DAG."""
        for name in CASES:
            with self.subTest(case=name):
                table = make_table(CASES[name])
                project = table.create_project_fondahl(simplify=True)
                self.assertTrue(
                    nx.is_directed_acyclic_graph(project.pert_graph),
                    f"[{name}] Fondahl result is not a DAG"
                )

    def test_actual_activities_preserved(self):
        """create_project_fondahl preserves all original activity names."""
        for name, data in CASES.items():
            with self.subTest(case=name):
                table = make_table(data)
                project = table.create_project_fondahl(simplify=True)
                self.assertEqual(
                    set(project.actual_activities), set(data.keys()),
                    f"[{name}] actual_activities don't match input"
                )

    def test_dummy_names_match_endpoints(self):
        """Each dummy's name encodes its actual tail→head node IDs: @{tail}→{head}."""
        for name in CASES:
            with self.subTest(case=name):
                _, project = self._verify(name)
                nfe = project.nodes_from_edges
                for act in project.dummies:
                    tail, head = nfe[act]
                    expected = f"@{tail}\u2192{head}"
                    self.assertEqual(
                        act, expected,
                        f"[{name}] dummy {act!r} should be named {expected!r}"
                    )

    def test_fondahl_chain_no_dummies(self):
        """A linear chain produces zero dummies under Fondahl (after simplification)."""
        _, project = self._verify("chain_four")
        self.assertEqual(len(project.dummies), 0)

    def test_fondahl_single_activity_two_nodes(self):
        """A single activity produces exactly two nodes (source and sink)."""
        _, project = self._verify("single_activity")
        self.assertEqual(len(project.nodes), 2)

    def test_fondahl_fewer_or_equal_dummies_than_roy(self):
        """Fondahl produces ≤ dummies compared to Roy+simplify (working cases)."""
        for name, data in ROY_WORKING.items():
            with self.subTest(case=name):
                table = make_table(data)
                roy = table.create_project_roy(simplify=True)
                fond = table.create_project_fondahl(simplify=True)
                self.assertLessEqual(
                    len(fond.dummies), len(roy.dummies),
                    f"[{name}] Fondahl has more dummies than Roy"
                )

    def test_fondahl_distant_predecessor_chain(self):
        """Full transitive-closure check on chain_four via Fondahl."""
        table, project = self._verify("chain_four")
        dp = project.distant_predecessor(dummies=False)
        self.assertEqual(dp["A"], set())
        self.assertEqual(dp["B"], {"A"})
        self.assertEqual(dp["C"], {"A", "B"})
        self.assertEqual(dp["D"], {"A", "B", "C"})


# ============================================================
# TestCrossMethod
# ============================================================

class TestCrossMethod(unittest.TestCase):
    """Cross-validate Roy and Fondahl methods against each other.

    Both methods must encode the same predecessor semantics.  Where both
    work (all non-broken cases), they should agree on the invariant and
    on the set of actual activities, though they may differ in dummy count.
    """

    def test_both_satisfy_invariant(self):
        """Both methods satisfy the invariant on all working cases."""
        for name, data in ROY_WORKING.items():
            with self.subTest(case=name):
                table = make_table(data)
                roy = table.create_project_roy(simplify=True)
                fond = table.create_project_fondahl(simplify=True)
                self.assertTrue(check_invariant(table, roy),
                                f"[{name}] Roy invariant failed")
                self.assertTrue(check_invariant(table, fond),
                                f"[{name}] Fondahl invariant failed")

    def test_both_agree_on_actual_activities(self):
        """Both methods produce the same set of actual (real) activities."""
        for name, data in ROY_WORKING.items():
            with self.subTest(case=name):
                table = make_table(data)
                roy = table.create_project_roy(simplify=True)
                fond = table.create_project_fondahl(simplify=True)
                self.assertEqual(
                    set(roy.actual_activities), set(fond.actual_activities),
                    f"[{name}] Roy and Fondahl disagree on actual_activities"
                )

    def test_fondahl_leq_roy_dummies_working_cases(self):
        """Fondahl dummy count ≤ Roy dummy count on all working cases."""
        for name, data in ROY_WORKING.items():
            with self.subTest(case=name):
                table = make_table(data)
                roy = table.create_project_roy(simplify=True)
                fond = table.create_project_fondahl(simplify=True)
                self.assertLessEqual(
                    len(fond.dummies), len(roy.dummies),
                    f"[{name}] Fondahl has more dummies ({len(fond.dummies)}) "
                    f"than Roy ({len(roy.dummies)})"
                )

    def test_fondahl_passes_broken_cases(self):
        """Fondahl handles the two cases that are broken under Roy."""
        for name in ROY_BROKEN_CASES:
            with self.subTest(case=name):
                table = make_table(CASES[name])
                fond = table.create_project_fondahl(simplify=True)
                self.assertTrue(
                    check_invariant(table, fond),
                    f"[{name}] Fondahl invariant failed on broken-for-Roy case"
                )

    def test_distant_predecessors_agree(self):
        """project.distant_predecessor(dummies=False) is identical for both methods."""
        for name, data in ROY_WORKING.items():
            with self.subTest(case=name):
                table = make_table(data)
                roy = table.create_project_roy(simplify=True)
                fond = table.create_project_fondahl(simplify=True)
                self.assertEqual(
                    roy.distant_predecessor(dummies=False),
                    fond.distant_predecessor(dummies=False),
                    f"[{name}] Roy and Fondahl give different distant_predecessor"
                )


# ============================================================
# TestCalculatePert
# ============================================================

class TestCalculatePert(unittest.TestCase):
    """Smoke tests for ProjectGraph.calculate_pert with known-answer networks.

    calculate_pert(durations) returns a dict with two keys:
        'nodes'      – DataFrame with columns 'early' and 'late'
                       (early/late event times), indexed by node ID.
        'activities' – DataFrame with column 'H_total' (total float,
                       "Holgura total"), indexed by activity name.
                       Both real activities AND dummies are included.

    Derived quantities used in these tests:
        Project duration  = result['nodes']['early'].max()
        Float of act X    = result['activities'].loc[X, 'H_total']

    Network used below (diamond-like):
        A(3) → C(2) → E(4)
        A(3) → D(5) → E(4)    (D is on the critical path)
        B(1) → D(5)
        B(1) → C(2)

    Critical path: A → D → E  =  3 + 5 + 4  =  12
    Total floats:  A=0, B=2, C=3, D=0, E=0
    """

    NETWORK = {
        "A": [], "B": [],
        "C": ["A", "B"], "D": ["A", "B"],
        "E": ["C", "D"],
    }
    DURATIONS = {"A": 3, "B": 1, "C": 2, "D": 5, "E": 4}
    PROJECT_DURATION = 12  # A(3) + D(5) + E(4)

    def _build(self):
        table = make_table(self.NETWORK)
        project = table.create_project_fondahl(simplify=True)
        return project

    def test_calculate_pert_returns_dict(self):
        """calculate_pert returns a dict with 'nodes' and 'activities' keys."""
        project = self._build()
        result = project.calculate_pert(self.DURATIONS)
        self.assertIsInstance(result, dict)
        self.assertIn("nodes", result,
                      "calculate_pert result should contain 'nodes'")
        self.assertIn("activities", result,
                      "calculate_pert result should contain 'activities'")
        # Every real activity must appear in the activities DataFrame.
        for act in self.DURATIONS:
            self.assertIn(act, result["activities"].index,
                          f"Real activity {act!r} missing from activities DataFrame")

    def test_project_duration_correct(self):
        """The overall project duration equals the length of the critical path."""
        project = self._build()
        result = project.calculate_pert(self.DURATIONS)
        duration = result["nodes"]["early"].max()
        self.assertEqual(duration, self.PROJECT_DURATION,
                         f"Expected project duration {self.PROJECT_DURATION}, "
                         f"got {duration}")

    def test_critical_path_zero_float(self):
        """Activities on the critical path (A, D, E) have zero total float."""
        project = self._build()
        result = project.calculate_pert(self.DURATIONS)
        for act in ("A", "D", "E"):
            tf = result["activities"].loc[act, "H_total"]
            self.assertEqual(tf, 0,
                             f"Activity {act!r} on critical path has H_total={tf} "
                             f"(expected 0)")

    def test_noncritical_positive_float(self):
        """Non-critical activities (B, C) have positive total float."""
        project = self._build()
        result = project.calculate_pert(self.DURATIONS)
        for act in ("B", "C"):
            tf = result["activities"].loc[act, "H_total"]
            self.assertGreater(tf, 0,
                               f"Non-critical activity {act!r} has H_total={tf} "
                               f"(expected > 0)")

    def test_chain_pert_simple(self):
        """A→B→C with durations 2,3,4: project duration=9, all activities TF=0."""
        table = make_table({"A": [], "B": ["A"], "C": ["B"]})
        project = table.create_project_fondahl(simplify=True)
        durations = {"A": 2, "B": 3, "C": 4}
        result = project.calculate_pert(durations)
        duration = result["nodes"]["early"].max()
        self.assertEqual(duration, 9)
        for act in ("A", "B", "C"):
            tf = result["activities"].loc[act, "H_total"]
            self.assertEqual(tf, 0,
                             f"Chain activity {act!r} should have H_total=0, got {tf}")

    def test_diamond_pert_float(self):
        """Diamond A(2)→B(1)→D(3); A(2)→C(4)→D(3): critical is A-C-D=9.

        Durations: A=2, B=1, C=4, D=3.
        Critical path: A → C → D = 2+4+3 = 9.
        B is non-critical: A→B→D = 2+1+3 = 6 < 9, so H_total(B) > 0.
        """
        table = make_table(CASES["diamond"])
        project = table.create_project_fondahl(simplify=True)
        durations = {"A": 2, "B": 1, "C": 4, "D": 3}
        result = project.calculate_pert(durations)
        duration = result["nodes"]["early"].max()
        self.assertEqual(duration, 9,
                         f"Expected project duration 9, got {duration}")
        for act in ("A", "C", "D"):
            tf = result["activities"].loc[act, "H_total"]
            self.assertEqual(tf, 0,
                             f"Critical activity {act!r} should have H_total=0")
        tf_b = result["activities"].loc["B", "H_total"]
        self.assertGreater(tf_b, 0,
                           f"Non-critical activity B should have H_total>0, got {tf_b}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
