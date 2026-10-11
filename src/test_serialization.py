"""
test_serialization.py — Tests for pmULE graph and route-matrix serialisation
=============================================================================

Covers the four ProjectGraph methods and two module-level functions added
to pmule.py for persisting PERT graphs and route matrices to disk:

    ProjectGraph.to_graphml(path)          → GraphML file
    ProjectGraph.from_graphml(path)        → ProjectGraph
    ProjectGraph.save_route_matrix(R, path)→ .npz + .json sidecar
    ProjectGraph.load_route_matrix(path)   → pd.DataFrame

    build_and_save(name, preds, out_dir)   → ProjectGraph  (module-level)
    load_instance(name, out_dir)           → (ProjectGraph, pd.DataFrame)

All filesystem work happens inside a TemporaryDirectory so no artefacts
are left behind.

Test organisation
-----------------
Class                        What is tested
---------------------------------------------------------------------------
TestGraphMLRoundtrip         to_graphml → from_graphml preserves structure
TestGraphMLDictAttribs       to_graphml handles dict-valued edge/node attrs
TestRouteMatrixRoundtrip     save_route_matrix → load_route_matrix roundtrip
TestBuildAndSave             build_and_save creates expected files on disk
TestLoadInstance             load_instance returns correct types and content
TestBuildAndSaveLoadIntegr.  build_and_save + load_instance end-to-end
"""

import sys
import os
import pathlib
import tempfile
import unittest

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmule import (
    PredecessorTable,
    ProjectGraph,
    build_and_save,
    load_instance,
)


# ============================================================
# Shared helpers
# ============================================================

def _make_project(preds: dict) -> ProjectGraph:
    """Build a ProjectGraph from {activity: [list-of-preds]}."""
    return PredecessorTable({k: set(v) for k, v in preds.items()}).create_project()


# A small but non-trivial network used across several tests.
SIMPLE_PREDS = {
    "A": [],
    "B": ["A"],
    "C": ["A"],
    "D": ["B", "C"],
}

# A slightly larger network to exercise multiple paths.
MEDIUM_PREDS = {
    "A": [],
    "B": ["A"],
    "C": ["A"],
    "D": ["B"],
    "E": ["C"],
    "F": ["D", "E"],
}


# ============================================================
# GraphML roundtrip
# ============================================================

class TestGraphMLRoundtrip(unittest.TestCase):
    """to_graphml → from_graphml must reproduce the graph structure."""

    def setUp(self):
        self.project = _make_project(SIMPLE_PREDS)
        self._td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_roundtrip_creates_file(self):
        path = self.tmp / "test.graphml"
        self.project.to_graphml(path)
        self.assertTrue(path.exists(), "GraphML file was not created")

    def test_roundtrip_node_count(self):
        path = self.tmp / "test.graphml"
        self.project.to_graphml(path)
        loaded = ProjectGraph.from_graphml(path)
        self.assertEqual(
            self.project.pert_graph.number_of_nodes(),
            loaded.pert_graph.number_of_nodes(),
        )

    def test_roundtrip_edge_count(self):
        path = self.tmp / "test.graphml"
        self.project.to_graphml(path)
        loaded = ProjectGraph.from_graphml(path)
        self.assertEqual(
            self.project.pert_graph.number_of_edges(),
            loaded.pert_graph.number_of_edges(),
        )

    def test_roundtrip_activity_names(self):
        """Edge labels (activity names) must survive the roundtrip."""
        path = self.tmp / "test.graphml"
        self.project.to_graphml(path)
        loaded = ProjectGraph.from_graphml(path)

        orig_acts = {
            d.get("label") or d.get("name", "")
            for _, _, d in self.project.pert_graph.edges(data=True)
        }
        load_acts = {
            d.get("label") or d.get("name", "")
            for _, _, d in loaded.pert_graph.edges(data=True)
        }
        self.assertEqual(orig_acts, load_acts)

    def test_roundtrip_preserves_precedence_invariant(self):
        """Loaded graph must encode the same precedence as the original."""
        table = PredecessorTable({k: set(v) for k, v in SIMPLE_PREDS.items()})
        path = self.tmp / "test.graphml"
        self.project.to_graphml(path)
        loaded = ProjectGraph.from_graphml(path)
        self.assertEqual(
            loaded.distant_predecessor(dummies=False),
            table.distant_predecessor,
        )

    def test_roundtrip_medium_network(self):
        """Roundtrip on a larger network."""
        project = _make_project(MEDIUM_PREDS)
        path = self.tmp / "medium.graphml"
        project.to_graphml(path)
        loaded = ProjectGraph.from_graphml(path)
        self.assertEqual(
            project.pert_graph.number_of_edges(),
            loaded.pert_graph.number_of_edges(),
        )

    def test_from_graphml_returns_project_graph_instance(self):
        path = self.tmp / "test.graphml"
        self.project.to_graphml(path)
        loaded = ProjectGraph.from_graphml(path)
        self.assertIsInstance(loaded, ProjectGraph)


# ============================================================
# GraphML dict-attribute handling
# ============================================================

class TestGraphMLDictAttribs(unittest.TestCase):
    """to_graphml must not crash when nodes/edges carry dict-valued attrs."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _make_graph_with_dict_attrs(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_node(0, meta={"role": "source", "level": 0})
        G.add_node(1, meta={"role": "sink",   "level": 1})
        G.add_edge(0, 1, label="A", extra={"weight": 3, "tag": "real"})
        return G

    def test_no_exception_on_dict_node_attrs(self):
        G = self._make_graph_with_dict_attrs()
        pg = ProjectGraph(G)
        path = self.tmp / "dict_attrs.graphml"
        # Should NOT raise NetworkXError about unsupported dict type.
        try:
            pg.to_graphml(path)
        except Exception as exc:
            self.fail(f"to_graphml raised unexpectedly: {exc}")

    def test_dict_attrs_file_is_valid_xml(self):
        G = self._make_graph_with_dict_attrs()
        pg = ProjectGraph(G)
        path = self.tmp / "dict_attrs.graphml"
        pg.to_graphml(path)
        # File must be readable XML (NetworkX can parse it back).
        loaded = nx.read_graphml(str(path))
        self.assertIsNotNone(loaded)

    def test_original_graph_unchanged_after_to_graphml(self):
        """to_graphml must not mutate the original pert_graph."""
        G = self._make_graph_with_dict_attrs()
        pg = ProjectGraph(G)
        path = self.tmp / "dict_attrs.graphml"
        pg.to_graphml(path)
        # The original edge attr should still be a dict, not a JSON string.
        edge_data = pg.pert_graph[0][1]
        self.assertIsInstance(
            edge_data.get("extra"),
            dict,
            "to_graphml must not mutate the original pert_graph attributes",
        )


# ============================================================
# Route-matrix roundtrip
# ============================================================

class TestRouteMatrixRoundtrip(unittest.TestCase):
    """save_route_matrix → load_route_matrix must reproduce the DataFrame."""

    def setUp(self):
        self.project = _make_project(SIMPLE_PREDS)
        self.R = self.project.path_matrix(dummies=False)
        self._td = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_save_creates_npz_file(self):
        path = self.tmp / "R.npz"
        ProjectGraph.save_route_matrix(self.R, path)
        self.assertTrue(path.exists())

    def test_save_creates_json_sidecar(self):
        path = self.tmp / "R.npz"
        ProjectGraph.save_route_matrix(self.R, path)
        sidecar = self.tmp / "R.json"
        self.assertTrue(sidecar.exists(), "JSON metadata sidecar not found")

    def test_roundtrip_shape(self):
        path = self.tmp / "R.npz"
        ProjectGraph.save_route_matrix(self.R, path)
        R2 = ProjectGraph.load_route_matrix(path)
        self.assertEqual(self.R.shape, R2.shape)

    def test_roundtrip_columns(self):
        path = self.tmp / "R.npz"
        ProjectGraph.save_route_matrix(self.R, path)
        R2 = ProjectGraph.load_route_matrix(path)
        self.assertListEqual(list(self.R.columns), list(R2.columns))

    def test_roundtrip_values(self):
        path = self.tmp / "R.npz"
        ProjectGraph.save_route_matrix(self.R, path)
        R2 = ProjectGraph.load_route_matrix(path)
        np.testing.assert_array_equal(self.R.values, R2.values)

    def test_roundtrip_returns_dataframe(self):
        path = self.tmp / "R.npz"
        ProjectGraph.save_route_matrix(self.R, path)
        R2 = ProjectGraph.load_route_matrix(path)
        self.assertIsInstance(R2, pd.DataFrame)

    def test_roundtrip_medium_network(self):
        project = _make_project(MEDIUM_PREDS)
        R = project.path_matrix(dummies=False)
        path = self.tmp / "R_medium.npz"
        ProjectGraph.save_route_matrix(R, path)
        R2 = ProjectGraph.load_route_matrix(path)
        # check_dtype=False: se guarda como uint8 (compacto), se carga como uint8;
        # el DataFrame original puede ser int64 — los valores son iguales.
        pd.testing.assert_frame_equal(R.reset_index(drop=True),
                                      R2.reset_index(drop=True),
                                      check_dtype=False)


# ============================================================
# build_and_save
# ============================================================

class TestBuildAndSave(unittest.TestCase):
    """build_and_save must create GraphML + parquet files and return a ProjectGraph."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.out_dir = pathlib.Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_returns_project_graph(self):
        result = build_and_save("net1", SIMPLE_PREDS, self.out_dir)
        self.assertIsInstance(result, ProjectGraph)

    def test_creates_graphml_file(self):
        build_and_save("net1", SIMPLE_PREDS, self.out_dir)
        self.assertTrue((self.out_dir / "net1.graphml").exists())

    def test_creates_parquet_file(self):
        build_and_save("net1", SIMPLE_PREDS, self.out_dir)
        self.assertTrue((self.out_dir / "net1_paths.parquet").exists())

    def test_creates_out_dir_if_missing(self):
        new_dir = self.out_dir / "subdir" / "deep"
        build_and_save("net1", SIMPLE_PREDS, new_dir)
        self.assertTrue(new_dir.is_dir())

    def test_graphml_encodes_correct_precedence(self):
        build_and_save("net1", SIMPLE_PREDS, self.out_dir)
        loaded = ProjectGraph.from_graphml(self.out_dir / "net1.graphml")
        table = PredecessorTable({k: set(v) for k, v in SIMPLE_PREDS.items()})
        self.assertEqual(
            loaded.distant_predecessor(dummies=False),
            table.distant_predecessor,
        )

    def test_parquet_has_expected_activity_columns(self):
        build_and_save("net1", SIMPLE_PREDS, self.out_dir)
        R = pd.read_parquet(self.out_dir / "net1_paths.parquet")
        # Real activities (no dummies) must appear as columns.
        real_acts = set(SIMPLE_PREDS.keys())
        self.assertTrue(real_acts.issubset(set(R.columns)))

    def test_list_predecessors_accepted(self):
        """build_and_save must accept list values (not only sets) without crashing."""
        # SIMPLE_PREDS already uses lists — this exercises the normalisation branch.
        try:
            build_and_save("net_list", SIMPLE_PREDS, self.out_dir)
        except Exception as exc:
            self.fail(f"build_and_save raised with list preds: {exc}")


# ============================================================
# load_instance
# ============================================================

class TestLoadInstance(unittest.TestCase):
    """load_instance must return (ProjectGraph, pd.DataFrame) from saved files."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.out_dir = pathlib.Path(self._td.name)
        build_and_save("net1", SIMPLE_PREDS, self.out_dir)

    def tearDown(self):
        self._td.cleanup()

    def test_returns_tuple_of_two(self):
        result = load_instance("net1", self.out_dir)
        self.assertEqual(len(result), 2)

    def test_first_element_is_project_graph(self):
        graph, _ = load_instance("net1", self.out_dir)
        self.assertIsInstance(graph, ProjectGraph)

    def test_second_element_is_dataframe(self):
        _, paths = load_instance("net1", self.out_dir)
        self.assertIsInstance(paths, pd.DataFrame)

    def test_dataframe_is_not_empty(self):
        _, paths = load_instance("net1", self.out_dir)
        self.assertGreater(len(paths), 0)

    def test_dataframe_columns_are_activity_names(self):
        _, paths = load_instance("net1", self.out_dir)
        real_acts = set(SIMPLE_PREDS.keys())
        self.assertTrue(real_acts.issubset(set(paths.columns)))


# ============================================================
# Integration: build_and_save + load_instance end-to-end
# ============================================================

class TestBuildAndSaveLoadIntegration(unittest.TestCase):
    """Full cycle: build → save → load → verify correctness."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.out_dir = pathlib.Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _roundtrip(self, name, preds):
        original = build_and_save(name, preds, self.out_dir)
        loaded_graph, loaded_paths = load_instance(name, self.out_dir)
        return original, loaded_graph, loaded_paths

    def test_simple_network_node_count(self):
        original, loaded, _ = self._roundtrip("simple", SIMPLE_PREDS)
        self.assertEqual(
            original.pert_graph.number_of_nodes(),
            loaded.pert_graph.number_of_nodes(),
        )

    def test_simple_network_edge_count(self):
        original, loaded, _ = self._roundtrip("simple", SIMPLE_PREDS)
        self.assertEqual(
            original.pert_graph.number_of_edges(),
            loaded.pert_graph.number_of_edges(),
        )

    def test_simple_network_path_matrix_shape(self):
        original, _, loaded_paths = self._roundtrip("simple", SIMPLE_PREDS)
        # build_and_save calls path_matrix() with dummies=True (default),
        # so we compare against the same call.
        original_paths = original.path_matrix()
        self.assertEqual(original_paths.shape, loaded_paths.shape)

    def test_medium_network_precedence_invariant(self):
        table = PredecessorTable({k: set(v) for k, v in MEDIUM_PREDS.items()})
        _, loaded, _ = self._roundtrip("medium", MEDIUM_PREDS)
        self.assertEqual(
            loaded.distant_predecessor(dummies=False),
            table.distant_predecessor,
        )

    def test_path_matrix_values_match(self):
        original, _, loaded_paths = self._roundtrip("simple", SIMPLE_PREDS)
        orig_paths = original.path_matrix(dummies=False)
        # Align columns before comparing (parquet preserves column order, but be safe).
        cols = list(orig_paths.columns)
        np.testing.assert_array_equal(
            orig_paths[cols].values,
            loaded_paths[cols].values,
        )

    def test_multiple_instances_coexist(self):
        """Saving two different instances to the same directory must not conflict."""
        build_and_save("inst_a", SIMPLE_PREDS, self.out_dir)
        build_and_save("inst_b", MEDIUM_PREDS, self.out_dir)
        ga, _ = load_instance("inst_a", self.out_dir)
        gb, _ = load_instance("inst_b", self.out_dir)
        self.assertNotEqual(
            ga.pert_graph.number_of_edges(),
            gb.pert_graph.number_of_edges(),
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    unittest.main()
