import random
import unittest
import networkx as nx
import pandas as pd
import numpy as np

from src.pmule import (PredecessorTable,
                       ProjectGraph,
                       make_project,
                       calculate_direct_predecessors,
                       calculate_predecessors,
                       read_rcp)

random.seed(42)

class pmULETests(unittest.TestCase):
    def setUp(self):
        self.immediate = {
                         'A': set(),
                         'B': {'A'},
                         'C': {'B'},
                         'D': {'B', 'J'},
                         'E': {'F'},
                         'F': {'I'},
                         'G': {'C', 'D'},
                         'H': {'E'},
                         'I': {'B', 'J'},
                         'J': {'A'}}

        self.distant = { 'A': set(),
                         'B': {'A'},
                         'C': {'A', 'B'},
                         'D': {'A', 'B', 'J'},
                         'E': {'A', 'B', 'F', 'I', 'J'},
                         'F': {'A', 'B', 'I', 'J'},
                         'G': {'A', 'B', 'C', 'D', 'J'},
                         'H': {'A', 'B', 'E', 'F', 'I', 'J'},
                         'I': {'A', 'B', 'J'},
                         'J': {'A'}}

    def test_calculate_immediate_predecessor_of(self):
        result = calculate_direct_predecessors(self.distant)
        self.assertEqual(self.immediate, result)

    def test_calculate_predecessors_of(self):
        result = calculate_predecessors(self.immediate)
        self.assertEqual(self.distant, result)

class PredecessorTableTests(unittest.TestCase):
    # setup values for tests
    def setUp(self):
        self.data = pd.DataFrame([
            ('A', '---  '),
            ('B', 'A    '),
            ('C', 'B    '),
            ('D', 'B,J  '),
            ('E', 'F,B,J'),
            ('F', 'A    '),
            ('G', 'C,D  '),
            ('H', 'E    '),
            ('I', 'B,J  '),
            ('J', 'A    '),
        ], columns=['activity', 'predecessors']).set_index('activity')

    def test_activity_names(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.activity_names
        expected_result = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.assertEqual(result, expected_result)

    def test_augmented_activities(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.augmented_predecessor
        expected_result = {'@∇A⤑ΔB': {'A'},
                           '@∇A⤑ΔF': {'A'},
                           '@∇A⤑ΔJ': {'A'},
                           '@∇B⤑ΔC': {'B'},
                           '@∇B⤑ΔD': {'B'},
                           '@∇B⤑ΔE': {'B'},
                           '@∇B⤑ΔI': {'B'},
                           '@∇C⤑ΔG': {'C'},
                           '@∇D⤑ΔG': {'D'},
                           '@∇E⤑ΔH': {'E'},
                           '@∇F⤑ΔE': {'F'},
                           '@∇J⤑ΔD': {'J'},
                           '@∇J⤑ΔE': {'J'},
                           '@∇J⤑ΔI': {'J'},
                           'A': {'@∇StartOfProject⤑ΔA'},
                           'B': {'@∇A⤑ΔB'},
                           'C': {'@∇B⤑ΔC'},
                           'D': {'@∇B⤑ΔD', '@∇J⤑ΔD'},
                           'E': {'@∇F⤑ΔE', '@∇J⤑ΔE', '@∇B⤑ΔE'},
                           'F': {'@∇A⤑ΔF'},
                           'G': {'@∇D⤑ΔG', '@∇G⤑ΔEndOfProject', '@∇C⤑ΔG'},
                           'H': {'@∇E⤑ΔH', '@∇H⤑ΔEndOfProject'},
                           'I': {'@∇B⤑ΔI', '@∇J⤑ΔI', '@∇I⤑ΔEndOfProject'},
                           'J': {'@∇A⤑ΔJ'}
                           }
        self.assertEqual(result, expected_result)

    def test_nodes(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.nodes
        expected_result = {'@∇A',
                           '@∇B',
                           '@∇C',
                           '@∇D',
                           '@∇E',
                           '@∇F',
                           '@∇G',
                           '@∇H',
                           '@∇I',
                           '@∇J',
                           '@∇StartOfProject',
                           'ΔA',
                           'ΔB',
                           'ΔC',
                           'ΔD',
                           'ΔE',
                           'ΔEndOfProject',
                           'ΔF',
                           'ΔG',
                           'ΔH',
                           'ΔI',
                           'ΔJ'}
        self.assertEqual(result, expected_result)

    def test_start_activities(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.start_activities
        expected_result = {'A'}
        self.assertEqual(result, expected_result)

    def test_end_activities(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.end_activities
        expected_result = {'G', 'H', 'I'}
        self.assertEqual(result, expected_result)

    def test_copy(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.copy().predecessors
        expected_result = {'A': set(),
                           'B': {'A'},
                           'C': {'A', 'B'},
                           'D': {'A', 'B', 'J'},
                           'E': {'A', 'B', 'F', 'J'},
                           'F': {'A'},
                           'G': {'A', 'B', 'C', 'D', 'J'},
                           'H': {'A', 'B', 'E', 'F', 'J'},
                           'I': {'A', 'B', 'J'},
                           'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_create_project(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=True)
        result = make_project(table).predecessors
        expected_result = {'A': set(),
                           'B': {'A'},
                           'C': {'A', 'B'},
                           'D': {'A', 'B', 'J'},
                           'E': {'A', 'B', 'F', 'J'},
                           'F': {'A'},
                           'G': {'A', 'B', 'C', 'D', 'J'},
                           'H': {'A', 'B', 'E', 'F', 'J'},
                           'I': {'A', 'B', 'J'},
                           'J': {'A'}}



        self.assertEqual(result, expected_result)

    def test_calculate_immediate_predecessor_of(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = {item: table.calculate_direct_predecessor_of(item) for item in table.activity_names}
        expected_result = {'A': set(),
                           'B': {'A'},
                           'C': {'B'},
                           'D': {'B', 'J'},
                           'E': {'B', 'F', 'J'},
                           'F': {'A'},
                           'G': {'C', 'D'},
                           'H': {'E'},
                           'I': {'B', 'J'},
                           'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_immediate(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.direct_predecessors
        expected_result = {'A': set(),
                           'B': {'A'},
                           'C': {'B'},
                           'D': {'B', 'J'},
                           'E': {'B', 'F', 'J'},
                           'F': {'A'},
                           'G': {'C', 'D'},
                           'H': {'E'},
                           'I': {'B', 'J'},
                           'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_immediate_linkage_matrix(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.immediate_linkage_matrix
        expected_result = pd.DataFrame([
            ['A', False, False, False, False, False, False, False, False, False, False],
            ['B', True, False, False, False, False, False, False, False, False, False],
            ['C', False, True, False, False, False, False, False, False, False, False],
            ['D', False, True, False, False, False, False, False, False, False, True],
            ['E', False, True, False, False, False, True, False, False, False, True],
            ['F', True, False, False, False, False, False, False, False, False, False],
            ['G', False, False, True, True, False, False, False, False, False, False],
            ['H', False, False, False, False, True, False, False, False, False, False],
            ['I', False, True, False, False, False, False, False, False, False, True],
            ['J', True, False, False, False, False, False, False, False, False, False],
        ], columns=['activities', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']).set_index('activities')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_distant(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.predecessors
        expected_result = {'A': set(),
                           'B': {'A'},
                           'C': {'A', 'B'},
                           'D': {'A', 'B', 'J'},
                           'E': {'A', 'B', 'F', 'J'},
                           'F': {'A'},
                           'G': {'A', 'B', 'C', 'D', 'J'},
                           'H': {'A', 'B', 'E', 'F', 'J'},
                           'I': {'A', 'B', 'J'},
                           'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_distant_linkage_matrix(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.distant_linkage_matrix
        expected_result = pd.DataFrame([
            ['A', False, False, False, False, False, False, False, False, False, False],
            ['B', True, False, False, False, False, False, False, False, False, False],
            ['C', True, True, False, False, False, False, False, False, False, False],
            ['D', True, True, False, False, False, False, False, False, False, True],
            ['E', True, True, False, False, False, True, False, False, False, True],
            ['F', True, False, False, False, False, False, False, False, False, False],
            ['G', True, True, True, True, False, False, False, False, False, True],
            ['H', True, True, False, False, True, True, False, False, False, True],
            ['I', True, True, False, False, False, False, False, False, False, True],
            ['J', True, False, False, False, False, False, False, False, False, False],
        ], columns=['activities', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']).set_index('activities')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_calculate_predecessors_of(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = {item: table.calculate_predecessor_of(item) for item in table.activity_names}
        expected_result = {'A': set(),
                           'B': {'A'},
                           'C': {'A', 'B'},
                           'D': {'A', 'B', 'J'},
                           'E': {'A', 'B', 'F', 'J'},
                           'F': {'A'},
                           'G': {'A', 'B', 'C', 'D', 'J'},
                           'H': {'A', 'B', 'E', 'F', 'J'},
                           'I': {'A', 'B', 'J'},
                           'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_dummies(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=True)
        result = table.dummies
        expected_result = {'@∇StartOfProject⤑ΔA',
                           '@∇A⤑ΔB',
                           '@∇A⤑ΔF',
                           '@∇A⤑ΔJ',
                           '@∇B⤑ΔC',
                           '@∇B⤑ΔD',
                           '@∇B⤑ΔE',
                           '@∇B⤑ΔI',
                           '@∇C⤑ΔG',
                           '@∇D⤑ΔG',
                           '@∇E⤑ΔH',
                           '@∇F⤑ΔE',
                           '@∇J⤑ΔD',
                           '@∇J⤑ΔE',
                           '@∇J⤑ΔI',
                           '@∇G⤑ΔEndOfProject',
                           '@∇H⤑ΔEndOfProject',
                           '@∇I⤑ΔEndOfProject',
                           }
        self.assertEqual(result, expected_result)

    def test_display_immediate_linkage_matrix(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=False)
        result = table.display_immediate_linkage_matrix.data
        expected_result = pd.DataFrame([
            ['A', '', '', '', '', '', '', '', '', '', ''],
            ['B', True, '', '', '', '', '', '', '', '', ''],
            ['C', '', True, '', '', '', '', '', '', '', ''],
            ['D', '', True, '', '', '', '', '', '', '', True],
            ['E', '', True, '', '', '', True, '', '', '', True],
            ['F', True, '', '', '', '', '', '', '', '', ''],
            ['G', '', '', True, True, '', '', '', '', '', ''],
            ['H', '', '', '', '', True, '', '', '', '', ''],
            ['I', '', True, '', '', '', '', '', '', '', True],
            ['J', True, '', '', '', '', '', '', '', '', ''],
        ], columns=['activities', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']).set_index('activities')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_from_dataframe_of_strings(self):
        result = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                            activity='activity',
                                                            predecessor='predecessors',
                                                            simplify=False).predecessors
        expected_result = PredecessorTable(data={'A': set(),
                                                 'B': {'A'},
                                                 'C': {'B'},
                                                 'D': {'B', 'J'},
                                                 'E': {'B', 'F', 'J'},
                                                 'F': {'A'},
                                                 'G': {'C', 'D'},
                                                 'H': {'E'},
                                                 'I': {'B', 'J'},
                                                 'J': {'A'}}).predecessors
        self.assertEqual(result, expected_result)

    def test_from_dict_of_sets(self):
        data = {'A': set(),
                'B': {'A'},
                'C': {'B'},
                'D': {'B', 'J'},
                'E': {'B', 'F', 'J'},
                'F': {'A'},
                'G': {'C', 'D'},
                'H': {'E'},
                'I': {'B', 'J'},
                'J': {'A'}}
        result = PredecessorTable.from_dict_of_sets(data).predecessors
        expected_result = PredecessorTable(data={'A': set(),
                                                 'B': {'A'},
                                                 'C': {'B'},
                                                 'D': {'B', 'J'},
                                                 'E': {'B', 'F', 'J'},
                                                 'F': {'A'},
                                                 'G': {'C', 'D'},
                                                 'H': {'E'},
                                                 'I': {'B', 'J'},
                                                 'J': {'A'}}).predecessors
        self.assertEqual(result, expected_result)

    def test_from_dict_of_strings(self):
        data = {'A': '',
                'B': 'A',
                'C': 'B',
                'D': 'B,J',
                'E': 'B,F,J',
                'F': 'A',
                'G': 'C,D',
                'H': 'E',
                'I': 'B,J',
                'J': 'A'}
        result = PredecessorTable.from_dict_of_strings(data).predecessors
        expected_result = PredecessorTable(data={'A': set(),
                                                 'B': {'A'},
                                                 'C': {'B'},
                                                 'D': {'B', 'J'},
                                                 'E': {'B', 'F', 'J'},
                                                 'F': {'A'},
                                                 'G': {'C', 'D'},
                                                 'H': {'E'},
                                                 'I': {'B', 'J'},
                                                 'J': {'A'}}).predecessors
        self.assertEqual(result, expected_result)

    def test_from_project(self):
        table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                           activity='activity',
                                                           predecessor='predecessors', simplify=True)
        project = make_project(table, simplify=True)
        result = PredecessorTable.from_project(project).predecessors
        expected_result = table.predecessors
        self.assertEqual(result, expected_result)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            ('A', '---  '),
            ('B', 'A    '),
            ('C', 'B    '),
            ('D', 'B,J  '),
            ('E', 'F,B,J'),
            ('F', 'A    '),
            ('G', 'C,D  '),
            ('H', 'E    '),
            ('I', 'B,J  '),
            ('J', 'A    '),
        ], columns=['activity', 'predecessors']).set_index('activity')
        self.table = PredecessorTable.from_dataframe_of_strings(data=self.data,
                                                                activity='activity',
                                                                predecessor='predecessors', simplify=True)
        self.project = make_project(self.table, simplify=True)
        self.duration = {'A': 2, 'B': 2, 'C': 3, 'D': 4, 'E': 1, 'F': 2, 'G': 1, 'H': 4, 'I': 1, 'J': 2}

    def test_activities(self):
        result = self.project.activities
        expected_result = ['@∇B⤑ΔD', '@∇J⤑ΔE', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        self.assertEqual(result, expected_result)

    def test_actual_activities(self):
        result = self.project.actual_activities
        expected_result = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.assertEqual(result, expected_result)

    def test_dummies(self):
        result = self.project.dummies
        expected_result = ['@∇B⤑ΔD', '@∇J⤑ΔE']
        self.assertEqual(result, expected_result)

    def test_nodes(self):
        result = self.project.nodes
        expected_result = ['1', '2', '3', '4', '5', '6', '7', '8']
        self.assertEqual(result, expected_result)

    def test_edges_from_nodes(self):
        result = self.project.edges_from_nodes
        expected_result = {('1', '2'): 'A',
                           ('2', '3'): 'B',
                           ('2', '4'): 'J',
                           ('2', '5'): 'F',
                           ('3', '4'): '@∇B⤑ΔD',
                           ('3', '6'): 'C',
                           ('4', '5'): '@∇J⤑ΔE',
                           ('4', '6'): 'D',
                           ('4', '8'): 'I',
                           ('5', '7'): 'E',
                           ('6', '8'): 'G',
                           ('7', '8'): 'H'}
        self.assertEqual(result, expected_result)

    def test_predecessors(self):
        result = self.project.predecessors
        expected_result = {
                           'A': set(),
                           'B': {'A'},
                           'C': {'A', 'B'},
                           'D': { 'A', 'B', 'J'},
                           'E': {'A', 'J', 'B', 'F', },
                           'F': {'A'},
                           'G': {'C', 'J', 'D', 'A', 'B'},
                           'H': {'F', 'E', 'J', 'B', 'A'},
                           'I': {'A', 'B', 'J'},
                           'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_duration(self):
        result = self.project.duration(durations=self.duration)
        expected_result = 9
        self.assertEqual(result, expected_result)

    def test_critical_path(self):
        result = self.project.critical_path(durations=self.duration)
        expected_result = {'Route_2': ['A', 'B', '@∇B⤑ΔD', '@∇J⤑ΔE', 'E', 'H'],
                           'Route_4': ['A', 'B', '@∇B⤑ΔD', 'D', 'G'],
                           'Route_5': ['A', 'F', 'E', 'H'],
                           'Route_6': ['A', 'J', '@∇J⤑ΔE', 'E', 'H'],
                           'Route_8': ['A', 'J', 'D', 'G']}
        self.assertEqual(result, expected_result)

    def test_from_dataframe_of_strings(self):
        result = ProjectGraph.from_dataframe_of_strings(data=self.data,
                                                        activity='activity',
                                                        predecessor='predecessors', simplify=True).predecessors
        expected_result = PredecessorTable.from_project(self.project).predecessors
        self.assertEqual(result, expected_result)

    def test_from_dict_of_strings(self):
        data = {'A': '',
                'B': 'A',
                'C': 'B',
                'D': 'B,J',
                'E': 'B,F,J',
                'F': 'A',
                'G': 'C,D',
                'H': 'E',
                'I': 'B,J',
                'J': 'A'}
        result = ProjectGraph.from_dict_of_strings(data).predecessors
        expected_result = {
                           'A': set(),
                           'B': {'A'},
                           'C': {'B', 'A'},
                           'D': {'A', 'B', 'J'},
                           'E': {'F', 'J', 'A', 'B'},
                           'F': {'A'},
                           'G': {'J', 'A', 'C', 'D', 'B'},
                           'H': {'F', 'J', 'A', 'E', 'B'},
                           'I': {'A', 'B', 'J'},
                           'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_nodes_from_edges(self):
        result = self.project.nodes_from_edges
        expected_result = {'@∇B⤑ΔD': ('3', '4'),
                           '@∇J⤑ΔE': ('4', '5'),
                           'A': ('1', '2'),
                           'B': ('2', '3'),
                           'C': ('3', '6'),
                           'D': ('4', '6'),
                           'E': ('5', '7'),
                           'F': ('2', '5'),
                           'G': ('6', '8'),
                           'H': ('7', '8'),
                           'I': ('4', '8'),
                           'J': ('2', '4')}
        self.assertEqual(result, expected_result)

    def test_from_dict_of_sets(self):
        data = {'A': set(),
                'B': {'A'},
                'C': {'B'},
                'D': {'B', 'J'},
                'E': {'B', 'F', 'J'},
                'F': {'A'},
                'G': {'C', 'D'},
                'H': {'E'},
                'I': {'B', 'J'},
                'J': {'A'}}
        result = ProjectGraph.from_dict_of_sets(data).predecessors
        expected_result = {'A': set(),
                           'B': {'A'},
                           'C': {'A', 'B'},
                           'D': {'A', 'J', 'B'},
                           'E': {'F', 'J', 'A', 'B'},
                           'F': {'A'},
                           'G': {'C', 'D', 'J', 'A', 'B'},
                           'H': {'F', 'J', 'A', 'E', 'B'},
                           'I': {'A', 'J', 'B'},
                           'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_paths(self):
        result = self.project.paths(dummies=True)
        expected_result = {'Route_1': ['A', 'B', 'C', 'G'],
                           'Route_2': ['A', 'B', '@∇B⤑ΔD', '@∇J⤑ΔE', 'E', 'H'],
                           'Route_3': ['A', 'B', '@∇B⤑ΔD', 'I'],
                           'Route_4': ['A', 'B', '@∇B⤑ΔD', 'D', 'G'],
                           'Route_5': ['A', 'F', 'E', 'H'],
                           'Route_6': ['A', 'J', '@∇J⤑ΔE', 'E', 'H'],
                           'Route_7': ['A', 'J', 'I'],
                           'Route_8': ['A', 'J', 'D', 'G']}
        self.assertEqual(result, expected_result)

    def test_path_matrix(self):
        result = self.project.path_matrix(dummies=True)
        expected_result = pd.DataFrame(data=[
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        ], columns=['@∇B⤑ΔD', '@∇J⤑ΔE', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            index=['Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5', 'Route_6', 'Route_7', 'Route_8'])
        pd.testing.assert_frame_equal(result, expected_result)

    # def test_calculate_predecessors_of(self):
    #     result = {item: self.project.calculate_predecessor_of(item) for item in self.project.activities}
    #     expected_result = {'@∇B⤑ΔD': {'B', 'A'},
    #                        '@∇J⤑ΔE': {'J', 'B', 'A'},
    #                        'A': set(),
    #                        'B': {'A'},
    #                        'C': {'B', 'A'},
    #                        'D': {'J', 'B', 'A'},
    #                        'E': {'J', 'F', 'B', 'A'},
    #                        'F': {'A'},
    #                        'G': {'B', 'A', 'C', 'J', 'D'},
    #                        'H': {'E', 'B', 'A', 'J', 'F'},
    #                        'I': {'J', 'B', 'A'},
    #                        'J': {'A'}}
    #     self.assertEqual(result, expected_result)

    def test_calculate_immediate_predecessor_of(self):
        result = self.project.direct_predecessor
        expected_result = {
                             'A': set(),
                             'B': {'A'},
                             'C': {'B'},
                             'D': {'J', '@∇B⤑ΔD'},
                             'E': {'F', '@∇J⤑ΔE'},
                             'F': {'A'},
                             'G': {'C', 'D'},
                             'H': {'E'},
                             'I': {'J', '@∇B⤑ΔD'},
                             'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_simplify(self):
        project = ProjectGraph.from_dataframe_of_strings(data=self.data,
                                                         activity='activity',
                                                         predecessor='predecessors',
                                                         simplify=False)

        result = project.simplify().predecessors
        expected_result = {'A': set(),
                           'B': {'A'},
                           'C': {'A', 'B'},
                           'D': {'A', 'J', 'B'},
                           'E': {'F', 'J', 'A', 'B'},
                           'F': {'A'},
                           'G': {'C', 'D', 'J', 'A', 'B'},
                           'H': {'F', 'J', 'A', 'E', 'B'},
                           'I': {'A', 'J', 'B'},
                           'J': {'A'}}
        self.assertEqual(result, expected_result)

    def test_assess_dummy_false(self):
        def check_dummies(proyecto, lista_names):
            flags = dict()
            new_proyecto = proyecto.copy()
            reference = new_proyecto.predecessors
            for index, dummy in enumerate(lista_names):
                flag, new_proyecto = ProjectGraph.assess_dummy(dummy, new_proyecto)
                flags[dummy] = flag
                assert (new_proyecto.predecessors == reference)
            return flags, new_proyecto

        number_of_shuffles = 5
        project = ProjectGraph.from_dataframe_of_strings(data=self.data,
                                                         activity='activity',
                                                         predecessor='predecessors',
                                                         simplify=False)
        original = project.predecessors

        list_of_dummies = [
            '@∇J⤑ΔE', '@∇B⤑ΔI', '@∇StartOfProject⤑ΔA', '@∇A⤑ΔF', '@∇A⤑ΔB', '@∇A⤑ΔJ', '@∇G⤑ΔEndOfProject',
            '@∇I⤑ΔEndOfProject', '@∇H⤑ΔEndOfProject', '@∇C⤑ΔG', '@∇D⤑ΔG', '@∇E⤑ΔH', '@∇B⤑ΔE', '@∇B⤑ΔC', '@∇J⤑ΔD',
            '@∇J⤑ΔD', '@∇J⤑ΔI', '@∇F⤑ΔE', '@∇A⤑ΔF', '@∇B⤑ΔD', ]

        for i in range(number_of_shuffles):
            random.shuffle(list_of_dummies)
            reference, new_project = check_dummies(project, list_of_dummies)
            self.assertEqual(original, new_project.predecessors)

    #@unittest.skip("This test is too slow, run it only when necessary")
    def test_read_rcp(self):
        rcp_filename = 'test_media/test_1kAD_1000001.rcp'
        csv_filename = 'test_media/test_1kAD_1000001.csv'
        proj_data = read_rcp(rcp_filename)
        predecessor_table = PredecessorTable.from_dict_of_strings(proj_data['predecessors'])
        project = make_project(predecessor_table, simplify=True)
        df = nx.to_pandas_edgelist(project.pert_graph)
        df.set_index('activity', inplace=True)
        df = pd.merge(right=proj_data['duration'], left=df, left_index=True, right_index=True, how='outer').fillna(0)
        df = pd.merge(left=df, right=proj_data['resources'], left_index=True, right_index=True, how='outer').fillna(0)
        df.rename(columns={resource: f'{resource} (max: {str(amount[0])})' for (resource, amount) in
                           zip(proj_data['resource_availability'].index, proj_data['resource_availability'].values)},
                  inplace=True)
        if 'contraction' in df.columns:
            df.drop(columns=['contraction'], inplace=True)

        result = df.astype(np.float64)
        result.index.name = 'activity'
        expected_result = pd.read_csv(csv_filename).rename(columns={'Unnamed: 0': 'activity'}).set_index(
            'activity').astype(np.float64)
        pd.testing.assert_frame_equal(result, expected_result)


#     def test_generate_random_str(self):
#         random_strings = [generate_random_str(10) for i in range(100)]
#         for random_str in random_strings:
#             self.assertEqual(len(random_str), 10)
#         # test all strings in random_strings are unique
#         self.assertEqual(len(random_strings), len(set(random_strings)))
#
#     def test_calculate_linkage_matrix(self):
#         project_data = pd.DataFrame([
#             #  actividad, precedentes, duracion, duracion_pesimista, duracion_modal, recursos
#             ('A', '---  ', 2, 2, 2, 1),
#             ('B', 'A    ', 2, 3, 2, 2),
#             ('C', 'B    ', 3, 4, 3, 2),
#             ('D', 'B,J  ', 4, 5, 17 / 4, 1),
#             ('E', 'F,B,J', 1, 1, 1, 1),
#             ('F', 'A    ', 2, 3, 2, 2),
#             ('G', 'C,D  ', 1, 1, 1, 1),
#             ('H', 'E    ', 4, 5, 4, 1),
#             ('I', 'B,J  ', 1, 1, 1, 1),
#             ('J', 'A    ', 2, 2, 2, 1),
#         ], columns=['actividad', 'precedentes', 'duracion', 'duracion_pesimista', 'duracion_modal',
#                     'recursos']).set_index('actividad')
#         result = calculate_linkage_matrix(data=project_data, predecessors='precedentes')
#
#         expected_data = [
#             ['actividad', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
#             ['A', False, False, False, False, False, False, False, False, False, False],
#             ['B', True, False, False, False, False, False, False, False, False, False],
#             ['C', False, True, False, False, False, False, False, False, False, False],
#             ['D', False, True, False, False, False, False, False, False, False, True],
#             ['E', False, True, False, False, False, True, False, False, False, True],
#             ['F', True, False, False, False, False, False, False, False, False, False],
#             ['G', False, False, True, True, False, False, False, False, False, False],
#             ['H', False, False, False, False, True, False, False, False, False, False],
#             ['I', False, True, False, False, False, False, False, False, False, True],
#             ['J', True, False, False, False, False, False, False, False, False, False],
#         ]
#         expected_result = pd.DataFrame(expected_data[1:], columns=expected_data[0])
#         expected_result.set_index(keys='actividad', inplace=True)
#         self.assertTrue(result.equals(expected_result))
#
#     def test_replace_activity_name_by_numbered_dummy(self):
#         data = set()
#         predecessor = 'b'
#         index = 1
#         expected_result = set()
#         result = replace_activity_name_by_numbered_dummy(data=data, predecessor=predecessor, index=index)
#         print("Despreciar el mensaje de advertencia sobre b")
#         self.assertEqual(expected_result, result)
#
#     def test_replace_activity_name_by_numbered_dummy_2(self):
#         data = {'b', 'c', 'd'}
#         predecessor = 'd'
#         index = 34
#         expected_result = {'b', 'c', 'f34_d'}
#         result = replace_activity_name_by_numbered_dummy(data=data, predecessor=predecessor, index=index)
#         self.assertEqual(result, expected_result)
#
#     # create test for remove_nasty_characters_from_activity_names
#     def test_remove_nasty_characters_from_activity_names(self):
#         data = pd.DataFrame({'predecessors': ['a;f e', 'h,,', 'b', 'c,']}, index=['A', 'B', 'C', 'D'])
#         expected_result = pd.DataFrame({'predecessors': [{'a', 'fe'}, {'h'}, {'b'}, {'c'}]}, index=['A', 'B', 'C', 'D'])
#         result = remove_nasty_characters_from_activity_names(data=data, predecessors='predecessors')
#         self.assertTrue(result.equals(expected_result))
#
#     def test_identify_activity_names(self):
#         data = pd.DataFrame({'predecessors': [{'a', 'fe'}, {'h'}, {'b'}, {'c'}]}, index=['A', 'B', 'C', 'D'])
#         expected_result = {'A', 'B', 'C', 'D', 'a', 'fe', 'h', 'b', 'c'}
#         result = identify_activity_names(data=data, predecessors_column_name='predecessors')
#         self.assertEqual(result, expected_result)
#
#     def test_identify_initial_activities(self):
#         data = pd.DataFrame({'predecessors': [set(), {'A'}, {}, {'A', 'B'}]}, index=['A', 'B', 'C', 'D'])
#         expected_result = {'A', 'C'}
#         result = identify_initial_activities(data=data, predecessors_column_name='predecessors')
#         self.assertEqual(result, expected_result)
#
#     def test_identify_final_activities(self):
#         data = pd.DataFrame({'predecessors': [set(), {'A'}, {}, {'A', 'B'}]}, index=['A', 'B', 'C', 'D'])
#         expected_result = {'C', 'D'}
#         result = identify_final_activities(data=data, predecessors_column_name='predecessors')
#         self.assertEqual(result, expected_result)
#
#     def test_identify_predecessors_appearances(self):
#         data = pd.DataFrame({'predecessors': [set(), {'A'}, {}, {'A', 'B'}]}, index=['A', 'B', 'C', 'D'])
#         expected_result =  {'A': ['B', 'D']}
#         result = identify_predecessors_appearances(data=data,
#                                                    predecessors_column_name='predecessors')
#         self.assertEqual(result, expected_result)
#
#     def test_transform_project_data(self):
#         data = pd.DataFrame({'predecessors': ['', 'A', '', 'A,B']}, index=['A', 'B', 'C', 'D'])
#
#         expected_result = pd.DataFrame(data={'predecessors': [set(),     {'A'},     set(),   {'A', 'B'}],
#                                              'source':       ['start',   'after_A', 'start', 'after_B'],
#                                              'destination':  ['after_A', 'after_B', 'finish', 'finish']},
#                                        index=['A', 'B', 'C', 'D'])
#
#         result = transform_project_data(data=data,
#                                         predecessors_column_name='predecessors',
#                                         source_column_name='source',
#                                         destination_column_name='destination')
#         print(f'{result=}')
#         print(f'{expected_result=}')
#
#         self.assertTrue(result.equals(expected_result))
#
#     def test_ProjectGraph_init(self):
#         data = pd.DataFrame([
#             #  actividad, precedentes, duracion, duracion_pesimista, duracion_modal, recursos
#             ('A', '---  ', 2, 2, 2, 1),
#             ('B', 'A    ', 2, 3, 2, 2),
#             ('C', 'B    ', 3, 4, 3, 2),
#             ('D', 'B,J  ', 4, 5, 17 / 4, 1),
#             ('E', 'F,B,J', 1, 1, 1, 1),
#             ('F', 'A    ', 2, 3, 2, 2),
#             ('G', 'C,D  ', 1, 1, 1, 1),
#             ('H', 'E    ', 4, 5, 4, 1),
#             ('I', 'B,J  ', 1, 1, 1, 1),
#             ('J', 'A    ', 2, 2, 2, 1),
#         ], columns=['actividad', 'predecessors', 'duracion', 'duracion_pesimista', 'duracion_modal',
#                     'recursos']).set_index('actividad')
#         data.drop(['duracion', 'duracion_pesimista', 'duracion_modal', 'recursos'], axis=1, inplace=True)
#
#         my_pert = ProjectGraph(data=data,
#                                predecessors_column_name='predecessors',
#                                source_column_name='source',
#                                destination_column_name='destination')
#
# if __name__ == '__main__':
#     unittest.main()
