import unittest

from pmule import *


class MyTestCase(unittest.TestCase):
    def test_generate_random_str(self):
        random_strings = [generate_random_str(10) for i in range(100)]
        for random_str in random_strings:
            self.assertEqual(len(random_str), 10)
        # test all strings in random_strings are unique
        self.assertEqual(len(random_strings), len(set(random_strings)))

    def test_calculate_linkage_matrix(self):
        project_data = pd.DataFrame([
            #  actividad, precedentes, duracion, duracion_pesimista, duracion_modal, recursos
            ('A', '---  ', 2, 2, 2, 1),
            ('B', 'A    ', 2, 3, 2, 2),
            ('C', 'B    ', 3, 4, 3, 2),
            ('D', 'B,J  ', 4, 5, 17 / 4, 1),
            ('E', 'F,B,J', 1, 1, 1, 1),
            ('F', 'A    ', 2, 3, 2, 2),
            ('G', 'C,D  ', 1, 1, 1, 1),
            ('H', 'E    ', 4, 5, 4, 1),
            ('I', 'B,J  ', 1, 1, 1, 1),
            ('J', 'A    ', 2, 2, 2, 1),
        ], columns=['actividad', 'precedentes', 'duracion', 'duracion_pesimista', 'duracion_modal',
                    'recursos']).set_index('actividad')
        result = calculate_linkage_matrix(data=project_data, predecessors='precedentes')

        expected_data = [
            ['actividad', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
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
        ]
        expected_result = pd.DataFrame(expected_data[1:], columns=expected_data[0])
        expected_result.set_index(keys='actividad', inplace=True)
        self.assertTrue(result.equals(expected_result))

    def test_replace_activity_name_by_numbered_dummy(self):
        data = set()
        predecessor = 'b'
        index = 1
        expected_result = set()
        result = replace_activity_name_by_numbered_dummy(data=data, predecessor=predecessor, index=index)
        print("Despreciar el mensaje de advertencia sobre b")
        self.assertEqual(expected_result, result)

    def test_replace_activity_name_by_numbered_dummy_2(self):
        data = {'b', 'c', 'd'}
        predecessor = 'd'
        index = 34
        expected_result = {'b', 'c', 'f34_d'}
        result = replace_activity_name_by_numbered_dummy(data=data, predecessor=predecessor, index=index)
        self.assertEqual(result, expected_result)

    # create test for remove_nasty_characters_from_activity_names
    def test_remove_nasty_characters_from_activity_names(self):
        data = pd.DataFrame({'predecessors': ['a;f e', 'h,,', 'b', 'c,']}, index=['A', 'B', 'C', 'D'])
        expected_result = pd.DataFrame({'predecessors': [{'a', 'fe'}, {'h'}, {'b'}, {'c'}]}, index=['A', 'B', 'C', 'D'])
        result = remove_nasty_characters_from_activity_names(data=data, predecessors='predecessors')
        self.assertTrue(result.equals(expected_result))

    def test_identify_activity_names(self):
        data = pd.DataFrame({'predecessors': [{'a', 'fe'}, {'h'}, {'b'}, {'c'}]}, index=['A', 'B', 'C', 'D'])
        expected_result = {'A', 'B', 'C', 'D', 'a', 'fe', 'h', 'b', 'c'}
        result = identify_activity_names(data=data, predecessors_column_name='predecessors')
        self.assertEqual(result, expected_result)

    def test_identify_initial_activities(self):
        data = pd.DataFrame({'predecessors': [set(), {'A'}, {}, {'A', 'B'}]}, index=['A', 'B', 'C', 'D'])
        expected_result = {'A', 'C'}
        result = identify_initial_activities(data=data, predecessors_column_name='predecessors')
        self.assertEqual(result, expected_result)

    def test_identify_final_activities(self):
        data = pd.DataFrame({'predecessors': [set(), {'A'}, {}, {'A', 'B'}]}, index=['A', 'B', 'C', 'D'])
        expected_result = {'C', 'D'}
        result = identify_final_activities(data=data, predecessors_column_name='predecessors')
        self.assertEqual(result, expected_result)

    def test_identify_predecessors_appearances(self):
        data = pd.DataFrame({'predecessors': [set(), {'A'}, {}, {'A', 'B'}]}, index=['A', 'B', 'C', 'D'])
        expected_result =  {'A': ['B', 'D']}
        result = identify_predecessors_appearances(data=data,
                                                   predecessors_column_name='predecessors')
        self.assertEqual(result, expected_result)

    def test_transform_project_data(self):
        data = pd.DataFrame({'predecessors': ['', 'A', '', 'A,B']}, index=['A', 'B', 'C', 'D'])

        expected_result = pd.DataFrame(data={'predecessors': [set(),     {'A'},     set(),   {'A', 'B'}],
                                             'source':       ['start',   'after_A', 'start', 'after_B'],
                                             'destination':  ['after_A', 'after_B', 'finish', 'finish']},
                                       index=['A', 'B', 'C', 'D'])

        result = transform_project_data(data=data,
                                        predecessors_column_name='predecessors',
                                        source_column_name='source',
                                        destination_column_name='destination')
        print(f'{result=}')
        print(f'{expected_result=}')

        self.assertTrue(result.equals(expected_result))

    def test_ProjectGraph_init(self):
        data = pd.DataFrame([
            #  actividad, precedentes, duracion, duracion_pesimista, duracion_modal, recursos
            ('A', '---  ', 2, 2, 2, 1),
            ('B', 'A    ', 2, 3, 2, 2),
            ('C', 'B    ', 3, 4, 3, 2),
            ('D', 'B,J  ', 4, 5, 17 / 4, 1),
            ('E', 'F,B,J', 1, 1, 1, 1),
            ('F', 'A    ', 2, 3, 2, 2),
            ('G', 'C,D  ', 1, 1, 1, 1),
            ('H', 'E    ', 4, 5, 4, 1),
            ('I', 'B,J  ', 1, 1, 1, 1),
            ('J', 'A    ', 2, 2, 2, 1),
        ], columns=['actividad', 'predecessors', 'duracion', 'duracion_pesimista', 'duracion_modal',
                    'recursos']).set_index('actividad')
        data.drop(['duracion', 'duracion_pesimista', 'duracion_modal', 'recursos'], axis=1, inplace=True)
        data = remove_nasty_characters_from_activity_names(data, predecessors='predecessors')
        my_pert = ProjectGraph(data=data,
                               predecessors_column_name='predecessors',
                               source_column_name='source',
                               destination_column_name='destination')

if __name__ == '__main__':
    unittest.main()
