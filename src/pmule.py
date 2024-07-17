from functools import reduce,  cached_property, lru_cache

from itertools import chain
from math import ceil
from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pygraphviz as pgv
from IPython.display import Image, display, SVG
from itertools import product
import pandas as pd


def calculate_predecessors(data: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    data = data.copy()
    def calculate_distant_predecessor_of(name: str, data: Dict[str, Set[str]]) -> Set[str]:
        predecessors = data.get(name, set())
        if predecessors == ():
            return set()
        result = predecessors.copy()
        for ancestor in predecessors:
            result.update(calculate_distant_predecessor_of(ancestor, data))
        return result

    all_activities = set(data.keys())
    for values in data.values():
        all_activities.update(values)

    predecessors = {key: set() for key in all_activities}
    for key in predecessors:
        predecessors[key] = calculate_distant_predecessor_of(key, data)

    predecessors = {key: predecessors[key] for key in sorted(predecessors) if key[0]!='@'}
    return predecessors


def calculate_direct_predecessors(data: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    distant_predecessors = calculate_predecessors(data)
    predecessors = data.copy()
    for key, ancestors in predecessors.items():
        for ancestor in ancestors:
            predecessors[key] = predecessors[key] - distant_predecessors[ancestor]

    return predecessors


def make_project(predecessor_table, simplify=True):
    predecessor_table = predecessor_table.copy()
    def create_edge_info_for_dummy(name):
        return name.replace('@', '').split('⤑') + [{'activity': name}]

    edgelist_dummy_activities = list(create_edge_info_for_dummy(name)
                                     for name in chain.from_iterable(predecessor_table.augmented_predecessor.values())
                                     if name[0] == '@'
                                     )

    edgelist_actual_activities = list((f'Δ{name}', f'∇{name}', {'activity': name})
                                      for name in predecessor_table.activity_names)
    edgelist = edgelist_dummy_activities + edgelist_actual_activities

    pert_graph = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
    for source, target, edge_attributes in edgelist:
        pert_graph.edges[source, target]['activity'] = edge_attributes['activity']

    project = ProjectGraph(graph=pert_graph)
    project.predecessors = predecessor_table.predecessors
    project.direct_predecessors = predecessor_table.direct_predecessors
    if not simplify:
        return project
    return project.simplify()

def read_rcp(filename):
    data = {
        'num_activities': 0,
        'activities': [],
        'num_resources': 0,
        'resource_availability': [],
        'resources': None,
        'predecessors': {}
    }

    with open(filename, 'r') as file:
        lines = file.readlines()

        first_line = lines[0].strip().split()
        data['num_activities'] = int(first_line[0])
        data['num_resources'] = int(first_line[1])

        # Calculate the number of digits required for zero-padding
        num_digits = len(str(data['num_activities']))

        data['predecessors'] = {f'A{str(iteration + 1).zfill(num_digits)}': [] for iteration in
                                range(data['num_activities'])}
        second_line = lines[1].strip().split()

        resource_columns = [f"Resource {i + 1}" for i in range(data['num_resources'])]
        data['resource_availability'] = pd.DataFrame(
                                                    [int(avail) for avail in second_line],
                                                    columns=['Availability'],
                                                    index= resource_columns)
        resource_data = []

        # Following lines: Activity details
        line_index = 2
        while line_index < len(lines):
            line = lines[line_index].strip().split()
            if line:
                activity_name = f"A{str(len(data['activities']) + 1).zfill(num_digits)}"  # Zero-padded activity name

                # Parse activity details
                duration = int(line[0])
                resource_requirements = [int(req) for req in line[1:data['num_resources'] + 1]]
                num_successors = int(line[data['num_resources'] + 1])
                successors = [int(line[i]) for i in range(data['num_resources'] + 2, len(line))]

                # Store activity details
                activity = {
                    'name': activity_name,
                    'duration': duration,
                    'resource_requirements': resource_requirements,
                    'successors': successors
                }
                data['activities'].append(activity)

                # Calculate predecessors based on successors
                for succ in successors:
                    successor_name = f"A{str(succ).zfill(num_digits)}"
                    if successor_name not in data['predecessors']:
                        data['predecessors'][successor_name] = []
                    data['predecessors'][successor_name].append(activity_name)

                # Append resource requirements for DataFrame
                resource_data.append(resource_requirements)

            line_index += 1

        # Create resource DataFrame
        data['resources'] = pd.DataFrame(resource_data, columns=resource_columns,
                                         index=[activity['name'] for activity in data['activities']])


        # Convert predecessors values to comma-separated strings
        for key in data['predecessors']:
            data['predecessors'][key] = ','.join(data['predecessors'][key])

        activity_durations = {activity['name']: activity['duration'] for activity in data['activities']}
        data['duration'] = pd.DataFrame.from_dict(activity_durations, orient='index', columns=['duration'])


    return data

def significant_values(values, threshold ):
    values = np.absolute(values)
    percents = values.cumsum()/values.sum()
    return int(np.argmax( percents > threshold))



class PredecessorTable:

    @staticmethod
    def from_dict_of_strings(data: Dict[str, str], simplify=True):
        def replace_and_split(x):
            result = set(x.replace(' ', '')
                         .replace('-', '')
                         .replace(';', ',')
                         .split(',')
                         ) - {''}
            return result

        dict_of_sets = {key: replace_and_split(value) for key, value in data.items()}
        return PredecessorTable.from_dict_of_sets(data=dict_of_sets, simplify=simplify)

    @staticmethod
    def from_dict_of_sets(data: Dict[str, Set[str]], simplify=True) -> 'PredecessorTable':
        names = chain.from_iterable(data.values())
        expanded_dict = {key: set() for key in names}
        expanded_dict.update(data)
        return PredecessorTable(data=expanded_dict, simplify=simplify)

    @staticmethod
    def from_dataframe_of_strings(data: pd.DataFrame, activity: str, predecessor: str,
                                  simplify=True) -> 'PredecessorTable':
        dataframe = data.reset_index().copy(deep=True)
        miniframe = dataframe.loc[:, [activity, predecessor]]
        miniframe.set_index(activity, inplace=True)
        dict_of_sets = miniframe.to_dict()
        return PredecessorTable.from_dict_of_strings(data=dict_of_sets[predecessor], simplify=simplify)

    @staticmethod
    def from_project(project) -> 'PredecessorTable':
        new_data = project.predecessors
        return PredecessorTable.from_dict_of_sets(data=new_data, simplify=False)

    def __init__(self, data, simplify=True):
        self.direct_predecessors = dict(sorted(data.items()))
        # predecessors MUST be calculated before direct_predecessor. It is used by calculate_direct_predecessor_of
        self.predecessors = {key: self.calculate_predecessor_of(key) for key in self.direct_predecessors.keys()}
        self.predecessors = dict(sorted(self.predecessors.items()))
        if simplify:
            self.direct_predecessors = {key: self.calculate_direct_predecessor_of(key) for key in self.direct_predecessors.keys()}
            self.direct_predecessors = dict(sorted(self.direct_predecessors.items()))

    @cached_property
    def activity_names(self) -> Set[str]:
        return sorted(
            list(set(chain.from_iterable(self.direct_predecessors.values())) | set(self.direct_predecessors.keys()))
        )

    
    def calculate_predecessor_of(self, name: str) -> Set[str]:
        predecessors = self.direct_predecessors[name].copy()
        if predecessors == ():
            return set()
        result = predecessors.copy()
        for ancestor in predecessors:
            result.update(self.calculate_predecessor_of(ancestor))
        return result

    
    def calculate_direct_predecessor_of(self, name: str) -> Dict[str, str]:
        predecessors = self.predecessors[name].copy()
        for ancestor in predecessors:
            predecessors = predecessors - self.predecessors[ancestor]
        return predecessors

    @cached_property
    def start_activities(self) -> Set[str]:
        result = {key for key, value in self.direct_predecessors.items() if value == set()}
        return result

    @cached_property
    def end_activities(self) -> Set[str]:
        predecessors = set(chain.from_iterable(self.direct_predecessors.values()))
        result = {key for key, value in self.direct_predecessors.items() if key not in predecessors}
        return result

    @cached_property
    def augmented_predecessor(self) -> Dict[str, str]:
        result = {key: set() for key in self.direct_predecessors.keys()}

        for activity, predecessors in self.direct_predecessors.items():
            if predecessors == set():
                result[activity].add(f'@∇StartOfProject⤑Δ{activity}')
            else:
                for ancestor in predecessors:
                    dummy_name = f'@∇{ancestor}⤑Δ{activity}'
                    result[activity].add(dummy_name)
                    result[dummy_name] = {ancestor}
        for activity in self.end_activities:
            result[activity].add(f'@∇{activity}⤑ΔEndOfProject')
        return result

    @cached_property
    def dummies(self):
        result = (self.augmented_predecessor.keys()
                  | chain.from_iterable(self.augmented_predecessor.values())
                  ) - set(self.activity_names)
        return result

    @cached_property
    def nodes(self) -> Set[str]:
        result = list()
        for activity, predecessors in self.augmented_predecessor.items():
            result.append(activity)
            result.extend(predecessors)
        result = set(chain.from_iterable(name.split('⤑') for name in result if name[0] == '@'))
        return result

    def copy(self):
        return PredecessorTable(data=self.direct_predecessors.copy())

    def create_project(self, simplify=True):
        def create_edge_info_for_dummy(name):
            return name.replace('@', '').split('⤑') + [{'activity': name}]

        edgelist_dummy_activities = list(create_edge_info_for_dummy(name)
                                         for name in chain.from_iterable(self.augmented_predecessor.values())
                                         if name[0] == '@'
                                         )

        edgelist_actual_activities = list((f'Δ{name}', f'∇{name}', {'activity': name})
                                          for name in self.activity_names)
        edgelist = edgelist_dummy_activities + edgelist_actual_activities

        pert_graph = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
        for source, target, edge_attributes in edgelist:
            pert_graph.edges[source, target]['activity'] = edge_attributes['activity']
        if not simplify:
            return ProjectGraph(pert_graph)
        return ProjectGraph(pert_graph).simplify()

    @cached_property
    def immediate_linkage_matrix(self):
        activities_name = sorted(list(self.activity_names))
        linkage_matrix = pd.DataFrame(data=False, columns=activities_name, index=activities_name, dtype=bool)
        linkage_matrix.index.name = 'activities'
        for activity, predecessors in self.direct_predecessors.items():
            for ancestor in predecessors:
                linkage_matrix.loc[activity, ancestor] = True
        return linkage_matrix

    @cached_property
    def distant_linkage_matrix(self):
        activities_name = sorted(list(self.activity_names))
        linkage_matrix = pd.DataFrame(data=False, columns=activities_name, index=activities_name, dtype=bool)
        linkage_matrix.index.name = 'activities'
        for activity, predecessors in self.predecessors.items():
            for ancestor in predecessors:
                linkage_matrix.loc[activity, ancestor] = True
        return linkage_matrix

    @cached_property
    def display_immediate_linkage_matrix(self):
        def highlight_true(value):
            if value:
                return 'background-color: cyan;  border: 1px solid black;'
            else:
                return 'background-color: white;  border:1px solid black;'

        data = self.immediate_linkage_matrix
        return data.replace(False, '').style.map(highlight_true)



class ProjectGraph:
    def __init__(self, graph):
        if graph is None:
            self.pert_graph = nx.DiGraph()
        else:
            self.pert_graph = graph.copy()

    def copy(self):
        return ProjectGraph(self.pert_graph)

    @staticmethod
    def from_dict_of_strings(data: Dict[str, str], simplify=True):
        # Simplify is for create_project to decide
        # whether or not to deliver a non-simplified project with plenty of dummies
        # The simplify for PredecessorTable is for reducing the data to the direct_predecessor predecessors
        # I think it is better to assume that the data will always be reduced to the direct_predecessor predecessors
        # and leave the simplify parameter for create_project
        result_table = PredecessorTable.from_dict_of_strings(data=data, simplify=True)
        result_project = result_table.create_project(simplify=simplify)
        return result_project

    @staticmethod
    def from_dict_of_sets(data: Dict[str, Set[str]], simplify=True) -> 'PredecessorTable':
        # for simplify see the reasons in from_dict_of_strings
        result_table = PredecessorTable.from_dict_of_sets(data=data, simplify=True)
        result_project = result_table.create_project(simplify=simplify)
        return result_project

    @staticmethod
    def from_dataframe_of_strings(data: pd.DataFrame,
                                  activity: str,
                                  predecessor: str,
                                  simplify=True) -> 'PredecessorTable':
        # for simplify see the reasons in from_dict_of_strings
        result_table = PredecessorTable.from_dataframe_of_strings(data=data,
                                                                  activity=activity,
                                                                  predecessor=predecessor,
                                                                  simplify=simplify)
        result_project = result_table.create_project(simplify=simplify)
        return result_project


    def paths(self, dummies=True):
        first_node, last_node = self.nodes[0], self.nodes[-1]
        edges_from_nodes = self.edges_from_nodes
        result = {}
        for index, path in enumerate(nx.all_simple_edge_paths(self.pert_graph, source=first_node, target=last_node), 1):
                result[f'Route_{index}'] =list(edges_from_nodes[source, target] for source, target in path)
                if not dummies:
                    result[f'Route_{index}'] = list(filter(lambda x: x[0] != '@', result[f'Route_{index}']))
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
        result= np.linalg.svd(path_matrix, full_matrices=False, compute_uv=compute_uv)
        return result

    def display_path_matrix(self, dummies=True):
        def highlight_true(value):
            if value:
                return 'background-color: cyan;  border: 1px solid black;'
            else:
                return 'background-color: white;  border:1px solid black;'

        data = self.path_matrix(dummies=dummies)
        return data.replace(False, '').style.map(highlight_true)

    @staticmethod
    def assess_dummy(name, project):
        original_copy = project.copy()
        other_project = project.contract_activities([name])
        if project.predecessors == other_project.predecessors:
            return (True, other_project)

        other_project = project.remove_activities([name])
        if project.predecessors == other_project.predecessors:
            return (True, other_project)

        return (False, original_copy)

    def simplify(self):
        new_project = self.copy()
        dummies = new_project.dummies
        while dummies:
            dummy = dummies.pop()
            update_needed, new_project = ProjectGraph.assess_dummy(name=dummy, project=new_project)
            if update_needed:  # dummy_was_removed_and_the_rest_changed_their_names,
                dummies = new_project.dummies

        #new_graph = nx.convert_node_labels_to_integers(new_project.pert_graph, ordering='increasing degree', first_label=1)
        #new_graph = nx.relabel_nodes(new_graph, {id: str(id) for id in new_graph.nodes})
        new_graph = nx.relabel_nodes(new_project.pert_graph,
                                     {id: str(indice+1) for indice,id in enumerate(nx.topological_sort(new_project.pert_graph))})
        return ProjectGraph(new_graph)

    @cached_property
    def nodes_from_edges(self) -> Dict[str, Tuple[str, str]]:  # activity_name: (source, target)
        list_of_dicts = list({attributes['activity']: (f'{source}', f'{target}')}
                             for (source, target, attributes) in self.pert_graph.edges(data=True))
        combined_dict = reduce(lambda x, y: {**x, **y}, list_of_dicts, {})
        return combined_dict

    @cached_property
    def edges_from_nodes(self):
        list_of_dicts = list({(f'{source}', f'{target}'): attributes['activity']}
                             for (source, target, attributes) in self.pert_graph.edges(data=True))
        combined_dict = reduce(lambda x, y: {**x, **y}, list_of_dicts, {})
        return combined_dict


    def contract_activities(self, names):
        def contract_activity(name: str, project: ProjectGraph):
            if name not in project.activities:
                # print(f'Activity {name} not found in project')
                return project.copy()
            source, target = project.nodes_from_edges[name]
            new_graph = nx.contracted_nodes(project.pert_graph, target, source, self_loops=False, copy=True)
            return ProjectGraph(new_graph)

        return reduce(lambda project, name: contract_activity(name=name, project=project), names, self)

    def remove_activities(self, names):
        def remove_activity(project_graph, name: str):
            if name not in self.activities:
                return project_graph
            if name in self.nodes_from_edges:
                project_graph.remove_edge(*self.nodes_from_edges[name])
            return project_graph

        new_graph = self.pert_graph.copy()
        new_graph = reduce(lambda graph, name: remove_activity(graph, name), names, new_graph)
        return ProjectGraph(new_graph)

    
    def calculate_predecessor_of(self, activity):
        source, target = self.nodes_from_edges[activity]
        edges_from_nodes_dict = self.edges_from_nodes
        predecessors = [edges_from_nodes_dict[(s, t)]
                        for s, t, reverse
                        in nx.edge_dfs(self.pert_graph, source=source, orientation='reverse')]

        result = filter(lambda x: x[0] != '@', predecessors)
        return set(result)

    @cached_property
    def direct_predecessor(self):
        def calculate_direct_predecessor_of(project, activity):
            source, target = project.nodes_from_edges[activity]
            edges_dict = project.edges_from_nodes
            predecessors = project.pert_graph.in_edges(source, target)
            result = set(filter(lambda x: x[0] != '@', predecessors))
            result = {edges_dict[(s, t)] for s, t, a in result}
            return set(result)

        result = {key: calculate_direct_predecessor_of(self, key) for key in self.actual_activities}
        return result




        return result


    @cached_property 
    def predecessors(self):
        activities = self.actual_activities
        return {key: self.calculate_predecessor_of(key) for key in activities}

    @cached_property
    def nodes(self):
        return [f'{nodo}' for nodo in list(nx.topological_sort(self.pert_graph))]

    @cached_property
    def activities(self):
        return sorted([attributes['activity'] for source, target, attributes in self.pert_graph.edges(data=True)])

    @cached_property
    def dummies(self):
        return sorted([name for name in self.activities if name[0] == '@'])

    @cached_property
    def actual_activities(self):
        return sorted([name for name in self.activities if name[0] != '@'])


    def calculate_pert(self, durations: Dict[str, float]):
        duraciones = {key: 0 for key in self.activities}
        duraciones.update(durations)
        dtype = next(type(item) for item in duraciones.values())
        nodos = self.nodes
        indice_dataframe = sorted(nodos, key=int)
        edges_from_nodes = self.edges_from_nodes
        tempranos = pd.Series(0, index=indice_dataframe).apply(dtype)
        tardios = pd.Series(0, index=indice_dataframe).apply(dtype)
        H_total = pd.Series(0, index=self.activities).apply(dtype)
        for nodo_id in nodos[1:]:
            tempranos[nodo_id] = max(tempranos[source] + duraciones.get(edges_from_nodes[(source, target)])
                                     for (source, target) in self.pert_graph.in_edges(nodo_id)
                                     )
        tardios[nodos[-1]] = tempranos[nodos[-1]]
        for nodo_id in nodos[-2::-1]:
            tardios[nodo_id] = min(tardios[target] - duraciones.get(edges_from_nodes[(source, target)])
                                   for (source, target) in self.pert_graph.out_edges(nodo_id)
                                   )

        for (source, target) in self.pert_graph.edges():
            activity_name = edges_from_nodes[(source, target)]
            H_total[activity_name] = tardios[target] - duraciones.get(activity_name) - tempranos[source]

        return dict(
            nodes=pd.DataFrame(dict(early=tempranos, late=tardios)),
            activities=pd.DataFrame(dict(H_total=H_total)),
        )


    def calendar(self, durations: Dict[str, float]):
        calendario = pd.DataFrame(0, index=durations.index, columns=['inicio_mas_temprano',
                                                                      'inicio_mas_tardio',
                                                                      'fin_mas_temprano',
                                                                      'fin_mas_tardio'])

        resultados_pert = self.calculate_pert(durations)
        calendario['H_total'] = resultados_pert['actividades']['H_total']
        calendario['duracion'] = durations

        tempranos = resultados_pert['nodos']['tempranos']
        tardios = resultados_pert['nodos']['tardios']
        str_to_id = {nodo: self.pert_graph.nodes[nodo]['id'] for nodo in self.pert_graph.nodes}

        for (nodo_inicial, nodo_final) in self.pert_graph.edges:
            activity_name = self.pert_graph.edges[nodo_inicial, nodo_final]['name']
            calendario.loc[activity_name, 'inicio_mas_temprano'] = tempranos[str_to_id[nodo_inicial]]
            calendario.loc[activity_name, 'inicio_mas_tardio'] = tardios[str_to_id[nodo_final]] - durations.get(
                activity_name)
            calendario.loc[activity_name, 'fin_mas_temprano'] = tempranos[str_to_id[nodo_inicial]] + durations.get(
                activity_name)
            calendario.loc[activity_name, 'fin_mas_tardio'] = tardios[str_to_id[nodo_final]]

        return calendario

    def duration(self, durations):
        duraciones = {key: 0 for key in self.activities}
        duraciones.update(durations)
        resultados_pert = self.calculate_pert(duraciones)
        duraciones = resultados_pert['nodes']['early'].values[-1]
        return duraciones

    def critical_path(self, durations):
        def calculate_path_float(path,floats):
            path_floats = [floats[activity] for activity in path]
            return sum(path_floats)

        duraciones = {key: 0 for key in self.activities}
        duraciones.update(durations)
        resultados_pert = self.calculate_pert(duraciones)
        total_floats = resultados_pert['activities']['H_total']

        result = {name: path for name,path in self.paths().items() if calculate_path_float(path,total_floats) == 0}
        return result





    def pert(self,
             filename=None,
             durations=False,
             size=None,
             orientation='landscape',
             rankdir='LR',
             ordering='out',
             ranksep=0.5,
             nodesep=0.5,
             rotate=0,
             **kwargs):
        if filename is None:
            filename = 'output_pert_figure.svg'

        if isinstance(durations, pd.Series) or durations:
            duraciones = {key: 0 for key in self.activities}
            duraciones.update(durations)
            resultados_pert = self.calculate_pert(duraciones)
            tempranos = resultados_pert['nodes']['early']
            tardios = resultados_pert['nodes']['late']
            H_total = resultados_pert['activities']['H_total']

        dot_graph = pgv.AGraph(size=size,
                               orientation=orientation,
                               rankdir=rankdir,
                               ordering=ordering,
                               ranksep=ranksep,
                               nodesep=nodesep,
                               rotate=rotate,
                               directed=True,
                               **kwargs)

        dot_graph.node_attr['shape'] = 'Mrecord'
        dot_graph.add_edges_from(self.pert_graph.edges)

        for node_number in dot_graph.nodes():
            current_node = dot_graph.get_node(node_number)

            if isinstance(durations, pd.Series) or durations:
                current_node.attr['label'] = (f"{node_number} | {{ "
                                              f"<early> {tempranos[node_number]} | "
                                              f"<last>  {tardios[node_number]} }}")
            else:
                current_node.attr['label'] = (f"{node_number} | {{ "
                                              f"<early>  | "
                                              f"<last>   }}")

        for origin, destination in dot_graph.edges_iter():
            current_edge = dot_graph.get_edge(origin, destination)
            current_edge.attr['headport'] = 'early'
            current_edge.attr['tailport'] = 'last'

            activity_name = self.edges_from_nodes[(origin, destination)]
            if isinstance(durations, pd.Series) or durations:
                current_edge.attr['label'] = (f"{activity_name}"
                                              f"({duraciones[activity_name]})")
                if H_total[activity_name] == 0:
                    current_edge.attr['color'] = 'red:red'
                    current_edge.attr['fontcolor'] = 'red'
                    current_edge.attr['style'] = 'dashed' if activity_name[0] == '@' else 'bold'

                else:
                    current_edge.attr['color'] = 'azure4'  # if activity_name[0] == '@' else 'black'
                    current_edge.attr['fontcolor'] = 'azure4'
                    current_edge.attr['style'] = 'dashed' if activity_name[0] == '@' else 'solid'
            else:
                current_edge.attr['label'] = f"{activity_name}"
                current_edge.attr['fontcolor'] = 'forestgreen' if activity_name[0] == '@' else 'black'
                current_edge.attr['style'] = 'dashed' if activity_name[0] == '@' else 'solid'
                current_edge.attr['color'] = 'forestgreen' if activity_name[0] == '@' else 'black'

        dot_graph.draw(filename, prog='dot')
        if filename.lower().endswith('.svg'):
            return SVG(filename)
        return Image(filename)

    def gantt(self, duraciones,
              representar=None,
              total=None,
              acumulado=False,
              holguras=False,
              cuadrados=False):

        if representar is None:
            representar = {actividad: '  '  for actividad in self.activities}

        if isinstance(representar, str) and representar == 'names':
            representar = {actividad: actividad for actividad in self.activities}

        if isinstance(representar, dict):
            representar = pd.Series(representar, index=self.actual_activities)

        if isinstance(representar, pd.Series):
            representar = representar.reindex(self.actual_activities, fill_value='')

        resultados_pert = self.calculate_pert(duraciones)
        tempranos = resultados_pert['nodes']['early']
        duracion_proyecto = tempranos.values[-1]
        periodos = range(1, ceil(duracion_proyecto) + 1)
        actividades_con_duracion = [nombre for nombre in self.activities if duraciones.get(nombre, 0) != 0]
        actividades_con_duracion.sort()
        gantt = pd.DataFrame('', index=actividades_con_duracion, columns=periodos)

        for edge in self.pert_graph.edges:
            activity_name = self.pert_graph.edges[edge]['activity']
            duracion_tarea = duraciones.get(activity_name, 0)
            if duracion_tarea != 0:
                comienzo_tarea = tempranos[edge[0]]
                gantt.loc[activity_name, (comienzo_tarea + 1):(comienzo_tarea + duracion_tarea)] = representar[
                    activity_name]

        def color_gantt(val):
            background = 'white' if val == '' else 'sandybrown'
            style = f'background-color: {background}'
            return style

        # Set CSS properties for th elements in dataframe
        th_props = [
            ('font-size', '11px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#6d6d6d'),
            ('background-color', '#f7f7f9'),
            ('border', '1px solid')
        ]

        # Set CSS properties for td elements in dataframe
        td_props = [
            ('font-size', '11px'),
            ('border-color', '#c0c0c0'),
            ('border', '1px solid'),
        ]

        # Set table styles
        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
            dict(selector='', props=[('border', '3px solid')]),
        ]

        def summary(df, fn=np.sum, axis=0, name='Total'):
            df_total = df.replace('', 0)
            total = df_total.apply(fn, axis=axis).to_frame(name)
            if axis == 0:
                total = total.T
            out = pd.concat([df, total], axis=axis)
            return out

        if total is None:
            mat = gantt

        elif total == 'columna':
            mat = summary(gantt, axis=1)

        elif total == 'fila':
            mat = summary(gantt, axis=0)
            if acumulado:
                fila_acumulado = mat.loc['Total'].cumsum()
                fila_acumulado.name = 'Acumulado'
                mat = pd.concat([mat, fila_acumulado.to_frame().T], axis=0).fillna('')
            if cuadrados:
                fila_cuadrados = mat.loc['Total'] ** 2
                fila_cuadrados.name = 'Cuadrados'
                mat = pd.concat([mat, fila_cuadrados.to_frame().T], axis=0).fillna('')
        elif total == 'ambas':
            mat = summary(summary(gantt, axis=0), axis=1)
            if acumulado:
                fila_acumulado = mat.loc['Total'].drop('Total').cumsum()
                fila_acumulado.name = 'Acumulado'
                mat = pd.concat([mat, fila_acumulado.to_frame().T], axis=0).fillna('')

        if holguras:
            mat['H_total'] = resultados_pert['activities']['H_total']
            if total in ['fila', 'ambas']:
                mat.loc['Total', 'H_total'] = None

        resultado = (mat
                     .style
                     .set_table_styles(styles)
                     .map(color_gantt)
                     .apply(lambda x: ['background: #f7f7f9' if x.name in ["Total", "Acumulado", "H_total", "Cuadrados"]
                                       else '' for i in x], axis=0)
                     .apply(lambda x: ['background: #f7f7f9' if x.name in ["Total", "Acumulado", "H_total", "Cuadrados"]
                                       else '' for i in x], axis=1)

                     )

        return resultado


    def roy(self,
            filename=None,
            duraciones=False,
            size=None,
            orientation='landscape',
            rankdir='LR',
            ordering='out',
            ranksep=0.5,
            nodesep=0.5,
            rotate=0,
            **kwargs):
        #precedentes = (project.data.precedentes
        #               .drop([actividad for actividad in project.data.index if actividad[0] == 'f'])
        #               )
        df = PredecessorTable.from_project(self).immediate_linkage_matrix
        rows_to_keep = ~df.index.str.startswith('@')
        columns_to_keep = ~df.columns.str.startswith('@')
        df_cleaned = df.loc[rows_to_keep, columns_to_keep]

        self.roy_graph = make_Roy(df_cleaned)

        if filename is None:
            filename = 'output_roy_figure.png'

        if duraciones is None:
            duraciones = self.data['duracion'].copy()

        if duraciones is not False:
            calendario = self.calendar()
            inicio_mas_temprano = calendario['inicio_mas_temprano']
            inicio_mas_tardio = calendario['inicio_mas_tardio']
            duraciones['inicio'] = 0
            duraciones['fin'] = 0
            inicio_mas_temprano['inicio'] = 0
            inicio_mas_tardio['inicio'] = 0

            inicio_mas_temprano['fin'] = self.duration()
            inicio_mas_tardio['fin'] = inicio_mas_temprano['fin']

        dot_graph = pgv.AGraph(size=size,
                               orientation=orientation,
                               rankdir=rankdir,
                               ordering=ordering,
                               ranksep=ranksep,
                               nodesep=nodesep,
                               rotate=rotate,
                               directed=True,
                               **kwargs)

        dot_graph.node_attr['shape'] = 'Mrecord'
        dot_graph.add_edges_from(self.roy_graph.edges)

        for nodo in dot_graph.nodes():
            current_node = dot_graph.get_node(nodo)
            if duraciones is not False:
                current_node.attr['label'] = (f"{nodo} | {{ "
                                              f"<min> {inicio_mas_temprano[str(nodo)]} |"
                                              f"<dur> {duraciones[str(nodo)]}          | "
                                              f"<max> {inicio_mas_tardio[str(nodo)]}   }}")
            else:
                current_node.attr['label'] = (f"{nodo} | {{ "
                                              f"<min>  | "
                                              f"<dur>  | "
                                              f"<max>   }}")

        dot_graph.draw(filename, prog='dot')
        return Image(filename)


    def gantt_cargas(self, duraciones, representar, report=True):
        gantt = self.gantt(representar=representar, total='fila', acumulado=False, holguras=True, cuadrados=True)
        gantt.data.loc['Total', 'H_total'] = ''
        suma_cuadrados = gantt.data.loc['Cuadrados', :].sum()
        gantt.data.loc['Cuadrados', 'H_total'] = suma_cuadrados
        if report:
            print('Suma de cuadrados:', suma_cuadrados, '\n')
        return gantt

    def desplazar(self, mostrar='cargas', report=True, **desplazamientos):
        for actividad, duracion in desplazamientos.items():
            nombre_slide = 'slide_' + actividad
            if nombre_slide in self.data.index:
                self.data.loc[nombre_slide, 'duracion'] += duracion
            else:
                nueva_fila = pd.Series({'duracion': duracion},
                                       name=nombre_slide,
                                       index=self.data.columns).fillna(0)
                self.data = pd.concat([self.data, nueva_fila.to_frame().T], axis=0)

        lista_edges = list(self.pert_graph.edges)
        for edge in lista_edges:
            activity_name = self.pert_graph.edges[edge]['nombre']
            slide_name = 'slide_' + activity_name

            if (activity_name in desplazamientos
                    and slide_name not in self.activities):
                self.pert_graph.remove_edge(edge[0], edge[1])
                tamano_cadena = 1
                nodo_auxiliar_str = activity_name  # + '___' + generate_random_str(tamano_cadena)
                while nodo_auxiliar_str in self.pert_graph.nodes():
                    nodo_auxiliar_str = activity_name  # + '___' + generate_random_str(tamano_cadena)
                self.pert_graph.add_node(nodo_auxiliar_str, id=nodo_auxiliar_str)
                self.pert_graph.add_edge(edge[0], nodo_auxiliar_str, nombre='slide_' + activity_name)
                self.pert_graph.add_edge(nodo_auxiliar_str, edge[1], nombre=activity_name)

        lista_nodos = list(nx.topological_sort(self.pert_graph))
        nx.set_node_attributes(self.pert_graph, {nodo: {'id': (id + 1)} for id, nodo in enumerate(lista_nodos)})

        if report and mostrar == 'cargas':
            representacion = self.gantt_cargas()
            display(representacion)
            return

        if report and mostrar in self.data.columns:
            representacion = self.gantt(representar=self.data[mostrar], total='fila', holguras=True)
            display(representacion)
            return

    def evaluar_desplazamiento(self, report=True, **desplazamientos):
        proyecto = self.copy()
        proyecto.desplazar(**desplazamientos, report=report)
        return proyecto.gantt_cargas(report=False).data.loc['Cuadrados', 'H_total']

    def evaluar_rango_de_desplazamientos(self, actividad, report=True):
        minimo = 0
        maximo = int(self.calculate_pert()['actividades'].loc[actividad, 'H_total'])
        suma_cuadrados = pd.DataFrame(0, index=range(minimo, maximo + 1), columns=['Suma_de_cuadrados'])
        if report:
            print('Sin desplazar:')
        suma_cuadrados.loc[0, 'Suma_de_cuadrados'] = self.gantt_cargas(report=report).data.loc['Cuadrados', 'H_total']
        for slide in range(minimo + 1, maximo + 1):
            if report:
                print('Desplazamiento:', slide)
            carga2 = self.evaluar_desplazamiento(**{actividad: slide}, report=report)
            suma_cuadrados.loc[slide, 'Suma_de_cuadrados'] = carga2
        return suma_cuadrados

    def standard_deviation(self, durations, variances):
        varianza = {key: 0 for key in self.activities}
        varianza.update(variances)
        caminos = self.critical_path(durations=durations)
        varianza_caminos = {key: sum(varianza[activity] for activity in path)
                            for key, path in caminos.items()}
        [print('Variance path:', key, ':', value) for key, value in varianza_caminos.items()]
        varianza_proyecto = max(varianza_caminos.values())
        std_deviation = varianza_proyecto ** 0.5
        print('Project duration variance:', varianza_proyecto)
        print('Project duration standard deviation:', std_deviation)
        return std_deviation

    
    def ackoff(self, durations, min_durations, costs, reduction=100):
        def calculate_reduction_cost(activities, costs):
            return sum([costs[name] for name in set(activities)])

        def highlight_maximum(column):
            highlight = 'background-color: greenyellow;'
            default = ''

            maximum_in_column = column.max()

            # must return one string per cell in this column
            return [highlight if v == maximum_in_column else default for v in column]

        def highlight_minimum(row):
            highlight = 'background-color: fuchsia;'
            default = ''

            minimum_in_row = row.min()

            # must return one string per cell in this column
            return [highlight if v == minimum_in_row else default for v in row]

        def highlight_cur(value):
                return 'background-color: white;  border:1px solid black;'

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

        # Data preparation
        path_matrix = self.path_matrix(dummies=False)
        path_names = path_matrix.columns
        activity_names = self.actual_activities
        result = path_matrix.T.copy()

        result.replace(0, np.nan, inplace=True)
        result = result.multiply(costs, axis='columns')
        step = 0
        result.loc[:, step] = durations @ path_matrix
        result.loc[step, :] = durations - min_durations
        print('\n')
        best_option = dict()
        while step < reduction:
            critical_paths = self.critical_path(durations)
            critical_path_matrix = result.loc[critical_paths.keys(), activity_names]
            mini_path_matrix = (result
                                .loc[critical_paths.keys(), self.actual_activities]
                                .loc[:, periods_available != 0]

                                )
            mini_path_matrix_filtered = mini_path_matrix.apply(lambda x: mini_path_matrix.columns[x.notna()], axis=1)

            mini_path_matrix_filtered_has_empty_rows = any(mini_path_matrix_filtered.apply(len) == 0)
            if mini_path_matrix_filtered_has_empty_rows:
                print('No more paths to reduce.\n\n')
                break

            costes = {key: calculate_reduction_cost(key, costs) for key in
                      product(*mini_path_matrix_filtered.values)}
            best_option[step] = list(set(min(costes, key=costes.get)))
            print(f'Step: {step},\t Best option: {best_option[step]}, \t Cost: {calculate_reduction_cost(best_option[step], costs)}, \t Critical paths: {list(critical_paths.keys())}')
            for activity in best_option[step]:
                periods_available[activity] -= 1
                durations[activity] -= 1


            result.loc[step + 1, activity_names] =  periods_available
            result.loc[path_names, step + 1] = durations @ path_matrix
            step  += 1

        idx = pd.IndexSlice
        filas_steps = idx[range(step+1), :]
        abajo = idx[range(step), activity_names]
        derecha = idx[path_names, range(step+1)]
        curs = idx[path_names, activity_names]

        result = (result
                  .astype(float)
                  .style
                  .set_table_styles([dict(selector='th', props=[('text-align', 'center'), ('border', '1px')]),
                                     dict(selector="", props= [("border", "1px solid")]),
                                     ])

                  .format(na_rep="", precision=1)
                  .map(highlight_cur, subset=curs)
                  #.apply(lambda x: ['background-color: cyan' for _ in x], axis=1, subset=abajo )
                  #.apply(lambda x: ['background-color: cyan' for _ in x], axis=1, subset=derecha )
                  .apply(highlight_maximum, axis=0, subset=derecha)
                  #.apply(highlight_minimum, axis=1, subset=filas_steps)
                  )
        for index in range(step):
            for activity in best_option[index]:
                recortadas = idx[index, activity]
                result.apply(lambda x: ['background-color: yellow' for _ in x], axis=1, subset=recortadas)
        print('\n' * 2)
        return result, best_option, durations, periods_available





class EarnedValue():
    def __init__(self, pert):
        self.pert = pert

    def calcula_gantts(self,
                       duraciones_planificadas,
                       duraciones_reales,
                       costes_planificados,
                       costes_reales,
                       porcentaje_de_completacion):
        if any(porcentaje_de_completacion > 1):
            porcentaje_de_completacion = porcentaje_de_completacion / 100

        costes_planificados_por_periodo = costes_planificados / duraciones_planificadas
        gantt_PV = self.pert.gantt(duraciones_planificadas,
                                   representar=costes_planificados_por_periodo,
                                   total='ambas', acumulado=True)

        costes_reales_por_periodo = (costes_reales / duraciones_reales).reindex(costes_planificados.index, fill_value=0)
        duraciones_reales = duraciones_reales.reindex(duraciones_planificadas.index, fill_value=0)
        gantt_AC = self.pert.gantt(duraciones_reales,
                                   representar=costes_reales_por_periodo,
                                   total='ambas', acumulado=True)

        valor_ganado_total_tarea = (costes_planificados * porcentaje_de_completacion).reindex(costes_planificados.index,
                                                                                              fill_value=0)
        valor_ganado_por_periodo = (valor_ganado_total_tarea / duraciones_reales).reindex(costes_planificados.index,
                                                                                          fill_value=0)
        gantt_EV = self.pert.gantt(duraciones_reales,
                                   representar=valor_ganado_por_periodo,
                                   total='ambas', acumulado=True)

        acumulados = pd.DataFrame(dict(PV=gantt_PV.data.loc['Total', :].cumsum(),
                                       EV=gantt_EV.data.loc['Total', :].cumsum(),
                                       AC=gantt_AC.data.loc['Total', :].cumsum(), ),
                                  index=gantt_EV.data.columns).drop('Total')

        return dict(Gantt_PV=gantt_PV, Gantt_AC=gantt_AC, Gantt_EV=gantt_EV, acumulados=acumulados)


def make_Roy(linkage_matrix):
    graph = nx.DiGraph()
    graph.add_nodes_from(['start', 'finish'])

    enlaces_iniciales = [('start', actividad) for actividad in linkage_matrix.index
                         if not any(linkage_matrix.loc[actividad, :])]

    graph.add_edges_from(enlaces_iniciales)

    enlaces_finales = [(actividad, 'finish') for actividad in linkage_matrix.index
                       if not any(linkage_matrix.loc[:, actividad])]

    graph.add_edges_from(enlaces_finales)

    resto_de_enlaces = [(b, a) for a, b in list(linkage_matrix[linkage_matrix].stack().index)]
    graph.add_edges_from(resto_de_enlaces)
    return graph


def zaderenko(project: ProjectGraph, durations: Dict[str, float]):
    duraciones = {key: 0 for key in project.activities}
    duraciones.update(durations)
    resultados_pert = project.calculate_pert(duraciones)['nodes']
    lista_de_nodos_ordenada = sorted(project.nodes, key=int)
    z = pd.DataFrame(np.nan, index=lista_de_nodos_ordenada, columns=lista_de_nodos_ordenada)

    for name, (source, target) in project.nodes_from_edges.items():
        z.loc[source, target] = duraciones[name]

    z['early'] = resultados_pert['early']
    z = pd.concat([z, resultados_pert['late'].to_frame().T], axis=0).fillna('')
    return z
