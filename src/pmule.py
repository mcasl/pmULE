from functools import reduce
from itertools import chain, product
import math
from math import ceil
from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import pygraphviz as pgv
from IPython.display import display, Image, SVG, Latex

from numpyarray_to_latex import to_ltx
from numpyarray_to_latex.jupyter import to_jup

def truncate_and_pad(value, n_decimal_places):
    def truncate_float(value, n_digits):
        if n_digits < 0:
            raise ValueError("n_digits must be non-negative.")
        multiplier = 10 ** n_digits
        truncated_value = math.trunc(value * multiplier) / multiplier
        return int(truncated_value) if n_digits == 0 else truncated_value
    
    truncated = truncate_float(value, n_decimal_places)
    format_string = f"{{:.{n_decimal_places}f}}"
    return format_string.format(truncated)

def read_rcp(filename):
	data = {
			'num_activities':        0,
			'activities':            [],
			'num_resources':         0,
			'resource_availability': [],
			'resources':             None,
			'predecessors':          {}
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
				index=resource_columns)
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
						'name':                  activity_name,
						'duration':              duration,
						'resource_requirements': resource_requirements,
						'successors':            successors
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


def significant_values(values, threshold):
	values = np.absolute(values)
	percents = values.cumsum() / values.sum()
	return int(np.argmax(percents > threshold))


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
	def from_project(project, dummies=False) -> 'PredecessorTable':
		new_data = project.distant_predecessor(dummies=dummies)
		return PredecessorTable.from_dict_of_sets(data=new_data, simplify=True)
	
	def __init__(self, data, simplify=True):
		self.immediate_predecessor = dict(sorted(data.items()))
		# distant_predecessor MUST be calculated before immediate_predecessor. It is used by calculate_immediate_predecessor_of
		self.distant_predecessor = {key: self.calculate_distant_predecessor_of(key) for key in
									self.immediate_predecessor.keys()}
		self.distant_predecessor = dict(sorted(self.distant_predecessor.items()))
		if simplify:
			self.immediate_predecessor = {key: self.calculate_immediate_predecessor_of(key) for key in
											self.immediate_predecessor.keys()}
			self.immediate_predecessor = dict(sorted(self.immediate_predecessor.items()))
	
	@property
	def activity_names(self) -> Set[str]:
		return sorted(
				list(set(chain.from_iterable(self.immediate_predecessor.values())) | set(
					self.immediate_predecessor.keys()))
		)
	
	def calculate_distant_predecessor_of(self, name: str) -> Set[str]:
		predecessors = self.immediate_predecessor[name].copy()
		if predecessors == ():
			return set()
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
		result = {key for key, value in self.immediate_predecessor.items() if value == set()}
		return result
	
	@property
	def end_activities(self) -> Set[str]:
		predecessors = set(chain.from_iterable(self.immediate_predecessor.values()))
		result = {key for key, value in self.immediate_predecessor.items() if key not in predecessors}
		return result
	
	@property
	def augmented_predecessor(self) -> Dict[str, str]:
		result = {key: set() for key in self.immediate_predecessor.keys()}
		
		for activity, predecessors in self.immediate_predecessor.items():
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
	
	@property
	def dummies(self):
		result = (self.augmented_predecessor.keys()
				  | chain.from_iterable(self.augmented_predecessor.values())
				  ) - set(self.activity_names)
		return result
	
	@property
	def nodes(self) -> Set[str]:
		result = list()
		for activity, predecessors in self.augmented_predecessor.items():
			result.append(activity)
			result.extend(predecessors)
		result = set(chain.from_iterable(name.split('⤑') for name in result if name[0] == '@'))
		return result
	
	def copy(self):
		return PredecessorTable(data=self.immediate_predecessor.copy())
	
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
	
	@property
	def immediate_linkage_matrix(self):
		activities_name = sorted(list(self.activity_names))
		linkage_matrix = pd.DataFrame(data=False, columns=activities_name, index=activities_name, dtype=bool)
		linkage_matrix.index.name = 'activities'
		for activity, predecessors in self.immediate_predecessor.items():
			for ancestor in predecessors:
				linkage_matrix.loc[activity, ancestor] = True
		return linkage_matrix
	
	@property
	def distant_linkage_matrix(self):
		def highlight_true(value):
			if value:
				return 'background-color: cyan;  border: 1px solid black;'
			else:
				return 'background-color: white;  border:1px solid black;'
		
		activities_name = sorted(list(self.activity_names))
		linkage_matrix = pd.DataFrame(data=False, columns=activities_name, index=activities_name, dtype=bool)
		linkage_matrix.index.name = 'activities'
		for activity, predecessors in self.distant_predecessor.items():
			for ancestor in predecessors:
				linkage_matrix.loc[activity, ancestor] = True
		return linkage_matrix.replace(False, '').style.map(highlight_true)
	
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
		# The simplify for PredecessorTable is for reducing the data to the immediate_predecessor predecessors
		# I think it is better to assume that the data will always be reduced to the immediate_predecessor predecessors
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
		result = []
		for index, path in enumerate(nx.all_simple_edge_paths(self.pert_graph, source=first_node, target=last_node), 1):
			new_item = list(edges_from_nodes[source, target] for source, target in path)
			if not dummies:
				new_item = list(filter(lambda x: x[0] != '@', new_item))
			result.append(new_item)
		
		result = {'Route_' + str(i + 1): value for i, value in enumerate(sorted(result))}
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
				return 'background-color: cyan;  border: 1px solid black;'
			else:
				return 'background-color: white;  border:1px solid black;'
		
		data = self.path_matrix(dummies=dummies)
		return data.replace(False, '').style.map(highlight_true)
	
	@staticmethod
	def assess_dummy(name, project):
		original_copy = project.copy()
		other_project = project.contract_activities([name])
		if project.distant_predecessor(dummies=False) == other_project.distant_predecessor(dummies=False):
			return (True, other_project)
		
		other_project = project.remove_activities([name])
		if project.distant_predecessor(dummies=False) == other_project.distant_predecessor(dummies=False):
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
		
		# new_graph = nx.convert_node_labels_to_integers(new_project.pert_graph, ordering='increasing degree', first_label=1)
		# new_graph = nx.relabel_nodes(new_graph, {id: str(id) for id in new_graph.nodes})
		new_graph = nx.relabel_nodes(new_project.pert_graph,
									 {id: str(indice + 1) for indice, id in
									  enumerate(nx.topological_sort(new_project.pert_graph))})
		return ProjectGraph(new_graph)
	
	@property
	def nodes_from_edges(self) -> Dict[str, Tuple[str, str]]:  # activity_name: (source, target)
		list_of_dicts = list({attributes['activity']: (f'{source}', f'{target}')}
							 for (source, target, attributes) in self.pert_graph.edges(data=True))
		combined_dict = reduce(lambda x, y: {**x, **y}, list_of_dicts, {})
		return combined_dict
	
	@property
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
	
	def calculate_distant_predecessor_of(self, activity, dummies=False):
		source, target = self.nodes_from_edges[activity]
		edges_from_nodes_dict = self.edges_from_nodes
		predecessors = [edges_from_nodes_dict[(s, t)]
						for s, t, reverse
						in nx.edge_dfs(self.pert_graph, source=source, orientation='reverse')]
		
		result = predecessors if dummies else filter(lambda x: x[0] != '@', predecessors)
		return set(result)
	
	def calculate_immediate_predecessor_of(self, activity, dummies=False):
		source, target = self.nodes_from_edges[activity]
		edges_dict = self.edges_from_nodes
		predecessors = self.pert_graph.in_edges(source, target)
		result = predecessors if dummies else set(filter(lambda x: x[0] != '@', predecessors))
		result = {edges_dict[(s, t)] for s, t, a in result}
		return set(result)
		
		return result
	
	def distant_predecessor(self, dummies=False, format=None):
		activities = self.activities if dummies else self.actual_activities
		resultado = {key: self.calculate_distant_predecessor_of(key, dummies=dummies) for key in activities}
		if format == 'DataFrame':
			resultado = pd.Series({key: ', '.join(sorted(value)) for key, value in resultado.items()}, name='predecessors').replace('', '----')
		return resultado
	
	@property
	def nodes(self):
		return [f'{nodo}' for nodo in list(nx.topological_sort(self.pert_graph))]
	
	@property
	def activities(self):
		return sorted([attributes['activity'] for source, target, attributes in self.pert_graph.edges(data=True)])
	
	@property
	def dummies(self):
		return sorted([name for name in self.activities if name[0] == '@'])
	
	@property
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
	
	def calendar(self, duraciones):
		
		calendario = pd.DataFrame(0, index=duraciones.index, columns=['inicio_mas_temprano',
																	  'inicio_mas_tardio',
																	  'fin_mas_temprano',
																	  'fin_mas_tardio'])
		
		resultados_pert = self.calculate_pert(duraciones)
		calendario['H_total'] = resultados_pert['activities']['H_total']
		calendario['duracion'] = duraciones
		
		tempranos = resultados_pert['nodes']['early']
		tardios = resultados_pert['nodes']['late']
		
		for (nodo_inicial, nodo_final) in self.pert_graph.edges:
			activity_name = self.pert_graph.edges[nodo_inicial, nodo_final]['activity']
			if activity_name[0] == '@':
				continue
			calendario.loc[activity_name, 'inicio_mas_temprano'] = tempranos[nodo_inicial]
			calendario.loc[activity_name, 'inicio_mas_tardio'] = tardios[nodo_final] - duraciones.get(
					activity_name)
			calendario.loc[activity_name, 'fin_mas_temprano'] = tempranos[nodo_inicial] + duraciones.get(
					activity_name)
			calendario.loc[activity_name, 'fin_mas_tardio'] = tardios[nodo_final]
		
		return calendario
	
	def duration(self, durations):
		duraciones = {key: 0 for key in self.activities}
		duraciones.update(durations)
		resultados_pert = self.calculate_pert(duraciones)
		duraciones = resultados_pert['nodes']['early'].max()
		return duraciones
	
	def critical_path(self, durations):
		def calculate_path_float(path, floats):
			path_floats = [floats[activity] for activity in path]
			return sum(path_floats)
		
		duraciones = {key: 0 for key in self.activities}
		duraciones.update(durations)
		resultados_pert = self.calculate_pert(duraciones)
		total_floats = resultados_pert['activities']['H_total']
		
		result = {name: path for name, path in self.paths().items() if calculate_path_float(path, total_floats) == 0}
		return result
	
	def zaderenko(self, durations: Dict[str, float]):
		duraciones = {key: 0 for key in self.activities}
		duraciones.update(durations)
		resultados_pert = self.calculate_pert(duraciones)['nodes']
		lista_de_nodos_ordenada = sorted(self.nodes, key=int)
		z = pd.DataFrame(np.nan, index=lista_de_nodos_ordenada, columns=lista_de_nodos_ordenada)
		
		for name, (source, target) in self.nodes_from_edges.items():
			z.loc[source, target] = duraciones[name]
		
		z['early'] = resultados_pert['early']
		z = pd.concat([z, resultados_pert['late'].to_frame().T], axis=0).fillna('')
		return z
	
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
	
	
		
				
				
	def gantt(self,
			  data,
			  duration_label,
			  resource_label = None,
			  total          = None,
			  acumulado      = False,
			  holguras       = False,
			  cuadrados      = False,
 			  tikz           = False,
      		  params		 = None,
        ):
		if params is None:
			params = dict()
   
		my_data = data.copy()
		
		if resource_label is None:
			resource_label = 'resources'
			my_data[resource_label] = '  '
		
		if isinstance(resource_label, str) and resource_label == 'names':
			resource_label = 'resources'
			my_data[resource_label] = my_data.index
		
		resultados_pert = self.calculate_pert(my_data.loc[:, duration_label])
		tempranos = resultados_pert['nodes']['early']
		duracion_proyecto = tempranos.max()
		periodos = range(1, ceil(duracion_proyecto) + 1)
		actividades_con_duracion = [nombre for nombre in self.activities if
									my_data.loc[:, duration_label].get(nombre, 0) != 0]
		actividades_con_duracion.sort()
		if tikz:
			gantt_data = pd.DataFrame(index=actividades_con_duracion, columns=['start', 'duration'])
		gantt = pd.DataFrame('', index=actividades_con_duracion, columns=periodos)
		
		for edge in self.pert_graph.edges:
			activity_name = self.pert_graph.edges[edge]['activity']
			if activity_name[0] == '@':
				continue
			duracion_tarea = my_data.loc[:, duration_label].get(activity_name, 0)
			if duracion_tarea != 0:
				comienzo_tarea = tempranos[edge[0]]
				gantt.loc[activity_name,
						(comienzo_tarea + 1):(comienzo_tarea + duracion_tarea)
						] = my_data.loc[activity_name, resource_label]
				if tikz:
					gantt_data.loc[activity_name, ['start']    ] = comienzo_tarea
					gantt_data.loc[activity_name, ['duration'] ] = duracion_tarea
					gantt_data.loc[activity_name, ['resource'] ] = my_data.loc[activity_name, resource_label]
                    		
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
		
		extra_rows = None
		extra_cols = None
	
		if total is None:
			mat = gantt		
		elif total == 'columna':
			mat = summary(gantt, axis=1)
			extra_cols = mat.copy().loc[actividades_con_duracion,['Total']]
		elif total == 'fila':
			mat = summary(gantt, axis=0)
			extra_rows = mat.copy().loc[['Total'],]
			if acumulado:
				fila_acumulado = mat.loc['Total'].cumsum()
				fila_acumulado.name = 'Acumulado'
				mat = pd.concat([mat, fila_acumulado.to_frame().T], axis=0).fillna('')
				extra_rows = mat.copy().loc[['Total', 'Acumulado'],]

			if cuadrados:
				fila_cuadrados = mat.loc['Total'] ** 2
				fila_cuadrados.name = 'Cuadrados'
				mat = pd.concat([mat, fila_cuadrados.to_frame().T], axis=0).fillna('')
				extra_rows = mat.copy().loc[['Total', 'Cuadrados'],]
		elif total == 'ambas':
			mat = summary(summary(gantt, axis=0), axis=1)
			extra_rows = mat.copy().loc[['Total'],]
			extra_cols = mat.copy().loc[actividades_con_duracion, ['Total']]
			if acumulado:
				fila_acumulado = mat.loc['Total'].drop('Total').cumsum()
				fila_acumulado.name = 'Acumulado'
				mat = pd.concat([mat, fila_acumulado.to_frame().T], axis=0).fillna('')
				extra_rows = mat.copy().loc[['Total', 'Acumulado'],]
				extra_cols = mat.copy().loc[actividades_con_duracion, ['Total']]	
			
			if cuadrados:
				fila_cuadrados = mat.loc['Total'] ** 2
				fila_cuadrados.name = 'Cuadrados'
				mat = pd.concat([mat, fila_cuadrados.to_frame().T], axis=0).fillna('')
				extra_rows = mat.copy().loc[['Total', 'Cuadrados'],]		
    
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
		dibujo = ''
		if tikz:
			gantt_data['Htotal'] = resultados_pert['activities']['H_total'].astype('Int64')
			if holguras:
				extra_cols = pd.concat([extra_cols, gantt_data[['Htotal']] ], axis=1).copy()
			params['inner_text'] = params.get('inner_text', 'resource')
			dibujo = make_gantt_tikz(
      							gantt_data=gantt_data,
                                extra_cols=extra_cols,
								extra_rows=extra_rows,
        						params=params
        				)
		return resultado, dibujo
	
 
 
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
		# precedentes = (self.data.precedentes
		#               .drop([actividad for actividad in self.data.index if actividad[0] == 'f'])
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
	
	def gantt_cargas(self, data, duration_label, resource_label, report=True, tikz=False):
		my_data = data.copy()
		
		gantt, dibujo = self.gantt(my_data, duration_label, resource_label, total='fila', acumulado=False, holguras=True, cuadrados=True, tikz=tikz)
		gantt.data.loc['Total', 'H_total'] = np.nan
		suma_cuadrados = gantt.data.loc['Cuadrados', :].sum()
		gantt.data.loc['Cuadrados', 'H_total'] = suma_cuadrados
		if report:
			print('Suma de cuadrados:', suma_cuadrados, '\n')
		return gantt, dibujo
	
	def desplazar(self, data, duration_label, resource_label, report=True, tikz=False, **activity_shifts):
		my_data = data.copy()
		
		for actividad, duracion in activity_shifts.items():
			slide_label = '💤' + actividad
			if slide_label in my_data.index:
				my_data.loc[slide_label, duration_label] += duracion
			else:
				nueva_fila = pd.Series({duration_label: duracion},
									   name=slide_label,
									   index=my_data.columns).fillna(0).astype(int)
				my_data = pd.concat([my_data, nueva_fila.to_frame().T], axis=0)
		
		lista_edges = list(self.pert_graph.edges)
		for edge in lista_edges:
			activity_name = self.pert_graph.edges[edge]['activity']
			slide_label = '💤' + activity_name
			
			if (activity_name in activity_shifts
					and slide_label not in self.activities):
				self.pert_graph.remove_edge(edge[0], edge[1])
				nodo_auxiliar_str = str(max([int(n) for n in self.pert_graph.nodes]) + 1)
				
				self.pert_graph.add_node(nodo_auxiliar_str, id=nodo_auxiliar_str)
				self.pert_graph.add_edge(edge[0], nodo_auxiliar_str, activity='💤' + activity_name)
				self.pert_graph.add_edge(nodo_auxiliar_str, edge[1], activity=activity_name)
		
		lista_nodos = list(nx.topological_sort(self.pert_graph))
		nx.set_node_attributes(self.pert_graph, {nodo: {'id': (id + 1)} for id, nodo in enumerate(lista_nodos)})
		
		if report and (resource_label in my_data.columns):
			gantt_df, dibujo = self.gantt_cargas(my_data, duration_label, resource_label, tikz=tikz)
			return my_data, gantt_df, dibujo
	
	def evaluar_desplazamiento(self, data, duration_label, resource_label, report=True, **desplazamientos):
		proyecto = self.copy()
		my_data = data.copy()
		
		new_data = proyecto.desplazar(my_data,
									  duration_label=duration_label,
									  resource_label=resource_label,
									  **desplazamientos, report=report)
		return proyecto.gantt_cargas(new_data, duration_label, resource_label, report=False).data.loc['Cuadrados', 'H_total']
	
	def evaluar_rango_de_desplazamientos(self, data, duration_label, resource_label, activity, report=True):
		my_data = data.copy()
		
		minimo = 0
		maximo = int(self.calculate_pert(my_data.loc[:,duration_label])['activities'].loc[activity, 'H_total'])
		suma_cuadrados = pd.DataFrame(0, index=range(minimo, maximo + 1), columns=['Suma_de_cuadrados'])
		if report:
			print('Sin desplazar:')
		suma_cuadrados.loc[0, 'Suma_de_cuadrados'] = self.gantt_cargas(my_data,
																	   duration_label,
																	   resource_label,
																	   report=report).data.loc['Cuadrados', 'H_total']
		for slide in range(minimo + 1, maximo + 1):
			if report:
				print('Desplazamiento:', slide)
			carga2 = self.evaluar_desplazamiento(my_data, duration_label, resource_label, **{activity: slide}, report=report)
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
		path_names = path_matrix.index
		activity_names = self.actual_activities
		result = path_matrix.copy()
		
		result.replace(0, np.nan, inplace=True)
		result = result.multiply(costs, axis='columns')
		step = 0
		print("D shape", durations.shape)
		print("Path matrix shape", path_matrix.shape)
		result.loc[:, step] = path_matrix @ durations
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
			print(
				f'Step: {step},\t Best option: {best_option[step]}, \t Cost: {calculate_reduction_cost(best_option[step], costs)}, \t Critical paths: {list(critical_paths.keys())}')
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
		
		result = (result
				  .astype(float)
				  .style
				  .set_table_styles([dict(selector='th', props=[('text-align', 'center'), ('border', '1px')]),
									 dict(selector="", props=[("border", "1px solid")]),
									 ])
				  
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
				result.apply(lambda x: ['background-color: yellow' for _ in x], axis=1, subset=recortadas)
		print('\n' * 2)
		return result, best_option, durations, periods_available
	
	def incidence_matrix(self):
		nodos_ordenados = sorted(self.pert_graph.nodes)
		aristas_sin_ordenar = self.pert_graph.edges
		nombres_sin_ordenar = [self.pert_graph.edges[edge]['activity'] for edge in self.pert_graph.edges]
		aristas_ordenadas = [x for _, x in sorted(zip(nombres_sin_ordenar, aristas_sin_ordenar))]
		nombres_ordenados = [self.pert_graph.edges[edge]['activity'] for edge in aristas_ordenadas]
		H = pd.DataFrame(nx.incidence_matrix(self.pert_graph, oriented=True, nodelist=nodos_ordenados,
											 edgelist=aristas_ordenadas).toarray().T,
						 index=nombres_ordenados, columns=nodos_ordenados).astype(int)
		return H
	
	def nullspace(self):
		nodos_ordenados = sorted(self.pert_graph.nodes)
		H = self.incidence_matrix()
		nullspace = H.drop(columns=[nodos_ordenados[0], nodos_ordenados[-1]])
		return nullspace
	
class EarnedValue():
	def __init__(self, pert):
		self.pert = pert
	
	def calcula_gantts(self,
					   data,
					   planned_durations_label,
					   actual_durations_label,
					   PV_label,
					   AC_label,
					   percentage_complete_label):
		my_data = data.copy()
		
		if any(my_data.loc[:, percentage_complete_label] > 1):
			my_data = my_data.astype({percentage_complete_label: 'float'})
			my_data.loc[:, percentage_complete_label] = my_data.loc[:, percentage_complete_label] / 100
		
		my_data.loc[:, 'PV_per_period'] = my_data.loc[:, PV_label]  / my_data.loc[:, planned_durations_label]
		gantt_PV = self.pert.gantt(my_data,
								   planned_durations_label,
								   'PV_per_period',
								   total='ambas',
								   acumulado=True)
		
		my_data.loc[:, 'AC_per_period'] = (my_data.loc[:, AC_label] / my_data.loc[:, actual_durations_label])
		
		gantt_AC = self.pert.gantt(my_data,
								   actual_durations_label,
								   'AC_per_period',
								   total='ambas',
								   acumulado=True)
		my_data.loc[:, 'EV'] = my_data.loc[:, PV_label] * my_data.loc[:, percentage_complete_label]
		my_data.loc[:, 'EV_per_period'] = my_data.loc[:, 'EV'] / my_data.loc[:, actual_durations_label]

		gantt_EV = self.pert.gantt(my_data,
								   actual_durations_label,
								   'EV_per_period',
								   total='ambas',
								   acumulado=True)
		
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

		return {
			"text/latex": f"${latex_str}$"
		}, None



def SVD(rutas):
	U, S, VT = np.linalg.svd(rutas, full_matrices=True)

	# Create a diagonal matrix from S (singular values)
	S = np.diag(S)

	# Pad the diagonal matrix to match the original dimensions of rutas
	m, n = rutas.shape
	if m > n:
		# If m > n, pad with zeros to make S_full m x m
		S = np.pad(S, ((0, m - n), (0, 0)), mode='constant')
	elif m < n:
		# If m < n, pad with zeros to make S_full n x n
		S = np.pad(S, ((0, 0), (0, n - m)), mode='constant')
	return {'U': U, 'S': S, 'VT':VT}

def pretty(x, dec=0, latex=False, **kwds):
	if latex:
		pretty_function = to_ltx
	else:
		pretty_function = to_jup

	if dec == 0:
		try:
		  	result = pretty_function(x, fmt='{:d}', **kwds)
		except:
	  		result = pretty_function(x, fmt=f'{{:.{dec}f}}', **kwds)
	else:
		result = pretty_function(x, fmt=f'{{:.{dec}f}}', **kwds)

	return result	

def pretty_latex(x, dec=0, **kwds):
  return pretty(x, dec=dec, latex=True, **kwds)

def beautify(*args):
	converter = lambda x: pretty_latex(x) if isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame) else str(x)
	return Latex(' '.join([converter(arg) for arg in args]))

def tanto_por_uno(x):
	return np.abs(x) / np.abs(x).sum()

def ordena_rutas(rutas, importancia_actividades, importancia_rutas):
	def highlight_positive(val):
		color = 'background-color: yellow' if val > 0 else ''
		return color

	rutas = rutas * importancia_actividades
	sorted_indices = np.flip(np.argsort(importancia_actividades))
	rutas = rutas.iloc[:, sorted_indices ]

	rutas['Importancia'] = importancia_rutas
	rutas = rutas.sort_values('Importancia', ascending=False)

	rutas = rutas.style.map(lambda x: '' if pd.Series(x).name == 'Importancia' else highlight_positive(x), subset=rutas.columns.drop('Importancia'))

	return rutas


def highlight_blue_red(df_or_array): # chatgpt dixit
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

def make_gantt_tikz(gantt_data,
					extra_rows=None,
					extra_cols=None, 
					params=None,
				   ):
	if params is None:
		params = dict()	
    
	background_horizontal_line_color= params.get('background_horizontal_line_color', 	"white!80!blue")  
	background_vertical_bars_color  = params.get('background_vertical_bars_color',		"white!90!cyan")
	row_height                      = params.get('row_height',							0.8)
	activity_relative_height        = params.get('activity_relative_height', 			0.7)
	period_width                    = params.get('period_width', 						1)
	names_width                     = params.get('names_width', 						2)
	extra_cols_width                = params.get('extra_cols_width', 					2)
	number_of_periods			    = params.get('number_of_periods', (gantt_data['start'] + gantt_data['duration']).max())
	number_of_periods			    = int(number_of_periods)
	regular_background_color 		= params.get('regular_background_color', 			"white!80!green")
	critical_background_color 		= params.get('critical_background_color', 			"red")
	regular_text_color 				= params.get('regular_text_color', 					"black")
	critical_text_color 			= params.get('critical_text_color', 				"white")
	activity_inner_text_style 		= params.get('activity_inner_text_style', 			r"\bfseries\large")
	arrow_width 					= params.get('arrow_width', 						"1pt")
	row_totals 						= params.get('row_totals', 							False)
	nan_string 						= params.get('nan_string', 							'---')
	inner_text 						= params.get('inner_text', 							None)
	inner_text_digits				= params.get('inner_text_digits', 					0)
  
    
	activity_list = list(gantt_data.index)
	number_of_activities = len(activity_list)
	number_of_extra_rows = 0 if extra_rows is None else extra_rows.shape[0]
	number_of_extra_columns = 0 if extra_cols is None else extra_cols.shape[1]
	activity_box_height = number_of_activities * row_height
	activity_box_width  = number_of_periods * period_width
	
	text=""
	text+="\n" + r"\begin{tikzpicture}[>= latex, scale=1]"
	text+="\n" + r"% Cuadrícula"
	text+="\n" + r"%Barras verticales fondo"
	for xpos in range(1, number_of_periods, 2):
		text+="\n" +f"\\fill[color={background_vertical_bars_color}] ( {xpos * period_width}cm, {row_height}cm) rectangle ++({period_width}cm,-{row_height*(1 + number_of_activities)}cm);"
	
	text+="\n" + r"%Recuadro exterior"
	text+="\n" + fr"\draw (0,0) rectangle ++({activity_box_width}cm,  {row_height}cm);"
	text+="\n" + fr"\draw (0,0) rectangle ++({activity_box_width}cm, -{activity_box_height}cm);"

	text+="\n" + fr"%Columna de actividades"
	text+="\n" + fr"\draw ({-names_width },0) rectangle ++({names_width },{row_height} )   node[pos=0.5, anchor=center, transform shape] {{Actividad}};"
	for y,actividad in enumerate(activity_list):
		text += "\n" + fr"\draw ({-names_width },{-(y+1)*row_height}cm) rectangle ++({names_width },{row_height})   node[pos=0.5, anchor=center, transform shape] {{" + actividad +"};" 
		text+="\n" + fr"  \draw[{background_horizontal_line_color},very thin] (0, {-(y+1)*row_height}) -- ({activity_box_width}, {-(y+1)*row_height});"

	text+="\n" + r"%Columnas extra"
	for column_number in range(0,number_of_extra_columns):
		col_name = extra_cols.columns[column_number]
		text+="\n" + fr"\draw ({activity_box_width + column_number*extra_cols_width},0) rectangle ++({extra_cols_width },{row_height} )   node[pos=0.5, anchor=center, transform shape] {{ {col_name} }};"
		for y,activity in enumerate(activity_list):
			text += "\n" + fr"\draw ({activity_box_width + column_number*extra_cols_width },{-(y+1)*row_height}cm) rectangle ++({extra_cols_width},{row_height})   node[pos=0.5, anchor=center, transform shape] {{" + str(extra_cols.loc[activity, col_name]) +"};" 
				
	text+="\n" + r"%Fila de periodos"
	for x in range(1, number_of_periods+1):
		text+="\n" + fr"\draw ({ (x-0.5)*period_width},{0.5*row_height}) node[transform shape, minimum width={period_width}, minimum height={row_height}] { {x} }; "
	text+="\n" + r"% Fin de la cuadrícula principal"

	text+="\n" + r"%Filas extra"
	text+="\n" + r"%Barras verticales fondo"
	for xpos in range(1, number_of_periods, 2):
		text+="\n" +fr"\fill[color={background_vertical_bars_color}] ( {xpos * period_width}cm, {-activity_box_height}cm ) rectangle ++({period_width}cm, {-row_height*number_of_extra_rows}cm);"
	
	for row_number in range(0,number_of_extra_rows):
		row_name = extra_rows.index[row_number]
		text+="\n" + r"%Casilla del nombre de la fila"
		text+="\n" + fr"\draw ({-names_width},-{(number_of_activities + (row_number+1))*row_height}) rectangle ++({names_width},{row_height})   node[pos=0.5, anchor=center, transform shape] {{ {str(row_name)} }};"
	
		for x in range(number_of_periods):
			if x >= extra_rows.shape[1]:
				x_data= nan_string
			else:
				x_data = str(extra_rows.iloc[row_number, x])
			text += "\n" + fr"\draw ({ x*period_width },{-activity_box_height - (row_number + 1)*row_height }) rectangle ++({period_width},{row_height}) node[pos=0.5, transform shape, minimum width={period_width}cm, minimum height={row_height}cm, anchor=center]  {{" + x_data +"};" 

		if row_totals:
			x_data = str(extra_rows.iloc[row_number,].sum())
			text += "\n" + fr"\draw ({ activity_box_width },{-activity_box_height - (row_number + 1)*row_height }) rectangle ++({extra_cols_width},{row_height}) node[pos=0.5, transform shape, minimum width={period_width}cm, minimum height={row_height}cm, anchor=center]  {{" + x_data +"};" 
 
	
	text+="\n" + r"%Fin retícula adicional inferior"

	text+="%Dibujo de las actividades"
	for y,actividad in enumerate(activity_list):
		background_color = critical_background_color if gantt_data.loc[actividad, 'Htotal'] == 0 else regular_background_color
		text_color = critical_text_color if gantt_data.loc[actividad, 'Htotal'] == 0 else regular_text_color
		start=gantt_data.loc[actividad, 'start']
		duration= gantt_data.loc[actividad, 'duration']
		name=actividad        
		text+="\n" + fr"""\node[draw=black, fill={background_color}, inner sep=0pt, outer sep=0pt, anchor=south west, minimum width={duration*period_width}cm, minimum height={row_height*activity_relative_height}cm, font=""" + activity_inner_text_style + fr""", transform shape, text= {text_color} ] ( {name} ) at ( {start*period_width}, {(-y -1 + (1-activity_relative_height)/2)*row_height} ) {{  \strut }};"""

		if inner_text is not None:
			for idx,x in enumerate(range(start,start+duration)):
				texto = r"\strut" 
				if isinstance(inner_text, str):
					texto = truncate_and_pad(gantt_data.loc[actividad, inner_text], inner_text_digits)
				else:
					if duration == len(inner_text.loc[name, 'data']):
						texto = truncate_and_pad(inner_text.loc[name, 'data'][idx], inner_text_digits)
				text+=("\n" + fr"""\node[anchor=center, inner sep=0pt, outer sep=0pt, minimum width={period_width}cm, minimum height={row_height*activity_relative_height}cm, font=""" 
				+ activity_inner_text_style + fr""", transform shape, text= {text_color} ] 
				at ( {(x + 0.5)*period_width}, {(-y -0.5 )*row_height} ) {{ {texto} }};"""
				)

	text+="%Escritura de los números internos"    
	
	text+="\n" + r"\end{tikzpicture}"  
	return text
