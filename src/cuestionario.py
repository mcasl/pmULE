from functools import reduce
from itertools import chain, product
from math import ceil
from typing import Dict, Set, Tuple
from scipy.stats import norm

import networkx as nx
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import pygraphviz as pgv
from IPython.display import display, Image, SVG, Latex

from numpyarray_to_latex import to_ltx
from numpyarray_to_latex.jupyter import to_jup

	

class ProjectQuestionMaker():
	def __init__(self, project):
		self.project = project
		self.text = ""
		
	def duration(self, data, durations_label, question=None):
		if question is None:
			question = '¿Cuál es la duración del proyecto?'
		self.text += ('\n\n' + question + f'{{ ={self.project.duration(data[durations_label])} }}')
	
	def standard_deviation(self, data, durations_label, variances_label,  question=None, dec=2):
		result = self.project.standard_deviation(durations=data[durations_label],
		                                         variances=data[variances_label]).round(dec)
		if question is None:
			question = '¿Cuál es la desviación estándar de la duración del proyecto?'
		self.text += ('\n\n' + question + f'{{ ={result} }}')
		
	
	def distant_predecessor(self, question=None):
		if question is None:
			question = "¿Cuáles son sus los predecesores distantes? (Escribir los nombres de las actividades en orden alfabético separados por un espacio como, por ejemplo, A B C D E)"
		predecessor_dataframe = self.project.distant_predecessor(format='DataFrame').replace('----', np.nan).dropna().to_frame()
		for activity in predecessor_dataframe.index:
			self.text += f"\n\nPara la actividad {activity}, {question} {{ ={predecessor_dataframe.loc[activity, 'predecessors']} ={predecessor_dataframe.loc[activity, 'predecessors'].replace(', ','-')} ={predecessor_dataframe.loc[activity, 'predecessors'].replace(', ',' ')} }}"
	
	def path_matrix(self, question=None):
		if question is None:
			question = "¿De qué actividades consta? (Escribir los nombres de las actividades en orden alfabético separados por un espacio como, por ejemplo, A B C D E)"
		rutas = self.project.path_matrix(dummies=False)
		rutas = rutas @ rutas.columns
		for ruta in rutas.index:
			self.text += f"\n\nLa ruta {ruta}, {question} {{ ={' '.join(rutas[ruta])} ={', '.join(rutas[ruta])} ={'-'.join(rutas[ruta])} }}"
	
	def paths_duration(self, data, durations_label,  question=None, dec=2):
		if question is None:
			question = "¿Cuál es su duración?"
		rutas = self.project.path_matrix(dummies=False)
		duracion_rutas = rutas @ data[durations_label]
		for ruta in duracion_rutas.index:
			self.text += f"\n\nCon respecto a la ruta {ruta}: {question} {{ ={duracion_rutas[ruta].round(dec)} }}"
	
	def date_for_probability(self, data, duration_label, variance_label, probability, question=None, dec=0):
		duracion_proyecto = self.project.duration(data[duration_label])
		desviacion_tipica = self.project.standard_deviation(durations=data[duration_label],
		                                                    variances=data[variance_label]).round(dec)
		if question is None:
			question = f"\n\n¿Cuál es la fecha para una probabilidad del {probability}% de terminar el proyecto??"
		result = round(norm.ppf(probability, duracion_proyecto, desviacion_tipica))
		self.text += f"\n\n {question} {{ ={result} }}"
		
	def probability_for_interval(self, data, duration_label, variance_label, lower_bound, upper_bound, question=None, dec=0):
		duracion_proyecto = self.project.duration(data[duration_label])
		desviacion_tipica = self.project.standard_deviation(durations=data[duration_label],
		                                                    variances=data[variance_label])
		if question is None:
			question = f"\n\n¿Cuál es la probabilidad, en tanto por ciento con {dec} decimales, de terminar el proyecto entre {lower_bound} y {upper_bound}?"
		probabilidad = norm.cdf(upper_bound, loc=duracion_proyecto, scale=desviacion_tipica) - norm.cdf(lower_bound, loc=duracion_proyecto, scale=desviacion_tipica)
		probabilidad = probabilidad * 100
		self.text += f"\n\n {question} {{ ={probabilidad.round(dec)} }}"
		
		
	def calendar(self, data, durations_label, question=None):
		earliest_start  = 'inicio_mas_temprano'
		latest_start    = 'inicio_mas_tardio'
		earliest_finish = 'fin_mas_temprano'
		latest_finish   = 'fin_mas_tardio'
		total_slack     = 'H_total'
		
		if question is None:
			question = ""
			
		result = self.project.calendar(duraciones=data[durations_label])
		
		for actividad in result.index:
			self.text += f"\n\nPara la actividad {actividad}, ¿Cuál es su tiempo de inicio más temprano? {question} {{ ={      result.loc[actividad, earliest_start]} }}"
			self.text += f"\n\nPara la actividad {actividad}, ¿Cuál es su tiempo de inicio más tardío? {question} {{ ={        result.loc[actividad, latest_start]} }}"
			self.text += f"\n\nPara la actividad {actividad}, ¿Cuál es su tiempo de finalización más temprano? {question} {{ ={result.loc[actividad, earliest_finish]} }}"
			self.text += f"\n\nPara la actividad {actividad}, ¿Cuál es su tiempo de finalización más tardío? {question} {{ ={  result.loc[actividad, latest_finish]} }}"
			self.text += f"\n\nPara la actividad {actividad}, ¿Cuál es su holgura total? {question} {{ ={                      result.loc[actividad, total_slack]} }}"
			
		
	def write_test(self):
		return self.text
	
		
		