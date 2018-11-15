from math import ceil
import networkx as nx
import numpy as np
import pandas as pd
import pygraphviz as pgv
from IPython.display import Image
from decimal import Decimal

class GrafoProyecto:
    def __init__(self, aristas):
        self.graph = nx.DiGraph()
        for activity in aristas.to_records(index=True):
            edge = (activity['nodo_inicial'], activity['nodo_final'])
            self.graph.add_edge(*edge)
            self.graph.edges[edge]['nombre'] = activity['actividad']

    @property
    def nodos(self):
        return list(nx.topological_sort(self.graph))

    @property
    def actividades(self):
        return [self.graph.edges[edge]['nombre'] for edge in self.graph.edges]

    def calcula_pert(self, duraciones):
        dtype=type(duraciones[0])
        tempranos  = pd.Series(0, index=self.nodos).apply(dtype)
        tardios    = pd.Series(0, index=self.nodos).apply(dtype)
        H_total    = pd.Series(0, index=self.actividades).apply(dtype)


        for nodo in self.nodos[1:]:
            tempranos[nodo] = max([(tempranos[inicial] + duraciones.get(attributes['nombre']))
                                    for (inicial, final, attributes) in  self.graph.in_edges(nodo, data=True)])


        tardios[self.nodos[-1]] =  tempranos[self.nodos[-1]]
        for nodo in self.nodos[-2::-1]:
            tardios[nodo] = min([tardios[final] - duraciones.get(attributes['nombre'])
                                 for (inicial, final, attributes) in self.graph.out_edges(nodo, data=True)])



        for (nodo_inicial, nodo_final) in self.graph.edges:
            activity_name = self.graph.edges[nodo_inicial, nodo_final]['nombre']
            H_total[activity_name] = tardios[nodo_final] - duraciones.get(activity_name) - tempranos[nodo_inicial]


        resultado = dict(tiempos = pd.DataFrame(dict(tempranos=tempranos, tardios=tardios)),
                         H_total = H_total)
        return resultado

    def duracion_proyecto(self, duraciones):
        resultados_pert = self.calcula_pert(duraciones)
        duracion = resultados_pert['tiempos']['tempranos'].values[-1]
        return duracion

    def camino_critico(self, duraciones):
        resultados_pert = self.calcula_pert(duraciones)
        H_total = resultados_pert['H_total']
        return H_total[H_total==0].index

    def zaderenko(self, duraciones):
        resultados_pert = self.calcula_pert(duraciones)['tiempos']
        lista_de_nodos_ordenada = self.nodos
        lista_de_nodos_ordenada.sort()
        z = pd.DataFrame(np.nan, index=lista_de_nodos_ordenada, columns=lista_de_nodos_ordenada)

        for edge in self.graph.edges:
            activity_name = self.graph.edges[edge]['nombre']
            z.loc[edge[0], edge[1]] = duraciones[activity_name]

        z['temprano'] = resultados_pert['tempranos']
        z = z.append(resultados_pert['tardios']).fillna('')
        return z

    def pert(self, filename, duraciones=None, size=None,
             orientation='landscape', rankdir='LR', ordering='out', ranksep=1, nodesep=1, rotate=0, **kwargs):
        if duraciones is not None:
            resultados_pert = self.calcula_pert(duraciones)
            tempranos = resultados_pert['tiempos']['tempranos']
            tardios   = resultados_pert['tiempos']['tardios']
            H_total   = resultados_pert['H_total']

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
        dot_graph.add_edges_from(self.graph.edges)

        for nodo in dot_graph.nodes():
            current_node = dot_graph.get_node(nodo)
            node_number = int(nodo)

            if duraciones is not None:
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

            current_edge_tuple_of_ints = (int(origin), int(destination))
            activity_name = self.graph.edges[current_edge_tuple_of_ints]['nombre']
            if duraciones is not None:
                current_edge.attr['label'] = (f"{activity_name}"
                                              f"({duraciones[activity_name]})")
                if H_total[activity_name] == 0:
                    current_edge.attr['color'] = 'red:red'
                    current_edge.attr['style'] = 'bold'

                if self.graph.edges[current_edge_tuple_of_ints]['nombre'][0] == 'f':
                    current_edge.attr['style'] = 'dashed'
            else:
                current_edge.attr['label'] = f"{activity_name}"




        self.dot_graph = dot_graph
        dot_graph.draw(filename, prog='dot')
        return Image(filename)


    def gantt(self, duraciones, representar=None, total=None, acumulado=False):
        duraciones = duraciones.reindex( self.actividades, fill_value=0)
        resultados_pert = self.calcula_pert(duraciones)
        tempranos = resultados_pert['tiempos']['tempranos']
        duracion_proyecto = tempranos.values[-1]
        periodos = range(1, ceil(duracion_proyecto) + 1)
        actividades_con_duracion = [nombre for nombre in self.actividades if duraciones.get(nombre, 0) != 0]
        actividades_con_duracion.sort()
        gantt = pd.DataFrame('', index=actividades_con_duracion, columns=periodos)

        for edge in self.graph.edges:
            activity_name = self.graph.edges[edge]['nombre']
            duracion_tarea = duraciones.get(activity_name, 0)
            if duracion_tarea != 0:
                comienzo_tarea = tempranos[edge[0]]
                valor = representar[activity_name] if representar is not None else ' '
                gantt.loc[activity_name, (comienzo_tarea + 1):(comienzo_tarea + duracion_tarea)] = valor

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
                mat = mat.append(fila_acumulado).fillna('')

        elif total == 'ambas':
            mat = summary( summary(gantt, axis=0), axis=1)
            if acumulado:
                fila_acumulado = mat.loc['Total'].drop('Total').cumsum()
                fila_acumulado.name = 'Acumulado'
                mat = mat.append(fila_acumulado).fillna('')

        resultado = (mat
                    .style
                    .set_table_styles(styles)
                    .applymap(color_gantt)
                    .apply(lambda x: ['background: #f7f7f9' if x.name in ["Total", "Acumulado"]
                                                           else '' for i in x], axis=0)
                    .apply(lambda x: ['background: #f7f7f9' if x.name in ["Total", "Acumulado"]
                                                           else '' for i in x], axis=1)

                    )

        return resultado

class ValorGanado():
    def __init__(self, pert):
        self.pert = pert

    def calcula_gantts(self,
                       duraciones_planificadas,
                       duraciones_reales,
                       costes_planificados,
                       costes_reales,
                       porcentaje_de_completacion):
        if any(porcentaje_de_completacion>1):
            porcentaje_de_completacion = porcentaje_de_completacion/100

        costes_planificados_por_periodo = costes_planificados/duraciones_planificadas
        gantt_PV = self.pert.gantt(duraciones_planificadas,
                                   representar=costes_planificados_por_periodo,
                                   total='ambas', acumulado=True)

        costes_reales_por_periodo = (costes_reales / duraciones_reales).reindex(costes_planificados.index, fill_value=0)
        duraciones_reales = duraciones_reales.reindex(duraciones_planificadas.index, fill_value=0)
        gantt_AC = self.pert.gantt(duraciones_reales,
                                   representar=costes_reales_por_periodo,
                                   total='ambas', acumulado=True)

        valor_ganado_total_tarea = (costes_planificados * porcentaje_de_completacion).reindex(costes_planificados.index, fill_value=0)
        valor_ganado_por_periodo = (valor_ganado_total_tarea / duraciones_reales).reindex(costes_planificados.index, fill_value=0)
        gantt_EV = self.pert.gantt(duraciones_reales,
                                   representar=valor_ganado_por_periodo,
                                   total='ambas', acumulado=True)

        acumulados = pd.DataFrame(dict(PV=gantt_PV.data.loc['Total',:].cumsum(),
                                       EV=gantt_EV.data.loc['Total',:].cumsum(),
                                       AC=gantt_AC.data.loc['Total',:].cumsum(), ),
                                  index = gantt_EV.data.columns).drop('Total')

        return dict(Gantt_PV=gantt_PV, Gantt_AC=gantt_AC, Gantt_EV=gantt_EV, acumulados=acumulados)
