import networkx as nx
import numpy as np
import pandas as pd
import pygraphviz as pgv


class PertGraph:
    def __init__(self, data):
        self.graph = nx.DiGraph()
        for activity in data.to_records(index=True):
            edge = (activity['nodo_inicial'], activity['nodo_final'])
            self.graph.add_edge(*edge)
            self.graph.edges[edge]['nombre'] = activity['index']
            self.graph.edges[edge]['duracion'] = activity['duracion']
            if 'recursos' in activity.dtype.names:
                self.graph.edges[edge]['recursos'] = activity['recursos']
            if 'CUR' in activity.dtype.names:
                self.graph.edges[edge]['CUR'] = activity['CUR']

        self.calcula_pert()

    def calcula_pert(self):
        lista_de_nodos = list(nx.topological_sort(self.graph))

        # [0] sirve para inicializar el tiempo temprano del nodo inicial correctamente
        # [self.graph.nodes[lista_de_nodos[-1]]['temprano']] sirve para
        # inicializar el tiempo tardío del nodo final correctamente con el valor del tiempo temprano

        for nodo in lista_de_nodos:

            self.graph.nodes[nodo]['temprano'] = max([0] +
                                                     [(self.graph.nodes[inicial]['temprano'] + attributes['duracion'])
                                                      for (inicial, final, attributes) in
                                                      self.graph.in_edges(nodo, data=True)])
        for nodo in lista_de_nodos[::-1]:
            self.graph.nodes[nodo]['tardio'] = min([self.graph.nodes[lista_de_nodos[-1]]['temprano']] +
                                                   [(self.graph.nodes[final]['tardio'] - attributes['duracion'])
                                                    for (inicial, final, attributes) in
                                                    self.graph.out_edges(nodo, data=True)])

        for (nodo_inicial, nodo_final) in self.graph.edges:
            self.graph.edges[nodo_inicial, nodo_final]['H_total'] = self.graph.nodes[nodo_final]['tardio'] - \
                                                                    self.graph.edges[nodo_inicial, nodo_final][
                                                                        'duracion'] - self.graph.nodes[nodo_inicial][
                                                                        'temprano']

    def duracion(self):
        return nx.dag_longest_path_length(self.graph, weight='duracion')

    def camino_critico(self):
        return {'Actividades': [(attributes['nombre']) for (nodo_inicial, nodo_final, attributes) in
                                self.graph.edges(data=True) if attributes['H_total'] == 0],
                'Nodos': nx.dag_longest_path(self.graph, weight='duracion')}

    def tiempos(self):
        return pd.DataFrame(dict(self.graph.nodes(data=True)))

    def holguras(self):
        holguras = pd.DataFrame([{'H_total': attributes['H_total'], 'nombre': attributes['nombre']}
                                 for (inicial, final, attributes) in self.graph.edges(data=True)]).set_index('nombre')
        return holguras

    def draw(self):
        nx.draw(self.graph, with_labels=True)

    def zaderenko(self):
        a = nx.to_pandas_adjacency(self.graph, weight='duracion', nonedge=np.nan)
        tiempos = self.tiempos()
        a['temprano'] = tiempos.loc['temprano', :]
        a = a.append(tiempos.loc['tardio', :]).fillna('')
        return a

    def write_dot(self, filename, size=None, orientation='landscape', rankdir='LR', ordering='out', ranksep=1,
                  nodesep=1, rotate=0, tiempos=True, **kwargs):
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
            if tiempos:
                current_node.attr['label'] = (f"{node_number} | {{ "
                                              f"<early> {self.graph.nodes[node_number]['temprano']} | "
                                              f"<last>  {self.graph.nodes[node_number]['tardio']} }}")
            else:
                current_node.attr['label'] = (f"{node_number} | {{ "
                                              f"<early>  | "
                                              f"<last>   }}")

        for origin, destination in dot_graph.edges_iter():
            current_edge = dot_graph.get_edge(origin, destination)
            current_edge_tuple_of_ints = (int(origin), int(destination))
            current_edge.attr['headport'] = 'early'
            current_edge.attr['tailport'] = 'last'
            if tiempos:
                current_edge.attr['label'] = (f"{self.graph.edges[current_edge_tuple_of_ints]['nombre']}"
                                              f"({self.graph.edges[current_edge_tuple_of_ints]['duracion']})")
            else:
                current_edge.attr['label'] = f"{self.graph.edges[current_edge_tuple_of_ints]['nombre']}"


            if self.graph.edges[current_edge_tuple_of_ints]['H_total'] == 0 and tiempos:
                current_edge.attr['color'] = 'red:red'
                current_edge.attr['style'] = 'bold'

            if self.graph.edges[current_edge_tuple_of_ints]['nombre'][0] == 'f':
                current_edge.attr['style'] = 'dashed'

        self.dot_graph = dot_graph
        dot_graph.draw(filename, prog='dot')


    def gantt_recursos(self, type='recursos'):
        key_a_representar = type

        actividades = [self.graph.edges[edge]['nombre'] for edge in self.graph.edges]
        actividades_sin_ficticias = [nombre for nombre in actividades if nombre[0] != 'f']
        periodos = range(1, self.duracion() + 1)
        gantt = pd.DataFrame('', index=actividades_sin_ficticias, columns=periodos)

        for edge in self.graph.edges:
            fila = self.graph.edges[edge]['nombre']
            if fila[0] != 'f':
                nodo_inicial = edge[0]
                comienzo_tarea = self.graph.nodes[nodo_inicial]['temprano']
                duracion = self.graph.edges[edge]['duracion']
                gantt.loc[fila, (comienzo_tarea + 1):(comienzo_tarea + duracion)] = self.graph.edges[edge][
                    key_a_representar]

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

        mat = (summary(gantt, axis=0)
               .style
               .set_table_styles(styles)
               .applymap(color_gantt)
               .apply(lambda x: ['background: #f7f7f9' if x.name == "Total" else '' for i in x], axis=1)
               )
        return mat


