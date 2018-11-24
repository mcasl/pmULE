from math import ceil
import networkx as nx
import numpy as np
import pandas as pd
import string
from random import choice
import pygraphviz as pgv
from IPython.display import Image, display
from decimal import Decimal


def genera_random_str(tamano):
    letters_and_digits = string.ascii_letters +  string.digits
    password = "".join(choice(letters_and_digits) for x in range(tamano))
    return password


class GrafoProyecto:
    def __init__(self, data=None):
        if data is None:
            self.data = None
            self.graph = nx.DiGraph()
        else:
            self.data = data.copy()
            tamano_cadena = 2
            aristas = self.data.loc[:, ['nodo_inicial', 'nodo_final']]
            ids = {key: str(key) + '___' + genera_random_str(tamano_cadena)
                   for key in set(aristas.loc[:, ['nodo_inicial', 'nodo_final']].values.flatten())}

            while len(ids) != len(set(ids.values())):
                tamano_cadena += 1
                ids = {key: str(key) + '___' + genera_random_str(tamano_cadena)
                       for key in set(aristas.loc[:, ['nodo_inicial', 'nodo_final']].values.flatten())}

            aristas.nodo_inicial = aristas.nodo_inicial.map(ids)
            aristas.nodo_final = aristas.nodo_final.map(ids)
            self.graph = nx.DiGraph()

            for numero, cadena in ids.items():
                self.graph.add_node(cadena, id=numero)

            for activity in aristas.to_records(index=True):
                edge = (activity['nodo_inicial'], activity['nodo_final'])
                self.graph.add_edge(*edge)
                self.graph.edges[edge]['nombre'] = activity['actividad']

    def copy(self):
        grafo = GrafoProyecto(data=self.data)
        return grafo

    @property
    def nodos(self):
        return [self.graph.node[nodo]['id'] for nodo in list(nx.topological_sort(self.graph))]


    @property
    def actividades(self):
        return [self.graph.edges[edge]['nombre'] for edge in self.graph.edges]

    def calcula_pert(self, duraciones=None):
        if duraciones is None:
            duraciones = self.data['duracion']

        dtype = type(duraciones[0])
        nodos = self.nodos
        id_to_str = {self.graph.nodes[nodo]['id']:nodo for nodo in self.graph.nodes}
        str_to_id = {nodo:self.graph.nodes[nodo]['id'] for nodo in self.graph.nodes}

        tempranos  = pd.Series(0, index=nodos).apply(dtype)
        tardios    = pd.Series(0, index=nodos).apply(dtype)
        H_total    = pd.Series(0, index=self.actividades).apply(dtype)

        for nodo_id in nodos[1:]:
            tempranos[nodo_id] = max([(tempranos[str_to_id[inicial]] + duraciones.get(attributes['nombre']))
                                    for (inicial, final, attributes) in  self.graph.in_edges(id_to_str[nodo_id], data=True)])

        tardios[nodos[-1]] =  tempranos[nodos[-1]]
        for nodo_id in nodos[-2::-1]:
            tardios[nodo_id] = min([tardios[str_to_id[final]] - duraciones.get(attributes['nombre'])
                                    for (inicial, final, attributes) in self.graph.out_edges(id_to_str[nodo_id], data=True)])

        for (nodo_inicial, nodo_final) in self.graph.edges:
            activity_name = self.graph.edges[nodo_inicial, nodo_final]['nombre']
            H_total[activity_name] = tardios[str_to_id[nodo_final]] - duraciones.get(activity_name) - tempranos[str_to_id[nodo_inicial]]

        resultado = dict(nodos = pd.DataFrame(dict(tempranos=tempranos, tardios=tardios)),
                         actividades = pd.DataFrame(dict(H_total=H_total)) )
        return resultado

    def calendario(self, duraciones=None):
        if duraciones is None:
            duraciones = self.data['duracion']

        calendario = pd.DataFrame(0, index=duraciones.index, columns=['inicio_mas_temprano',
                                                                      'inicio_mas_tardio',
                                                                      'fin_mas_temprano',
                                                                      'fin_mas_tardio'])

        resultados_pert= self.calcula_pert(duraciones)
        calendario['H_total']  = resultados_pert['actividades']['H_total']
        calendario['duracion'] = duraciones

        tempranos = resultados_pert['nodos']['tempranos']
        tardios = resultados_pert['nodos']['tardios']
        str_to_id = {nodo:self.graph.nodes[nodo]['id'] for nodo in self.graph.nodes}

        for (nodo_inicial, nodo_final) in self.graph.edges:
            activity_name = self.graph.edges[nodo_inicial, nodo_final]['nombre']
            calendario.loc[activity_name, 'inicio_mas_temprano'] = tempranos[str_to_id[nodo_inicial]]
            calendario.loc[activity_name, 'inicio_mas_tardio'] = tardios[str_to_id[nodo_final]] - duraciones.get(activity_name)
            calendario.loc[activity_name, 'fin_mas_temprano'] = tempranos[str_to_id[nodo_inicial]] + duraciones.get(activity_name)
            calendario.loc[activity_name, 'fin_mas_tardio'] = tardios[str_to_id[nodo_final]]

        return calendario



    def duracion_proyecto(self, duraciones=None):
        if duraciones is None:
            duraciones = self.data['duracion']

        resultados_pert = self.calcula_pert(duraciones)
        duraciones = resultados_pert['nodos']['tempranos'].values[-1]
        return duraciones

    def camino_critico(self, duraciones=None):
        if duraciones is None:
            duraciones = self.data['duracion']

        resultados_pert = self.calcula_pert(duraciones)
        H_total = resultados_pert['actividades']['H_total']
        return H_total[H_total==0].index

    def resolver_zaderenko(self, duraciones=None):
        if duraciones is None:
            duraciones = self.data['duracion']

        resultados_pert = self.calcula_pert(duraciones)['nodos']
        lista_de_nodos_ordenada = self.nodos
        lista_de_nodos_ordenada.sort()
        z = pd.DataFrame(np.nan, index=lista_de_nodos_ordenada, columns=lista_de_nodos_ordenada)
        str_to_id = {nodo:self.graph.nodes[nodo]['id'] for nodo in self.graph.nodes}

        for edge in self.graph.edges:
            id_inicial = str_to_id[edge[0]]
            id_final   = str_to_id[edge[1]]
            activity_name = self.graph.edges[edge]['nombre']
            z.loc[id_inicial, id_final] = duraciones[activity_name]

        z['temprano'] = resultados_pert['tempranos']
        z = z.append(resultados_pert['tardios']).fillna('')
        return z

    def pert(self,
             filename=None,
             duraciones=None,
             size=None,
             orientation='landscape',
             rankdir='LR',
             ordering='out',
             ranksep=0.5,
             nodesep=0.5,
             rotate=0,
             **kwargs):

        if filename is None:
            filename = 'output_pert_figure.png'

        if duraciones is None:
            duraciones = self.data['duracion']

        if duraciones is not False:
            resultados_pert = self.calcula_pert(duraciones)
            tempranos = resultados_pert['nodos']['tempranos']
            tardios   = resultados_pert['nodos']['tardios']
            H_total   = resultados_pert['actividades']['H_total']

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
        str_to_id = {nodo: self.graph.nodes[nodo]['id'] for nodo in self.graph.nodes}
        for nodo in dot_graph.nodes():
            current_node = dot_graph.get_node(nodo)
            node_number = int(str_to_id[nodo])

            if duraciones is not False:
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

            activity_name = self.graph.edges[origin, destination]['nombre']
            if duraciones is not False:
                current_edge.attr['label'] = (f"{activity_name}"
                                              f"({duraciones[activity_name]})")
                if H_total[activity_name] == 0:
                    current_edge.attr['color'] = 'red:red'
                    current_edge.attr['style'] = 'bold'

                if self.graph.edges[origin, destination]['nombre'][0] == 'f':
                    current_edge.attr['style'] = 'dashed'
            else:
                current_edge.attr['label'] = f"{activity_name}"


        self.dot_graph = dot_graph
        dot_graph.draw(filename, prog='dot')
        return Image(filename)


    def gantt(self, duraciones=None,
              representar=None,
              total=None,
              acumulado=False,
              holguras=False,
              cuadrados = False):

        if duraciones is None:
            duraciones = self.data['duracion']

        duraciones = duraciones.reindex(self.actividades, fill_value=0)
        if representar is None:
            representar = pd.Series({actividad: '  ' for actividad in duraciones.index})
        elif isinstance(representar, str):
            if representar == 'nombres' or representar == 'actividad':
                representar = pd.Series({actividad:actividad for actividad in duraciones.index})
            elif representar == 'vacio':
                representar = pd.Series({actividad:'  ' for actividad in duraciones.index})
            else:
                representar = pd.Series({actividad:self.data.loc[actividad, representar]
                                     for actividad in duraciones.index})
        else:
            representar = representar.reindex(self.actividades, fill_value=0)

        resultados_pert = self.calcula_pert(duraciones)
        tempranos = resultados_pert['nodos']['tempranos']
        duracion_proyecto = tempranos.values[-1]
        periodos = range(1, ceil(duracion_proyecto) + 1)
        actividades_con_duracion = [nombre for nombre in self.actividades if duraciones.get(nombre, 0) != 0]
        actividades_con_duracion.sort()
        gantt = pd.DataFrame('', index=actividades_con_duracion, columns=periodos)
        str_to_id = {nodo: self.graph.nodes[nodo]['id'] for nodo in self.graph.nodes}

        for edge in self.graph.edges:
            activity_name = self.graph.edges[edge]['nombre']
            duracion_tarea = duraciones.get(activity_name, 0)
            if duracion_tarea != 0:
                comienzo_tarea = tempranos[str_to_id[edge[0]]]
                gantt.loc[activity_name, (comienzo_tarea + 1):(comienzo_tarea + duracion_tarea)] = representar[activity_name]

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
            if cuadrados:
                fila_cuadrados =  mat.loc['Total'] ** 2
                fila_cuadrados.name = 'Cuadrados'
                mat = mat.append(fila_cuadrados).fillna('')
        elif total == 'ambas':
            mat = summary( summary(gantt, axis=0), axis=1)
            if acumulado:
                fila_acumulado = mat.loc['Total'].drop('Total').cumsum()
                fila_acumulado.name = 'Acumulado'
                mat = mat.append(fila_acumulado).fillna('')

        if holguras:
            mat['H_total'] = resultados_pert['actividades']['H_total']
            if total == 'fila' or total =='ambas':
                mat.loc['Total', 'H_total'] = None

        resultado = (mat
                    .style
                    .set_table_styles(styles)
                    .applymap(color_gantt)
                    .apply(lambda x: ['background: #f7f7f9' if x.name in ["Total", "Acumulado", "H_total", "Cuadrados"]
                                                           else '' for i in x], axis=0)
                    .apply(lambda x: ['background: #f7f7f9' if x.name in ["Total", "Acumulado", "H_total", "Cuadrados"]
                                                           else '' for i in x], axis=1)

                    )

        return resultado

    def gantt_cargas(self, duraciones=None):
        gantt = self.gantt(representar='recursos', total='fila', acumulado=False, holguras=True, cuadrados=True)
        gantt.data.loc['Total', 'H_total'] = ''
        gantt.data.loc['Cuadrados', 'H_total'] = gantt.data.loc['Cuadrados', :].sum()
        return gantt


    def desplazar(self, mostrar='cargas', **desplazamientos):

        for actividad, duracion in desplazamientos.items():
            nombre_slide = 'slide_' + actividad
            if nombre_slide in self.data.index:
                self.data.loc[nombre_slide, 'duracion'] += duracion
            else:
                nueva_fila = pd.Series({'duracion': duracion},
                                       name=nombre_slide,
                                       index=self.data.columns).fillna(0)
                self.data = self.data.append(nueva_fila)

        lista_edges = list(self.graph.edges)
        for edge in lista_edges:
            activity_name = self.graph.edges[edge]['nombre']
            slide_name = 'slide_' + activity_name

            if (activity_name in desplazamientos
                and slide_name not in self.actividades):
                self.graph.remove_edge(edge[0], edge[1])
                tamano_cadena = 1
                nodo_auxiliar_str = activity_name + '___' + genera_random_str(tamano_cadena)
                while nodo_auxiliar_str in self.graph.nodes():
                    nodo_auxiliar_str = activity_name + '___' + genera_random_str(tamano_cadena)
                self.graph.add_node(nodo_auxiliar_str, id=nodo_auxiliar_str)
                self.graph.add_edge(edge[0], nodo_auxiliar_str, nombre='slide_' + activity_name)
                self.graph.add_edge(nodo_auxiliar_str, edge[1], nombre=activity_name)

        lista_nodos = nx.topological_sort(self.graph)
        nx.set_node_attributes(self.graph, {nodo: {'id': (id + 1)}
                                                for id, nodo in enumerate(lista_nodos)})

        if isinstance(mostrar, str) and mostrar == 'cargas':
            representacion = self.gantt_cargas()
        else:
            representacion = self.gantt(representar=self.data['recursos'],
                                        total='fila',
                                        holguras=True)

        return representacion

    def evaluar_desplazamiento(self, **desplazamientos):
        proyecto = self.copy()
        return proyecto.desplazar(**desplazamientos)

    def evaluar_rango_de_desplazamientos(self, actividad, minimo=1, maximo=None):
        if maximo is None:
            resultados_pert = self.calcula_pert()
            maximo = int(resultados_pert['actividades'].loc[actividad, 'H_total'])

        suma_cuadrados = pd.DataFrame(0, index=range(minimo, maximo + 1), columns=['Suma_de_cuadrados'])
        for slide in range(minimo, maximo+1):
            print('Desplazamiento:', slide)
            gantt = self.evaluar_desplazamiento(**{actividad:slide})
            suma_cuadrados.loc[slide, 'Suma_de_cuadrados'] = gantt.data.loc['Cuadrados', 'H_total']  # Aunque sea la columna H_total el valor es la suma de los cuadrados, no unaholgura
            display(gantt)

        return suma_cuadrados

    def cur_ordenado(self, rama):
        return self.data.loc[rama, ['cur', 'duracion', 'duracion_tope']].sort_values(by='cur')

    def reducir(self, **kwargs):
        sobrecoste = 0
        for actividad, decremento in kwargs.items():
            self.data.loc[actividad, 'duracion'] -= decremento
            sobrecoste += self.data.loc[actividad, 'cur'] * decremento
        print('Sobrecoste de la reducción:', sobrecoste)

    def desviacion_proyecto(self, *ramas):
        if 'varianza' not in self.data.columns:
            varianza = self.data['desviacion'] ** 2
        else:
            varianza = self.data['varianza']

        desviacion_ramas = {'-'.join(rama):varianza[rama].sum() ** 0.5 for rama in ramas}
        [print('Desviación rama:', key, ':', value) for key, value in desviacion_ramas.items() ]
        desviacion = max(desviacion_ramas.values())
        print('Desviación del proyecto:', desviacion)
        return desviacion


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
