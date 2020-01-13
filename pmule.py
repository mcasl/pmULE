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

def calcula_encadenamientos(precedentes):
    mat = pd.DataFrame('', index=precedentes.index, columns=precedentes.index)
    for fila, columnas in precedentes.items():
        columnas_preprocesadas = (columnas
                                     .replace('-','')
                                     .replace(' ','')
                                     .split(',')
                                  )
        columnas_dict = {key:True for key in columnas_preprocesadas if key}
        mat.loc[fila, :] = mat.columns.map(columnas_dict).fillna('')
    return mat

def make_Roy(encadenamientos):
    graph = nx.DiGraph()
    graph.add_nodes_from(['inicio', 'fin'])

    enlaces_iniciales = [('inicio', actividad) for actividad in encadenamientos.index
                         if not any(encadenamientos.loc[actividad, :])]

    graph.add_edges_from(enlaces_iniciales)

    enlaces_finales = [(actividad, 'fin') for actividad in encadenamientos.index
                       if not any(encadenamientos.loc[:, actividad])]

    graph.add_edges_from(enlaces_finales)

    resto_de_enlaces = [(b, a) for a, b in list(encadenamientos[encadenamientos == True].stack().index)]
    graph.add_edges_from(resto_de_enlaces)
    return graph


class GrafoProyecto:
    def __init__(self, data=None):
        if data is None:
            self.data = None
            self.pert_graph = nx.DiGraph()
            self.roy_graph = nx.DiGraph()
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
            self.pert_graph = nx.DiGraph()

            for numero, cadena in ids.items():
                self.pert_graph.add_node(cadena, id=numero)

            for activity in aristas.to_records(index=True):
                edge = (activity['nodo_inicial'], activity['nodo_final'])
                self.pert_graph.add_edge(*edge)
                self.pert_graph.edges[edge]['nombre'] = activity['actividad']

    def copy(self):
        grafo = GrafoProyecto()
        grafo.data = self.data.copy()
        grafo.pert_graph = self.pert_graph.copy()
        return grafo

    @property
    def nodos(self):
        return [self.pert_graph.node[nodo]['id'] for nodo in list(nx.topological_sort(self.pert_graph))]


    @property
    def actividades(self):
        return [self.pert_graph.edges[edge]['nombre'] for edge in self.pert_graph.edges]


    def calcula_pert(self, duraciones=None):
        if duraciones is None:
            duraciones = self.data['duracion']

        dtype = type(duraciones[0])
        nodos = self.nodos
        id_to_str = {self.pert_graph.nodes[nodo]['id']:nodo for nodo in self.pert_graph.nodes}
        str_to_id = {nodo:self.pert_graph.nodes[nodo]['id'] for nodo in self.pert_graph.nodes}

        tempranos  = pd.Series(0, index=nodos).apply(dtype)
        tardios    = pd.Series(0, index=nodos).apply(dtype)
        H_total    = pd.Series(0, index=self.actividades).apply(dtype)

        for nodo_id in nodos[1:]:
            tempranos[nodo_id] = max([(tempranos[str_to_id[inicial]] + duraciones.get(attributes['nombre']))
                                      for (inicial, final, attributes) in self.pert_graph.in_edges(id_to_str[nodo_id], data=True)])

        tardios[nodos[-1]] =  tempranos[nodos[-1]]
        for nodo_id in nodos[-2::-1]:
            tardios[nodo_id] = min([tardios[str_to_id[final]] - duraciones.get(attributes['nombre'])
                                    for (inicial, final, attributes) in self.pert_graph.out_edges(id_to_str[nodo_id], data=True)])

        for (nodo_inicial, nodo_final) in self.pert_graph.edges:
            activity_name = self.pert_graph.edges[nodo_inicial, nodo_final]['nombre']
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
        str_to_id = {nodo:self.pert_graph.nodes[nodo]['id'] for nodo in self.pert_graph.nodes}

        for (nodo_inicial, nodo_final) in self.pert_graph.edges:
            activity_name = self.pert_graph.edges[nodo_inicial, nodo_final]['nombre']
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
        str_to_id = {nodo:self.pert_graph.nodes[nodo]['id'] for nodo in self.pert_graph.nodes}

        for edge in self.pert_graph.edges:
            id_inicial = str_to_id[edge[0]]
            id_final   = str_to_id[edge[1]]
            activity_name = self.pert_graph.edges[edge]['nombre']
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
        dot_graph.add_edges_from(self.pert_graph.edges)
        str_to_id = {nodo: self.pert_graph.nodes[nodo]['id'] for nodo in self.pert_graph.nodes}
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

            activity_name = self.pert_graph.edges[origin, destination]['nombre']
            if duraciones is not False:
                current_edge.attr['label'] = (f"{activity_name}"
                                              f"({duraciones[activity_name]})")
                if H_total[activity_name] == 0:
                    current_edge.attr['color'] = 'red:red'
                    current_edge.attr['style'] = 'bold'

                if self.pert_graph.edges[origin, destination]['nombre'][0] == 'f':
                    current_edge.attr['style'] = 'dashed'
            else:
                current_edge.attr['label'] = f"{activity_name}"


        self.dot_graph = dot_graph
        dot_graph.draw(filename, prog='dot')
        return Image(filename)


    def roy(self,
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

        precedentes = (self.data.precedentes
                       .drop([actividad for actividad in self.data.index if actividad[0] == 'f'])
                      )
        encadenamientos = calcula_encadenamientos(precedentes)
        self.roy_graph = make_Roy(encadenamientos)

        if filename is None:
            filename = 'output_roy_figure.png'

        if duraciones is None:
            duraciones = self.data['duracion'].copy()

        if duraciones is not False:
            calendario = self.calendario()
            inicio_mas_temprano = calendario['inicio_mas_temprano']
            inicio_mas_tardio = calendario['inicio_mas_tardio']
            duraciones['inicio'] = 0
            duraciones['fin'] = 0
            inicio_mas_temprano['inicio'] = 0
            inicio_mas_tardio['inicio'] = 0

            inicio_mas_temprano['fin'] = self.duracion_proyecto()
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
        str_to_id = {nodo: self.pert_graph.nodes[nodo]['id'] for nodo in self.pert_graph.nodes}

        for edge in self.pert_graph.edges:
            activity_name = self.pert_graph.edges[edge]['nombre']
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

    def gantt_cargas(self, duraciones=None, report=True):
        gantt = self.gantt(representar='recursos', total='fila', acumulado=False, holguras=True, cuadrados=True)
        gantt.data.loc['Total', 'H_total'] = ''
        suma_cuadrados = gantt.data.loc['Cuadrados', :].sum()
        gantt.data.loc['Cuadrados', 'H_total'] = suma_cuadrados
        if report:
            print('Suma de cuadrados:', suma_cuadrados, '\n' )
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
                self.data = self.data.append(nueva_fila)

        lista_edges = list(self.pert_graph.edges)
        for edge in lista_edges:
            activity_name = self.pert_graph.edges[edge]['nombre']
            slide_name = 'slide_' + activity_name

            if (activity_name in desplazamientos
                and slide_name not in self.actividades):
                self.pert_graph.remove_edge(edge[0], edge[1])
                tamano_cadena = 1
                nodo_auxiliar_str = activity_name + '___' + genera_random_str(tamano_cadena)
                while nodo_auxiliar_str in self.pert_graph.nodes():
                    nodo_auxiliar_str = activity_name + '___' + genera_random_str(tamano_cadena)
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
        suma_cuadrados = proyecto.gantt_cargas(report=False).data.loc['Cuadrados', 'H_total']  # No una holgura, pero está ahí el dato
        return suma_cuadrados

    def evaluar_rango_de_desplazamientos(self, actividad, report=True):
        minimo = 0
        maximo = int(self.calcula_pert()['actividades'].loc[actividad, 'H_total'])
        suma_cuadrados = pd.DataFrame(0, index=range(minimo, maximo + 1), columns=['Suma_de_cuadrados'])
        if report:
            print('Sin desplazar:')
        suma_cuadrados.loc[0, 'Suma_de_cuadrados'] = self.gantt_cargas(report=report).data.loc['Cuadrados', 'H_total']
        for slide in range(minimo + 1, maximo + 1):
            if report:
                print('Desplazamiento:', slide)
            carga2 = self.evaluar_desplazamiento(**{actividad:slide}, report=report)
            suma_cuadrados.loc[slide, 'Suma_de_cuadrados'] = carga2
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
        print('Varianza del proyecto:', desviacion ** 2)
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
