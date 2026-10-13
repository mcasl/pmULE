# Ejercicios resueltos

[Ejercicio 01](Ejercicio_01.html)  
[Ejercicio 02](Ejercicio_02.html)  
[Ejercicio 03](Ejercicio_03.html)  
[Ejercicio 04](Ejercicio_04.html)  
[Ejercicio 05](Ejercicio_05.html)  
[Ejercicio 06](Ejercicio_06.html)  
[Ejercicio 07](Ejercicio_07.html)  
[Ejercicio 09](Ejercicio_09.html)  
[Ejercicio 10](Ejercicio_10.html)  
[Ejercicio 11](Ejercicio_11.html)  
[Ejercicio 12](Ejercicio_12.html)  
[Ejercicio 13](Ejercicio_13.html)  
[Ejercicio 14](Ejercicio_14.html)  
[Ejercicio 15](Ejercicio_15.html)  
[Ejercicio 16](Ejercicio_16.html)  
[Ejercicio 17](Ejercicio_17.html)  
[Ejercicio 20](Ejercicio_20.html)  
[Ejercicio 21](Ejercicio_21.html)  
[Ejercicio 22](Ejercicio_22.html)  
[Ejercicio 23](Ejercicio_23.html)  
[Ejercicio 24](Ejercicio_24.html)  
[Ejercicio 25](Ejercicio_25.html)  
[Ejercicio 30](Ejercicio_30.html)  
[Ejercicio 31](Ejercicio_31.html)  

<!-- colecciones:inicio -->

# Colecciones de ejercicios cortos por competencia

Ejercicios breves, de una sola competencia, para practicar cada paso por separado. Los cuadernos están en `notebooks/colecciones/`.

## Cuadro de prelaciones

Depurar un cuadro de prelaciones (redundancias, erratas que crean ciclos), construir el cuadro expandido y el de ancestros.  

[Prelaciones 01](colecciones/01_prelaciones/Prelaciones_01.html) — 10 actividades, 2 prelaciones redundantes  
[Prelaciones 02](colecciones/01_prelaciones/Prelaciones_02.html) — 10 actividades, 2 prelaciones redundantes  
[Prelaciones 03](colecciones/01_prelaciones/Prelaciones_03.html) — 10 actividades, 2 prelaciones redundantes, una errata con ciclo  

## Grafo PERT y actividades ficticias

Dibujar el grafo PERT con el mínimo número de ficticias, justificar cada una, numerar los nodos y enumerar las rutas.  

[PERT 01](colecciones/02_pert/PERT_01.html) — 8 actividades, 2 ficticias, 6 rutas  
[PERT 02](colecciones/02_pert/PERT_02.html) — 9 actividades, 3 ficticias, 7 rutas  
[PERT 03](colecciones/02_pert/PERT_03.html) — 10 actividades, 3 ficticias, 5 rutas  

## Grafo Roy

Dibujar el grafo Roy, compararlo con el PERT y modelar esperas y solapes entre actividades.  

[Roy 01](colecciones/03_roy/Roy_01.html) — 8 actividades, 2 ficticias en PERT  
[Roy 02](colecciones/03_roy/Roy_02.html) — 8 actividades, 2 ficticias en PERT  
[Roy 03](colecciones/03_roy/Roy_03.html) — 8 actividades, 3 ficticias en PERT, con espera y solape  

## Método de Zaderenko y calendario

Calcular con la matriz de Zaderenko los tiempos tempranos y tardíos, el calendario de las actividades y el diagrama de Gantt.  

[Zaderenko 01](colecciones/04_zaderenko/Zaderenko_01.html) — 8 actividades, 8 nodos, 2 ficticias  
[Zaderenko 02](colecciones/04_zaderenko/Zaderenko_02.html) — 9 actividades, 9 nodos, 1 ficticias  
[Zaderenko 03](colecciones/04_zaderenko/Zaderenko_03.html) — 10 actividades, 9 nodos, 1 ficticias  

## Holguras y camino crítico

Calcular e interpretar las holguras total, libre e independiente; identificar todos los caminos críticos.  

[Holguras 01](colecciones/05_holguras/Holguras_01.html) — 9 actividades, 1 ruta(s) crítica(s)  
[Holguras 02](colecciones/05_holguras/Holguras_02.html) — 10 actividades, 1 ruta(s) crítica(s)  
[Holguras 03](colecciones/05_holguras/Holguras_03.html) — 10 actividades, 2 ruta(s) crítica(s)  

## Matriz de caminos

Enumerar las rutas, construir la matriz de caminos y usarla para calcular la duración del proyecto; reconstruir el grafo a partir de las rutas; rutas dominantes mediante la SVD.  

[Caminos 01](colecciones/06_matriz_caminos/Caminos_01.html) — 8 actividades, 6 rutas  
[Caminos 02](colecciones/06_matriz_caminos/Caminos_02.html) — 8 actividades, 6 rutas  
[Caminos 03](colecciones/06_matriz_caminos/Caminos_03.html) — 9 actividades, 8 rutas, reconstrucción desde las rutas y SVD  

## Estimaciones de duración (a, m, b)

Obtener la duración media y la varianza de cada actividad a partir de sus tres estimaciones, resolver el problema inverso y comparar distribuciones.  

[Estimaciones 01](colecciones/07_estimaciones/Estimaciones_01.html) — 7 actividades, 0 con estimaciones asimétricas  
[Estimaciones 02](colecciones/07_estimaciones/Estimaciones_02.html) — 7 actividades, 2 con estimaciones asimétricas  
[Estimaciones 03](colecciones/07_estimaciones/Estimaciones_03.html) — 8 actividades, 2 con estimaciones asimétricas  

## Probabilidad de cumplir el plazo (aproximación normal)

Varianza del proyecto por el camino crítico, probabilidad de terminar antes de un plazo o entre dos fechas y duración comprometible para una probabilidad dada.  

[Probabilidad 01](colecciones/08_probabilidad/Probabilidad_01.html) — 9 actividades, 1 ruta(s) crítica(s), σ = 1.18  
[Probabilidad 02](colecciones/08_probabilidad/Probabilidad_02.html) — 9 actividades, 2 ruta(s) crítica(s), σ = 1.25  
[Probabilidad 03](colecciones/08_probabilidad/Probabilidad_03.html) — 10 actividades, 2 ruta(s) crítica(s), σ = 1.57  

## Método de Monte Carlo

Simular la duración del proyecto con la matriz de caminos, compararla con la aproximación normal y calcular índices de criticidad y sensibilidad.  

[MonteCarlo 01](colecciones/09_montecarlo/MonteCarlo_01.html) — 8 actividades, 5 rutas, distribución normal  
[MonteCarlo 02](colecciones/09_montecarlo/MonteCarlo_02.html) — 9 actividades, 6 rutas, distribución triangular  
[MonteCarlo 03](colecciones/09_montecarlo/MonteCarlo_03.html) — 9 actividades, 6 rutas, distribución Beta-PERT  

## Reducción de la duración con mínimo coste

Coste unitario de reducción, algoritmo de Ackoff-Sasieni paso a paso, costes indirectos y duración de coste mínimo, reducción frente a penalización.  

[Reduccion 01](colecciones/10_reduccion/Reduccion_01.html) — 8 actividades, reducción de 3 semanas  
[Reduccion 02](colecciones/10_reduccion/Reduccion_02.html) — 8 actividades, costes indirectos de 9 k€/semana  
[Reduccion 03](colecciones/10_reduccion/Reduccion_03.html) — 9 actividades, plazo 27 con penalización  

## Nivelación y asignación de recursos

Diagrama de cargas, nivelación por mínimos cuadrados, asignación con límite de recursos (Wiest-Levy), evaluación de desplazamientos y varios tipos de recurso.  

[Recursos 01](colecciones/11_recursos/Recursos_01.html) — 9 actividades, nivelación  
[Recursos 02](colecciones/11_recursos/Recursos_02.html) — 8 actividades, asignación con límite 6→8  
[Recursos 03](colecciones/11_recursos/Recursos_03.html) — 8 actividades, dos recursos y propuesta de desplazamiento  

## Método del valor ganado

Diagramas de Gantt del valor planificado, ganado y coste real; CPI, SPI, EAC, VAC, TCPI; programación ganada; evolución de los indicadores en varios puntos de control.  

[ValorGanado 01](colecciones/12_valor_ganado/ValorGanado_01.html) — 9 actividades, un punto de control (semana 10), EAC/VAC/TCPI  
[ValorGanado 02](colecciones/12_valor_ganado/ValorGanado_02.html) — 8 actividades, un punto de control (semana 6), programación ganada  
[ValorGanado 03](colecciones/12_valor_ganado/ValorGanado_03.html) — 9 actividades, tres puntos de control (semanas 5, 10, 14)  

<!-- colecciones:fin -->
