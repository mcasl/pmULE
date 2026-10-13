# Colecciones de ejercicios cortos por competencia

Cada carpeta es una colección de ejercicios breves que ejercitan una sola competencia. Los cuadernos siguen el guion de los `Ejercicio_NN` (enunciado con `pregunta_N`, tabla de datos, resolución por apartados con pmULE) y están emparejados con jupytext (`.py` ↔ `.ipynb`).

Se generan con `python -m scripts.colecciones.generador` desde la raíz del repositorio (ver la ayuda del script). Los datos de cada ejercicio son deterministas (semilla fija) y se eligen por muestreo hasta que el proyecto presenta los rasgos que la competencia necesita (ficticias, varias rutas críticas, holguras distintas, sobrecoste, …).

Para renderizar a HTML en `docs/colecciones/`: `./render_colecciones.sh` (todos) o `./render_colecciones.sh notebooks/colecciones/05_holguras/Holguras_01.ipynb`.

## 01_prelaciones — Cuadro de prelaciones

Depurar un cuadro de prelaciones (redundancias, erratas que crean ciclos), construir el cuadro expandido y el de ancestros.

- `Prelaciones_01` — 10 actividades, 2 prelaciones redundantes
- `Prelaciones_02` — 10 actividades, 2 prelaciones redundantes
- `Prelaciones_03` — 10 actividades, 2 prelaciones redundantes, una errata con ciclo

## 02_pert — Grafo PERT y actividades ficticias

Dibujar el grafo PERT con el mínimo número de ficticias, justificar cada una, numerar los nodos y enumerar las rutas.

- `PERT_01` — 8 actividades, 2 ficticias, 6 rutas
- `PERT_02` — 9 actividades, 3 ficticias, 7 rutas
- `PERT_03` — 10 actividades, 3 ficticias, 5 rutas

## 03_roy — Grafo Roy

Dibujar el grafo Roy, compararlo con el PERT y modelar esperas y solapes entre actividades.

- `Roy_01` — 8 actividades, 2 ficticias en PERT
- `Roy_02` — 8 actividades, 2 ficticias en PERT
- `Roy_03` — 8 actividades, 3 ficticias en PERT, con espera y solape

## 04_zaderenko — Método de Zaderenko y calendario

Calcular con la matriz de Zaderenko los tiempos tempranos y tardíos, el calendario de las actividades y el diagrama de Gantt.

- `Zaderenko_01` — 8 actividades, 8 nodos, 2 ficticias
- `Zaderenko_02` — 9 actividades, 9 nodos, 1 ficticias
- `Zaderenko_03` — 10 actividades, 9 nodos, 1 ficticias

## 05_holguras — Holguras y camino crítico

Calcular e interpretar las holguras total, libre e independiente; identificar todos los caminos críticos.

- `Holguras_01` — 9 actividades, 1 ruta(s) crítica(s)
- `Holguras_02` — 10 actividades, 1 ruta(s) crítica(s)
- `Holguras_03` — 10 actividades, 2 ruta(s) crítica(s)

## 06_matriz_caminos — Matriz de caminos

Enumerar las rutas, construir la matriz de caminos y usarla para calcular la duración del proyecto; reconstruir el grafo a partir de las rutas; rutas dominantes mediante la SVD.

- `Caminos_01` — 8 actividades, 6 rutas
- `Caminos_02` — 8 actividades, 6 rutas
- `Caminos_03` — 9 actividades, 8 rutas, reconstrucción desde las rutas y SVD

## 07_estimaciones — Estimaciones de duración (a, m, b)

Obtener la duración media y la varianza de cada actividad a partir de sus tres estimaciones, resolver el problema inverso y comparar distribuciones.

- `Estimaciones_01` — 7 actividades, 0 con estimaciones asimétricas
- `Estimaciones_02` — 7 actividades, 2 con estimaciones asimétricas
- `Estimaciones_03` — 8 actividades, 2 con estimaciones asimétricas

## 08_probabilidad — Probabilidad de cumplir el plazo (aproximación normal)

Varianza del proyecto por el camino crítico, probabilidad de terminar antes de un plazo o entre dos fechas y duración comprometible para una probabilidad dada.

- `Probabilidad_01` — 9 actividades, 1 ruta(s) crítica(s), σ = 1.18
- `Probabilidad_02` — 9 actividades, 2 ruta(s) crítica(s), σ = 1.25
- `Probabilidad_03` — 10 actividades, 2 ruta(s) crítica(s), σ = 1.57

## 09_montecarlo — Método de Monte Carlo

Simular la duración del proyecto con la matriz de caminos, compararla con la aproximación normal y calcular índices de criticidad y sensibilidad.

- `MonteCarlo_01` — 8 actividades, 5 rutas, distribución normal
- `MonteCarlo_02` — 9 actividades, 6 rutas, distribución triangular
- `MonteCarlo_03` — 9 actividades, 6 rutas, distribución Beta-PERT

## 10_reduccion — Reducción de la duración con mínimo coste

Coste unitario de reducción, algoritmo de Ackoff-Sasieni paso a paso, costes indirectos y duración de coste mínimo, reducción frente a penalización.

- `Reduccion_01` — 8 actividades, reducción de 3 semanas
- `Reduccion_02` — 8 actividades, costes indirectos de 9 k€/semana
- `Reduccion_03` — 9 actividades, plazo 27 con penalización

## 11_recursos — Nivelación y asignación de recursos

Diagrama de cargas, nivelación por mínimos cuadrados, asignación con límite de recursos (Wiest-Levy), evaluación de desplazamientos y varios tipos de recurso.

- `Recursos_01` — 9 actividades, nivelación
- `Recursos_02` — 8 actividades, asignación con límite 6→8
- `Recursos_03` — 8 actividades, dos recursos y propuesta de desplazamiento

## 12_valor_ganado — Método del valor ganado

Diagramas de Gantt del valor planificado, ganado y coste real; CPI, SPI, EAC, VAC, TCPI; programación ganada; evolución de los indicadores en varios puntos de control.

- `ValorGanado_01` — 9 actividades, un punto de control (semana 10), EAC/VAC/TCPI
- `ValorGanado_02` — 8 actividades, un punto de control (semana 6), programación ganada
- `ValorGanado_03` — 9 actividades, tres puntos de control (semanas 5, 10, 14)
