# Colecciones de ejercicios cortos por competencia

Cada carpeta es una colección de ejercicios breves que ejercitan una sola competencia. Los cuadernos siguen el guion de los `Ejercicio_NN` (enunciado con `pregunta_N`, tabla de datos, resolución por apartados con pmULE) y están emparejados con jupytext (`.py` ↔ `.ipynb`).

Se generan con `python -m scripts.colecciones.generador` desde la raíz del repositorio (`--listar` muestra las variantes; ver la ayuda del script). Cada colección tiene varias *variantes*, una por concepto que se quiere comprobar; el ejercicio k usa la variante k (cíclicamente). Los datos de cada ejercicio son deterministas (semilla fija) y se eligen por muestreo hasta que el proyecto presenta los rasgos que la variante necesita (ficticias, varias rutas críticas, holguras distintas, sobrecoste, …). La mitad de los cuadernos llevan además un contexto narrativo con la descripción de cada actividad.

`catalogo.json` lista todos los ejercicios con su colección, variante y nivel; `python -m scripts.colecciones.examen` monta exámenes (enunciado y solución por separado) a partir de las mismas plantillas con datos nuevos.

Para renderizar a HTML en `docs/colecciones/`: `./render_colecciones.sh` (todos) o `./render_colecciones.sh notebooks/colecciones/05_holguras/Holguras_01.ipynb`.

## 01_prelaciones — Cuadro de prelaciones

Depurar un cuadro de prelaciones (redundancias, erratas que crean ciclos), construir el cuadro expandido y el de ancestros.

Variantes:
  - `redundancias` (nivel 1): cuadro con prelaciones redundantes: depurarlo, cuadro expandido y de ancestros
  - `redundancias-b` (nivel 1): ídem con otro proyecto
  - `ciclo` (nivel 2): errata que crea un ciclo: detectarla con la librería y corregirla antes de depurar
  - `texto` (nivel 2): las prelaciones llegan como texto corrido con contexto; hay que tabularlas
  - `grafo-erroneo` (nivel 3): se da un grafo PERT con una prelación de más y otra de menos: encontrarlas comparando cuadros de ancestros

- `Prelaciones_01` [redundancias, nivel 1] — 10 actividades, 2 prelaciones redundantes
- `Prelaciones_02` [redundancias-b, nivel 1] — 10 actividades, 2 prelaciones redundantes
- `Prelaciones_03` [ciclo, nivel 2] — 10 actividades, 2 prelaciones redundantes, una errata con ciclo
- `Prelaciones_04` [texto, nivel 2] — 9 actividades descritas en texto (la organización de un congreso científico)
- `Prelaciones_05` [grafo-erroneo, nivel 3] — 8 actividades, grafo con una prelación de más y otra de menos

## 02_pert — Grafo PERT y actividades ficticias

Dibujar el grafo PERT con el mínimo número de ficticias, justificar cada una, numerar los nodos y enumerar las rutas.

Variantes:
  - `minimo-ficticias` (nivel 1): dibujar con el mínimo de ficticias, justificar cada una, rutas y ancestros
  - `paralelas` (nivel 2): ídem con dos actividades de idénticas predecesoras (arcos paralelos)
  - `mas-ficticias` (nivel 2): ídem con un proyecto mayor y más ficticias
  - `compartir-nodos` (nivel 2): qué parejas pueden compartir nodo inicial, nodo final, ambos o ninguno
  - `ficticias-sobrantes` (nivel 3): grafo sistemático sin simplificar frente al mínimo: qué ficticias sobran y por qué

- `PERT_01` [minimo-ficticias, nivel 1] — 8 actividades, 2 ficticias, 6 rutas
- `PERT_02` [paralelas, nivel 2] — 9 actividades, 3 ficticias, 7 rutas
- `PERT_03` [mas-ficticias, nivel 2] — 10 actividades, 3 ficticias, 5 rutas
- `PERT_04` [compartir-nodos, nivel 2] — 9 actividades, 1 pareja(s) con mismas predecesoras y 3 con mismas sucesoras
- `PERT_05` [ficticias-sobrantes, nivel 3] — 9 actividades, 7 ficticias sin simplificar frente a 2

## 03_roy — Grafo Roy

Dibujar el grafo Roy, compararlo con el PERT y modelar esperas y solapes entre actividades.

Variantes:
  - `roy-vs-pert` (nivel 1): grafo Roy, PERT equivalente, comparación de nodos y arcos, calendario y camino crítico
  - `roy-vs-pert-b` (nivel 1): ídem con otro proyecto
  - `esperas-y-solapes` (nivel 3): espera entre dos actividades y solape comienzo-comienzo modelados con actividades auxiliares
  - `predecesora-dominante` (nivel 2): cálculo hacia delante y hacia atrás a mano sobre Roy; qué predecesora fija cada inicio

- `Roy_01` [roy-vs-pert, nivel 1] — 8 actividades, 2 ficticias en PERT
- `Roy_02` [roy-vs-pert-b, nivel 1] — 8 actividades, 2 ficticias en PERT
- `Roy_03` [esperas-y-solapes, nivel 3] — 8 actividades, 3 ficticias en PERT, con espera y solape
- `Roy_04` [predecesora-dominante, nivel 2] — 8 actividades, cálculo manual sobre Roy; predecesora dominante de E

## 04_zaderenko — Método de Zaderenko y calendario

Calcular con la matriz de Zaderenko los tiempos tempranos y tardíos, el calendario de las actividades y el diagrama de Gantt.

Variantes:
  - `zaderenko` (nivel 1): matriz de Zaderenko, tiempos de los nodos, calendario, Gantt y efecto de un retraso
  - `zaderenko-b` (nivel 1): ídem con un proyecto mayor
  - `zaderenko-c` (nivel 2): ídem con diez actividades
  - `matriz-dada` (nivel 3): inverso: de la matriz de Zaderenko rellena al cuadro de prelaciones, el grafo y los tiempos
  - `prediccion` (nivel 2): predecir sin calcular si un retraso cambia el plazo y comprobarlo con las holguras

- `Zaderenko_01` [zaderenko, nivel 1] — 8 actividades, 8 nodos, 2 ficticias
- `Zaderenko_02` [zaderenko-b, nivel 1] — 9 actividades, 9 nodos, 1 ficticias
- `Zaderenko_03` [zaderenko-c, nivel 2] — 10 actividades, 9 nodos, 1 ficticias
- `Zaderenko_04` [matriz-dada, nivel 3] — 8 actividades, 8 nodos; reconstrucción desde la matriz
- `Zaderenko_05` [prediccion, nivel 2] — 9 actividades; predicción cualitativa sobre H, C y I

## 05_holguras — Holguras y camino crítico

Calcular e interpretar las holguras total, libre e independiente; identificar todos los caminos críticos.

Variantes:
  - `tres-holguras` (nivel 1): holguras total, libre e independiente; caminos críticos; retraso de una actividad
  - `tres-holguras-b` (nivel 2): ídem con holgura independiente distinta de la libre
  - `varias-criticas` (nivel 2): ídem con dos o más rutas críticas
  - `holgura-compartida` (nivel 2): dos actividades consecutivas no pueden gastar las dos la misma holgura total
  - `vigilancia` (nivel 2): ordenar por riesgo; no críticas sin margen; la que más puede retrasarse sin que se note
  - `diagnostico` (nivel 3): calendario entregado con dos errores: localizarlos por las relaciones entre fechas y holguras

- `Holguras_01` [tres-holguras, nivel 1] — 9 actividades, 1 ruta(s) crítica(s)
- `Holguras_02` [tres-holguras-b, nivel 2] — 10 actividades, 1 ruta(s) crítica(s)
- `Holguras_03` [varias-criticas, nivel 2] — 10 actividades, 2 ruta(s) crítica(s)
- `Holguras_04` [holgura-compartida, nivel 2] — 10 actividades; D y G comparten 13 semanas de holgura
- `Holguras_05` [vigilancia, nivel 2] — 9 actividades; ordenación por vigilancia y actividades sin margen
- `Holguras_06` [diagnostico, nivel 3] — 8 actividades; calendario con errores en B y C

## 06_matriz_caminos — Matriz de caminos

Enumerar las rutas, construir la matriz de caminos y usarla para calcular la duración del proyecto; reconstruir el grafo a partir de las rutas; rutas dominantes mediante la SVD.

Variantes:
  - `matriz-y-duracion` (nivel 1): enumerar rutas, matriz de caminos, duración como máximo de R·d, sensibilidad
  - `matriz-y-duracion-b` (nivel 1): ídem con otro proyecto
  - `rutas-a-grafo-svd` (nivel 3): reconstruir el grafo a partir de las rutas; rutas dominantes por SVD
  - `escenarios` (nivel 2): tres escenarios de duración a la vez con R·T; puente con Monte Carlo

- `Caminos_01` [matriz-y-duracion, nivel 1] — 8 actividades, 6 rutas
- `Caminos_02` [matriz-y-duracion-b, nivel 1] — 8 actividades, 6 rutas
- `Caminos_03` [rutas-a-grafo-svd, nivel 3] — 9 actividades, 8 rutas, reconstrucción desde las rutas y SVD
- `Caminos_04` [escenarios, nivel 2] — 8 actividades, 4 rutas, tres escenarios

## 07_estimaciones — Estimaciones de duración (a, m, b)

Obtener la duración media y la varianza de cada actividad a partir de sus tres estimaciones, resolver el problema inverso y comparar distribuciones.

Variantes:
  - `tres-estimaciones` (nivel 1): media y varianza PERT, cálculo inverso de a, camino crítico, Beta-PERT vs triangular vs uniforme
  - `tres-estimaciones-b` (nivel 1): ídem con estimaciones asimétricas
  - `tres-estimaciones-c` (nivel 1): ídem con otro proyecto
  - `deterministas` (nivel 2): actividades de duración fija y qué actividades determinan la varianza del plazo

- `Estimaciones_01` [tres-estimaciones, nivel 1] — 7 actividades, 0 con estimaciones asimétricas
- `Estimaciones_02` [tres-estimaciones-b, nivel 1] — 7 actividades, 2 con estimaciones asimétricas
- `Estimaciones_03` [tres-estimaciones-c, nivel 1] — 8 actividades, 2 con estimaciones asimétricas
- `Estimaciones_04` [deterministas, nivel 2] — 8 actividades, 2 de duración fija

## 08_probabilidad — Probabilidad de cumplir el plazo (aproximación normal)

Varianza del proyecto por el camino crítico, probabilidad de terminar antes de un plazo o entre dos fechas y duración comprometible para una probabilidad dada.

Variantes:
  - `normal` (nivel 1): varianza por la ruta crítica, P(T ≤ t), P(a < T ≤ b), duración para un percentil
  - `dos-rutas-criticas` (nivel 2): ídem con dos rutas críticas de varianzas distintas: cuál usar
  - `dos-rutas-criticas-b` (nivel 2): ídem con otro proyecto
  - `casi-critica-variable` (nivel 3): la segunda ruta es mucho más variable que la crítica: cota de probabilidad y Monte Carlo
  - `varianza-maxima` (nivel 3): inverso: σ máxima admisible de una actividad para cumplir el plazo con probabilidad dada
  - `diagnostico` (nivel 2): resolución entregada con σ sumadas y con la ruta equivocada: localizar y corregir

- `Probabilidad_01` [normal, nivel 1] — 9 actividades, 1 ruta(s) crítica(s), σ = 1.18
- `Probabilidad_02` [dos-rutas-criticas, nivel 2] — 9 actividades, 2 ruta(s) crítica(s), σ = 1.25
- `Probabilidad_03` [dos-rutas-criticas-b, nivel 2] — 10 actividades, 2 ruta(s) crítica(s), σ = 1.57
- `Probabilidad_04` [casi-critica-variable, nivel 3] — 9 actividades; segunda ruta a 1 semana(s) de la crítica y mucho más variable
- `Probabilidad_05` [varianza-maxima, nivel 3] — 10 actividades; σ máxima admisible de I para P ≥ 90 % en 37 semanas
- `Probabilidad_06` [diagnostico, nivel 2] — 9 actividades; resolución con σ sumadas y con la ruta equivocada

## 09_montecarlo — Método de Monte Carlo

Simular la duración del proyecto con la matriz de caminos, compararla con la aproximación normal y calcular índices de criticidad y sensibilidad.

Variantes:
  - `normal` (nivel 1): Monte Carlo con R·T, comparación con la aproximación normal, criticidad y sensibilidad (normal)
  - `triangular` (nivel 2): ídem con duraciones triangulares (a, m, b)
  - `beta-pert` (nivel 2): ídem con duraciones Beta-PERT
  - `a-mano` (nivel 1): cinco réplicas a mano: R·T, máximo por columnas, media, desviación y criticidad
  - `mixtas` (nivel 2): distribuciones distintas por actividad (incluidas fijas) con monte_carlo de la librería
  - `convergencia` (nivel 3): cuántas réplicas hacen falta: error típico de la media y del percentil 90

- `MonteCarlo_01` [normal, nivel 1] — 8 actividades, 5 rutas, distribución normal
- `MonteCarlo_02` [triangular, nivel 2] — 9 actividades, 6 rutas, distribución triangular
- `MonteCarlo_03` [beta-pert, nivel 2] — 9 actividades, 6 rutas, distribución Beta-PERT
- `MonteCarlo_04` [a-mano, nivel 1] — 7 actividades, 5 rutas, 5 réplicas calculadas a mano
- `MonteCarlo_05` [mixtas, nivel 2] — 8 actividades con distribuciones mixtas (1 fijas)
- `MonteCarlo_06` [convergencia, nivel 3] — 9 actividades; convergencia de la media y del percentil 90 con N

## 10_reduccion — Reducción de la duración con mínimo coste

Coste unitario de reducción, algoritmo de Ackoff-Sasieni paso a paso, costes indirectos y duración de coste mínimo, reducción frente a penalización.

Variantes:
  - `cur-y-ackoff` (nivel 1): CUR a partir de costes normal y tope; Ackoff-Sasieni tres semanas con un paso de dos actividades
  - `costes-indirectos` (nivel 2): curva coste-duración con costes indirectos y duración de coste mínimo
  - `penalizacion` (nivel 2): reducir frente a pagar penalización; actividades no acelerables
  - `barata-no-optima` (nivel 2): dos rutas críticas: la crítica más barata no sirve si no es común a ambas
  - `desaceleracion` (nivel 2): reducido el proyecto, el plazo se relaja: qué paso deshacer (el más caro) y por qué la curva coste-duración es convexa

- `Reduccion_01` [cur-y-ackoff, nivel 1] — 8 actividades, reducción de 3 semanas
- `Reduccion_02` [costes-indirectos, nivel 2] — 8 actividades, costes indirectos de 9 k€/semana
- `Reduccion_03` [penalizacion, nivel 2] — 9 actividades, plazo 27 con penalización
- `Reduccion_04` [barata-no-optima, nivel 2] — 8 actividades, dos rutas críticas; A es la más barata pero no es común
- `Reduccion_05` [desaceleracion, nivel 2] — 8 actividades; deshacer la reducción más cara al relajar el plazo

## 11_recursos — Nivelación y asignación de recursos

Diagrama de cargas, nivelación por mínimos cuadrados, asignación con límite de recursos (Wiest-Levy), evaluación de desplazamientos y varios tipos de recurso.

Variantes:
  - `nivelacion` (nivel 1): cargas, suma de cuadrados y nivelación por mínimos cuadrados
  - `asignacion` (nivel 2): asignación Wiest-Levy con límite por tramos y con límite fijo
  - `dos-recursos` (nivel 3): dos tipos de recurso; evaluación de un desplazamiento propuesto frente a la nivelación
  - `consumible` (nivel 2): recurso consumible no almacenable frente a almacenable (restricción semanal frente a acumulada)
  - `limite-insuficiente` (nivel 2): factibilidad del límite; tabla operarios → duración y número óptimo de operarios

- `Recursos_01` [nivelacion, nivel 1] — 9 actividades, nivelación
- `Recursos_02` [asignacion, nivel 2] — 8 actividades, asignación con límite 6→8
- `Recursos_03` [dos-recursos, nivel 3] — 8 actividades, dos recursos y propuesta de desplazamiento
- `Recursos_04` [consumible, nivel 2] — 9 actividades; combustible con suministro de 70 l/semana
- `Recursos_05` [limite-insuficiente, nivel 2] — 8 actividades; factibilidad y duración en función de los operarios

## 12_valor_ganado — Método del valor ganado

Diagramas de Gantt del valor planificado, ganado y coste real; CPI, SPI, EAC, VAC, TCPI; programación ganada; evolución de los indicadores en varios puntos de control.

Variantes:
  - `indicadores-eac` (nivel 1): tres Gantts, PV/EV/AC, CV/SV, CPI/SPI, tres fórmulas de EAC, VAC y TCPI
  - `earned-schedule` (nivel 2): programación ganada: ES, SPI(t), previsión de duración frente a la de SPI
  - `tendencia` (nivel 2): tres puntos de control: evolución de CPI, SPI y EAC
  - `coste-bien-plazo-mal` (nivel 2): gastar menos no es ir bien: CPI > 1 con SPI < 1
  - `plazo-bien-coste-mal` (nivel 2): ir adelantado a costa de sobrecoste: SPI ≥ 1 con CPI < 1
  - `diagnostico` (nivel 2): resolución entregada con CPI y SPI intercambiados, CV con el signo cambiado y EAC mal

- `ValorGanado_01` [indicadores-eac, nivel 1] — 9 actividades, un punto de control (semana 10), EAC/VAC/TCPI
- `ValorGanado_02` [earned-schedule, nivel 2] — 8 actividades, un punto de control (semana 6), programación ganada
- `ValorGanado_03` [tendencia, nivel 2] — 9 actividades, tres puntos de control (semanas 5, 10, 14)
- `ValorGanado_04` [coste-bien-plazo-mal, nivel 2] — 9 actividades; punto de control en la semana 11: coste bien plazo mal
- `ValorGanado_05` [plazo-bien-coste-mal, nivel 2] — 9 actividades; punto de control en la semana 10: plazo bien coste mal
- `ValorGanado_06` [diagnostico, nivel 2] — 8 actividades; resolución con CPI y SPI intercambiados

## 13_formula_compleccion — La fórmula del tiempo de compleción τ = ‖R·t‖∞

Matriz de rutas R, τ como norma infinito de R·t, gradiente ∂τ/∂t, descomposición en valores singulares de R, núcleo de la matriz de incidencia e instancias de referencia.

Variantes:
  - `tau-norma-infinito` (nivel 1): R, R·t y τ = ‖R·t‖∞; equivalencia con Zaderenko; τ como función lineal a trozos de una duración
  - `gradiente` (nivel 2): ∂τ/∂t con rutas críticas empatadas: subgradiente, diferencias finitas y suma de sensibilidades
  - `svd-rango-1` (nivel 3): SVD de R: σ₁/σ₂, rankings por u₁ y v₁, aproximación de rango 1 de τ y precisión frente al conjunto crítico
  - `nucleo-incidencia` (nivel 3): matriz de incidencia, R·H = 0 y perturbaciones de las duraciones que no cambian τ
  - `patterson` (nivel 2): una instancia de referencia (Patterson, formato RCP): τ frente a CPM y ranking SVD
  - `svd-vs-montecarlo` (nivel 3): índice de criticidad Monte Carlo frente a |v₁|: correlación y efecto de la incertidumbre

- `Formula_01` [tau-norma-infinito, nivel 1] — 7 actividades, 4 rutas; τ como norma infinito
- `Formula_02` [gradiente, nivel 2] — 9 actividades, 2 rutas críticas; gradiente con empates
- `Formula_03` [svd-rango-1, nivel 3] — 10 actividades, 6 rutas; SVD de R y ranking frente al conjunto crítico
- `Formula_04` [nucleo-incidencia, nivel 3] — 8 actividades, 9 nodos; incidencia y núcleo
- `Formula_05` [patterson, nivel 2] — instancia pat1 de Patterson (14 actividades)
- `Formula_06` [svd-vs-montecarlo, nivel 3] — 10 actividades, 7 rutas; criticidad Monte Carlo frente a |v₁|
