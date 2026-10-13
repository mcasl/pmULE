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
  - `numeracion-nodos` (nivel 2): numerar los nodos del PERT con i < j en cada arco; nodos intercambiables; contar las numeraciones válidas
  - `datos-sucios` (nivel 1): cuadro transcrito con minúsculas, separadores y duplicados: normalizarlo con reglas antes de construir el proyecto

- `Prelaciones_01` [redundancias, nivel 1] — 10 actividades, 2 prelaciones redundantes
- `Prelaciones_02` [redundancias-b, nivel 1] — 10 actividades, 2 prelaciones redundantes
- `Prelaciones_03` [ciclo, nivel 2] — 10 actividades, 2 prelaciones redundantes, una errata con ciclo
- `Prelaciones_04` [texto, nivel 2] — 9 actividades descritas en texto (la organización de un congreso científico)
- `Prelaciones_05` [grafo-erroneo, nivel 3] — 8 actividades, grafo con una prelación de más y otra de menos
- `Prelaciones_06` [numeracion-nodos, nivel 2] — 7 actividades, 6 nodos; numeración de los nodos
- `Prelaciones_07` [datos-sucios, nivel 1] — 9 actividades; tabla con 7 formatos distintos que normalizar
- `Prelaciones_08` [redundancias, nivel 1] — 9 actividades, 2 prelaciones redundantes
- `Prelaciones_09` [redundancias-b, nivel 1] — 9 actividades, 2 prelaciones redundantes
- `Prelaciones_10` [ciclo, nivel 2] — 9 actividades, 2 prelaciones redundantes, una errata con ciclo
- `Prelaciones_11` [texto, nivel 2] — 9 actividades descritas en texto (la organización de un congreso científico)
- `Prelaciones_12` [grafo-erroneo, nivel 3] — 9 actividades, grafo con una prelación de más y otra de menos
- `Prelaciones_13` [numeracion-nodos, nivel 2] — 7 actividades, 7 nodos; numeración de los nodos
- `Prelaciones_14` [datos-sucios, nivel 1] — 9 actividades; tabla con 8 formatos distintos que normalizar

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
- `PERT_06` [minimo-ficticias, nivel 1] — 8 actividades, 2 ficticias, 6 rutas
- `PERT_07` [paralelas, nivel 2] — 9 actividades, 3 ficticias, 9 rutas
- `PERT_08` [mas-ficticias, nivel 2] — 10 actividades, 3 ficticias, 9 rutas
- `PERT_09` [compartir-nodos, nivel 2] — 9 actividades, 2 pareja(s) con mismas predecesoras y 4 con mismas sucesoras
- `PERT_10` [ficticias-sobrantes, nivel 3] — 8 actividades, 8 ficticias sin simplificar frente a 3

## 03_roy — Grafo Roy

Dibujar el grafo Roy, compararlo con el PERT y modelar esperas y solapes entre actividades.

Variantes:
  - `roy-vs-pert` (nivel 1): grafo Roy, PERT equivalente, comparación de nodos y arcos, calendario y camino crítico
  - `roy-vs-pert-b` (nivel 1): ídem con otro proyecto
  - `esperas-y-solapes` (nivel 3): espera entre dos actividades y solape comienzo-comienzo modelados con actividades auxiliares
  - `predecesora-dominante` (nivel 2): cálculo hacia delante y hacia atrás a mano sobre Roy; qué predecesora fija cada inicio
  - `desde-sucesoras` (nivel 1): el cuadro da sucesoras en lugar de predecesoras: invertirlo, Roy, PERT y calendario

- `Roy_01` [roy-vs-pert, nivel 1] — 8 actividades, 2 ficticias en PERT
- `Roy_02` [roy-vs-pert-b, nivel 1] — 8 actividades, 2 ficticias en PERT
- `Roy_03` [esperas-y-solapes, nivel 3] — 8 actividades, 3 ficticias en PERT, con espera y solape
- `Roy_04` [predecesora-dominante, nivel 2] — 8 actividades, cálculo manual sobre Roy; predecesora dominante de E
- `Roy_05` [desde-sucesoras, nivel 1] — 8 actividades; cuadro de sucesoras, 1 ficticias en PERT
- `Roy_06` [roy-vs-pert, nivel 1] — 9 actividades, 4 ficticias en PERT
- `Roy_07` [roy-vs-pert-b, nivel 1] — 8 actividades, 2 ficticias en PERT
- `Roy_08` [esperas-y-solapes, nivel 3] — 8 actividades, 2 ficticias en PERT, con espera y solape
- `Roy_09` [predecesora-dominante, nivel 2] — 8 actividades, cálculo manual sobre Roy; predecesora dominante de H
- `Roy_10` [desde-sucesoras, nivel 1] — 8 actividades; cuadro de sucesoras, 1 ficticias en PERT

## 04_zaderenko — Método de Zaderenko y calendario

Calcular con la matriz de Zaderenko los tiempos tempranos y tardíos, el calendario de las actividades y el diagrama de Gantt.

Variantes:
  - `zaderenko` (nivel 1): matriz de Zaderenko, tiempos de los nodos, calendario, Gantt y efecto de un retraso
  - `zaderenko-b` (nivel 1): ídem con un proyecto mayor
  - `zaderenko-c` (nivel 2): ídem con diez actividades
  - `matriz-dada` (nivel 3): inverso: de la matriz de Zaderenko rellena al cuadro de prelaciones, el grafo y los tiempos
  - `prediccion` (nivel 2): predecir sin calcular si un retraso cambia el plazo y comprobarlo con las holguras
  - `plazo-impuesto` (nivel 2): plazo menor que la duración: tardíos desde el plazo, holguras negativas, rutas que lo exceden y actividad común
  - `gantt-dado` (nivel 3): sólo se conoce el Gantt: deducir iniciales, predecesoras posibles y ajustadas, y una ruta crítica

- `Zaderenko_01` [zaderenko, nivel 1] — 8 actividades, 8 nodos, 2 ficticias
- `Zaderenko_02` [zaderenko-b, nivel 1] — 9 actividades, 9 nodos, 1 ficticias
- `Zaderenko_03` [zaderenko-c, nivel 2] — 10 actividades, 9 nodos, 1 ficticias
- `Zaderenko_04` [matriz-dada, nivel 3] — 8 actividades, 8 nodos; reconstrucción desde la matriz
- `Zaderenko_05` [prediccion, nivel 2] — 9 actividades; predicción cualitativa sobre H, C y I
- `Zaderenko_06` [plazo-impuesto, nivel 2] — 9 actividades; duración 25, plazo impuesto 22 (3 rutas lo exceden)
- `Zaderenko_07` [gantt-dado, nivel 3] — 7 actividades; sólo se conoce el Gantt de inicios tempranos
- `Zaderenko_08` [zaderenko, nivel 1] — 8 actividades, 8 nodos, 2 ficticias
- `Zaderenko_09` [zaderenko-b, nivel 1] — 9 actividades, 9 nodos, 2 ficticias
- `Zaderenko_10` [zaderenko-c, nivel 2] — 10 actividades, 8 nodos, 1 ficticias
- `Zaderenko_11` [matriz-dada, nivel 3] — 8 actividades, 8 nodos; reconstrucción desde la matriz
- `Zaderenko_12` [prediccion, nivel 2] — 8 actividades; predicción cualitativa sobre H, E y F
- `Zaderenko_13` [plazo-impuesto, nivel 2] — 9 actividades; duración 31, plazo impuesto 28 (3 rutas lo exceden)
- `Zaderenko_14` [gantt-dado, nivel 3] — 7 actividades; sólo se conoce el Gantt de inicios tempranos

## 05_holguras — Holguras y camino crítico

Calcular e interpretar las holguras total, libre e independiente; identificar todos los caminos críticos.

Variantes:
  - `tres-holguras` (nivel 1): holguras total, libre e independiente; caminos críticos; retraso de una actividad
  - `tres-holguras-b` (nivel 2): ídem con holgura independiente distinta de la libre
  - `varias-criticas` (nivel 2): ídem con dos o más rutas críticas
  - `holgura-compartida` (nivel 2): dos actividades consecutivas no pueden gastar las dos la misma holgura total
  - `vigilancia` (nivel 2): ordenar por riesgo; no críticas sin margen; la que más puede retrasarse sin que se note
  - `diagnostico` (nivel 3): calendario entregado con dos errores: localizarlos por las relaciones entre fechas y holguras
  - `mini-a-mano` (nivel 1): cinco o seis actividades sin ficticias: rutas, pasos adelante y atrás y holguras con los números sustituidos

- `Holguras_01` [tres-holguras, nivel 1] — 9 actividades, 1 ruta(s) crítica(s)
- `Holguras_02` [tres-holguras-b, nivel 2] — 10 actividades, 1 ruta(s) crítica(s)
- `Holguras_03` [varias-criticas, nivel 2] — 10 actividades, 2 ruta(s) crítica(s)
- `Holguras_04` [holgura-compartida, nivel 2] — 10 actividades; D y G comparten 13 semanas de holgura
- `Holguras_05` [vigilancia, nivel 2] — 9 actividades; ordenación por vigilancia y actividades sin margen
- `Holguras_06` [diagnostico, nivel 3] — 8 actividades; calendario con errores en B y C
- `Holguras_07` [mini-a-mano, nivel 1] — 5 actividades, 2 rutas, sin ficticias; cálculo completo a mano
- `Holguras_08` [tres-holguras, nivel 1] — 10 actividades, 1 ruta(s) crítica(s)
- `Holguras_09` [tres-holguras-b, nivel 2] — 10 actividades, 1 ruta(s) crítica(s)
- `Holguras_10` [varias-criticas, nivel 2] — 10 actividades, 2 ruta(s) crítica(s)
- `Holguras_11` [holgura-compartida, nivel 2] — 9 actividades; C y D comparten 1 semanas de holgura
- `Holguras_12` [vigilancia, nivel 2] — 10 actividades; ordenación por vigilancia y actividades sin margen
- `Holguras_13` [diagnostico, nivel 3] — 9 actividades; calendario con errores en D y C
- `Holguras_14` [mini-a-mano, nivel 1] — 5 actividades, 2 rutas, sin ficticias; cálculo completo a mano

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
- `Caminos_05` [matriz-y-duracion, nivel 1] — 8 actividades, 6 rutas
- `Caminos_06` [matriz-y-duracion-b, nivel 1] — 8 actividades, 8 rutas
- `Caminos_07` [rutas-a-grafo-svd, nivel 3] — 8 actividades, 5 rutas, reconstrucción desde las rutas y SVD
- `Caminos_08` [escenarios, nivel 2] — 8 actividades, 4 rutas, tres escenarios

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
- `Estimaciones_05` [tres-estimaciones, nivel 1] — 7 actividades, 3 con estimaciones asimétricas
- `Estimaciones_06` [tres-estimaciones-b, nivel 1] — 8 actividades, 1 con estimaciones asimétricas
- `Estimaciones_07` [tres-estimaciones-c, nivel 1] — 7 actividades, 3 con estimaciones asimétricas
- `Estimaciones_08` [deterministas, nivel 2] — 9 actividades, 2 de duración fija

## 08_probabilidad — Probabilidad de cumplir el plazo (aproximación normal)

Varianza del proyecto por el camino crítico, probabilidad de terminar antes de un plazo o entre dos fechas y duración comprometible para una probabilidad dada.

Variantes:
  - `normal` (nivel 1): varianza por la ruta crítica, P(T ≤ t), P(a < T ≤ b), duración para un percentil
  - `dos-rutas-criticas` (nivel 2): ídem con dos rutas críticas de varianzas distintas: cuál usar
  - `dos-rutas-criticas-b` (nivel 2): ídem con otro proyecto
  - `casi-critica-variable` (nivel 3): la segunda ruta es mucho más variable que la crítica: cota de probabilidad y Monte Carlo
  - `varianza-maxima` (nivel 3): inverso: σ máxima admisible de una actividad para cumplir el plazo con probabilidad dada
  - `diagnostico` (nivel 2): resolución entregada con σ sumadas y con la ruta equivocada: localizar y corregir
  - `hito` (nivel 2): probabilidad de alcanzar un hito intermedio: cadenas hasta la actividad, media y varianza del hito frente a las del proyecto
  - `ser-critica` (nivel 3): dos cadenas paralelas: probabilidad de que la no crítica resulte la más larga y probabilidad exacta del plazo frente a PERT

- `Probabilidad_01` [normal, nivel 1] — 9 actividades, 1 ruta(s) crítica(s), σ = 1.18
- `Probabilidad_02` [dos-rutas-criticas, nivel 2] — 9 actividades, 2 ruta(s) crítica(s), σ = 1.25
- `Probabilidad_03` [dos-rutas-criticas-b, nivel 2] — 10 actividades, 2 ruta(s) crítica(s), σ = 1.57
- `Probabilidad_04` [casi-critica-variable, nivel 3] — 9 actividades; segunda ruta a 1 semana(s) de la crítica y mucho más variable
- `Probabilidad_05` [varianza-maxima, nivel 3] — 10 actividades; σ máxima admisible de I para P ≥ 90 % en 37 semanas
- `Probabilidad_06` [diagnostico, nivel 2] — 9 actividades; resolución con σ sumadas y con la ruta equivocada
- `Probabilidad_07` [hito, nivel 2] — 9 actividades; hito al final de C (fecha media 11, comprometida 12)
- `Probabilidad_08` [ser-critica, nivel 3] — dos cadenas paralelas de 2 y 2 actividades; P(ruta no crítica resulta la más larga)
- `Probabilidad_09` [normal, nivel 1] — 9 actividades, 1 ruta(s) crítica(s), σ = 0.57
- `Probabilidad_10` [dos-rutas-criticas, nivel 2] — 9 actividades, 2 ruta(s) crítica(s), σ = 1.47
- `Probabilidad_11` [dos-rutas-criticas-b, nivel 2] — 10 actividades, 2 ruta(s) crítica(s), σ = 2.21
- `Probabilidad_12` [casi-critica-variable, nivel 3] — 9 actividades; segunda ruta a 1 semana(s) de la crítica y mucho más variable
- `Probabilidad_13` [varianza-maxima, nivel 3] — 10 actividades; σ máxima admisible de E para P ≥ 90 % en 28 semanas
- `Probabilidad_14` [diagnostico, nivel 2] — 9 actividades; resolución con σ sumadas y con la ruta equivocada
- `Probabilidad_15` [hito, nivel 2] — 9 actividades; hito al final de E (fecha media 14, comprometida 15)
- `Probabilidad_16` [ser-critica, nivel 3] — dos cadenas paralelas de 3 y 2 actividades; P(ruta no crítica resulta la más larga)

## 09_montecarlo — Método de Monte Carlo

Simular la duración del proyecto con la matriz de caminos, compararla con la aproximación normal y calcular índices de criticidad y sensibilidad.

Variantes:
  - `normal` (nivel 1): Monte Carlo con R·T, comparación con la aproximación normal, criticidad y sensibilidad (normal)
  - `triangular` (nivel 2): ídem con duraciones triangulares (a, m, b)
  - `beta-pert` (nivel 2): ídem con duraciones Beta-PERT
  - `a-mano` (nivel 1): cinco réplicas a mano: R·T, máximo por columnas, media, desviación y criticidad
  - `mixtas` (nivel 2): distribuciones distintas por actividad (incluidas fijas) con monte_carlo de la librería
  - `convergencia` (nivel 3): cuántas réplicas hacen falta: error típico de la media y del percentil 90
  - `correlacionadas` (nivel 3): un factor común correlaciona las duraciones expuestas: misma media, más dispersión, menos probabilidad de cumplir
  - `percentiles` (nivel 2): leer la distribución simulada: percentiles, probabilidad de penalización, retraso esperado y margen a comprometer

- `MonteCarlo_01` [normal, nivel 1] — 8 actividades, 5 rutas, distribución normal
- `MonteCarlo_02` [triangular, nivel 2] — 9 actividades, 6 rutas, distribución triangular
- `MonteCarlo_03` [beta-pert, nivel 2] — 9 actividades, 6 rutas, distribución Beta-PERT
- `MonteCarlo_04` [a-mano, nivel 1] — 7 actividades, 5 rutas, 5 réplicas calculadas a mano
- `MonteCarlo_05` [mixtas, nivel 2] — 8 actividades con distribuciones mixtas (1 fijas)
- `MonteCarlo_06` [convergencia, nivel 3] — 9 actividades; convergencia de la media y del percentil 90 con N
- `MonteCarlo_07` [correlacionadas, nivel 3] — 9 actividades; 4 expuestas a un factor común (σ_F = 0.15)
- `MonteCarlo_08` [percentiles, nivel 2] — 8 actividades; plazo del cliente 34; percentiles y retraso esperado
- `MonteCarlo_09` [normal, nivel 1] — 8 actividades, 5 rutas, distribución normal
- `MonteCarlo_10` [triangular, nivel 2] — 8 actividades, 6 rutas, distribución triangular
- `MonteCarlo_11` [beta-pert, nivel 2] — 9 actividades, 6 rutas, distribución Beta-PERT
- `MonteCarlo_12` [a-mano, nivel 1] — 7 actividades, 3 rutas, 5 réplicas calculadas a mano
- `MonteCarlo_13` [mixtas, nivel 2] — 9 actividades con distribuciones mixtas (2 fijas)
- `MonteCarlo_14` [convergencia, nivel 3] — 8 actividades; convergencia de la media y del percentil 90 con N
- `MonteCarlo_15` [correlacionadas, nivel 3] — 9 actividades; 4 expuestas a un factor común (σ_F = 0.15)
- `MonteCarlo_16` [percentiles, nivel 2] — 9 actividades; plazo del cliente 33; percentiles y retraso esperado

## 10_reduccion — Reducción de la duración con mínimo coste

Coste unitario de reducción, algoritmo de Ackoff-Sasieni paso a paso, costes indirectos y duración de coste mínimo, reducción frente a penalización.

Variantes:
  - `cur-y-ackoff` (nivel 1): CUR a partir de costes normal y tope; Ackoff-Sasieni tres semanas con un paso de dos actividades
  - `costes-indirectos` (nivel 2): curva coste-duración con costes indirectos y duración de coste mínimo
  - `penalizacion` (nivel 2): reducir frente a pagar penalización; actividades no acelerables
  - `barata-no-optima` (nivel 2): dos rutas críticas: la crítica más barata no sirve si no es común a ambas
  - `desaceleracion` (nivel 2): reducido el proyecto, el plazo se relaja: qué paso deshacer (el más caro) y por qué la curva coste-duración es convexa
  - `programacion-lineal` (nivel 3): reducción óptima por programación lineal con la matriz de caminos frente al voraz de Ackoff-Sasieni
  - `cur-por-tramos` (nivel 3): CUR creciente por tramos: Ackoff con coste marginal frente a un CUR medio; por qué funciona si el coste es convexo

- `Reduccion_01` [cur-y-ackoff, nivel 1] — 8 actividades, reducción de 3 semanas
- `Reduccion_02` [costes-indirectos, nivel 2] — 8 actividades, costes indirectos de 9 k€/semana
- `Reduccion_03` [penalizacion, nivel 2] — 9 actividades, plazo 27 con penalización
- `Reduccion_04` [barata-no-optima, nivel 2] — 8 actividades, dos rutas críticas; A es la más barata pero no es común
- `Reduccion_05` [desaceleracion, nivel 2] — 8 actividades; deshacer la reducción más cara al relajar el plazo
- `Reduccion_06` [programacion-lineal, nivel 3] — 9 actividades; reducir 3 semanas: Ackoff 16 k€ frente a PL 16 k€
- `Reduccion_07` [cur-por-tramos, nivel 3] — 8 actividades; CUR por tramos, reducir 3 semanas
- `Reduccion_08` [cur-y-ackoff, nivel 1] — 9 actividades, reducción de 3 semanas
- `Reduccion_09` [costes-indirectos, nivel 2] — 9 actividades, costes indirectos de 10 k€/semana
- `Reduccion_10` [penalizacion, nivel 2] — 8 actividades, plazo 28 con penalización
- `Reduccion_11` [barata-no-optima, nivel 2] — 8 actividades, dos rutas críticas; E es la más barata pero no es común
- `Reduccion_12` [desaceleracion, nivel 2] — 8 actividades; deshacer la reducción más cara al relajar el plazo
- `Reduccion_13` [programacion-lineal, nivel 3] — 9 actividades; reducir 3 semanas: Ackoff 3 k€ frente a PL 3 k€
- `Reduccion_14` [cur-por-tramos, nivel 3] — 9 actividades; CUR por tramos, reducir 3 semanas

## 11_recursos — Nivelación y asignación de recursos

Diagrama de cargas, nivelación por mínimos cuadrados, asignación con límite de recursos (Wiest-Levy), evaluación de desplazamientos y varios tipos de recurso.

Variantes:
  - `nivelacion` (nivel 1): cargas, suma de cuadrados y nivelación por mínimos cuadrados
  - `asignacion` (nivel 2): asignación Wiest-Levy con límite por tramos y con límite fijo
  - `dos-recursos` (nivel 3): dos tipos de recurso; evaluación de un desplazamiento propuesto frente a la nivelación
  - `consumible` (nivel 2): recurso consumible no almacenable frente a almacenable (restricción semanal frente a acumulada)
  - `limite-insuficiente` (nivel 2): factibilidad del límite; tabla operarios → duración y número óptimo de operarios
  - `reglas-prioridad` (nivel 3): planificador en paralelo con tres reglas de prioridad (holgura, SPT, más recurso) frente a Wiest-Levy
  - `interrumpibles` (nivel 3): misma regla con y sin interrupciones permitidas: duración, actividades interrumpidas y costes ocultos

- `Recursos_01` [nivelacion, nivel 1] — 9 actividades, nivelación
- `Recursos_02` [asignacion, nivel 2] — 8 actividades, asignación con límite 6→8
- `Recursos_03` [dos-recursos, nivel 3] — 8 actividades, dos recursos y propuesta de desplazamiento
- `Recursos_04` [consumible, nivel 2] — 9 actividades; combustible con suministro de 70 l/semana
- `Recursos_05` [limite-insuficiente, nivel 2] — 8 actividades; factibilidad y duración en función de los operarios
- `Recursos_06` [reglas-prioridad, nivel 3] — 9 actividades; 4 operarios; tres reglas de prioridad frente a Wiest-Levy
- `Recursos_07` [interrumpibles, nivel 3] — 8 actividades; 4 operarios; interrumpir o no interrumpir
- `Recursos_08` [nivelacion, nivel 1] — 9 actividades, nivelación
- `Recursos_09` [asignacion, nivel 2] — 9 actividades, asignación con límite 4→6
- `Recursos_10` [dos-recursos, nivel 3] — 9 actividades, dos recursos y propuesta de desplazamiento
- `Recursos_11` [consumible, nivel 2] — 8 actividades; combustible con suministro de 70 l/semana
- `Recursos_12` [limite-insuficiente, nivel 2] — 9 actividades; factibilidad y duración en función de los operarios
- `Recursos_13` [reglas-prioridad, nivel 3] — 9 actividades; 5 operarios; tres reglas de prioridad frente a Wiest-Levy
- `Recursos_14` [interrumpibles, nivel 3] — 8 actividades; 5 operarios; interrumpir o no interrumpir

## 12_valor_ganado — Método del valor ganado

Diagramas de Gantt del valor planificado, ganado y coste real; CPI, SPI, EAC, VAC, TCPI; programación ganada; evolución de los indicadores en varios puntos de control.

Variantes:
  - `indicadores-eac` (nivel 1): tres Gantts, PV/EV/AC, CV/SV, CPI/SPI, tres fórmulas de EAC, VAC y TCPI
  - `earned-schedule` (nivel 2): programación ganada: ES, SPI(t), previsión de duración frente a la de SPI
  - `tendencia` (nivel 2): tres puntos de control: evolución de CPI, SPI y EAC
  - `coste-bien-plazo-mal` (nivel 2): gastar menos no es ir bien: CPI > 1 con SPI < 1
  - `plazo-bien-coste-mal` (nivel 2): ir adelantado a costa de sobrecoste: SPI ≥ 1 con CPI < 1
  - `diagnostico` (nivel 2): resolución entregada con CPI y SPI intercambiados, CV con el signo cambiado y EAC mal
  - `inverso-desde-curvas` (nivel 2): sólo las curvas S: índices semana a semana, cuándo empezó el retraso y el sobrecoste, EAC y programación ganada
  - `coste-no-uniforme` (nivel 3): perfiles de desembolso (inicio/final/uniforme): PV correcto frente a uniforme y efecto sobre el SPI
  - `vigilar` (nivel 2): SPI y CPI por actividad cruzados con la holgura: qué amenaza el plazo, qué el presupuesto y qué vigilar

- `ValorGanado_01` [indicadores-eac, nivel 1] — 9 actividades, un punto de control (semana 10), EAC/VAC/TCPI
- `ValorGanado_02` [earned-schedule, nivel 2] — 8 actividades, un punto de control (semana 6), programación ganada
- `ValorGanado_03` [tendencia, nivel 2] — 9 actividades, tres puntos de control (semanas 5, 10, 14)
- `ValorGanado_04` [coste-bien-plazo-mal, nivel 2] — 9 actividades; punto de control en la semana 11: coste bien plazo mal
- `ValorGanado_05` [plazo-bien-coste-mal, nivel 2] — 9 actividades; punto de control en la semana 10: plazo bien coste mal
- `ValorGanado_06` [diagnostico, nivel 2] — 8 actividades; resolución con CPI y SPI intercambiados
- `ValorGanado_07` [inverso-desde-curvas, nivel 2] — 9 actividades (ocultas); curvas S hasta la semana 13 de 18
- `ValorGanado_08` [coste-no-uniforme, nivel 3] — 7 actividades; perfiles de desembolso no uniformes; control en la semana 8
- `ValorGanado_09` [vigilar, nivel 2] — 9 actividades; qué vigilar cruzando SPI/CPI por actividad con la holgura
- `ValorGanado_10` [indicadores-eac, nivel 1] — 8 actividades, un punto de control (semana 10), EAC/VAC/TCPI
- `ValorGanado_11` [earned-schedule, nivel 2] — 9 actividades, un punto de control (semana 10), programación ganada
- `ValorGanado_12` [tendencia, nivel 2] — 8 actividades, tres puntos de control (semanas 5, 9, 13)
- `ValorGanado_13` [coste-bien-plazo-mal, nivel 2] — 8 actividades; punto de control en la semana 10: coste bien plazo mal
- `ValorGanado_14` [plazo-bien-coste-mal, nivel 2] — 8 actividades; punto de control en la semana 11: plazo bien coste mal
- `ValorGanado_15` [diagnostico, nivel 2] — 8 actividades; resolución con CPI y SPI intercambiados
- `ValorGanado_16` [inverso-desde-curvas, nivel 2] — 9 actividades (ocultas); curvas S hasta la semana 14 de 20
- `ValorGanado_17` [coste-no-uniforme, nivel 3] — 8 actividades; perfiles de desembolso no uniformes; control en la semana 8
- `ValorGanado_18` [vigilar, nivel 2] — 9 actividades; qué vigilar cruzando SPI/CPI por actividad con la holgura

## 13_formula_compleccion — La fórmula del tiempo de compleción τ = ‖R·t‖∞

Matriz de rutas R, τ como norma infinito de R·t, gradiente ∂τ/∂t, descomposición en valores singulares de R, núcleo de la matriz de incidencia e instancias de referencia.

Variantes:
  - `tau-norma-infinito` (nivel 1): R, R·t y τ = ‖R·t‖∞; equivalencia con Zaderenko; τ como función lineal a trozos de una duración
  - `gradiente` (nivel 2): ∂τ/∂t con rutas críticas empatadas: subgradiente, diferencias finitas y suma de sensibilidades
  - `svd-rango-1` (nivel 3): SVD de R: σ₁/σ₂, rankings por u₁ y v₁, aproximación de rango 1 de τ y precisión frente al conjunto crítico
  - `nucleo-incidencia` (nivel 3): matriz de incidencia, R·H = 0 y perturbaciones de las duraciones que no cambian τ
  - `patterson` (nivel 2): una instancia de referencia (Patterson, formato RCP): τ frente a CPM y ranking SVD
  - `svd-vs-montecarlo` (nivel 3): índice de criticidad Monte Carlo frente a |v₁|: correlación y efecto de la incertidumbre
  - `editar-R` (nivel 2): añadir una prelación: filas de R que aparecen y desaparecen (prefijos × sufijos), nuevo τ y monotonía
  - `casi-criticas` (nivel 2): vector de márgenes τ·1 − R·t: rutas casi críticas, actividades que las forman, menor retraso que cambia la crítica y sensibilidad

- `Formula_01` [tau-norma-infinito, nivel 1] — 7 actividades, 4 rutas; τ como norma infinito
- `Formula_02` [gradiente, nivel 2] — 9 actividades, 2 rutas críticas; gradiente con empates
- `Formula_03` [svd-rango-1, nivel 3] — 10 actividades, 6 rutas; SVD de R y ranking frente al conjunto crítico
- `Formula_04` [nucleo-incidencia, nivel 3] — 8 actividades, 9 nodos; incidencia y núcleo
- `Formula_05` [patterson, nivel 2] — instancia pat1 de Patterson (14 actividades)
- `Formula_06` [svd-vs-montecarlo, nivel 3] — 10 actividades, 7 rutas; criticidad Monte Carlo frente a |v₁|
- `Formula_07` [editar-R, nivel 2] — 7 actividades; añadir G → D cambia τ de 21 a 29
- `Formula_08` [casi-criticas, nivel 2] — 8 actividades; 2 rutas con margen ≤ 2
- `Formula_09` [tau-norma-infinito, nivel 1] — 7 actividades, 4 rutas; τ como norma infinito
- `Formula_10` [gradiente, nivel 2] — 8 actividades, 2 rutas críticas; gradiente con empates
- `Formula_11` [svd-rango-1, nivel 3] — 10 actividades, 6 rutas; SVD de R y ranking frente al conjunto crítico
- `Formula_12` [nucleo-incidencia, nivel 3] — 7 actividades, 7 nodos; incidencia y núcleo
- `Formula_13` [patterson, nivel 2] — instancia pat20 de Patterson (22 actividades)
- `Formula_14` [svd-vs-montecarlo, nivel 3] — 10 actividades, 11 rutas; criticidad Monte Carlo frente a |v₁|
- `Formula_15` [editar-R, nivel 2] — 7 actividades; añadir E → F cambia τ de 29 a 32
- `Formula_16` [casi-criticas, nivel 2] — 9 actividades; 2 rutas con margen ≤ 2

## 14_comprension — Comprensión conceptual

Diseñar redes con rasgos dados, acotar la duración con datos incompletos, afirmaciones verdadero/falso con contraejemplo, programaciones que violan las prelaciones y cuestionarios tipo test exportables a Moodle (GIFT).

Variantes:
  - `disena-una-red` (nivel 2): diseñar un proyecto con rasgos dados (iniciales, ficticias, rutas críticas) y verificarlos
  - `datos-insuficientes` (nivel 2): dos duraciones en intervalos: cotas de τ, actividades siempre/nunca críticas y qué dato averiguar primero
  - `verdadero-falso` (nivel 2): seis afirmaciones sobre holguras, rutas críticas y ficticias; contraejemplos calculados
  - `comparar-planificaciones` (nivel 1): dos programaciones propuestas: violaciones de prelaciones, corrección mínima y margen restante
  - `tipo-test` (nivel 1): cuestionario de cinco preguntas con distractores razonados y exportación a GIFT (Moodle)

- `Comprension_01` [disena-una-red, nivel 2] — 6 actividades, 2 iniciales, 1 ficticias, 2 rutas críticas
- `Comprension_02` [datos-insuficientes, nivel 2] — 8 actividades; duraciones de D y C en intervalos; cotas de τ
- `Comprension_03` [verdadero-falso, nivel 2] — 6 afirmaciones; proyecto de apoyo con 8 actividades
- `Comprension_04` [comparar-planificaciones, nivel 1] — 8 actividades; dos planes con violaciones (E, F y F)
- `Comprension_05` [tipo-test, nivel 1] — 8 actividades; 5 preguntas de respuesta múltiple con exportación GIFT
- `Comprension_06` [disena-una-red, nivel 2] — 6 actividades, 2 iniciales, 1 ficticias, 2 rutas críticas
- `Comprension_07` [datos-insuficientes, nivel 2] — 9 actividades; duraciones de G y H en intervalos; cotas de τ
- `Comprension_08` [verdadero-falso, nivel 2] — 6 afirmaciones; proyecto de apoyo con 9 actividades
- `Comprension_09` [comparar-planificaciones, nivel 1] — 9 actividades; dos planes con violaciones (E, F y A)
- `Comprension_10` [tipo-test, nivel 1] — 9 actividades; 5 preguntas de respuesta múltiple con exportación GIFT

## 15_avanzado — Casos avanzados: PDM, calendario real y decisión económica

Relaciones de precedencia generalizadas (fin-comienzo, comienzo-comienzo, fin-fin, comienzo-fin) con desfases, traducción del calendario en días laborables a fechas reales con festivos, y elección de la duración que minimiza el coste total con indirectos, penalización y prima.

Variantes:
  - `pdm-desfases` (nivel 3): relaciones FC/CC/FF/CF con desfases: pasos adelante y atrás con candidatos, criticidad por solape, comparación con FC+0
  - `fechas-calendario` (nivel 2): del calendario en días laborables a fechas reales: fines de semana, festivos y efecto de un festivo nuevo
  - `decision-economica` (nivel 2): duración óptima con costes directos (Ackoff), indirectos, penalización y prima; análisis marginal

- `Avanzado_01` [pdm-desfases, nivel 3] — 8 actividades; 4 relaciones PDM no ordinarias (1 CC, 1 FF)
- `Avanzado_02` [fechas-calendario, nivel 2] — 9 actividades; inicio 2026-09-28, 2 festivos
- `Avanzado_03` [decision-economica, nivel 2] — 9 actividades; indirectos 8, plazo 28, penalización 18, prima 2
- `Avanzado_04` [pdm-desfases, nivel 3] — 8 actividades; 4 relaciones PDM no ordinarias (1 CC, 1 FF)
- `Avanzado_05` [fechas-calendario, nivel 2] — 8 actividades; inicio 2026-09-21, 2 festivos
- `Avanzado_06` [decision-economica, nivel 2] — 8 actividades; indirectos 8, plazo 22, penalización 11, prima 6

## 16_gantt_grafos — Gantt y grafos: ida y vuelta

Construir el diagrama de Gantt a partir del grafo PERT o del Roy (leyendo el cuadro de prelaciones del dibujo) y reconstruir el cuadro, el Roy y el PERT a partir de un Gantt con flechas de prelación.

Variantes:
  - `pert-a-gantt` (nivel 1): del PERT sin tiempos (más duraciones) al cuadro, el calendario y los Gantt temprano y tardío
  - `roy-a-gantt` (nivel 1): del Roy al cuadro, el calendario y los Gantt temprano y tardío; holgura libre como hueco, total como desplazamiento
  - `gantt-a-roy` (nivel 2): del Gantt con flechas al cuadro de prelaciones y al Roy; qué holguras se leen en el Gantt
  - `gantt-a-pert` (nivel 2): del Gantt con flechas al cuadro y al PERT con mínimo de ficticias; por qué el Gantt no las necesita

- `Gantt_01` [pert-a-gantt, nivel 1] — 7 actividades, 2 ficticias; del PERT al Gantt
- `Gantt_02` [roy-a-gantt, nivel 1] — 8 actividades, 2 ficticias; del Roy al Gantt
- `Gantt_03` [gantt-a-roy, nivel 2] — 8 actividades, 1 ficticias en PERT; del Gantt al Roy
- `Gantt_04` [gantt-a-pert, nivel 2] — 8 actividades, 1 ficticias en PERT; del Gantt al PERT
- `Gantt_05` [pert-a-gantt, nivel 1] — 7 actividades, 1 ficticias; del PERT al Gantt
- `Gantt_06` [roy-a-gantt, nivel 1] — 7 actividades, 1 ficticias; del Roy al Gantt
- `Gantt_07` [gantt-a-roy, nivel 2] — 7 actividades, 2 ficticias en PERT; del Gantt al Roy
- `Gantt_08` [gantt-a-pert, nivel 2] — 7 actividades, 1 ficticias en PERT; del Gantt al PERT
