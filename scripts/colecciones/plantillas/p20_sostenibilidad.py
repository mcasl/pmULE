"""Colección 20 — Sostenibilidad en el desarrollo de proyectos: situaciones típicas (compresión, compras,
residuos, tramitación ambiental, seguimiento, comunicación, condiciones de trabajo, energía en obra,
fin de vida, ventanas biológicas, movimiento de tierras, cláusulas sociales, elección de materiales,
horarios de obra, prefabricación, cuota de agua en sequía, pista de obra por un espacio protegido, retirada de
amianto) en las que alguien propone resolver el proyecto trasladando un impacto a
otros: al medio, a la comunidad, a los trabajadores de la cadena o al futuro. Cada caso tiene tres bloques de preguntas:
**A** (cálculo con las herramientas del curso: la magnitud del impacto y su coste), **B** (aplicar un
criterio de referencia enunciado en la lección 20 —paráfrasis propia que remite al principio de
sostenibilidad del PMBOK 8, a la ICB4, al GHG Protocol o a una norma legal— con respuesta determinada por
el criterio) y **C** (deliberación abierta, evaluada por rúbrica).

Los textos del PMI, de IPMA y del GHG Protocol no se reproducen (son obras protegidas); los criterios
están redactados por el curso y remiten a la fuente. Las cifras legales y los factores técnicos van en
CONSTANTES: los legales con su artículo; los factores de emisión son valores del enunciado (orientativos)
que en la práctica se toman de la fuente oficial del año."""
import contextlib
import io

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import proyectos as P
from ..cuaderno import (Variante, despachar, md, code, preguntas, apartado, fig_grafo,
                        literal_dataframe)

META = dict(slug="20_sostenibilidad", prefijo="Sostenibilidad",
            titulo="Sostenibilidad en el desarrollo de proyectos",
            descripcion="Situaciones típicas de un proyecto (comprimir el plazo, elegir equipo o proveedor, gestionar "
                        "los residuos de obra, tramitar la evaluación ambiental, informar de la huella, anunciar el "
                        "proyecto como «neutro», planificar en verano, cubrir picos de personal, dar energía a la obra, "
                        "cerrar el caso de negocio sin el fin de vida, desbrozar en época de cría, llevar las tierras al "
                        "vertedero, cumplir la cláusula social del contrato, elegir el hormigón, comprimir con turno de "
                        "noche, prefabricar o no, regar con la cuota de sequía, abrir una pista por un espacio protegido, "
                        "retirar una cubierta de amianto) en las que alguien propone trasladar un impacto al "
                        "medio, a la comunidad, a los trabajadores o al futuro. Cada caso pide (A) calcular la magnitud "
                        "del impacto y su coste con las herramientas del curso, (B) aplicar un criterio de referencia con "
                        "respuesta determinada (principio de sostenibilidad del PMBOK 8, GHG Protocol, ICB4 o una norma "
                        "legal, parafraseados en la lección 20) y (C) deliberar y redactar la comunicación.",
            contexto=False, n_defecto=5)

SEMILLA_MC = 2030
N_MC = 20000

# ---------------------------------------------------------------------------
# Criterios de referencia (redacción propia del curso; cada uno remite a su fuente, que no se reproduce)
# ---------------------------------------------------------------------------
CRITERIOS = {
    "S1": ("Jerarquía de respuesta a los impactos (pirámide de sostenibilidad)",
           "Ante un impacto negativo del proyecto las estrategias se ordenan de menos a más deseable: compensar o "
           "contrarrestar el daño, restaurarlo, minimizarlo y evitarlo. Compensar cuando se podía minimizar o evitar es "
           "quedarse en el escalón más bajo; clasificar una propuesta en su escalón es una pregunta con respuesta única.",
           "PMBOK 8, principio «integrar la sostenibilidad en todas las áreas del proyecto» (§3.7, figura 3-7)"),
    "S2": ("Externalidad internalizada",
           "Cuando la organización ha fijado un precio para una externalidad (precio sombra del carbono, canon, "
           "tasa), toda decisión de planificación —comprimir, comprar, diseñar— se compara por el coste total: coste del "
           "proyecto más el valor de la externalidad a ese precio. Con el precio fijado, la alternativa correcta es la de "
           "menor coste total, y comprar compensaciones a un precio inferior no cambia la cuenta.",
           "PMBOK 8 §3.7 (estrategias para abordar las externalidades); LCSP art. 148.2 (externalidades "
           "medioambientales monetizables); precio sombra del carbono del BEI"),
    "S3": ("Coste del ciclo de vida",
           "Dos alternativas se comparan por el coste de todo su ciclo de vida, descontado: adquisición, uso (energía, "
           "consumibles), mantenimiento, fin de vida y las externalidades que tengan precio; no por el precio de compra. "
           "Elegir por precio de adquisición cuando el uso domina el coste es un error de método, no una opinión.",
           "LCSP art. 148; PMBOK 8 §3.7.3 (finanzas: inversiones en sostenibilidad con ahorro a largo plazo)"),
    "S4": ("Orientación temporal: el proyecto responde de sus resultados",
           "La dirección del proyecto puede orientarse al proceso (hasta la entrega), a los entregables (su vida útil) "
           "o a los resultados (lo que dejan a la comunidad y al futuro). Un caso de negocio que termina en la puesta en "
           "servicio deja fuera costes y beneficios que pertenecen al proyecto (operación, desmantelamiento, residuos "
           "finales); si una obligación futura es consecuencia de la inversión, su valor actual forma parte del coste.",
           "PMBOK 8 §3.7.1 (figura 3-8); Plan General de Contabilidad, norma de valoración 2.ª (obligaciones de "
           "desmantelamiento)"),
    "S5": ("Metas de sostenibilidad en los documentos formales",
           "Si el acta de constitución, el caso de negocio o el contrato fijan un indicador de sostenibilidad "
           "(toneladas, kWh, residuos valorizados), se planifica, se mide y se informa como el plazo y el coste: contra "
           "lo planificado a la fecha y con previsión al cierre. Informar sólo el acumulado, sin la referencia, es dar un "
           "dato cierto que lleva a una conclusión equivocada (C2 y C3 de la lección 19).",
           "PMBOK 8 §3.7.1 (KPI de sostenibilidad en el acta de constitución y el caso de negocio)"),
    "S6": ("Alcances de las emisiones",
           "Las emisiones de un proyecto se clasifican por quién controla la fuente: alcance 1, combustión en fuentes "
           "propias o controladas (maquinaria y vehículos propios, generadores); alcance 2, electricidad comprada; "
           "alcance 3, el resto de la cadena (subcontratas, transporte ajeno, materiales comprados, viajes). A qué "
           "alcance pertenece cada fuente tiene respuesta determinada, y el alcance 3 no deja de contar por ser ajeno.",
           "GHG Protocol, estándar corporativo; UNE-EN ISO 14064-1; registro de huella de carbono del MITECO (RD 163/2014)"),
    "S7": ("Afirmaciones ambientales",
           "Una afirmación ambiental sobre un proyecto o producto debe ser específica y demostrable. Están prohibidas "
           "las afirmaciones genéricas («sostenible», «verde», «respetuoso con el clima») sin un desempeño reconocido "
           "que las respalde y las que presentan un producto como de impacto neutro, reducido o positivo sobre el clima "
           "basándose en la compensación de emisiones. Se puede decir cuánto se ha reducido y cuánto se compensa, con "
           "cifras; no se puede decir «neutro» por haber comprado créditos.",
           "Directiva (UE) 2024/825 (anexo I de la Directiva 2005/29/CE, aplicable desde el 27-9-2026); cf. C1–C3 de "
           "la lección 19"),
    "S8": ("Jerarquía de residuos y separación en origen",
           "El orden es prevención, preparación para la reutilización, reciclado, otra valorización y, en último lugar, "
           "eliminación. Los residuos de construcción y demolición no peligrosos se clasifican, preferentemente en la "
           "obra, en al menos seis fracciones (madera, minerales, metales, vidrio, plástico y yeso) y la demolición es "
           "selectiva, previo estudio de las cantidades de cada fracción. Mezclarlo todo en un contenedor que va a "
           "vertedero es elegir el último escalón e incumplir la obligación de clasificar.",
           "Ley 7/2022 de residuos y suelos contaminados, arts. 8 y 30; RD 105/2008 (estudio y plan de gestión de RCD)"),
    "S9": ("Evaluación ambiental del conjunto",
           "El procedimiento de evaluación ambiental que corresponde a un proyecto se determina por sus magnitudes "
           "totales: los proyectos que, presentándose fraccionados, alcanzan por acumulación los umbrales del anexo I "
           "se evalúan como uno solo, con el procedimiento ordinario. Fraccionar para quedar bajo el umbral no cambia el "
           "procedimiento exigible; cambia el riesgo de nulidad y el plazo (cf. C7 de la lección 19).",
           "Ley 21/2013 de evaluación ambiental, arts. 7.1.a) y 7.2.d) y anexos I y II"),
    "S10": ("Condiciones de trabajo en la cadena de ejecución",
            "El plan no puede apoyarse en condiciones de trabajo que la norma prohíbe: ni en una cadena de "
            "subcontratación más larga que la permitida, ni en jornadas al aire libre que ignoren los avisos por calor "
            "extremo. Una ventaja de coste o plazo obtenida así no es una opción de la tabla (cf. C7 y C5 de la "
            "lección 19).",
            "Ley 32/2006 de subcontratación en la construcción, arts. 4 y 5; RD 486/1997, disposición adicional única "
            "(RDL 4/2023); ICB4, Perspectiva 3 «salud, seguridad y medio ambiente»"),
    "S11": ("Riesgo de sostenibilidad",
            "Una amenaza ambiental, climática, regulatoria o social que puede cambiar el plazo, el coste o el resultado "
            "del proyecto es un riesgo del proyecto: entra en el registro con su probabilidad y su impacto y se "
            "cuantifica en el cronograma como cualquier otro. Dejarlo fuera «porque es el clima» o «porque es de la "
            "administración» es retener información (C4 de la lección 19).",
            "PMBOK 8 §3.7.1 (riesgos de sostenibilidad) y §3.7.3 (dominio de desempeño del riesgo)"),
    "S12": ("Condiciones de la autorización como restricciones del cronograma",
            "Las condiciones de la declaración de impacto ambiental, de la licencia o de la ordenanza (ventanas "
            "biológicas, horarios de trabajo, límites de ruido) son parte del proyecto autorizado: entran en el "
            "cronograma como restricciones de fecha u horario de las actividades afectadas, no como riesgos que se "
            "aceptan ni como opciones de compresión. Qué actividades quedan afectadas lo decide el calendario, y una "
            "ventaja de plazo obtenida incumpliendo la condición no es una opción de la tabla; incumplirla es infracción.",
            "Ley 21/2013, arts. 41 y 55.3.c); Ley 42/2007, art. 54.5; Ley 37/2003, arts. 6 y 28; PMBOK 8 §3.7.3 "
            "(dominio del cronograma: huella y uso de recursos en las decisiones de programación)"),
    "S13": ("Materiales excavados: reutilizar en la obra es prevención",
            "Los suelos no contaminados y otros materiales naturales excavados que se reutilizan en su estado natural en "
            "la misma obra no son residuo; los que salen de la obra sí lo son (salvo valorización autorizada) y entran "
            "en la jerarquía por el escalón que les corresponda. Reutilizar en la obra es prevención, el primer escalón; "
            "si la reutilización es posible sin acopio o sin retraso lo decide la secuencia del cronograma (las fechas y "
            "holguras de la excavación y del relleno), que tiene respuesta determinada.",
            "Ley 7/2022, arts. 3.2.b) y 8; Orden APM/1007/2017 (valorización de materiales naturales excavados en otra "
            "obra); PMBOK 8 §3.7 (figura 3-7)"),
    "S14": ("Condiciones especiales de ejecución del contrato",
            "Una condición especial de ejecución de tipo social o ambiental fijada en el pliego es una obligación del "
            "contrato, no una recomendación: se planifica como el alcance, se verifica con documentación y su "
            "incumplimiento lleva penalidad o, si el pliego la califica de esencial, resolución. Se exige igualmente a "
            "los subcontratistas, de modo que trasladar la obligación a la cadena no la cumple ni la elimina. Si el plan "
            "la cumple o no es un cálculo (operarios-semana, kWh, toneladas) con los datos del enunciado.",
            "LCSP (Ley 9/2017), art. 202.1 a 202.4; ICB4, Perspectiva 3 «cumplimiento, estándares y regulaciones»"),
    "S15": ("Prestaciones, no costumbre",
            "El pliego especifica el material por prestaciones (resistencia característica, clase de exposición, "
            "durabilidad) y plazos; entre materiales que las cumplen se decide por el coste total con la externalidad "
            "(S2) y por el ciclo de vida (S3), y el efecto de cada uno en el cronograma (plazos de curado, de "
            "suministro) se calcula, no se supone. «Siempre se ha hecho así» no es un criterio del pliego, y una huella "
            "menor no autoriza a aceptar un material que no cumple las prestaciones.",
            "Código Estructural (RD 470/2021), tipificación del hormigón y anejo 4 (cementos); LCSP art. 126 "
            "(prescripciones técnicas por rendimiento o exigencias funcionales); PMBOK 8 §3.7.3 (dominio de recursos: "
            "materiales y prácticas sostenibles)"),
    "S16": ("Cuotas de recursos impuestas por la autoridad",
            "Cuando la autoridad limita un recurso (agua en sequía, potencia, ocupación de vía pública, horas de "
            "maquinaria), la cuota es un límite del plan como lo es el número de operarios: se nivela dentro de las "
            "holguras y, si no cabe, se asigna con límite (lección 11) o se compra el recurso fuera, con su coste. "
            "Obtener el recurso saltándose la limitación es infracción, no una opción de la tabla; y si el plan cabe o "
            "no en la cuota tras nivelar lo dice el diagrama de cargas, no la opinión.",
            "Texto refundido de la Ley de Aguas (RDL 1/2001), arts. 55.2, 58 y 116.3.b) y c); bando municipal de "
            "sequía (Ley 7/1985, arts. 25.2.c) y 26.1.a), abastecimiento); PMBOK 8 §3.7.3 (dominio de recursos)"),
    "S17": ("Actividades exigidas por una norma: tienen duración y van en el cronograma",
            "Un trámite o trabajo que la norma exige antes de otro (un plan de trabajo aprobado por la autoridad "
            "laboral, la retirada de un material peligroso por una empresa inscrita, la evaluación de una modificación "
            "del proyecto) es una actividad del cronograma con su duración legal y su prelación. Omitirla no acorta el "
            "proyecto: lo deja sin autorización. Cuántas semanas de retraso causa y con cuánta antelación habría que "
            "lanzarla lo dice el calendario (lección 05), y es una cuenta, no una opinión (cf. C7 de la lección 19).",
            "RD 396/2006, arts. 11, 12.2 y 17.1; Ley 21/2013, art. 7.2.c); PMBOK 8 §3.7.3 (dominio del cronograma)"),
    "S18": ("Alternativas antes que compensación en la Red Natura 2000",
            "Un proyecto o una modificación que pueda afectar de forma apreciable a un espacio de la Red Natura 2000 "
            "se somete a una adecuada evaluación de sus repercusiones y sólo se autoriza si no perjudica la integridad "
            "del lugar; con conclusiones negativas, las medidas compensatorias sólo caben a falta de soluciones "
            "alternativas y por razones imperiosas de interés público de primer orden. Si existe una alternativa (la "
            "que el proyecto evaluado ya tenía), la opción que daña el lugar no se autoriza aunque se ofrezca restaurar "
            "o compensar: en la pirámide, evitar está por encima de restaurar y de compensar, y aquí la norma lo exige.",
            "Ley 42/2007, art. 46.4 y 46.5; Ley 21/2013, art. 7.2.c) (modificaciones de proyectos evaluados); PMBOK 8 "
            "§3.7 (figura 3-7)"),
}

# Cifras legales y factores técnicos: paráfrasis + referencia. Comprobadas en fuentes secundarias el 20-10-2026;
# verificar las legales en el BOE/EUR-Lex consolidados y los factores en la fuente oficial del año antes de un examen.
CONSTANTES = {
    "fracciones_rcd": dict(valor=6, unidad="fracciones",
                           texto="los residuos de construcción y demolición no peligrosos se clasifican, preferentemente en "
                                 "la obra, en al menos seis fracciones (madera, minerales, metales, vidrio, plástico y yeso) "
                                 "desde el 1 de julio de 2022 (art. 30.2); desde el 1 de enero de 2024 la demolición se lleva "
                                 "a cabo preferiblemente de forma selectiva —y con carácter obligatorio en los supuestos que se "
                                 "desarrollen reglamentariamente—, garantizando la retirada de al menos esas fracciones, previo "
                                 "estudio de las cantidades de cada una (art. 30.3)",
                           ref="Ley 7/2022, art. 30.2 y 30.3"),
    "impuesto_vertido": dict(valor=(10, 1.5), unidad="€ por tonelada",
                             texto="impuesto estatal al depósito en vertedero: en vertedero de residuos no peligrosos, 10 € "
                                   "por tonelada para los residuos no municipales que han pasado por tratamiento previo (15 € "
                                   "si están eximidos de él); si la parte inerte supera el 75 %, esa parte paga 1,5 € por "
                                   "tonelada (3 € si eximidos); en vertedero de inertes, 1,5 € por tonelada; el residuo que se "
                                   "recicla o valoriza no paga el impuesto",
                             ref="Ley 7/2022, art. 93.1"),
    "eia_umbrales": dict(valor=None, unidad="",
                         texto="evaluación de impacto ambiental ordinaria (anexo I) para parques eólicos con 50 o más "
                               "aerogeneradores, más de 30 MW o a menos de 2 km de otro parque, y para plantas solares en "
                               "suelo de más de 100 ha; simplificada (anexo II) para los parques eólicos no incluidos en el "
                               "anexo I y las plantas solares en suelo de 5 ha o más. Los proyectos que, presentándose "
                               "fraccionados, alcanzan por acumulación los umbrales de un anexo se evalúan por ese anexo",
                         ref="Ley 21/2013, art. 7.1.a) y 7.2.d), anexos I (grupo 3) y II (grupo 4), redacción del RD 445/2023"),
    "subcontratacion": dict(valor=3, unidad="niveles de subcontratación",
                            texto="en las obras de construcción el contratista puede subcontratar con un primer "
                                  "subcontratista, este con un segundo y este con un tercero; el tercer subcontratista, los "
                                  "trabajadores autónomos y las empresas cuya aportación a la obra consista básicamente en "
                                  "mano de obra no pueden subcontratar. Excepcionalmente (caso fortuito, "
                                  "especialización, fuerza mayor) la dirección facultativa puede aprobar un nivel más, que "
                                  "se anota en el Libro de Subcontratación. Las empresas que se dedican habitualmente a "
                                  "contratar o subcontratar obras deben tener al menos un 30 % de plantilla indefinida",
                            ref="Ley 32/2006, arts. 5.2 (letras d, e y f), 5.3 y 4.4"),
    "calor_extremo": dict(valor=None, unidad="",
                          texto="en el trabajo al aire libre el empresario adopta las medidas que resulten de la evaluación "
                                "de riesgos frente a los fenómenos meteorológicos adversos, incluidas las temperaturas "
                                "extremas; cuando la AEMET emite un aviso de nivel naranja o rojo y las medidas preventivas "
                                "no garantizan la protección, es obligatorio adaptar las condiciones de trabajo, incluida la "
                                "reducción o modificación de las horas de la jornada, hasta prohibir determinadas tareas en "
                                "las horas en que concurre el fenómeno",
                          ref="RD 486/1997, disposición adicional única (añadida por el RDL 4/2023, de 11 de mayo)"),
    "afirmaciones_ambientales": dict(valor=None, unidad="",
                                     texto="prohibidas las afirmaciones ambientales genéricas sin un desempeño ambiental "
                                           "excelente reconocido que las respalde, y las afirmaciones de que un producto "
                                           "tiene un impacto neutro, reducido o positivo sobre el clima basadas en la "
                                           "compensación de emisiones; aplicable desde el 27 de septiembre de 2026",
                                     ref="Directiva (UE) 2024/825, que añade los puntos 4 bis (afirmación genérica) y 4 quater "
                                         "(impacto neutro, reducido o positivo basado en compensaciones) al anexo I de la "
                                         "Directiva 2005/29/CE; transposición antes del 27-3-2026"),
    "cbam": dict(valor=50, unidad="toneladas por importador y año",
                 texto="desde el 1 de enero de 2026 la importación de hierro y acero, aluminio, cemento y fertilizantes "
                       "paga certificados del mecanismo de ajuste en frontera por carbono (CBAM) por las emisiones "
                       "implícitas de la mercancía, al precio del régimen de comercio de derechos de emisión (ETS); los "
                       "importadores de hasta 50 toneladas al año quedan exentos. A efectos del caso se toma el coste "
                       "simplificado emisiones implícitas × precio del ETS (la liquidación real descuenta la asignación "
                       "gratuita que aún reciben los productores europeos y el precio del carbono pagado en origen)",
                 ref="Reglamento (UE) 2023/956, modificado por el Reglamento (UE) 2025/2083"),
    "ciclo_de_vida": dict(valor=None, unidad="",
                          texto="el coste del ciclo de vida comprende los costes de adquisición, de utilización (energía y "
                                "otros recursos), de mantenimiento y de fin de vida (recogida y reciclado), y puede "
                                "incluir las externalidades medioambientales (emisiones, contaminación) cuando su valor "
                                "monetario pueda determinarse y verificarse",
                          ref="Ley 9/2017 de Contratos del Sector Público, art. 148"),
    "desmantelamiento": dict(valor=None, unidad="",
                             texto="forma parte del precio de adquisición de un inmovilizado la estimación inicial del valor "
                                   "actual de las obligaciones de desmantelamiento o retiro y de rehabilitación del lugar, "
                                   "cuando constituyan obligaciones incurridas como consecuencia de su uso",
                             ref="Plan General de Contabilidad (RD 1514/2007), norma de registro y valoración 2.ª, apartado 1"),
    "alcances": dict(valor=(1, 2, 3), unidad="alcances",
                     texto="alcance 1: emisiones directas de fuentes propias o controladas (combustible de maquinaria y "
                           "vehículos propios, generadores en obra); alcance 2: emisiones indirectas de la electricidad "
                           "comprada; alcance 3: las demás emisiones indirectas de la cadena de valor (materiales comprados, "
                           "transporte contratado, subcontratas, residuos, viajes)",
                     ref="GHG Protocol, Corporate Accounting and Reporting Standard; UNE-EN ISO 14064-1"),
    "factores": dict(valor=dict(gasoleo=2.47, electricidad=0.15, camion=0.10, barco=0.015), unidad="kg CO₂ por litro, kWh o t·km",
                     texto="factores de emisión que usa el enunciado: gasóleo 2,47 kg CO₂ por litro; electricidad de red "
                           "0,15 kg CO₂ por kWh (mix de comercializadora genérico); transporte por carretera 0,10 y marítimo "
                           "0,015 kg CO₂ por tonelada y kilómetro. Son valores orientativos: en un caso real se toman los "
                           "factores oficiales del año",
                     ref="calculadora de huella de carbono del MITECO (factores anuales); CNMC (mix de comercializadoras)"),
    "dia_condiciones": dict(valor=None, unidad="",
                            texto="la declaración de impacto ambiental fija las condiciones en que el proyecto puede ejecutarse "
                                  "(entre ellas, las medidas preventivas y correctoras y los calendarios de obra que protegen a "
                                  "la fauna en reproducción) y se incorpora a la autorización; incumplir esas condiciones es "
                                  "infracción grave. Con independencia de la DIA, está prohibido dañar, molestar o inquietar "
                                  "intencionadamente a los animales silvestres y destruir sus nidos, crías o huevos",
                            ref="Ley 21/2013, arts. 41 y 55.3.c); Ley 42/2007, art. 54.5"),
    "suelos_excavados": dict(valor=None, unidad="",
                             texto="quedan fuera de la ley de residuos los suelos excavados no contaminados y otros materiales "
                                   "naturales excavados en actividades de construcción cuando se tenga la certeza de que se "
                                   "utilizarán con fines de construcción en su estado natural en el lugar u obra donde fueron "
                                   "extraídos; los que salen de la obra son residuo (tierras a vertedero de inertes) salvo que "
                                   "se valoricen conforme a la orden ministerial de valorización de materiales naturales excavados",
                             ref="Ley 7/2022, art. 3.2.b); Orden APM/1007/2017"),
    "condiciones_ejecucion": dict(valor=None, unidad="",
                                  texto="el pliego debe incluir al menos una condición especial de ejecución de tipo "
                                        "medioambiental, social o relativa a la innovación (por ejemplo, combatir el paro, en "
                                        "particular el juvenil o el de larga duración, o la igualdad en el empleo); los pliegos "
                                        "pueden fijar penalidades por su incumplimiento o calificarlas de obligaciones "
                                        "contractuales esenciales a efectos de resolución, y todas se exigen igualmente a los "
                                        "subcontratistas que participen en la ejecución",
                                  ref="Ley 9/2017 (LCSP), art. 202.1, 202.2, 202.3 y 202.4"),
    "ruido": dict(valor=(600, 12000, 300000), unidad="€ (máximo de la multa por infracción leve, grave y muy grave)",
                  texto="los ayuntamientos regulan por ordenanza los horarios y límites de las obras ruidosas y pueden "
                        "autorizar excepciones motivadas; superar los valores límite o incumplir las condiciones de la "
                        "autorización es infracción grave (601 a 12 000 €) o muy grave si causa daño o riesgo grave para la "
                        "salud (12 001 a 300 000 €), con posible suspensión de la actividad",
                  ref="Ley 37/2003 del Ruido, arts. 6, 28 y 29; RD 1367/2007 (objetivos de calidad acústica y valores límite)"),
    "hormigon_prestaciones": dict(valor=None, unidad="",
                                  texto="el hormigón estructural se tipifica por prestaciones (resistencia característica a 28 "
                                        "días, consistencia, tamaño máximo del árido y clase de exposición) y el cemento se elige "
                                        "entre los admitidos para la aplicación; los cementos con adiciones (escoria, cenizas, "
                                        "puzolanas) tienen menos clínker y menos emisiones por tonelada y adquieren la resistencia "
                                        "más despacio, lo que alarga los plazos de desencofrado y puesta en carga, sobre todo en "
                                        "tiempo frío",
                                  ref="Código Estructural (RD 470/2021), tipificación del hormigón y anejo 4; UNE-EN 197-1"),
    "sequia": dict(valor=None, unidad="",
                   texto="en situaciones de sequía el organismo de cuenca puede condicionar o limitar el uso del dominio "
                         "público hidráulico y el Gobierno adoptar por real decreto medidas excepcionales aunque afecten a "
                         "concesiones; derivar agua sin concesión o autorización e incumplir las condiciones de las concesiones "
                         "y autorizaciones son infracciones administrativas; el abastecimiento de agua es competencia "
                         "municipal y el ayuntamiento fija por bando las restricciones de consumo y, en el reglamento del "
                         "servicio, las sanciones por conexiones no autorizadas (hidrantes, tomas fraudulentas)",
                   ref="TRLA (RDL 1/2001), arts. 55.2, 58 y 116.3.b) y c); Ley 7/1985, arts. 25.2.c) y 26.1.a)"),
    "amianto": dict(valor=45, unidad="días",
                    texto="antes de empezar cualquier trabajo con riesgo de exposición al amianto (la retirada de cubiertas de "
                          "fibrocemento incluida) el empresario elabora un plan de trabajo y lo presenta a la autoridad "
                          "laboral, que resuelve en 45 días desde la entrada en su registro (transcurridos sin resolución "
                          "expresa, el plan se entiende aprobado); sólo pueden hacerlo empresas inscritas en el Registro de "
                          "Empresas con Riesgo por Amianto (RERA); los residuos con amianto son peligrosos (LER 17 06 05*) "
                          "y se embalan, etiquetan y entregan a gestor autorizado sin mezclarlos con otros residuos",
                    ref="RD 396/2006, arts. 11, 12.2 y 17.1; Ley 7/2022, arts. 20 y 21; Decisión 2014/955/UE (lista "
                        "europea de residuos)"),
    "natura": dict(valor=None, unidad="",
                   texto="cualquier plan, programa o proyecto que pueda afectar de forma apreciable a un espacio de la Red "
                         "Natura 2000 se somete a una adecuada evaluación de sus repercusiones y sólo se autoriza tras "
                         "asegurarse de que no causará perjuicio a la integridad del lugar; si, pese a conclusiones negativas "
                         "y a falta de soluciones alternativas, debe realizarse por razones imperiosas de interés público de "
                         "primer orden, se adoptan medidas compensatorias. Modificar un proyecto ya evaluado de modo que "
                         "pueda tener efectos adversos significativos exige, como mínimo, evaluación simplificada",
                   ref="Ley 42/2007, art. 46.4 y 46.5; Ley 21/2013, art. 7.2.c)"),
}

RUBRICA_C = ("La pregunta del bloque C es abierta: no tiene una respuesta única y se evalúa el **procedimiento**: "
             "(1) identificar a todos los afectados, incluidos los que no están en la mesa (la comunidad, los "
             "trabajadores de la cadena, quien opere o desmantele lo construido, las generaciones siguientes) y qué "
             "pierde o gana cada uno con las cifras del bloque A; (2) formular al menos dos alternativas además de la "
             "propuesta, situándolas en la pirámide (compensar, restaurar, minimizar, evitar); (3) someter cada "
             "alternativa a tres pruebas —legalidad (¿cumple la norma?), publicidad (¿la defendería ante el inspector, "
             "el cliente o la prensa?) y ciclo de vida (¿quién paga el impacto y cuándo?)—; (4) decidir y redactar la "
             "comunicación con las cifras. Una respuesta que sólo opina, sin cifras ni alternativas, no supera la rúbrica.")


def _silencio():
    return contextlib.redirect_stdout(io.StringIO())


def _criterios(ids, constantes=()):
    """Celda con los criterios de referencia del caso (paráfrasis propia + fuente) y, si procede, las cifras."""
    lineas = ["## Criterio de referencia",
              "",
              "Las preguntas del bloque **B** se responden aplicando los criterios siguientes, tomados de la lección 20 "
              "(redactados por el curso; cada uno remite a la fuente de la que procede, que no se reproduce). La respuesta "
              "se justifica citando el criterio y la fuente, no con una opinión.",
              ""]
    for c in ids:
        titulo, texto, fuente = CRITERIOS[c]
        lineas.append(f"- **{c} — {titulo}.** {texto} *(cf. {fuente}.)*")
    for l in constantes:
        r = CONSTANTES[l]
        lineas.append(f"- **Norma o dato de referencia — {r['texto']}** *({r['ref']}.)*")
    return md("\n".join(lineas))


def _mensaje(de, texto):
    return md(f"> **De: {de}**\n> {texto}")


def _rubrica():
    return md(RUBRICA_C)


def _gift(preguntas_test, titulo):
    """Celda que define el test del bloque B (lista de (pregunta, [correcta, distractores…], retroalimentación))
    y lo exporta a GIFT, como en las colecciones 14 y 19."""
    return code(preguntas_test + "\n"
                "import os\n"
                "def a_gift(preguntas_test, titulo, semilla=1):\n"
                "    r = np.random.default_rng(semilla)\n"
                "    lineas = [f'$CATEGORY: {titulo}', '']\n"
                "    for i, (q, ops, fb) in enumerate(preguntas_test, 1):\n"
                "        correcta = ops[0]\n"
                "        todas = [ops[j] for j in r.permutation(len(ops))]\n"
                "        cuerpo = ' '.join((f'={v}' if v == correcta else f'~{v}') for v in todas)\n"
                "        lineas.append(f'::{titulo} P{i}::{q} {{{cuerpo} ####{fb}}}')\n"
                "        lineas.append('')\n"
                "    return '\\n'.join(lineas)\n"
                f"gift = a_gift(preguntas_test, {titulo!r})\n"
                "os.makedirs(os.path.expanduser('~/temp'), exist_ok=True)\n"
                "with open(os.path.expanduser('~/temp/cuestionario_sostenibilidad.gift'), 'w', encoding='utf8') as f:\n"
                "    f.write(gift)\n"
                "print(gift)")


def _T(p, d):
    with _silencio():
        return int(p.duration(d))


def _prob(p, d, sd, plazo):
    with _silencio():
        R = p.path_matrix(dummies=False)
    L = R @ d.reindex(R.columns)
    V = R @ (sd.reindex(R.columns).fillna(0) ** 2)
    T = float(L.max()); sigma = float(np.sqrt(V[L == T].max()))
    return T, sigma, float(norm.cdf((plazo - T) / sigma))


def _ackoff(p, d, tope, cur, r):
    with _silencio():
        _, pasos, dur, _ = p.ackoff(d, tope, cur, reduction=r, report=False)
    coste = float(sum(cur[a] for acts in pasos.values() for a in acts))
    return pasos, dur.reindex(d.index), coste


def _cargas(info, df, etiqueta="resources"):
    with _silencio():
        g = info["proyecto"].copy()
        gantt_df, _ = g.gantt_cargas(df.copy(), "duration", etiqueta, report=False, tikz=False)
    return gantt_df.data.loc["Total"].drop("H_total", errors="ignore").astype(float)


def _nivelar(info, df, etiqueta="resources"):
    with _silencio():
        g = info["proyecto"].copy()
        desp, _, datos, gantt_df, _ = g.nivelar(df.copy(), "duration", etiqueta, report=False)
    return desp, gantt_df.data.loc["Total"].drop("H_total", errors="ignore").astype(float)


def _van(flujos, k):
    return float(sum(f / (1 + k) ** t for t, f in enumerate(flujos)))


def _cal(p, d):
    with _silencio():
        return p.calendar(d)


def _carga_filas(gantt_df, filas):
    """Carga semanal (Series por semana) de las filas indicadas de un Gantt de cargas (celdas vacías = 0)."""
    cols = [c for c in gantt_df.data.columns if c != "H_total"]
    filas = [f for f in filas if f in gantt_df.data.index]
    return gantt_df.data.loc[filas, cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()


def _elegir(rng, lista, k):
    """k elementos distintos de una lista (ordenada antes de consumir rng: determinista entre procesos)."""
    lista = sorted(lista)
    return sorted(str(x) for x in rng.choice(lista, size=min(k, len(lista)), replace=False))


SETUP = ("datos_proyecto = datos_enunciado.copy()\n"
         "prelaciones = PredecessorTable.from_dataframe_of_strings(datos_proyecto, activity='activity', predecessor='predecessor')\n"
         "proyecto = prelaciones.create_project()\n"
         "calendario = proyecto.calendar(datos_proyecto['duration'])\n"
         "duracion = int(proyecto.duration(datos_proyecto['duration']))\n")

ACEPTA_BASE = "1 <= i['n_dummies'] <= 4 and 4 <= i['n_rutas'] <= 9 and 12 <= i['T'] <= 22"


def _base(i, criticas_1=True, T=(12, 22)):
    return (1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 9 and T[0] <= i["T"] <= T[1]
            and (i["n_criticas"] == 1 or not criticas_1))


# ---------------------------------------------------------------------------
# Caso 1 — Comprimir el plazo con precio del carbono
# ---------------------------------------------------------------------------
def _compresion_con_carbono(rng):
    e = {}

    def acepta(i):
        if not _base(i):
            return False
        tope, cn, ct, cur = P.costes_reduccion(rng, i["D"], n_inf=0)
        cur = cur.astype(float)
        em = pd.Series(rng.choice([0, 4, 8, 12, 16, 20], size=len(i["D"])), index=i["D"].index, name="emisiones_reduccion").astype(float)
        if (em[tope < i["D"]] > 0).sum() < 2:
            return False
        r = int(rng.choice([2, 3]))
        precio = int(rng.choice([150, 250, 300]))
        cur_total = cur + em * precio / 1000
        try:
            pasos1, _, coste1 = _ackoff(i["proyecto"], i["D"].astype(float), tope, cur, r)
            pasos2, _, coste2 = _ackoff(i["proyecto"], i["D"].astype(float), tope, cur_total, r)
        except Exception:
            return False
        if len(pasos1) < r or len(pasos2) < r:
            return False
        acts1 = sorted(a for acts in pasos1.values() for a in acts)
        acts2 = sorted(a for acts in pasos2.values() for a in acts)
        if acts1 == acts2:
            return False
        em1 = float(sum(em[a] for a in acts1)); em2 = float(sum(em[a] for a in acts2))
        coste2_dinero = float(sum(cur[a] for a in acts2))
        tot1 = coste1 + em1 * precio / 1000; tot2 = coste2_dinero + em2 * precio / 1000
        if not (em2 <= em1 - 8 and coste2_dinero > coste1 and tot2 < tot1 - 0.4):
            return False
        e.update(tope=tope, cur=cur, em=em, r=r, precio=precio, acts1=acts1, acts2=acts2, coste1=coste1,
                 coste2=coste2_dinero, em1=em1, em2=em2, tot1=tot1, tot2=tot2, creditos=int(rng.choice([8, 10, 12])))
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    r, precio, q = e["r"], e["precio"], e["creditos"]
    df = P.tabla_base(info, duration=D, cap_duration=e["tope"], ucr=e["cur"], emisiones_reduccion=e["em"])
    celdas = [md("# Enunciado"),
              md("El cliente exige adelantar la entrega de un proyecto de montaje industrial **" + str(r) + " semanas**. "
                 "El cuadro de prelaciones, las duraciones (semanas), las duraciones mínimas alcanzables, el coste "
                 "unitario de reducción (`ucr`, k€ por semana) y las **emisiones adicionales** de cada semana que se "
                 "acorta (`emisiones_reduccion`, t CO₂ por semana: grupos electrógenos para el doble turno, transporte "
                 "aéreo de componentes, hormigón de fraguado rápido) se adjuntan."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"La organización tiene fijado en su política un **precio sombra del carbono de {precio} €/t CO₂** para "
                 "las decisiones de proyecto. En el mercado voluntario hay créditos de compensación a "
                 f"{q} €/t."),
              _mensaje("Director de operaciones",
                       f"Comprime por el método de siempre: Ackoff con los costes unitarios y listo, que es lo más "
                       f"barato. Las toneladas que salgan las compensamos con créditos a {q} €/t y el proyecto queda "
                       "«neutro». El precio sombra es para la memoria de sostenibilidad, no para la obra."),
              _criterios(["S2", "S1"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Obtenga con el algoritmo de Ackoff-Sasieni el plan de reducción de {r} semanas de menor coste monetario "
        "(sin precio del carbono): actividades acortadas en cada paso, coste y emisiones adicionales del plan.",
        f"(A) Repita el algoritmo con el coste unitario total (ucr + emisiones × {precio} €/t) y compare los dos "
        "planes: coste monetario, emisiones y coste total al precio sombra. ¿Cuánto costaría compensar las emisiones "
        f"de cada plan a {q} €/t y cuánto valen al precio sombra?",
        "(B) Según los criterios de referencia, ¿qué plan es el correcto y por qué el argumento «compensamos a "
        f"{q} €/t» no cambia la respuesta? ¿En qué escalón de la pirámide está cada propuesta?",
        "(C) ¿Qué propone al director de operaciones y qué ofrece al cliente? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"semanas_pedidas, precio_sombra, precio_creditos = {r}, {precio}, {q}\n" + SETUP +
                       "datos_proyecto['ucr_total'] = datos_proyecto['ucr'] + datos_proyecto['emisiones_reduccion'] * precio_sombra / 1000\n"
                       "def resumen_plan(pasos):\n"
                       "    acts = [a for paso in pasos.values() for a in paso]\n"
                       "    coste = float(datos_proyecto.loc[acts, 'ucr'].sum())\n"
                       "    emisiones = float(datos_proyecto.loc[acts, 'emisiones_reduccion'].sum())\n"
                       "    return pd.Series({'coste monetario (k€)': coste, 'emisiones adicionales (t CO₂)': emisiones,\n"
                       "                      'valor al precio sombra (k€)': emisiones * precio_sombra / 1000,\n"
                       "                      'coste total al precio sombra (k€)': coste + emisiones * precio_sombra / 1000,\n"
                       "                      'compensar a precio de créditos (k€)': emisiones * precio_creditos / 1000})"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Ackoff-Sasieni con el coste unitario monetario"\n#| label: tbl-ackoffDinero\n'
                       "tabla_1, pasos_1, duraciones_1, margen_1 = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr'], reduction=semanas_pedidas, report=False)\n"
                       "tabla_1"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso (coste monetario)"\n#| label: tbl-ackoffPaso1\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr'], pasos_1)"))
    celdas.append(code("plan_1 = resumen_plan(pasos_1)\n"
                       "display(plan_1.round(2).to_frame('plan 1: coste monetario').T)\n"
                       "Markdown('Plan 1: ' + '; '.join('paso ' + str(k + 1) + ': ' + ', '.join(v) for k, v in sorted(pasos_1.items())) + "
                       "f'''. Coste **{plan_1.iloc[0]:.0f} k€** y **{plan_1.iloc[1]:.0f} t CO₂** adicionales.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Ackoff-Sasieni con el coste unitario total (ucr + emisiones × precio sombra)"\n#| label: tbl-ackoffTotal\n'
                       "tabla_2, pasos_2, duraciones_2, margen_2 = proyecto.ackoff(\n"
                       "    durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "    costs=datos_proyecto['ucr_total'], reduction=semanas_pedidas, report=False)\n"
                       "tabla_2"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso (coste total)"\n#| label: tbl-ackoffPaso2\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr_total'], pasos_2)"))
    celdas.append(code('#| tbl-cap: "Comparación de los dos planes"\n#| label: tbl-planes\n'
                       "plan_2 = resumen_plan(pasos_2)\n"
                       "comparacion = pd.DataFrame({'plan 1 (ucr)': plan_1, 'plan 2 (ucr + carbono)': plan_2})\n"
                       "display(comparacion.round(2))\n"
                       "Markdown('Plan 2: ' + '; '.join('paso ' + str(k + 1) + ': ' + ', '.join(v) for k, v in sorted(pasos_2.items())) + "
                       "f'''. Cuesta **{plan_2.iloc[0] - plan_1.iloc[0]:.1f} k€ más** en dinero y emite **{plan_1.iloc[1] - plan_2.iloc[1]:.0f} t menos**; "
                       "al precio sombra el plan 2 es **{plan_1.iloc[3] - plan_2.iloc[3]:.1f} k€ más barato** en total. Compensar el plan 1 a "
                       "{precio_creditos} €/t costaría {plan_1.iloc[4]:.1f} k€; esas toneladas valen {plan_1.iloc[2]:.1f} k€ al precio sombra.''')"))
    celdas.append(md("El algoritmo es el mismo; cambia el coste unitario con el que se ordenan las opciones. Con el precio "
                     "del carbono dentro del coste, una semana «barata» en dinero pero cara en emisiones deja de ser la "
                     "primera opción, y el plan cambia de actividades."))
    celdas += apartado(3)
    celdas.append(md(f"**S2.** La organización ha fijado el precio de la externalidad ({precio} €/t), de modo que el "
                     f"criterio de decisión es el coste total: el plan 2 ({e['tot2']:.1f} k€) frente al plan 1 "
                     f"({e['tot1']:.1f} k€). El argumento de los créditos a {q} €/t sustituye el precio que la organización "
                     "ha decidido por el precio que a uno le conviene: si valiera, el precio sombra no serviría para nada. "
                     "Comprar créditos no reduce las emisiones del proyecto; las compensa, y a un precio que no es el de la "
                     "política. **S1.** El plan 1 más créditos está en el escalón más bajo de la pirámide (**compensar**); "
                     "el plan 2 **minimiza** las emisiones de la compresión; **evitar** sería negociar con el cliente las "
                     f"{r} semanas (lección 19, C13 y C10) o acortar por medios sin emisiones (solapes, más personal "
                     "en jornada ordinaria). La clasificación no es opinable: compensar es compensar."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Con un precio sombra del carbono fijado por la organización, el plan de compresión correcto es…', ['el de menor coste total (dinero + emisiones al precio sombra)', 'el de menor coste monetario, compensando después', 'el de menos emisiones a cualquier coste', 'el que pida el cliente'],\n"
        "     'S2: la externalidad con precio entra en el coste de la decisión.'),\n"
        "    ('Comprar créditos de compensación para las emisiones de un plan está, en la pirámide de la sostenibilidad del PMBOK 8, en el escalón…', ['más bajo: compensar', 'de minimizar', 'de restaurar', 'de evitar'],\n"
        "     'La figura 3-7 ordena compensar < restaurar < minimizar < evitar.'),\n"
        "    ('Incorporar el precio del carbono a Ackoff-Sasieni consiste en…', ['sumar al coste unitario de reducción las emisiones por semana multiplicadas por el precio', 'restar el precio de los créditos del coste unitario', 'eliminar las actividades con emisiones', 'cambiar el número de semanas pedidas'],\n"
        "     'El algoritmo no cambia; cambia el coste con el que ordena las opciones.'),\n"
        "]", "Sostenibilidad 20 — compresión con carbono"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de operaciones: el plan 2 con sus cifras (cuesta {e['coste2'] - e['coste1']:.1f} k€ más "
                     f"y emite {e['em1'] - e['em2']:.0f} t menos); el precio sombra es la política de la empresa y se aplica "
                     "en la obra o no es una política. Al cliente: el precio de cada semana con el coste total, y la "
                     "alternativa de no comprimir o comprimir menos. Prueba de publicidad: ¿pondría en la memoria de "
                     f"sostenibilidad que el proyecto se llamó «neutro» por {e['em1'] * q / 1000:.1f} k€ de créditos "
                     "mientras se elegía el plan que más emitía?"))
    return celdas, (f"adelantar {r} semanas: {e['coste1']:.0f} k€ y {e['em1']:.0f} t frente a {e['coste2']:.0f} k€ y "
                    f"{e['em2']:.0f} t con el carbono a {precio} €/t")


# ---------------------------------------------------------------------------
# Caso 2 — Elegir por precio de compra o por coste del ciclo de vida
# ---------------------------------------------------------------------------
def _coste_ciclo_de_vida(rng):
    quiero_A_sin_carbono = bool(rng.random() < 0.5)      # la mitad de los casos: A gana sin el carbono y B con él
    for intento in range(5000):
        I_A = int(rng.integers(8, 16)) * 10
        I_B = int(round(I_A * float(rng.choice([1.3, 1.4, 1.5, 1.6])) / 10)) * 10
        c_A = int(rng.integers(6, 26)) * 10                        # MWh/año
        c_B = int(round(c_A * float(rng.choice([0.6, 0.7, 0.8]))))
        m_A, m_B = int(rng.integers(4, 9)), int(rng.integers(2, 6))
        fin_A, fin_B = int(rng.integers(4, 10)), int(rng.integers(4, 10))
        N = int(rng.choice([10, 12, 15]))
        k = float(rng.choice([0.04, 0.06, 0.08]))
        e = float(rng.choice([0.12, 0.15, 0.18]))                  # k€/MWh (= €/kWh)
        f = 0.15                                                    # t CO₂/MWh
        p = int(rng.choice([100, 150, 250]))                        # €/t
        anualidad = (1 - (1 + k) ** -N) / k

        def lcc(I, c, m, fin, precio):
            return I + (c * e + m + c * f * precio / 1000) * anualidad + fin / (1 + k) ** N

        lcc_A0, lcc_B0 = lcc(I_A, c_A, m_A, fin_A, 0), lcc(I_B, c_B, m_B, fin_B, 0)
        lcc_A, lcc_B = lcc(I_A, c_A, m_A, fin_A, p), lcc(I_B, c_B, m_B, fin_B, p)
        if not (lcc_B < lcc_A - 2):
            continue
        if (lcc_A0 < lcc_B0 - 1) != quiero_A_sin_carbono and intento < 4000:
            continue
        delta_c = (c_A - c_B) * f / 1000 * anualidad                 # k€ por €/t de precio
        p_equilibrio = (lcc_B0 - lcc_A0) / delta_c                  # precio del carbono que iguala los LCC (puede ser < 0)
        break
    gana_sin_carbono = lcc_B0 < lcc_A0
    tabla = pd.DataFrame({"A (estándar)": [I_A, c_A, m_A, fin_A], "B (eficiente)": [I_B, c_B, m_B, fin_B]},
                         index=pd.Index(["inversión (k€)", "consumo eléctrico (MWh/año)", "mantenimiento (k€/año)",
                                         "retirada al fin de vida (k€)"], name="concepto"))
    celdas = [md("# Enunciado"),
              md("Un proyecto de reforma de un edificio público incluye la sustitución de la climatización. Hay dos "
                 "alternativas que cumplen las prestaciones: **A**, el equipo estándar, y **B**, el equipo de alta "
                 f"eficiencia. Los datos de cada una se adjuntan; la vida útil es de {N} años en los dos casos, la tasa de "
                 f"descuento del {k:.0%}, el precio de la electricidad de {e:.2f} €/kWh y el factor de emisión de la red "
                 f"de {f:.2f} t CO₂ por MWh. La organización aplica un precio sombra del carbono de {p} €/t."),
              code('#| tbl-cap: "Datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(tabla, indice="concepto")),
              _mensaje("Director de compras",
                       f"Adjudicamos por precio, como siempre: A cuesta {I_A} y B {I_B}, un {(I_B / I_A - 1):.0%} más. "
                       "Lo que consuma después no es de este proyecto, es de mantenimiento, y el carbono es un número de "
                       "la memoria, no dinero."),
              _criterios(["S3", "S2", "S4"], constantes=["ciclo_de_vida"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule el coste del ciclo de vida (valor actual) de cada alternativa sin el carbono: inversión, energía, "
        "mantenimiento y retirada. ¿Cuál es más barata y por cuánto?",
        "(A) Añada el carbono al precio sombra y recalcule. Obtenga el precio del carbono con el que las dos "
        "alternativas tendrían el mismo coste del ciclo de vida, y las emisiones de cada una en la vida útil.",
        "(B) Según los criterios de referencia, ¿qué alternativa se elige y con qué criterio? ¿Es correcto decir que el "
        "consumo «no es de este proyecto»?",
        "(C) ¿Cómo plantea la adjudicación al director de compras y al órgano de contratación? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"N, k, precio_energia, factor, precio_sombra = {N}, {k}, {e}, {f}, {p}\n"
                       "datos = datos_enunciado.copy()\n"
                       "anualidad = (1 - (1 + k) ** -N) / k\n"
                       "def ciclo_de_vida(col, precio_carbono):\n"
                       "    I, c, m, fin = datos[col]\n"
                       "    energia = c * precio_energia * anualidad\n"
                       "    mantenimiento = m * anualidad\n"
                       "    carbono = c * factor * precio_carbono / 1000 * anualidad\n"
                       "    retirada = fin / (1 + k) ** N\n"
                       "    return pd.Series({'inversión': I, 'energía (VA)': energia, 'mantenimiento (VA)': mantenimiento,\n"
                       "                      'retirada (VA)': retirada, 'carbono (VA)': carbono,\n"
                       "                      'coste del ciclo de vida': I + energia + mantenimiento + retirada + carbono})"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Coste del ciclo de vida sin el carbono (k€, valor actual)"\n#| label: tbl-lccSin\n'
                       "sin_carbono = pd.DataFrame({col: ciclo_de_vida(col, 0) for col in datos.columns})\n"
                       "display(sin_carbono.round(1))\n"
                       "dif0 = sin_carbono.loc['coste del ciclo de vida', 'A (estándar)'] - sin_carbono.loc['coste del ciclo de vida', 'B (eficiente)']\n"
                       "Markdown(f'''Factor de anualidad {anualidad:.3f} ({N} años al {k:.0%}). Sin el carbono, B es ''' + ('**más barata**' if dif0 > 0 else '**más cara**') + "
                       "f''' que A en el ciclo de vida: {abs(dif0):.1f} k€ de diferencia, aunque cueste {datos.iloc[0, 1] - datos.iloc[0, 0]:.0f} k€ más al comprarla.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Coste del ciclo de vida con el carbono al precio sombra (k€, valor actual)"\n#| label: tbl-lccCon\n'
                       "con_carbono = pd.DataFrame({col: ciclo_de_vida(col, precio_sombra) for col in datos.columns})\n"
                       "display(con_carbono.round(1))\n"
                       "dif = con_carbono.loc['coste del ciclo de vida', 'A (estándar)'] - con_carbono.loc['coste del ciclo de vida', 'B (eficiente)']\n"
                       "emisiones = datos.loc['consumo eléctrico (MWh/año)'] * factor * N\n"
                       "delta_c = (datos.iloc[1, 0] - datos.iloc[1, 1]) * factor / 1000 * anualidad\n"
                       "p_equilibrio = -dif0 / delta_c\n"
                       "Markdown(f'''Con el carbono a {precio_sombra} €/t, B es más barata en **{dif:.1f} k€**. Emisiones en {N} años: A {emisiones.iloc[0]:.0f} t, B {emisiones.iloc[1]:.0f} t. "
                       "Las dos alternativas se igualan con un precio del carbono de **{p_equilibrio:.0f} €/t**''' + "
                       "(' (negativo: B gana incluso sin carbono).' if p_equilibrio < 0 else '; por encima gana B.'))"))
    celdas += apartado(3)
    celdas.append(md("**S3.** El criterio es el coste del ciclo de vida, no el precio de adquisición: la LCSP (art. 148) "
                     "lo define con los mismos sumandos del apartado 1 y permite añadir las externalidades con precio. "
                     + (f"B es más barata ya sin el carbono ({lcc_A0 - lcc_B0:.1f} k€) y más aún con él "
                        f"({lcc_A - lcc_B:.1f} k€): " if gana_sin_carbono else
                        f"Sin el carbono A es más barata por {lcc_B0 - lcc_A0:.1f} k€, pero **S2** obliga a contar la "
                        f"externalidad al precio que la organización ha fijado, y con {p} €/t B gana por {lcc_A - lcc_B:.1f} k€ "
                        f"(el precio de equilibrio es {p_equilibrio:.0f} €/t): ")
                     + "la respuesta es **B**. **S4.** «El consumo no es de este proyecto» confunde el ciclo de vida del "
                     "proceso con el de los entregables: quien decide el equipo decide la factura de energía de "
                     f"{N} años, y la decisión se toma ahora. Que el coste lo pague otro presupuesto (mantenimiento) no lo "
                     "saca del cálculo; lo saca de la vista."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('El coste del ciclo de vida de un equipo, según la LCSP (art. 148), incluye…', ['adquisición, uso, mantenimiento, fin de vida y, si tienen precio, las externalidades', 'sólo el precio de adquisición', 'adquisición y mantenimiento del primer año', 'el precio más el IVA'],\n"
        "     'S3: todos los costes del ciclo, descontados.'),\n"
        "    ('Con un precio sombra del carbono fijado, la decisión entre dos equipos se toma…', ['por el coste del ciclo de vida incluyendo el carbono a ese precio', 'por el precio de compra', 'por las emisiones solamente', 'por el precio de los créditos de compensación'],\n"
        "     'S2: la externalidad con precio entra en la cuenta.'),\n"
        "    ('Decir que el consumo del equipo «no es de este proyecto» es…', ['orientarse sólo al proceso e ignorar el ciclo de vida de los entregables (S4)', 'correcto: el proyecto termina en la entrega', 'correcto si lo paga otro presupuesto', 'un criterio de la LCSP'],\n"
        "     'La decisión del equipo fija el coste de uso de toda la vida útil.'),\n"
        "]", "Sostenibilidad 20 — coste del ciclo de vida"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de compras: la tabla del apartado 2; B cuesta {I_B - I_A} k€ más hoy y "
                     f"{lcc_A - lcc_B:.0f} k€ menos en {N} años al precio sombra ({lcc_A0 - lcc_B0:.0f} k€ sin él). Al "
                     "órgano de contratación: que el pliego fije el coste del ciclo de vida como criterio de adjudicación "
                     "con los parámetros publicados (vida útil, tasa, precio de la energía, factor de emisión y precio del "
                     "carbono), que es lo que la ley permite y la política de la organización exige. Prueba del ciclo de "
                     f"vida: ¿quién paga los {c_A - c_B} MWh al año de diferencia durante {N} años?"))
    return celdas, (f"A {I_A} k€ frente a B {I_B} k€ de inversión; ciclo de vida {lcc_A:.0f} frente a {lcc_B:.0f} k€ con el "
                    f"carbono a {p} €/t "
                    + (f"(equilibrio {p_equilibrio:.0f} €/t)" if p_equilibrio > 0 else "(B gana ya sin el carbono)"))


# ---------------------------------------------------------------------------
# Caso 3 — Los residuos de la obra en un solo contenedor
# ---------------------------------------------------------------------------
FRACCIONES = ["hormigon", "ceramica", "madera", "metal", "yeso", "plastico"]
TARIFA_SEPARADO = dict(hormigon=6, ceramica=8, madera=25, metal=-120, yeso=45, plastico=60)      # € por t, gestor (negativo: ingreso)
TARIFA_MIXTO = 38                                                                                 # € por t, gestor de mezcla a vertedero
IMPUESTO_MIXTO, IMPUESTO_INERTE = 10, 1.5


def _residuos_de_obra(rng):
    e = {}

    def acepta(i):
        if not _base(i):
            return False
        cal = i["calendario"]
        criticas = [a for a in i["D"].index if cal.loc[a, "H_total"] == 0 and 3 <= int(i["D"][a]) <= 6]
        if not criticas:
            return False
        X = str(rng.choice(criticas))
        s = int(rng.choice([1, 2]))
        D2 = i["D"].copy(); D2[X] -= s
        T2 = _T(i["proyecto"], D2)
        if int(i["T"]) - T2 < 1:
            return False
        generan = list(rng.choice(list(i["D"].index), size=int(rng.integers(3, 5)), replace=False))
        if X not in generan:
            generan.append(X)
        res = pd.DataFrame(0.0, index=i["D"].index, columns=FRACCIONES)
        for a in generan:
            res.loc[a, "hormigon"] = int(rng.integers(15, 80))
            res.loc[a, "ceramica"] = int(rng.integers(5, 40))
            res.loc[a, "madera"] = int(rng.integers(1, 8))
            res.loc[a, "metal"] = int(rng.integers(1, 10))
            res.loc[a, "yeso"] = int(rng.integers(0, 12))
            res.loc[a, "plastico"] = int(rng.integers(0, 4))
        if (res.sum() == 0).any():
            return False
        e.update(X=X, s=s, T2=T2, res=res)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    X, s, res = e["X"], e["s"], e["res"]
    coste_semana = int(rng.choice([8, 10, 12, 15]))
    df = P.tabla_base(info, duration=D, **{c: res[c] for c in FRACCIONES})
    total = float(res.values.sum())
    inertes = float(res[["hormigon", "ceramica"]].values.sum())
    # Impuesto al vertido (Ley 7/2022, art. 93): 10 €/t; si la parte inerte supera el 75 %, esa parte paga 1,5 €/t
    inerte_mayoritario = inertes / total > 0.75
    impuesto_mixto_total = (inertes * IMPUESTO_INERTE + (total - inertes) * IMPUESTO_MIXTO) if inerte_mayoritario else total * IMPUESTO_MIXTO
    coste_mixto = (total * TARIFA_MIXTO + impuesto_mixto_total) / 1000
    coste_sep = float(sum(res[c].sum() * TARIFA_SEPARADO[c] for c in FRACCIONES)) / 1000
    valorizado_sep = float(res[["hormigon", "ceramica", "madera", "metal"]].values.sum())
    celdas = [md("# Enunciado"),
              md("Una obra de reforma integral comprende las actividades del cuadro adjunto (duraciones en semanas). "
                 f"La actividad **{X}** es la demolición selectiva: incluye desmontar, separar y acopiar por fracciones "
                 "lo que se retira. Las toneladas de residuo que genera cada actividad, por fracción, se adjuntan "
                 "(hormigón, cerámica, madera, metal, yeso y plástico; el estudio de gestión de residuos del proyecto "
                 "prevé que el hormigón y la cerámica vayan a planta de reciclado de áridos, la madera y el metal a "
                 "valorización y el yeso y el plástico a gestor específico)."),
              code('#| tbl-cap: "Cuadro de datos del enunciado (residuos en toneladas)"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md("Tarifas del gestor (€ por tonelada, impuesto no incluido): hormigón "
                 f"{TARIFA_SEPARADO['hormigon']}, cerámica {TARIFA_SEPARADO['ceramica']}, madera {TARIFA_SEPARADO['madera']}, "
                 f"metal {TARIFA_SEPARADO['metal']} (se cobra), yeso {TARIFA_SEPARADO['yeso']}, plástico "
                 f"{TARIFA_SEPARADO['plastico']}; residuo mezclado a vertedero de no peligrosos, {TARIFA_MIXTO} €/t más el "
                 "impuesto al depósito (a efectos del impuesto, hormigón y cerámica son la parte inerte del contenedor). El "
                 f"coste semanal del equipo de obra es de {coste_semana} k€."),
              _mensaje("Jefe de obra",
                       f"Demolición mecánica y un solo contenedor para todo: {X} se hace en {int(D[X]) - s} semanas en vez "
                       f"de {int(D[X])}, y nos ahorramos el acopio por fracciones, que ocupa media obra. El gestor se lo "
                       "lleva a su vertedero y nos olvidamos; es como se ha hecho siempre."),
              _criterios(["S8", "S1"], constantes=["fracciones_rcd", "impuesto_vertido"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Calcule la duración del proyecto con la demolición selectiva y con la mecánica ({X} {s} semana(s) más "
        "corta). ¿Cuántas semanas ahorra la propuesta y cuánto valen al coste semanal del equipo?",
        "(A) Calcule el coste de gestión de los residuos en las dos opciones (tarifas más impuesto al vertido) y la "
        "fracción del residuo que se valoriza en cada una. ¿Cuál es el resultado económico neto de la propuesta?",
        "(B) Según el criterio y la norma de referencia, ¿puede ejecutarse la propuesta del jefe de obra? ¿Qué "
        "fracciones hay que separar y en qué escalón de la jerarquía de residuos queda cada opción?",
        "(C) ¿Qué propone al jefe de obra y qué comunica a la dirección facultativa? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"demolicion, semanas_menos, coste_semana = {X!r}, {s}, {coste_semana}\n"
                       f"fracciones = {FRACCIONES!r}\n"
                       f"tarifa_separado = {TARIFA_SEPARADO!r}\n"
                       f"tarifa_mixto, impuesto_mixto, impuesto_inerte = {TARIFA_MIXTO}, {IMPUESTO_MIXTO}, {IMPUESTO_INERTE}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con la demolición selectiva", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code("duraciones_mecanica = datos_proyecto['duration'].copy()\n"
                       "duraciones_mecanica[demolicion] -= semanas_menos\n"
                       "duracion_mecanica = int(proyecto.duration(duraciones_mecanica))\n"
                       "ahorro_semanas = duracion - duracion_mecanica\n"
                       "camino = ' / '.join('–'.join(r) for r in proyecto.critical_path(duraciones_mecanica).values())\n"
                       "Markdown(f'''Con demolición selectiva: **{duracion}** semanas; con demolición mecánica: **{duracion_mecanica}** "
                       "(ruta crítica {camino}). La propuesta ahorra **{ahorro_semanas}** semana(s), que al coste del equipo valen "
                       "**{ahorro_semanas * coste_semana} k€**.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Toneladas por fracción y coste de gestión de cada opción"\n#| label: tbl-residuos\n'
                       "toneladas = datos_proyecto[fracciones].sum()\n"
                       "fraccion_inerte = float(toneladas[['hormigon', 'ceramica']].sum()) / float(toneladas.sum())\n"
                       "# Ley 7/2022, art. 93: 10 €/t en vertedero de no peligrosos; si la parte inerte supera el 75 %, esa parte paga 1,5 €/t\n"
                       "impuesto_por_fraccion = pd.Series({c: (impuesto_inerte if (fraccion_inerte > 0.75 and c in ('hormigon', 'ceramica')) else impuesto_mixto) for c in fracciones})\n"
                       "tabla = pd.DataFrame({'toneladas': toneladas,\n"
                       "                      'separado: tarifa (€/t)': pd.Series(tarifa_separado),\n"
                       "                      'separado: coste (€)': toneladas * pd.Series(tarifa_separado),\n"
                       "                      'mixto: tarifa + impuesto (€/t)': tarifa_mixto + impuesto_por_fraccion,\n"
                       "                      'mixto: coste (€)': toneladas * (tarifa_mixto + impuesto_por_fraccion)})\n"
                       "tabla.loc['Total'] = tabla.sum(); tabla.loc['Total', ['separado: tarifa (€/t)', 'mixto: tarifa + impuesto (€/t)']] = None\n"
                       "display(tabla.round(1))\n"
                       "coste_separado = float(tabla.loc['Total', 'separado: coste (€)']) / 1000\n"
                       "coste_mixto = float(tabla.loc['Total', 'mixto: coste (€)']) / 1000\n"
                       "valorizado = float(toneladas[['hormigon', 'ceramica', 'madera', 'metal']].sum()) / float(toneladas.sum())\n"
                       "neto = ahorro_semanas * coste_semana - (coste_mixto - coste_separado)\n"
                       "Markdown(f'''La parte inerte (hormigón y cerámica) es el {fraccion_inerte:.0%} del contenedor''' + "
                       "(f''': supera el 75 % y paga {impuesto_inerte} €/t; el resto, {impuesto_mixto} €/t. ''' if fraccion_inerte > 0.75 else f''': no supera el 75 % y todo el contenedor paga {impuesto_mixto} €/t. ''') + "
                       "f'''Gestión separada: **{coste_separado:.1f} k€** (el metal se cobra; las fracciones recicladas no pagan impuesto); "
                       "mezclado a vertedero: **{coste_mixto:.1f} k€**. Se valoriza el **{valorizado:.0%}** del residuo frente al 0 % en vertedero. "
                       "Resultado neto de la propuesta: {ahorro_semanas * coste_semana} − {coste_mixto - coste_separado:.1f} = **{neto:.1f} k€**''' + "
                       "(' (pierde dinero además de incumplir).' if neto < 0 else ' a favor de la propuesta, antes de la sanción.'))"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma.** Desde el 1 de julio de 2022 los residuos no peligrosos de la obra se clasifican en al "
                     f"menos seis fracciones, preferentemente en la propia obra (Ley 7/2022, art. 30.2), y desde el 1 de "
                     f"enero de 2024 la demolición se hace preferiblemente de forma selectiva, garantizando la retirada de "
                     f"esas fracciones previo estudio de sus cantidades (art. 30.3): la demolición mecánica con contenedor "
                     f"único a vertedero **no puede ejecutarse**, con independencia de lo que ahorre, porque incumple la "
                     f"clasificación del 30.2 aunque se discuta el alcance del 30.3. Hay que separar las seis fracciones que "
                     f"aparecen en la tabla: hormigón y cerámica (minerales), "
                     f"madera, metal, yeso y plástico. **S8.** El contenedor único a vertedero es **eliminación**, el último "
                     f"escalón de la jerarquía, cuando el {valorizado_sep / total:.0%} del residuo tiene reciclado o "
                     f"valorización disponibles; el plan del proyecto está en los escalones de reciclado y valorización. "
                     f"**S1.** Y en la pirámide del PMBOK 8 la propuesta ni siquiera compensa: traslada el impacto. La "
                     f"respuesta B no depende del apartado 2 (que sólo muestra que, además, "
                     + ("es mal negocio)." if coste_mixto - coste_sep > s * coste_semana else "el ahorro es pequeño frente a la sanción).")))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Desde 2022, los residuos de construcción y demolición no peligrosos deben clasificarse en al menos…', ['seis fracciones: madera, minerales, metales, vidrio, plástico y yeso', 'dos fracciones: inertes y resto', 'tantas fracciones como quiera el gestor', 'ninguna: el vertedero las separa'],\n"
        "     'Ley 7/2022, art. 30.2.'),\n"
        "    ('Llevar a vertedero un residuo que tiene reciclado disponible es, en la jerarquía de residuos…', ['el último escalón: eliminación', 'otra valorización', 'prevención', 'preparación para la reutilización'],\n"
        "     'S8: prevención > reutilización > reciclado > otra valorización > eliminación.'),\n"
        "    ('Desde el 1 de enero de 2024, según el art. 30.3 de la Ley 7/2022, la demolición…', ['se lleva a cabo preferiblemente de forma selectiva, garantizando la retirada de al menos las seis fracciones, previo estudio de sus cantidades', 'puede hacerse de forma mecánica si el gestor separa después en planta', 'sólo es selectiva si lo exige el pliego', 'no está regulada: la decide la dirección facultativa'],\n"
        "     'Ley 7/2022, art. 30.3; la clasificación en seis fracciones del 30.2 es exigible desde el 1-7-2022.'),\n"
        "]", "Sostenibilidad 20 — residuos de obra"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al jefe de obra: la opción no está en la tabla (legalidad); si el acopio por fracciones «ocupa "
                     f"media obra», el problema es de espacio y se resuelve con contenedores por fracción retirados con más "
                     f"frecuencia o con separación en planta de un gestor autorizado para ello, no con un vertedero. Si de "
                     f"verdad hacen falta las {s} semana(s), se buscan en otra actividad crítica con Ackoff (lección 10), "
                     f"no en la norma. A la dirección facultativa: el plan de gestión de residuos se cumple como está, con "
                     f"las cifras del apartado 2 ({valorizado_sep / total:.0%} valorizado, {coste_sep:.1f} k€). Prueba de "
                     "publicidad: ¿firmaría el certificado final de obra con el estudio de gestión de residuos incumplido?"))
    return celdas, (f"demolición mecánica con contenedor único: {s} semana(s) y {s * coste_semana} k€ frente a "
                    f"{coste_mixto - coste_sep:.1f} k€ más de gestión y el {valorizado_sep / total:.0%} valorizable a vertedero; art. 30 Ley 7/2022")


# ---------------------------------------------------------------------------
# Caso 4 — Fraccionar el proyecto para esquivar la evaluación ambiental
# ---------------------------------------------------------------------------
def _fraccionamiento_eia(rng):
    e = {}
    solar = bool(rng.random() < 0.5)

    def acepta(i):
        if not (1 <= i["n_dummies"] <= 4 and 4 <= i["n_rutas"] <= 9 and 10 <= i["T"] <= 20 and i["n_criticas"] == 1):
            return False
        cal = i["calendario"]
        iniciales = [a for a in i["iniciales"] if cal.loc[a, "H_total"] == 0]
        if not iniciales:
            return False
        E = str(rng.choice(iniciales))
        d_simp = int(i["D"][E])
        if not (2 <= d_simp <= 5):
            return False
        delta = int(rng.choice([4, 5, 6]))
        D_ord = i["D"].copy(); D_ord[E] += delta
        T_ord = _T(i["proyecto"], D_ord)
        if T_ord != int(i["T"]) + delta:
            return False
        hito = int(i["T"]) + delta - int(rng.choice([1, 2, 2, 3]))
        sd = P.desviaciones(rng, list(i["D"].index))
        sd[E] = float(rng.choice([1.0, 1.5, 2.0]))
        _, _, p_simp = _prob(i["proyecto"], i["D"].astype(float), sd, hito)
        _, _, p_ord = _prob(i["proyecto"], D_ord.astype(float), sd, hito)
        if not (p_simp >= 0.85 and 0.05 <= p_ord <= 0.45):
            return False
        e.update(E=E, delta=delta, D_ord=D_ord, T_ord=T_ord, hito=hito, sd=sd, p_simp=p_simp, p_ord=p_ord)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    E, delta, hito = e["E"], e["delta"], e["hito"]
    if solar:
        S = int(rng.integers(12, 19)) * 10
        descripcion = f"una planta fotovoltaica en suelo de **{S} ha**"
        trozos = f"dos plantas «independientes» de {S // 2} ha, promovidas por dos sociedades del grupo"
        umbral = "más de 100 ha (anexo I); 5 ha o más, anexo II"
    else:
        # El conjunto supera los 30 MW del anexo I, pero cada mitad queda por debajo (si no, la propuesta del
        # director ni siquiera sería aritméticamente creíble): con 4,5 o 5 MW, de 8 a 12 aerogeneradores; con 6 MW, de 6 a 10.
        pot = float(rng.choice([4.5, 5.0, 6.0]))
        n_aero = int(rng.integers(8, 13)) if pot < 6 else int(rng.integers(6, 11))
        S = n_aero * pot
        descripcion = f"un parque eólico de **{n_aero} aerogeneradores de {pot} MW ({S:.0f} MW)**"
        trozos = (f"dos parques «independientes» de {n_aero // 2} y {n_aero - n_aero // 2} aerogeneradores "
                  f"({(n_aero // 2) * pot:.0f} y {(n_aero - n_aero // 2) * pot:.0f} MW), contiguos, promovidos por dos sociedades del grupo")
        umbral = "50 o más aerogeneradores, más de 30 MW o a menos de 2 km de otro parque (anexo I)"
    desc_plana = descripcion.replace("**", "")
    doble = ("" if solar else
             " Y hay una segunda razón, propia de los parques eólicos: dos parques contiguos están cada uno a menos de "
             "2 km del otro, que es por sí solo un supuesto del anexo I; el fraccionamiento no esquiva el anexo, lo duplica.")
    df = P.tabla_base(info, duration=D, standard_deviation=e["sd"])
    celdas = [md("# Enunciado"),
              md(f"Una empresa promueve {descripcion}. El cuadro de prelaciones, las duraciones medias (meses) y las "
                 f"desviaciones típicas se adjuntan; la actividad **{E}** es la tramitación ambiental, con la duración "
                 f"estimada del procedimiento **simplificado** ({int(D[E])} meses). Si el proyecto requiere evaluación "
                 f"**ordinaria**, la experiencia de la empresa es que {E} dura {delta} meses más."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"Los permisos de acceso y conexión a la red caducan si la autorización de construcción no se obtiene "
                 f"antes del **mes {hito}** (hito administrativo); perderlos supone volver a la cola de acceso."),
              _mensaje("Director de desarrollo",
                       f"Lo presentamos como {trozos}: cada uno queda por debajo del umbral del anexo I, va por el "
                       f"procedimiento simplificado y llegamos al hito. Si alguien pregunta, son dos proyectos distintos."),
              _criterios(["S9", "S11"], constantes=["eia_umbrales"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Calcule la duración del proyecto con el procedimiento simplificado y con el ordinario, y el margen (o el "
        f"retraso) respecto al hito del mes {hito} en cada caso.",
        f"(A) Calcule la probabilidad de cumplir el hito en cada caso con la aproximación normal del PERT y con una "
        "simulación Monte Carlo (20 000 réplicas, duraciones normales). ¿Qué actividades tendrían que acortarse, y "
        "cuánto, para recuperar el margen con el procedimiento ordinario?",
        "(B) Según la norma y el criterio de referencia, ¿qué procedimiento corresponde al proyecto? ¿Cambia algo "
        "presentarlo en dos partes? ¿Es la tramitación ambiental un riesgo del proyecto o «cosa de la administración»?",
        "(C) ¿Qué propone al director de desarrollo? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"tramitacion, meses_mas, hito = {E!r}, {delta}, {hito}\n" + SETUP +
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "duraciones_ordinaria = datos_proyecto['duration'].copy(); duraciones_ordinaria[tramitacion] += meses_mas\n"
                       "duracion_ordinaria = int(proyecto.duration(duraciones_ordinaria))"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el procedimiento simplificado", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code("Markdown(f'''Procedimiento simplificado: **{duracion}** meses, margen de {hito - duracion} sobre el hito. "
                       "Procedimiento ordinario: **{duracion_ordinaria}** meses, ''' + (f'retraso de {duracion_ordinaria - hito}' if duracion_ordinaria > hito else f'margen de {hito - duracion_ordinaria}') + "
                       "f''' respecto al hito (la tramitación es crítica: los {meses_mas} meses van enteros al plazo).''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Duración y varianza de cada ruta con el procedimiento ordinario"\n#| label: tbl-rutas\n'
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "display(detalle_rutas(R, duraciones_ordinaria, 'duración'))\n"
                       "detalle_rutas(R, datos_proyecto['variance'], 'varianza', 3)"))
    celdas.append(code("sigma_s, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])\n"
                       "sigma_o, _ = proyecto.standard_deviation(durations=duraciones_ordinaria, variances=datos_proyecto['variance'])\n"
                       "p_pert_s = float(norm.cdf((hito - duracion) / sigma_s)); p_pert_o = float(norm.cdf((hito - duracion_ordinaria) / sigma_o))\n"
                       "dists_s = {act: ('normal', fila['duration'], fila['standard_deviation']) for act, fila in datos_proyecto.iterrows()}\n"
                       "dists_o = dict(dists_s); dists_o[tramitacion] = ('normal', float(duraciones_ordinaria[tramitacion]), float(datos_proyecto.loc[tramitacion, 'standard_deviation']))\n"
                       f"mc_s = proyecto.monte_carlo(dists_s, n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       f"mc_o = proyecto.monte_carlo(dists_o, n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "p_mc_s = float((np.asarray(mc_s.tau_samples) <= hito).mean()); p_mc_o = float((np.asarray(mc_o.tau_samples) <= hito).mean())\n"
                       "tabla = pd.DataFrame({'simplificado': [duracion, sigma_s, p_pert_s, p_mc_s], 'ordinario': [duracion_ordinaria, sigma_o, p_pert_o, p_mc_o]},\n"
                       "                     index=['duración esperada', 'σ del camino crítico', 'P(T ≤ hito) PERT', 'P(T ≤ hito) Monte Carlo'])\n"
                       "display(tabla.round(2))\n"
                       "Markdown(f'''La probabilidad de llegar al hito pasa de **{p_mc_s:.2f}** a **{p_mc_o:.2f}** con el procedimiento ordinario. "
                       "Para llegar al hito habría que acortar {max(0, duracion_ordinaria - hito)} meses en actividades críticas distintas de la tramitación "
                       "(ingeniería y compras en paralelo con la tramitación, lección 10) —y más para recuperar un margen—, o negociar el hito.''')"))
    celdas.append(code('#| tbl-cap: "Índice de criticidad con el procedimiento ordinario"\n#| label: tbl-criticidad\n'
                       "(mc_o.sensitivity_mean * 100).round(1).to_frame('criticidad (%)').T"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma.** El proyecto es {desc_plana}: supera el umbral del anexo I ({umbral}) y le "
                     f"corresponde la evaluación **ordinaria**. **S9.** Presentarlo en dos partes no cambia el procedimiento: "
                     f"los proyectos que, fraccionados, alcanzan por acumulación los umbrales del anexo I se evalúan por el "
                     f"anexo I (art. 7.1.a). Lo que cambia es el riesgo: el órgano ambiental acumula las dos solicitudes "
                     f"—y los {int(D[E])} meses de simplificada se convierten en {int(D[E]) + delta} de ordinaria empezando "
                     f"más tarde— o, peor, la autorización obtenida es nula y el hito se pierde de todos modos.{doble} **S11.** La "
                     f"tramitación es un riesgo del proyecto: tiene probabilidad, impacto ({delta} meses en la ruta crítica, "
                     f"P del hito de {e['p_simp']:.2f} a {e['p_ord']:.2f}) y respuestas (empezar antes, solapar, negociar el "
                     "hito); «es cosa de la administración» es retener del comité el dato que decide."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Un proyecto que se presenta fraccionado en partes que, sumadas, alcanzan los umbrales del anexo I de la Ley 21/2013…', ['se evalúa por el procedimiento ordinario, como un solo proyecto', 'se evalúa parte por parte por el simplificado', 'queda exento de evaluación', 'se evalúa sólo la parte mayor'],\n"
        "     'Art. 7.1.a): acumulación de magnitudes.'),\n"
        f"    ('{desc_plana[0].upper() + desc_plana[1:]} requiere…', ['evaluación de impacto ambiental ordinaria (anexo I)', 'evaluación simplificada (anexo II)', 'ninguna evaluación', 'sólo licencia municipal'],\n"
        "     'Supera el umbral del anexo I.'),\n"
        "    ('La duración de la tramitación ambiental, para el cronograma del proyecto, es…', ['un riesgo del proyecto que se cuantifica y se gestiona', 'un asunto ajeno que no entra en el plan', 'una holgura', 'una reserva de gestión'],\n"
        "     'S11: tiene probabilidad, impacto y respuestas.'),\n"
        "]", "Sostenibilidad 20 — fraccionamiento"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de desarrollo: la tabla del apartado 2 y la lista de lo que sí se puede hacer: presentar "
                     f"el proyecto completo cuanto antes con el documento inicial bien hecho (las ampliaciones de plazo "
                     f"suelen venir de requerimientos de información), solapar la ingeniería y las compras con la "
                     f"tramitación, acortar {e['T_ord'] - hito if e['T_ord'] > hito else 0} meses en otras actividades "
                     f"críticas con su coste, y pedir a la administración la prórroga del hito con el expediente ambiental "
                     "en marcha. Lo que no cabe: dos sociedades y «son proyectos distintos». Prueba de publicidad: ¿lo "
                     "defendería ante el órgano ambiental con el plano de los dos recintos contiguos encima de la mesa?"))
    return celdas, (f"{desc_plana} fraccionado: P(hito) {e['p_simp']:.2f} con simplificada frente a "
                    f"{e['p_ord']:.2f} con la ordinaria que corresponde; art. 7.1.a Ley 21/2013")




# ---------------------------------------------------------------------------
# Caso 5 — La huella del proyecto como valor ganado de carbono
# ---------------------------------------------------------------------------
FUENTES_ALCANCE = [("gasóleo de la maquinaria propia", 1), ("grupo electrógeno de la obra", 1),
                   ("electricidad comprada a la red", 2), ("hormigón y acero comprados", 3),
                   ("transporte de materiales por un transportista contratado", 3),
                   ("vehículos de la subcontrata de montaje", 3), ("viajes del equipo en avión", 3),
                   ("gestión de los residuos por el gestor", 3)]


def _datos_huella(rng):
    e = {}

    def acepta(i):
        if not _base(i):
            return False
        D, cal, T = i["D"], i["calendario"], int(i["T"])
        bac = pd.Series(rng.integers(2, 9, size=len(D)) * 10, index=D.index, name="emisiones_plan").astype(float)
        t = int(round(0.5 * T))
        frac = ((t - cal["inicio_mas_temprano"]) / D).clip(0, 1)
        en_curso = (frac > 0) & (frac < 1)
        if en_curso.sum() < 2 or (frac == 1).sum() < 1:
            return False
        h = pd.Series(rng.choice([0.4, 0.5, 0.6], size=len(D)), index=D.index)
        perf = (frac * 100).round().astype(int)
        perf[en_curso] = (frac[en_curso] * h[en_curso] * 100).round().astype(int)
        g = pd.Series(np.where(frac > 0, rng.choice([1.2, 1.3, 1.4], size=len(D)), 0.0), index=D.index)
        EE_i = bac * perf / 100
        AE_i = (EE_i * g).round(1)
        PE = float((bac * frac).sum()); EE = float(EE_i.sum()); AE = float(AE_i.sum())
        if not (AE < 0.97 * PE and EE / AE <= 0.88 and EE > 0):
            return False
        trans = (t - cal["inicio_mas_temprano"]).clip(lower=0).clip(upper=D).astype(int)
        e.update(bac=bac, t=t, perf=perf, AE_i=AE_i, trans=trans, PE=PE, EE=EE, AE=AE, EPI=EE / AE,
                 EAC=float(bac.sum()) / (EE / AE), en_curso=list(D.index[en_curso]))
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    return info, e


def _huella_del_proyecto(rng):
    info, e = _datos_huella(rng)
    D, T = info["D"], int(info["T"])
    bac, t = e["bac"], e["t"]
    presupuesto = float(bac.sum())
    df = P.tabla_base(info, duration=D, emisiones_plan=bac)
    seg = pd.DataFrame({"actual_duration": e["trans"], "performed": e["perf"], "emisiones_reales": e["AE_i"]})
    seg.index.name = "activity"
    fuentes = [FUENTES_ALCANCE[j] for j in sorted(rng.choice(len(FUENTES_ALCANCE), size=5, replace=False))]
    celdas = [md("# Enunciado"),
              md("El acta de constitución de un proyecto de obra civil fija, junto al plazo y al presupuesto, un "
                 f"**indicador de sostenibilidad: emisiones del proyecto ≤ {presupuesto:.0f} t CO₂**. El cuadro de "
                 "prelaciones, las duraciones (semanas) y las emisiones planificadas de cada actividad (t CO₂, "
                 "repartidas uniformemente en su duración) se adjuntan."),
              code('#| tbl-cap: "Planificación del proyecto y presupuesto de emisiones"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"Al final de la semana {t} se dispone del avance físico verificado de cada actividad (`performed`, %) y de "
                 "las emisiones reales acumuladas de cada una, calculadas a partir de los partes de combustible, las "
                 "facturas de electricidad y los albaranes (`emisiones_reales`, t CO₂)."),
              code(f'#| tbl-cap: "Seguimiento al final de la semana {t}"\n#| label: tbl-seguimiento\n' + literal_dataframe(seg, nombre="seguimiento")),
              _mensaje("Jefe de proyecto (borrador del informe al patrocinador)",
                       f"Emisiones acumuladas a la semana {t}: {e['AE']:.0f} t, por debajo de las {e['PE']:.0f} t previstas "
                       "a esta fecha en el plan. El indicador de sostenibilidad va bien; no hace falta más detalle."),
              md("Las fuentes de emisión del proyecto, para el inventario, son las siguientes: "
                 + "; ".join(f for f, _ in fuentes) + "."),
              _criterios(["S5", "S6"], constantes=["alcances"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule, actividad a actividad y en total, las emisiones planificadas a la fecha (PE), las emisiones "
        "«ganadas» (EE = plan × avance físico) y las reales (AE), el índice de desempeño de emisiones EPI = EE / AE y "
        "la previsión de emisiones al cierre EAC = presupuesto / EPI. ¿Se cumple el indicador del acta?",
        "(A) Explique por qué las emisiones reales están por debajo de las previstas a la fecha y, a la vez, el "
        "proyecto va camino de superar el presupuesto de emisiones. ¿Qué dos efectos se mezclan en el acumulado?",
        "(B) Según los criterios de referencia, ¿cumple el borrador del informe? ¿Qué cifras tendría que contener? "
        "Clasifique cada fuente de emisión del enunciado en su alcance.",
        "(C) Redacte el párrafo del informe al patrocinador y lo que dice al jefe de proyecto. Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"semana_control, presupuesto_emisiones = {t}, {presupuesto:.0f}\n" + SETUP +
                       "def valor_ganado_carbono(datos, seguimiento, calendario, semana):\n"
                       "    filas = {}\n"
                       "    for a, fila in datos.iterrows():\n"
                       "        es, dur = int(calendario.loc[a, 'inicio_mas_temprano']), int(calendario.loc[a, 'duracion'])\n"
                       "        frac = min(max(semana - es, 0), dur) / dur\n"
                       "        plan = float(fila['emisiones_plan']); perf = float(seguimiento.loc[a, 'performed']) / 100\n"
                       "        filas[a] = {'plan (t)': plan, 'planificado hasta la semana': f'{min(max(semana - es, 0), dur)} de {dur}',\n"
                       "                    'PE = plan·fracción': f'{plan:.0f}·{frac:.2f} = {plan * frac:.1f}', 'avance (%)': int(perf * 100),\n"
                       "                    'EE = plan·avance': f'{plan:.0f}·{perf:.2f} = {plan * perf:.1f}',\n"
                       "                    'AE': float(seguimiento.loc[a, 'emisiones_reales']), 'PE_i': plan * frac, 'EE_i': plan * perf}\n"
                       "    return pd.DataFrame(filas).T\n"
                       "detalle = valor_ganado_carbono(datos_proyecto, seguimiento, calendario, semana_control)\n"
                       "PE, EE, AE = float(detalle['PE_i'].sum()), float(detalle['EE_i'].sum()), float(detalle['AE'].sum())\n"
                       "EPI = EE / AE\n"
                       "EAC = presupuesto_emisiones / EPI"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT de la planificación", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "PE, EE y AE actividad a actividad (t CO₂)"\n#| label: tbl-detalle\n'
                       "detalle[['plan (t)', 'planificado hasta la semana', 'PE = plan·fracción', 'avance (%)', 'EE = plan·avance', 'AE']]"))
    celdas.append(code("indicadores = pd.Series({'PE (t)': PE, 'EE (t)': EE, 'AE (t)': AE, 'EE − AE (t)': EE - AE, 'EE − PE (t)': EE - PE,\n"
                       "                         'EPI = EE / AE': EPI, 'índice de avance EE / PE': EE / PE,\n"
                       "                         'EAC = presupuesto / EPI (t)': EAC, 'exceso previsto sobre el presupuesto (t)': EAC - presupuesto_emisiones})\n"
                       "display(indicadores.round(2).to_frame('valor').T)\n"
                       "Markdown(f'''EPI = {EE:.1f} / {AE:.1f} = **{EPI:.2f}**: por cada tonelada planificada para el trabajo hecho se han emitido {1 / EPI:.2f}. "
                       "Previsión al cierre **{EAC:.0f} t** frente a las {presupuesto_emisiones:.0f} del acta: el indicador **no se cumple** si no cambia el ritmo de emisión.''')"))
    celdas += apartado(2)
    celdas.append(md(f"Se mezclan dos efectos de signo contrario. El proyecto va **retrasado** (EE/PE = "
                     f"{e['EE'] / e['PE']:.2f}: se ha hecho menos trabajo del previsto a la fecha), y por eso ha emitido "
                     f"menos que lo planificado a la fecha: {e['AE']:.0f} t frente a {e['PE']:.0f}. Pero por el trabajo que sí "
                     f"se ha hecho ha emitido **más de lo previsto** (EPI = {e['EPI']:.2f}): las actividades empezadas "
                     f"({', '.join(e['en_curso'])}, y las terminadas) emiten entre un 20 y un 40 % más por unidad de "
                     "avance que el plan. El acumulado compara dos cosas que no son comparables (lo emitido por el trabajo "
                     "hecho con lo previsto para un trabajo mayor); EE es la referencia correcta, exactamente como en el "
                     "valor ganado de costes (lección 14): AE < PV no dice nada de la eficiencia; EV − AC sí."))
    celdas += apartado(3)
    alcances_txt = "; ".join(f"{f}: alcance {a}" for f, a in fuentes)
    celdas.append(md(f"**S5.** El indicador está en el acta de constitución: se informa como el coste, con PE, EE, AE, el "
                     f"índice y la previsión al cierre. El borrador da un dato cierto ({e['AE']:.0f} < {e['PE']:.0f}) que "
                     f"lleva a la conclusión contraria a la correcta (EAC {e['EAC']:.0f} t > {presupuesto:.0f}): es una "
                     f"verdad a medias (C2) o, si el jefe de proyecto conoce el EPI, información fuera de contexto (C3). "
                     f"Las cifras que debe contener: EPI {e['EPI']:.2f}, EAC {e['EAC']:.0f} t, retraso EE/PE "
                     f"{e['EE'] / e['PE']:.2f}, causas y medidas. **S6.** Clasificación de las fuentes: {alcances_txt}. La "
                     "regla es quién controla la fuente: el combustible de lo propio es alcance 1 aunque lo queme un "
                     "camión; el mismo camión de un transportista contratado es alcance 3; la electricidad comprada es "
                     "siempre alcance 2."))
    f3 = [f for f, a in fuentes if a == 3][0]
    f1 = [f for f, a in fuentes if a == 1]
    f1 = f1[0] if f1 else "gasóleo de la maquinaria propia"
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Si las emisiones reales acumuladas están por debajo de las planificadas a la fecha pero EPI < 1, el proyecto…', ['va retrasado y emite más de lo previsto por unidad de trabajo', 'cumple el indicador', 'va adelantado', 'emite menos de lo previsto por unidad de trabajo'],\n"
        "     'AE < PE por el retraso; EE < AE por la ineficiencia.'),\n"
        f"    ('Las emisiones de «{f3}» son…', ['alcance 3', 'alcance 1', 'alcance 2', 'no se contabilizan'],\n"
        "     'S6: fuente de la cadena de valor no controlada por la organización.'),\n"
        f"    ('Las emisiones de «{f1}» son…', ['alcance 1', 'alcance 2', 'alcance 3', 'no se contabilizan'],\n"
        "     'S6: combustión en una fuente propia o controlada.'),\n"
        "    ('La previsión de emisiones al cierre con el método del valor ganado es…', ['presupuesto de emisiones / EPI', 'emisiones reales acumuladas × 2', 'PE + AE', 'presupuesto − AE'],\n"
        "     'Mismo razonamiento que EAC = BAC / CPI.'),\n"
        "]", "Sostenibilidad 20 — huella del proyecto"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Párrafo para el patrocinador (modelo): «A la semana {t} el proyecto lleva emitidas {e['AE']:.0f} t "
                     f"por un trabajo que el plan valoraba en {e['EE']:.0f} t (EPI {e['EPI']:.2f}); el avance es del "
                     f"{e['EE'] / presupuesto:.0%} frente al {e['PE'] / presupuesto:.0%} previsto. Si no cambia el ritmo, "
                     f"el cierre se prevé en {e['EAC']:.0f} t, {e['EAC'] - presupuesto:.0f} por encima del indicador del "
                     "acta. Causas identificadas y medidas: …». Al jefe de proyecto: el acumulado sin referencia es el "
                     "mismo error que informar AC sin EV; si el patrocinador rehace la cuenta en el cierre, el problema no "
                     "serán las toneladas sino el informe. Prueba de publicidad: ¿firmaría el informe si llevara anexa la "
                     "tabla del apartado 1?"))
    return celdas, (f"AE {e['AE']:.0f} t < PE {e['PE']:.0f} t a la semana {t}, pero EPI {e['EPI']:.2f} y previsión "
                    f"{e['EAC']:.0f} t sobre {presupuesto:.0f}; alcances de las fuentes")


# ---------------------------------------------------------------------------
# Caso 6 — «Proyecto neutro en carbono» con créditos de compensación
# ---------------------------------------------------------------------------
MEDIDAS_BASE = [("hormigón con cemento de bajo contenido en clínker", 180, 40),
                ("acero de horno eléctrico (chatarra reciclada)", 220, 60),
                ("conexión a red en lugar de grupo electrógeno", 45, 6),
                ("biocombustible HVO en la maquinaria propia", 150, 45),
                ("transporte a carga completa y proveedores cercanos", 40, 2),
                ("electricidad con garantía de origen renovable", 60, 3),
                ("oficina de obra con fotovoltaica y aerotermia", 10, 8),
                ("reutilización de encofrados y tierras en la obra", 30, 4)]


def _neutro_en_carbono(rng):
    fuentes = pd.DataFrame({"t CO2": [int(rng.integers(25, 45)) * 10, int(rng.integers(4, 9)) * 10,
                                      int(rng.integers(70, 110)) * 10, int(rng.integers(10, 20)) * 10, int(rng.integers(1, 4)) * 10],
                            "alcance": [1, 2, 3, 3, 3]},
                           index=pd.Index(["gasóleo de la maquinaria propia", "electricidad de la obra",
                                           "hormigón y acero comprados", "transporte contratado", "viajes del equipo"], name="fuente"))
    H = float(fuentes["t CO2"].sum())
    medidas = pd.DataFrame([(n, int(round(r * float(rng.uniform(0.8, 1.2)))), int(round(c * float(rng.uniform(0.8, 1.3)))))
                            for n, r, c in MEDIDAS_BASE], columns=["medida", "reduccion_t", "coste_k"]).set_index("medida")
    medidas["coste_k"] = medidas["coste_k"].clip(lower=1)
    q = int(rng.choice([6, 9, 12]))
    p = int(rng.choice([80, 100, 150]))
    objetivo = float(rng.choice([0.3, 0.4]))
    cu = medidas["coste_k"] * 1000 / medidas["reduccion_t"]
    rentables = medidas[cu <= p]
    red_rentable = float(rentables["reduccion_t"].sum()); coste_rentable = float(rentables["coste_k"].sum())
    celdas = [md("# Enunciado"),
              md("Antes de empezar la obra, el departamento de marketing quiere anunciar el proyecto como **«proyecto "
                 "neutro en carbono»**. El inventario de emisiones previsto (t CO₂) por fuente se adjunta, junto con las "
                 "medidas de reducción que el equipo ha estudiado, con la reducción que consigue cada una (t CO₂) y su "
                 "coste (k€)."),
              code('#| tbl-cap: "Inventario de emisiones previsto"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(fuentes, indice="fuente")),
              code('#| tbl-cap: "Medidas de reducción estudiadas"\n#| label: tbl-medidas\n' + literal_dataframe(medidas, nombre="medidas", indice="medida")),
              md(f"En el mercado voluntario hay créditos de compensación a **{q} €/t**. La organización aplica un precio "
                 f"sombra del carbono de **{p} €/t** a sus decisiones y se ha comprometido públicamente a reducir las "
                 f"emisiones de sus proyectos un **{objetivo:.0%}** respecto al diseño inicial."),
              _mensaje("Director de marketing",
                       f"Compramos créditos por las {H:.0f} t a {q} €/t, que es lo que cuesta un fin de semana de "
                       "publicidad, y lo anunciamos como «el primer proyecto neutro en carbono del grupo» y "
                       "«construcción sostenible». Las medidas de reducción, las que salgan gratis; el resto no se nota."),
              _criterios(["S7", "S1", "S2"], constantes=["afirmaciones_ambientales"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Calcule el coste de compensar todas las emisiones a {q} €/t y el coste unitario (€/t) de cada medida de "
        f"reducción. ¿Qué medidas cuestan menos que el precio sombra de {p} €/t, cuánto reducen en conjunto y cuánto "
        "cuestan?",
        f"(A) Obtenga la combinación de medidas de menor coste que alcanza la reducción del {objetivo:.0%} "
        "comprometida, su coste y lo que costaría compensar el resto a precio de créditos y al precio sombra.",
        "(B) Según los criterios y la norma de referencia, ¿puede anunciarse el proyecto como «neutro en carbono» "
        "comprando los créditos? ¿Y como «construcción sostenible»? ¿Qué afirmación sí podría hacerse? ¿En qué "
        "escalón de la pirámide queda cada propuesta?",
        "(C) ¿Qué propone al director de marketing y a la dirección? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"precio_creditos, precio_sombra, objetivo = {q}, {p}, {objetivo}\n"
                       "inventario = datos_enunciado.copy()\n"
                       "huella = float(inventario['t CO2'].sum())\n"
                       "medidas = medidas.copy()\n"
                       "medidas['coste unitario (€/t)'] = medidas['coste_k'] * 1000 / medidas['reduccion_t']\n"
                       "medidas = medidas.sort_values('coste unitario (€/t)')"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Medidas ordenadas por coste unitario (curva de coste marginal de reducción)"\n#| label: tbl-macc\n'
                       "medidas['reducción acumulada (t)'] = medidas['reduccion_t'].cumsum()\n"
                       "medidas['coste acumulado (k€)'] = medidas['coste_k'].cumsum()\n"
                       "medidas['rentable al precio sombra'] = medidas['coste unitario (€/t)'] <= precio_sombra\n"
                       "display(medidas.round(1))\n"
                       "rentables = medidas[medidas['rentable al precio sombra']]\n"
                       "coste_compensar_todo = huella * precio_creditos / 1000\n"
                       "Markdown(f'''Compensar las {huella:.0f} t a {precio_creditos} €/t cuesta **{coste_compensar_todo:.1f} k€**. "
                       "Las medidas con coste unitario ≤ {precio_sombra} €/t son {len(rentables)}: reducen **{rentables[\"reduccion_t\"].sum():.0f} t** "
                       "({rentables[\"reduccion_t\"].sum() / huella:.0%}) por **{rentables[\"coste_k\"].sum():.0f} k€**; al precio sombra esas toneladas valen "
                       "{rentables[\"reduccion_t\"].sum() * precio_sombra / 1000:.1f} k€.''')"))
    celdas += apartado(2)
    celdas.append(code("from itertools import combinations\n"
                       "objetivo_t = objetivo * huella\n"
                       "mejor = None\n"
                       "for n in range(1, len(medidas) + 1):\n"
                       "    for combo in combinations(medidas.index, n):\n"
                       "        red = medidas.loc[list(combo), 'reduccion_t'].sum(); coste = medidas.loc[list(combo), 'coste_k'].sum()\n"
                       "        if red >= objetivo_t and (mejor is None or coste < mejor[1]):\n"
                       "            mejor = (combo, coste, red)\n"
                       "combo, coste_combo, red_combo = mejor\n"
                       "resto = huella - red_combo\n"
                       "display(medidas.loc[list(combo), ['reduccion_t', 'coste_k', 'coste unitario (€/t)']].round(1))\n"
                       "Markdown(f'''Reducir al menos {objetivo_t:.0f} t ({objetivo:.0%}) cuesta como mínimo **{coste_combo:.0f} k€** con las medidas de la tabla "
                       "({red_combo:.0f} t, {red_combo / huella:.0%}). Quedan {resto:.0f} t: compensarlas cuesta {resto * precio_creditos / 1000:.1f} k€ a precio de "
                       "créditos y valen {resto * precio_sombra / 1000:.1f} k€ al precio sombra.''')"))
    celdas.append(md("La combinación óptima se busca por enumeración (ocho medidas, 255 combinaciones); con medidas "
                     "divisibles bastaría recorrer la curva de coste marginal por orden de coste unitario. Obsérvese que la "
                     "medida más barata por tonelada no siempre entra en la combinación óptima para un objetivo dado: las "
                     "medidas son indivisibles."))
    celdas += apartado(3)
    celdas.append(md(f"**Norma y S7.** «Proyecto neutro en carbono» basado en compensaciones es exactamente la afirmación "
                     f"que la Directiva (UE) 2024/825 añade a la lista de prácticas prohibidas (impacto neutro, reducido o "
                     f"positivo por compensación): **no puede anunciarse** así, con independencia del precio de los "
                     f"créditos. «Construcción sostenible» es una afirmación genérica sin un desempeño reconocido que la "
                     f"respalde: **tampoco**. Lo que sí puede afirmarse es lo específico y demostrable: «el proyecto reduce "
                     f"sus emisiones un {objetivo:.0%} respecto al diseño inicial mediante [medidas]; las {H * (1 - objetivo):.0f} t "
                     f"restantes se compensan con créditos verificados de [proyecto y estándar]». **S1.** Compensar está en "
                     f"el escalón más bajo; las medidas del apartado 2 **minimizan**; cambiar el diseño para no necesitar "
                     f"el hormigón o el acero **evita**. **S2.** Con el precio sombra de {p} €/t, las medidas con coste "
                     f"unitario inferior ({red_rentable:.0f} t por {coste_rentable:.0f} k€) se ejecutan aunque no se anuncie nada: "
                     "son decisiones rentables según la política de la organización, no publicidad."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Anunciar un proyecto como «neutro en carbono» porque se han comprado créditos de compensación es, desde el 27-9-2026…', ['una práctica comercial prohibida por la Directiva (UE) 2024/825', 'legal si los créditos están verificados', 'legal si se dice el precio pagado', 'obligatorio en los contratos públicos'],\n"
        "     'S7: afirmación de impacto neutro basada en compensación.'),\n"
        "    ('Una afirmación ambiental genérica («sostenible», «verde») sin desempeño reconocido que la respalde…', ['está prohibida', 'es correcta si el proyecto compensa', 'es correcta en publicidad pero no en contratos', 'sólo se prohíbe en envases'],\n"
        "     'Anexo I de la Directiva 2005/29/CE, punto añadido por la 2024/825.'),\n"
        "    ('Con un precio sombra fijado, las medidas de reducción que se ejecutan son…', ['las de coste unitario menor o igual que el precio sombra', 'las gratuitas', 'las que pida marketing', 'ninguna si se compensa'],\n"
        "     'S2: la externalidad con precio decide.'),\n"
        "    ('En la pirámide de la sostenibilidad, cambiar el diseño para no necesitar un material emisor es…', ['evitar', 'compensar', 'restaurar', 'minimizar'],\n"
        "     'El escalón más deseable es evitar el impacto.'),\n"
        "]", "Sostenibilidad 20 — neutro en carbono"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de marketing: la frase que puede usarse (específica, con cifras) y la que no; la "
                     f"sanción por práctica desleal y el daño reputacional cuando un competidor o una asociación de "
                     f"consumidores pida la cuenta. A la dirección: ejecutar las medidas rentables al precio sombra "
                     f"({coste_rentable:.0f} k€), decidir si se cumple el compromiso del {objetivo:.0%} con la combinación "
                     "del apartado 2 y, sólo después, compensar el resto con créditos verificados y decirlo con esa palabra. "
                     "Prueba de publicidad, en sentido literal: ¿resistiría el anuncio una pregunta de un periodista sobre "
                     "cuántas toneladas se han reducido y cuántas se han comprado?"))
    return celdas, (f"{H:.0f} t compensadas a {q} €/t ({H * q / 1000:.0f} k€) frente a reducir el {objetivo:.0%}; "
                    f"«neutro» prohibido por la Directiva 2024/825")


# ---------------------------------------------------------------------------
# Caso 7 — La ola de calor que no está en el cronograma
# ---------------------------------------------------------------------------
VENTANA_VERANO = (5, 14)       # semanas del proyecto que caen en julio-agosto-principios de septiembre (semana 1 = 1 de junio)
PERDIDA_EPISODIO = 0.4         # semanas perdidas por cada episodio de aviso (dos jornadas)


def _ola_de_calor(rng):
    e = {}

    def acepta(i):
        if not _base(i, T=(14, 22)):
            return False
        D, cal, T = i["D"], i["calendario"], int(i["T"])
        exterior = pd.Series(rng.random(len(D)) < 0.6, index=D.index).astype(int)
        v0, v1 = VENTANA_VERANO
        n_s = pd.Series({a: max(0, min(int(cal.loc[a, "fin_mas_temprano"]), v1) - max(int(cal.loc[a, "inicio_mas_temprano"]), v0 - 1))
                         for a in D.index}) * exterior
        criticas_ext = [a for a in D.index if cal.loc[a, "H_total"] == 0 and n_s[a] >= 2]
        if len(criticas_ext) < 2 or n_s.sum() < 8:
            return False
        pi = float(rng.choice([0.3, 0.4, 0.5]))
        perdida_esperada = n_s * pi * PERDIDA_EPISODIO
        D_cal = (D + perdida_esperada)
        T_cal = _T(i["proyecto"], D_cal.round().astype(int))
        T_esp = float(max((i["P"] @ D_cal.reindex(i["P"].columns))))
        if T_esp - T < 1:
            return False
        sd = P.desviaciones(rng, list(D.index))
        plazo = T + int(rng.choice([1, 2]))
        _, _, p0 = _prob(i["proyecto"], D.astype(float), sd, plazo)
        if not (0.8 <= p0 <= 0.97):
            return False
        rng_mc = np.random.default_rng(SEMILLA_MC)
        dists = {}
        for a in D.index:
            mu, s, n = float(D[a]), float(sd[a]), int(n_s[a])

            def f(size=None, mu=mu, s=s, n=n):
                return rng_mc.normal(mu, s, size) + PERDIDA_EPISODIO * rng_mc.binomial(n, pi, size)
            dists[a] = f
        with _silencio():
            res = i["proyecto"].monte_carlo(dists, n_samples=N_MC, seed=SEMILLA_MC)
        p1 = float((np.asarray(res.tau_samples) <= plazo).mean())
        if p1 > 0.55:
            return False
        e.update(exterior=exterior, n_s=n_s, pi=pi, sd=sd, plazo=plazo, p0=p0, p1=p1, T_esp=T_esp, T_cal=T_cal,
                 criticas_ext=criticas_ext, perdida_esperada=perdida_esperada)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    pi, plazo = e["pi"], e["plazo"]
    v0, v1 = VENTANA_VERANO
    df = P.tabla_base(info, duration=D, standard_deviation=e["sd"], exterior=e["exterior"])
    celdas = [md("# Enunciado"),
              md("Una obra de urbanización empieza el **1 de junio** (semana 1 del proyecto). El cuadro de prelaciones, "
                 "las duraciones medias (semanas), las desviaciones típicas y si la actividad se ejecuta **al aire "
                 f"libre** (`exterior` = 1) se adjuntan. El plazo contractual es de **{plazo} semanas**."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"Según los registros de la AEMET de los últimos años, en la zona cada semana de julio, agosto y la primera "
                 f"quincena de septiembre (semanas {v0} a {v1} del proyecto) tiene una probabilidad **{pi:.0%}** de "
                 f"incluir un episodio de aviso naranja o rojo por calor, durante el cual el trabajo al aire libre se "
                 f"interrumpe en las horas centrales: cada episodio supone perder el equivalente a **{PERDIDA_EPISODIO:.1f} "
                 "semanas** de trabajo en las actividades exteriores en curso."),
              _mensaje("Jefe de proyecto",
                       "El cronograma va con las duraciones medias y así se presenta al cliente. El calor no es un "
                       "riesgo del proyecto, es el clima: si hay aviso, se trabaja igual y se recupera con horas por la "
                       "tarde, como siempre. No quiero una reserva de «calor» en el plan, que luego se usa para todo."),
              _criterios(["S10", "S11"], constantes=["calor_extremo"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Con el calendario de inicios más tempranos, obtenga para cada actividad exterior las semanas que caen en "
        "la ventana de calor y la pérdida esperada de duración (semanas × probabilidad × pérdida por episodio). "
        "¿Cuál es la duración esperada del proyecto con la pérdida y qué actividades concentran el riesgo?",
        f"(A) Calcule la probabilidad de cumplir el plazo de {plazo} semanas sin el calor (aproximación normal del "
        "PERT) y con él (simulación Monte Carlo, 20 000 réplicas: duración normal más la pérdida binomial de cada "
        "actividad exterior). ¿Qué actividades exteriores podrían salir de la ventana usando su holgura total?",
        "(B) Según la norma y los criterios de referencia, ¿puede presentarse al cliente el cronograma con las "
        "duraciones medias y «se trabaja igual»? ¿Es el calor un riesgo del proyecto?",
        "(C) ¿Qué propone al jefe de proyecto y qué comunica al cliente? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"plazo, p_episodio, perdida_episodio, ventana = {plazo}, {pi}, {PERDIDA_EPISODIO}, {VENTANA_VERANO!r}\n" + SETUP +
                       "datos_proyecto['variance'] = datos_proyecto['standard_deviation'] ** 2\n"
                       "def semanas_en_ventana(a):\n"
                       "    es, ef = int(calendario.loc[a, 'inicio_mas_temprano']), int(calendario.loc[a, 'fin_mas_temprano'])\n"
                       "    return max(0, min(ef, ventana[1]) - max(es, ventana[0] - 1)) * int(datos_proyecto.loc[a, 'exterior'])"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones medias", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Semanas de cada actividad en la ventana de calor y pérdida esperada"\n#| label: tbl-calor\n'
                       "tabla = calendario[['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']].copy()\n"
                       "tabla['exterior'] = datos_proyecto['exterior']\n"
                       "tabla['semanas en la ventana'] = [semanas_en_ventana(a) for a in tabla.index]\n"
                       "tabla['pérdida esperada'] = tabla['semanas en la ventana'] * p_episodio * perdida_episodio\n"
                       "display(tabla.round(2))\n"
                       "duraciones_calor = datos_proyecto['duration'] + tabla['pérdida esperada']\n"
                       "R = proyecto.path_matrix(dummies=False)\n"
                       "duracion_esperada_calor = float((R @ duraciones_calor[R.columns]).max())\n"
                       "criticas_exterior = [a for a in tabla.index if tabla.loc[a, 'H_total'] == 0 and tabla.loc[a, 'semanas en la ventana'] > 0]\n"
                       "Markdown(f'''Duración esperada **{duracion_esperada_calor:.1f}** semanas con la pérdida por calor, frente a {duracion} del plan "
                       "(+{duracion_esperada_calor - duracion:.1f}). El riesgo se concentra en las actividades exteriores críticas: {', '.join(criticas_exterior)}.''')"))
    celdas += apartado(2)
    celdas.append(code("sigma, _ = proyecto.standard_deviation(durations=datos_proyecto['duration'], variances=datos_proyecto['variance'])\n"
                       "p_sin = float(norm.cdf((plazo - duracion) / sigma))\n"
                       f"rng_mc = np.random.default_rng({SEMILLA_MC})\n"
                       "def duracion_con_calor(mu, sd, n_semanas):\n"
                       "    '''Normal más la pérdida de los episodios de aviso (binomial) en las semanas de la ventana.'''\n"
                       "    def f(size=None):\n"
                       "        return rng_mc.normal(mu, sd, size) + perdida_episodio * rng_mc.binomial(n_semanas, p_episodio, size)\n"
                       "    return f\n"
                       "dists = {a: duracion_con_calor(float(fila['duration']), float(fila['standard_deviation']), int(tabla.loc[a, 'semanas en la ventana']))\n"
                       "         for a, fila in datos_proyecto.iterrows()}\n"
                       f"con_calor = proyecto.monte_carlo(dists, n_samples={N_MC}, seed={SEMILLA_MC})\n"
                       "tau = np.asarray(con_calor.tau_samples, dtype=float)\n"
                       "p_con = float((tau <= plazo).mean())\n"
                       "display(con_calor.summary().round(2))\n"
                       "Markdown(f'''P(T ≤ {plazo}) = **{p_sin:.2f}** sin el calor (PERT, σ = {sigma:.2f}) y **{p_con:.2f}** con él (Monte Carlo); "
                       "con probabilidad 0,8 la obra termina en la semana {np.percentile(tau, 80):.1f}.''')"))
    celdas.append(code('#| tbl-cap: "Actividades exteriores: desplazamiento necesario para salir de la ventana frente a la holgura total"\n#| label: tbl-desplazar\n'
                       "ext = tabla[(tabla['exterior'] == 1) & (tabla['semanas en la ventana'] > 0)].copy()\n"
                       "ext['desplazamiento para empezar en la semana ' + str(ventana[1] + 1)] = (ventana[1] - ext['inicio_mas_temprano']).clip(lower=0)\n"
                       "ext['cabe en la holgura'] = ext.iloc[:, -1] <= ext['H_total']\n"
                       "ext[['inicio_mas_temprano', 'H_total', 'semanas en la ventana', ext.columns[-2], 'cabe en la holgura']]"))
    celdas.append(md("Las actividades críticas no tienen margen: su pérdida va entera al plazo. Las exteriores con holgura "
                     "pueden, a veces, salir de la ventana (o entrar menos en ella); si no cabe, la alternativa es la "
                     "adaptación de la jornada (empezar a las 6, parar en las horas de aviso) con su productividad, no "
                     "«trabajar igual»."))
    celdas += apartado(3)
    celdas.append(md(f"**Norma y S10.** Cuando hay aviso naranja o rojo, la adaptación de las condiciones de trabajo es "
                     f"obligatoria, incluida la reducción o modificación de las horas y la prohibición de ciertas tareas en "
                     f"las horas del fenómeno (RD 486/1997, disposición adicional única): «se trabaja igual» no es una "
                     f"opción del plan, y «se recupera con horas por la tarde» tiene el límite de la jornada y de las horas "
                     f"extraordinarias (lección 19, horas extra). **S11.** El calor tiene probabilidad ({pi:.0%} por "
                     f"semana), impacto ({e['T_esp'] - T:.1f} semanas esperadas; P del plazo de {e['p0']:.2f} a "
                     f"{e['p1']:.2f}) y respuestas (reordenar, adaptar la jornada, reserva dimensionada): es un riesgo del "
                     "proyecto por definición, y presentar al cliente las duraciones medias conociéndolo es retener "
                     "información (C4). Lo que el jefe de proyecto teme —una reserva «que se usa para todo»— se "
                     "resuelve declarándola y asignándola (lección 19, C8), no ocultando el riesgo."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Cuando la AEMET emite un aviso naranja o rojo y las medidas preventivas no bastan, el empresario…', ['debe adaptar las condiciones de trabajo, incluidas las horas de la jornada', 'puede seguir trabajando si paga horas extra', 'sólo debe avisar a los trabajadores', 'no tiene ninguna obligación nueva'],\n"
        "     'RD 486/1997, disposición adicional única (RDL 4/2023).'),\n"
        "    ('La pérdida esperada de duración de una actividad exterior por el calor se calcula como…', ['semanas en la ventana × probabilidad de episodio × pérdida por episodio', 'probabilidad × duración', 'pérdida por episodio × duración', 'sólo si la actividad es crítica'],\n"
        "     'Valor esperado de una binomial por la pérdida de cada episodio.'),\n"
        "    ('Un fenómeno climático con probabilidad e impacto conocidos sobre el plazo es, para el proyecto…', ['un riesgo que entra en el registro y en el cronograma', 'una circunstancia ajena que no se planifica', 'una holgura', 'fuerza mayor siempre'],\n"
        "     'S11: riesgo de sostenibilidad.'),\n"
        "]", "Sostenibilidad 20 — ola de calor"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al jefe de proyecto: la tabla del apartado 1 y las dos probabilidades; el plan que se puede "
                     f"defender tiene jornada adaptada en las semanas {v0}–{v1} para {', '.join(e['criticas_ext'])}, las "
                     f"exteriores con holgura movidas donde quepa y una reserva de contingencia dimensionada "
                     f"(≈ {e['T_esp'] - T:.1f} semanas esperadas; con el 80 % lo que dé la simulación) declarada al "
                     f"cliente con su dueño. Al cliente: la duración esperada con el calor, la probabilidad del plazo y el "
                     "coste de las alternativas (empezar antes, acortar otras actividades). Prueba de reversibilidad: "
                     "¿aceptaría, como cuadrilla, el plan de «se trabaja igual» en una tarde de aviso rojo?"))
    return celdas, (f"calor en las semanas {v0}–{v1} ({pi:.0%} por semana): duración esperada {e['T_esp']:.1f} frente a {T}, "
                    f"P(T ≤ {plazo}) de {e['p0']:.2f} a {e['p1']:.2f}")


# ---------------------------------------------------------------------------
# Caso 8 — Cubrir el pico de personal con una cadena de subcontratas
# ---------------------------------------------------------------------------
def _subcontratacion_en_cadena(rng):
    e = {}

    def acepta(i):
        if not _base(i, criticas_1=False, T=(12, 20)):
            return False
        rec = P.recursos(rng, list(i["D"].index), hi=4)
        df = P.tabla_base(i, duration=i["D"], resources=rec)
        try:
            total = _cargas(i, df)
            desp, total_niv = _nivelar(i, df)
        except Exception:
            return False
        pico, pico_niv = int(total.max()), int(total_niv.max())
        L = int(np.ceil(total.mean()))
        if L >= pico_niv or L < 3 or L < int(rec.max()) or pico_niv >= pico:
            return False
        exceso = (total - L).clip(lower=0); exceso_niv = (total_niv - L).clip(lower=0)
        if exceso.sum() < 4:
            return False
        try:
            with _silencio():
                _, g, _ = i["proyecto"].copy().asignar(data=df.copy(), duration_label="duration", resource_label="resources", maximo=[L], report=False)
            T_L = len(g.data.columns.drop("H_total"))
        except Exception:
            return False
        if T_L <= int(i["T"]):
            return False
        e.update(rec=rec, df=df, total=total, total_niv=total_niv, L=L, pico=pico, pico_niv=pico_niv,
                 exceso=exceso, exceso_niv=exceso_niv, T_L=T_L)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    L, df = e["L"], e["df"]
    c_propio, c_ett, c_cadena, penal = 2.0, 2.6, 1.6, int(rng.choice([8, 10, 12]))
    coste_cadena = float(e["exceso_niv"].sum()) * c_cadena
    coste_ett = float(e["exceso_niv"].sum()) * c_ett
    coste_retraso = (e["T_L"] - T) * penal
    celdas = [md("# Enunciado"),
              md("Una empresa constructora ejecuta como contratista principal el proyecto cuyo cuadro de prelaciones, "
                 "duraciones (semanas) y operarios necesarios por actividad se adjunta. La plantilla propia disponible "
                 f"es de **{L} operarios**; cada operario-semana propio cuesta {c_propio} k€, uno contratado a través de "
                 f"una empresa de trabajo temporal (ETT) {c_ett} k€ y el retraso del proyecto tiene una penalidad de "
                 f"{penal} k€ por semana."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              _mensaje("Director de producción",
                       f"Para las semanas de pico no contratamos a nadie: lo subcontratamos a la empresa de siempre, que "
                       f"a su vez lo pasa a su subcontrata, esta a una tercera y esta a las cuadrillas de autónomos, que "
                       f"es donde está el precio: {c_cadena} k€ por operario-semana, menos que los nuestros. Cuatro "
                       "escalones, pero a nosotros nos factura uno solo y el Libro de Subcontratación lo llevan ellos."),
              _criterios(["S10", "S1"], constantes=["subcontratacion"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Dibuje el diagrama de cargas con inicios más tempranos y obtenga el pico y el exceso de operarios sobre "
        f"los {L} de plantilla, semana a semana. Nivele las cargas dentro de las holguras (nivelación) y repita el "
        "cálculo: ¿cuántos operarios-semana de exceso quedan?",
        f"(A) Calcule el coste de cubrir el exceso (tras nivelar) con la cadena de subcontratas y con una ETT, y el "
        f"coste de no cubrirlo: asignación con límite de {L} operarios, retraso y penalidad. ¿Cuál es la opción más "
        "barata y cuánto vale el «ahorro» de la cadena?",
        "(B) Según la norma y el criterio de referencia, ¿puede ejecutarse la propuesta del director de producción? "
        "¿Cuál es el último nivel que puede subcontratar y en qué condiciones se admite uno más?",
        "(C) ¿Qué propone al director de producción? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"plantilla, coste_propio, coste_ett, coste_cadena, penalidad = {L}, {c_propio}, {c_ett}, {c_cadena}, {penal}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Gantt con inicios más tempranos y carga de operarios"\n#| label: fig-ganttCargas\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas frente a la plantilla"\n#| label: fig-cargas\n'
                       "cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[plantilla])"))
    celdas.append(code('#| tbl-cap: "Desplazamientos de la nivelación (suma de cuadrados mínima dentro de las holguras)"\n#| label: tbl-nivelacion\n'
                       "proyecto_nivelado = proyecto.copy()\n"
                       "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo_niv = proyecto_nivelado.nivelar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')\n"
                       "cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)\n"
                       "desplazamientos[desplazamientos['desplazamientos'] > 0]"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas tras la nivelación"\n#| label: fig-cargasNivelado\n'
                       "plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=[plantilla])"))
    celdas.append(code('#| tbl-cap: "Exceso de operarios sobre la plantilla, antes y después de nivelar"\n#| label: tbl-exceso\n'
                       "exceso = (cargas - plantilla).clip(lower=0); exceso_niv = (cargas_niveladas - plantilla).clip(lower=0)\n"
                       "tabla = pd.DataFrame({'carga inicial': cargas, 'exceso inicial': exceso, 'carga nivelada': cargas_niveladas, 'exceso nivelado': exceso_niv})\n"
                       "tabla.loc['Total'] = tabla.sum()\n"
                       "display(tabla.astype(int).T)\n"
                       "Markdown(f'''Pico de **{int(cargas.max())}** operarios con inicios más tempranos y **{int(cargas_niveladas.max())}** tras nivelar, frente a {plantilla} de plantilla; "
                       "el exceso pasa de {int(exceso.sum())} a **{int(exceso_niv.sum())} operarios-semana** sin alargar el proyecto ({duracion} semanas).''')"))
    celdas += apartado(2)
    celdas.append(code("proyecto_limitado = proyecto.copy()\n"
                       "datos_asignados, gantt_asignado, dibujo_asig = proyecto_limitado.asignar(data=datos_proyecto.copy(), duration_label='duration',\n"
                       "                                                                      resource_label='resources', maximo=[plantilla], report=False, params=GANTT_PARAMS)\n"
                       "duracion_limitada = len(gantt_asignado.data.columns.drop('H_total'))\n"
                       "exceso_total = float(exceso_niv.sum())\n"
                       "opciones = pd.Series({'cadena de subcontratas (ilegal)': exceso_total * coste_cadena,\n"
                       "                      'ETT para el exceso': exceso_total * coste_ett,\n"
                       "                      'sólo plantilla: retraso × penalidad': (duracion_limitada - duracion) * penalidad})\n"
                       "display(opciones.round(1).to_frame('coste (k€)'))\n"
                       "Markdown(f'''Con sólo la plantilla el proyecto dura **{duracion_limitada}** semanas ({duracion_limitada - duracion} de retraso, {(duracion_limitada - duracion) * penalidad} k€ de penalidad). "
                       "La cadena «ahorra» {exceso_total * (coste_ett - coste_cadena):.1f} k€ frente a la ETT: ese es el precio de los cuatro escalones.''')"))
    celdas.append(code('#| fig-cap: "Gantt con asignación limitada a la plantilla"\n#| label: fig-ganttAsignado\npinta_tikz(dibujo_asig)'))
    celdas += apartado(3)
    celdas.append(md(f"**Norma.** La Ley 32/2006 limita la cadena: contratista → primer subcontratista → segundo → "
                     f"tercero, y **el tercer subcontratista no puede subcontratar**; los autónomos tampoco, ni las empresas "
                     f"que sólo aportan mano de obra (art. 5.2, letras d, e y f). La propuesta tiene cuatro escalones por "
                     f"debajo del contratista, el último de autónomos: **no puede ejecutarse**. Un nivel más sólo se admite excepcionalmente (caso fortuito, especialización, fuerza "
                     f"mayor), aprobado por la dirección facultativa antes y anotado en el Libro de Subcontratación (art. "
                     f"5.3); «lo llevan ellos» no exime: el libro es del contratista. **S10.** El «ahorro» de "
                     f"{coste_ett - coste_cadena:.1f} k€ sale de las condiciones de quien está al final de la cadena y de "
                     f"los riesgos de seguridad que la ley quiere evitar con el límite: no es una opción de la tabla. "
                     f"**S1.** La cadena no compensa ni minimiza: traslada el impacto a quien no está en la mesa. La "
                     f"respuesta no depende del apartado 2; el cálculo muestra que la opción legal (ETT tras nivelar) cuesta "
                     f"{coste_ett:.0f} k€ frente a {coste_retraso:.0f} de retraso."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('En una obra de construcción, el último nivel que puede subcontratar la ejecución es…', ['el segundo subcontratista (el tercero ya no puede)', 'el tercer subcontratista', 'cualquiera mientras se anote en el Libro de Subcontratación', 'el trabajador autónomo'],\n"
        "     'Ley 32/2006, art. 5.2.'),\n"
        "    ('Un nivel adicional de subcontratación se admite…', ['excepcionalmente, con aprobación previa de la dirección facultativa y anotación en el Libro de Subcontratación', 'siempre que el contratista lo autorice', 'nunca', 'si lo pide el cliente'],\n"
        "     'Art. 5.3: caso fortuito, especialización, complicaciones técnicas o fuerza mayor.'),\n"
        "    ('Nivelar las cargas dentro de las holguras…', ['reduce el pico sin alargar el proyecto', 'alarga el proyecto', 'reduce la duración', 'elimina siempre el exceso'],\n"
        "     'Lección 11: la nivelación mueve actividades no críticas.'),\n"
        "]", "Sostenibilidad 20 — subcontratación en cadena"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de producción: nivelar primero (el exceso baja de {int(e['exceso'].sum())} a "
                     f"{int(e['exceso_niv'].sum())} operarios-semana), cubrir el resto con ETT o contratación temporal propia "
                     f"({coste_ett:.0f} k€) o con una subcontrata de primer nivel con medios propios, y comprobar el 30 % de "
                     f"plantilla indefinida de quien se contrate. Si se quiere subcontratar la especialidad, un escalón, "
                     "con el libro al día. Prueba de publicidad: ¿explicaría a la Inspección de Trabajo, tras un accidente "
                     "en la cuadrilla del cuarto escalón, cómo llegó a la obra?"))
    return celdas, (f"pico {e['pico']} → {e['pico_niv']} nivelado frente a {L} de plantilla; cadena de 4 escalones "
                    f"{coste_cadena:.0f} k€ frente a ETT {coste_ett:.0f} k€; art. 5 Ley 32/2006")




# ---------------------------------------------------------------------------
# Caso 9 — La energía de la obra: grupo electrógeno o acometida tras nivelar
# ---------------------------------------------------------------------------
FACTOR_RED, FACTOR_GASOLEO = 0.15, 2.47        # kg CO₂ por kWh y por litro
CONSUMO_GRUPO, PRECIO_GASOLEO, PRECIO_RED = 0.30, 1.30, 0.18   # l/kWh, €/l, €/kWh
HORAS_SEMANA = 40


def _energia_en_obra(rng):
    e = {}

    def acepta(i):
        if not _base(i, criticas_1=False, T=(12, 20)):
            return False
        kw = pd.Series(rng.choice([5, 10, 15, 20, 25, 30], size=len(i["D"])), index=i["D"].index, name="kW")
        df = P.tabla_base(i, duration=i["D"], kW=kw)
        try:
            total = _cargas(i, df, "kW")
            desp, total_niv = _nivelar(i, df, "kW")
        except Exception:
            return False
        pico, pico_niv = int(total.max()), int(total_niv.max())
        if pico_niv >= pico or pico_niv < 20:
            return False
        e.update(kw=kw, df=df, total=total, total_niv=total_niv, pico=pico, pico_niv=pico_niv, desp=desp)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    df, C = e["df"], e["pico_niv"]
    plazo_acometida = int(rng.choice([3, 4, 5]))
    alquiler = float(rng.choice([0.5, 0.6, 0.8]))          # k€ por semana del grupo dimensionado para el pico
    coste_acometida = int(rng.choice([4, 6, 8]))            # k€
    precio = int(rng.choice([100, 150, 250]))               # €/t
    energia = float(e["total"].sum()) * HORAS_SEMANA        # kWh del proyecto (la nivelación no cambia la energía)
    energia_grupo_inicial = float(e["total_niv"].iloc[:plazo_acometida].sum()) * HORAS_SEMANA
    em_grupo = energia * CONSUMO_GRUPO * FACTOR_GASOLEO / 1000
    em_mixto = (energia_grupo_inicial * CONSUMO_GRUPO * FACTOR_GASOLEO + (energia - energia_grupo_inicial) * FACTOR_RED) / 1000
    coste_grupo = alquiler * T + energia * CONSUMO_GRUPO * PRECIO_GASOLEO / 1000
    coste_mixto = alquiler * plazo_acometida + energia_grupo_inicial * CONSUMO_GRUPO * PRECIO_GASOLEO / 1000 + coste_acometida + (energia - energia_grupo_inicial) * PRECIO_RED / 1000
    celdas = [md("# Enunciado"),
              md("Una obra de edificación necesita suministro eléctrico provisional. El cuadro de prelaciones, las "
                 "duraciones (semanas) y la potencia eléctrica que demanda cada actividad (`kW`, grúa, hormigonera, "
                 f"soldadura, bombas, secado) se adjuntan; se trabaja {HORAS_SEMANA} horas por semana a plena potencia."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"Opciones: un **grupo electrógeno** de gasóleo dimensionado para el pico de demanda (alquiler "
                 f"{alquiler} k€ por semana, consumo {CONSUMO_GRUPO} l por kWh, gasóleo a {PRECIO_GASOLEO} €/l, factor "
                 f"{FACTOR_GASOLEO} kg CO₂ por litro) o una **acometida provisional** de la compañía, que ofrece hasta "
                 f"**{C} kW** por {coste_acometida} k€ y tarda {plazo_acometida} semanas en estar disponible (electricidad a "
                 f"{PRECIO_RED} €/kWh, factor {FACTOR_RED} kg CO₂ por kWh). La organización aplica un precio sombra del "
                 f"carbono de {precio} €/t."),
              _mensaje("Jefe de obra",
                       f"Grupo electrógeno para toda la obra y nos olvidamos de la compañía: con inicios más tempranos "
                       f"el pico es de {e['pico']} kW, más de los {C} que dan, y la acometida no llega hasta la semana "
                       f"{plazo_acometida}. Se alquila uno grande y punto; el gasóleo va a costes generales."),
              _criterios(["S2", "S1", "S6"], constantes=["factores", "alcances"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Dibuje el diagrama de cargas eléctricas con inicios más tempranos y obtenga el pico. Nivele las cargas "
        f"dentro de las holguras y compruebe si el pico nivelado cabe en los {C} kW de la acometida sin alargar el "
        "proyecto. Calcule la energía total de la obra (kWh).",
        f"(A) Calcule el coste y las emisiones de las dos opciones: grupo electrógeno toda la obra, y grupo durante las "
        f"{plazo_acometida} primeras semanas y acometida después (con las cargas niveladas). Compare el coste total al "
        "precio sombra.",
        "(B) Según los criterios de referencia, ¿qué opción es la correcta y en qué escalón de la pirámide está cada "
        "una? ¿A qué alcance pertenecen las emisiones del grupo electrógeno y las de la acometida?",
        "(C) ¿Qué propone al jefe de obra? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"capacidad_acometida, plazo_acometida, alquiler_grupo, coste_acometida, precio_sombra = {C}, {plazo_acometida}, {alquiler}, {coste_acometida}, {precio}\n"
                       f"consumo_grupo, precio_gasoleo, precio_red, factor_gasoleo, factor_red, horas_semana = {CONSUMO_GRUPO}, {PRECIO_GASOLEO}, {PRECIO_RED}, {FACTOR_GASOLEO}, {FACTOR_RED}, {HORAS_SEMANA}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Gantt con inicios más tempranos y demanda eléctrica (kW)"\n#| label: fig-ganttCargas\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'kW', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas eléctricas frente a la capacidad de la acometida"\n#| label: fig-cargas\n'
                       "cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[capacidad_acometida])"))
    celdas.append(code('#| tbl-cap: "Desplazamientos de la nivelación"\n#| label: tbl-nivelacion\n'
                       "proyecto_nivelado = proyecto.copy()\n"
                       "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo_niv = proyecto_nivelado.nivelar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='kW')\n"
                       "cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)\n"
                       "desplazamientos[desplazamientos['desplazamientos'] > 0]"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas eléctricas tras la nivelación"\n#| label: fig-cargasNivelado\n'
                       "plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=[capacidad_acometida])"))
    celdas.append(code("energia = float(cargas.sum()) * horas_semana\n"
                       "Markdown(f'''Pico de **{int(cargas.max())} kW** con inicios más tempranos y **{int(cargas_niveladas.max())} kW** tras nivelar''' + "
                       "(' (cabe en la acometida)' if cargas_niveladas.max() <= capacidad_acometida else ' (no cabe)') + "
                       "f''', con la misma duración ({duracion} semanas). Energía de la obra: {energia:,.0f} kWh (la nivelación mueve la demanda, no la cambia).''')"))
    celdas += apartado(2)
    celdas.append(code("energia_inicial = float(cargas_niveladas.iloc[:plazo_acometida].sum()) * horas_semana\n"
                       "def opcion(semanas_grupo, kwh_grupo, kwh_red, acometida):\n"
                       "    litros = kwh_grupo * consumo_grupo\n"
                       "    coste = semanas_grupo * alquiler_grupo + litros * precio_gasoleo / 1000 + acometida + kwh_red * precio_red / 1000\n"
                       "    emisiones = (litros * factor_gasoleo + kwh_red * factor_red) / 1000\n"
                       "    return pd.Series({'alquiler del grupo (k€)': semanas_grupo * alquiler_grupo, 'gasóleo (litros)': litros,\n"
                       "                      'gasóleo (k€)': litros * precio_gasoleo / 1000, 'acometida (k€)': acometida, 'electricidad de red (k€)': kwh_red * precio_red / 1000,\n"
                       "                      'coste (k€)': coste, 'emisiones alcance 1 (t)': litros * factor_gasoleo / 1000, 'emisiones alcance 2 (t)': kwh_red * factor_red / 1000,\n"
                       "                      'emisiones (t CO₂)': emisiones, 'coste total al precio sombra (k€)': coste + emisiones * precio_sombra / 1000})\n"
                       "comparacion = pd.DataFrame({'grupo toda la obra': opcion(duracion, energia, 0, 0),\n"
                       "                            f'grupo {plazo_acometida} semanas + acometida': opcion(plazo_acometida, energia_inicial, energia - energia_inicial, coste_acometida)})\n"
                       "display(comparacion.round(2))\n"
                       "Markdown(f'''La opción mixta cuesta **{comparacion.iloc[5, 0] - comparacion.iloc[5, 1]:.1f} k€ menos** y emite "
                       "**{comparacion.iloc[8, 0] - comparacion.iloc[8, 1]:.1f} t menos**; al precio sombra la diferencia total es de "
                       "{comparacion.iloc[9, 0] - comparacion.iloc[9, 1]:.1f} k€.''')"))
    celdas += apartado(3)
    celdas.append(md(f"**S2.** Con el precio sombra la opción correcta es la de menor coste total: grupo las "
                     f"{plazo_acometida} primeras semanas y acometida después ({coste_mixto + em_mixto * precio / 1000:.1f} k€ "
                     f"frente a {coste_grupo + em_grupo * precio / 1000:.1f}); aquí ni siquiera hace falta el precio, porque "
                     f"también es más barata en dinero ({coste_mixto:.1f} frente a {coste_grupo:.1f} k€). El argumento del "
                     f"pico ({e['pico']} kW) desaparece al nivelar: el pico nivelado es {C} kW y cabe. **S1.** El grupo "
                     f"para toda la obra ni siquiera compensa; la opción mixta **minimiza** ({em_grupo - em_mixto:.1f} t "
                     f"menos); **evitar** sería pedir la acometida con la antelación suficiente para no necesitar el grupo. "
                     f"**S6.** El gasóleo del grupo es **alcance 1** (fuente controlada por la obra, aunque se alquile); la "
                     "electricidad de la acometida es **alcance 2**. Que el gasóleo «vaya a costes generales» no cambia "
                     "el alcance ni la cuenta."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Las emisiones del gasóleo de un grupo electrógeno alquilado en la obra son…', ['alcance 1', 'alcance 2', 'alcance 3', 'del arrendador'],\n"
        "     'S6: fuente controlada por la organización.'),\n"
        "    ('Nivelar las cargas eléctricas dentro de las holguras…', ['reduce el pico de potencia sin cambiar la energía ni la duración', 'reduce la energía consumida', 'alarga el proyecto', 'cambia las duraciones'],\n"
        "     'Lección 11: la nivelación mueve actividades no críticas.'),\n"
        "    ('Con un precio sombra fijado, entre dos formas de suministro se elige…', ['la de menor coste total (dinero más emisiones al precio sombra)', 'la de menor alquiler', 'la que pida el jefe de obra', 'la de menos trámites'],\n"
        "     'S2.'),\n"
        "]", "Sostenibilidad 20 — energía en obra"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al jefe de obra: pedir la acometida hoy, nivelar las cargas (tabla del apartado 1) y alquilar el "
                     f"grupo sólo {plazo_acometida} semanas; cifras del apartado 2. Si el pico nivelado no cupiera en la "
                     "acometida, la alternativa sería un grupo pequeño de apoyo en las semanas de pico, no uno grande "
                     "toda la obra. Prueba de publicidad: ¿pondría en la memoria de sostenibilidad del proyecto "
                     f"{em_grupo:.0f} t de gasóleo pudiendo haber emitido {em_mixto:.0f}?"))
    return celdas, (f"pico {e['pico']} kW → {C} kW nivelado: grupo toda la obra {coste_grupo:.0f} k€ y {em_grupo:.0f} t "
                    f"frente a grupo + acometida {coste_mixto:.0f} k€ y {em_mixto:.0f} t")


# ---------------------------------------------------------------------------
# Caso 10 — El acero importado: alcance 3, ajuste en frontera y plazo de entrega
# ---------------------------------------------------------------------------
COSTE_CAMION, COSTE_BARCO = 0.12, 0.02          # € por t·km
FACTOR_CAMION, FACTOR_BARCO = 0.10, 0.015       # kg CO₂ por t·km


def _acero_importado(rng):
    e = {}

    def acepta(i):
        if not _base(i):
            return False
        cal = i["calendario"]
        cands = [a for a in i["D"].index if 0 <= cal.loc[a, "H_total"] <= 3 and a not in i["iniciales"]]
        if not cands:
            return False
        X = str(rng.choice(cands))
        H = int(cal.loc[X, "H_total"])
        delta = int(rng.choice([2, 3, 4, 5]))
        D2 = i["D"].copy(); D2[X] += delta
        T2 = _T(i["proyecto"], D2)
        retraso = T2 - int(i["T"])
        if retraso < 1:
            return False
        e.update(X=X, H=H, delta=delta, retraso=retraso, T2=T2)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    X, delta, retraso = e["X"], e["delta"], e["retraso"]
    for _ in range(1000):
        Q = int(rng.integers(8, 40)) * 10
        p_eu = int(rng.integers(95, 116)) * 10
        p_imp = int(round(p_eu * float(rng.choice([0.75, 0.8, 0.85])) / 10)) * 10
        e_eu, e_imp = float(rng.choice([1.4, 1.6, 1.8])), float(rng.choice([2.2, 2.4, 2.6]))
        d_eu, d_sea, d_cam = int(rng.integers(3, 9)) * 100, int(rng.integers(8, 13)) * 1000, int(rng.integers(2, 5)) * 100
        ets = int(rng.integers(60, 91))
        penal = int(rng.choice([6, 8, 10, 12]))
        precio = int(rng.choice([100, 150, 250]))
        coste_eu = Q * (p_eu + d_eu * COSTE_CAMION) / 1000
        cbam = Q * e_imp * ets / 1000
        coste_imp = Q * (p_imp + d_sea * COSTE_BARCO + d_cam * COSTE_CAMION) / 1000
        em_eu = Q * (e_eu + d_eu * FACTOR_CAMION / 1000)
        em_imp = Q * (e_imp + (d_sea * FACTOR_BARCO + d_cam * FACTOR_CAMION) / 1000)
        tot_imp = coste_imp + cbam + retraso * penal
        if coste_imp < coste_eu - 1 and tot_imp > coste_eu + 1:
            break
    df = P.tabla_base(info, duration=D)
    ofertas = pd.DataFrame({"precio (€/t)": [p_eu, p_imp], "emisiones implícitas (t CO2/t)": [e_eu, e_imp],
                            "km por carretera": [d_eu, d_cam], "km por mar": [0, d_sea], "semanas de entrega": [0, delta]},
                           index=pd.Index(["acería europea", "acería importada"], name="oferta"))
    celdas = [md("# Enunciado"),
              md(f"Para la actividad **{X}** de un proyecto de edificación hacen falta **{Q} t de perfiles de acero**, que "
                 f"deben estar en obra antes de que empiece. El cuadro de prelaciones y las duraciones (semanas) se "
                 "adjuntan; el plan supone la entrega de la acería europea, que llega a tiempo. Las dos ofertas recibidas "
                 "se resumen a continuación (la columna «semanas de entrega» es el retraso de la entrega respecto al plan; "
                 f"la empresa importaría directamente y sería el declarante CBAM)."),
              code('#| tbl-cap: "Cuadro de prelaciones"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              code('#| tbl-cap: "Ofertas de suministro del acero"\n#| label: tbl-ofertas\n' + literal_dataframe(ofertas, nombre="ofertas", indice="oferta")),
              md(f"Transporte: {COSTE_CAMION} €/t·km por carretera y {COSTE_BARCO} €/t·km por mar; factores de emisión "
                 f"{FACTOR_CAMION} y {FACTOR_BARCO} kg CO₂ por t·km. Precio del derecho de emisión (ETS) para los "
                 f"certificados CBAM: {ets} €/t. Penalidad por retraso del proyecto: {penal} k€ por semana. Precio sombra "
                 f"del carbono de la organización: {precio} €/t."),
              _mensaje("Director de compras",
                       f"El acero importado sale a {p_imp} €/t frente a {p_eu}: un {1 - p_imp / p_eu:.0%} menos. Las "
                       "emisiones de la acería no son nuestras, son de ellos; y lo del CBAM es un trámite de aduanas. "
                       f"Las {delta} semanas de barco las absorbe la obra, siempre hay algo que hacer."),
              _criterios(["S2", "S6"], constantes=["cbam"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule el coste de cada oferta puesta en obra (precio, transporte y, en la importada, los certificados "
        "CBAM con el cálculo simplificado del enunciado) y las emisiones de cada una (implícitas más transporte).",
        f"(A) Calcule el efecto en el plazo de las {delta} semanas de entrega: holgura total de {X}, retraso del "
        "proyecto y penalidad. Obtenga el coste total de cada oferta con el retraso y, además, con las emisiones al "
        "precio sombra.",
        "(B) Según los criterios y la norma de referencia, ¿qué oferta se elige? ¿Es cierto que las emisiones de la "
        "acería «no son nuestras»? ¿Está la importación exenta del CBAM?",
        "(C) ¿Qué propone al director de compras? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"actividad_acero, toneladas, precio_ets, penalidad, precio_sombra = {X!r}, {Q}, {ets}, {penal}, {precio}\n"
                       f"coste_camion, coste_barco, factor_camion, factor_barco = {COSTE_CAMION}, {COSTE_BARCO}, {FACTOR_CAMION}, {FACTOR_BARCO}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Coste puesto en obra y emisiones de cada oferta"\n#| label: tbl-ofertasCalculo\n'
                       "o = ofertas.copy()\n"
                       "o['acero (k€)'] = toneladas * o['precio (€/t)'] / 1000\n"
                       "o['transporte (k€)'] = toneladas * (o['km por carretera'] * coste_camion + o['km por mar'] * coste_barco) / 1000\n"
                       "o['CBAM (k€)'] = [0.0, toneladas * o.loc['acería importada', 'emisiones implícitas (t CO2/t)'] * precio_ets / 1000]\n"
                       "o['coste en obra (k€)'] = o['acero (k€)'] + o['transporte (k€)'] + o['CBAM (k€)']\n"
                       "o['emisiones implícitas (t)'] = toneladas * o['emisiones implícitas (t CO2/t)']\n"
                       "o['emisiones del transporte (t)'] = toneladas * (o['km por carretera'] * factor_camion + o['km por mar'] * factor_barco) / 1000\n"
                       "o['emisiones (t CO₂)'] = o['emisiones implícitas (t)'] + o['emisiones del transporte (t)']\n"
                       "display(o.iloc[:, 5:].round(2).T)\n"
                       "Markdown(f'''Sin el CBAM la importada es {o.iloc[0, 8] - (o.iloc[1, 5] + o.iloc[1, 6]):.1f} k€ más barata; con los certificados "
                       "({o.iloc[1, 7]:.1f} k€) ''' + ('sigue siendo' if o.iloc[1, 8] < o.iloc[0, 8] else 'pasa a ser') + f''' {abs(o.iloc[0, 8] - o.iloc[1, 8]):.1f} k€ ''' + "
                       "('más barata' if o.iloc[1, 8] < o.iloc[0, 8] else 'más cara') + f'''. Emite {o.iloc[1, 11] - o.iloc[0, 11]:.0f} t más.''')"))
    celdas += apartado(2)
    celdas.append(code("holgura = int(calendario.loc[actividad_acero, 'H_total'])\n"
                       "duraciones_importado = datos_proyecto['duration'].copy(); duraciones_importado[actividad_acero] += ofertas.loc['acería importada', 'semanas de entrega']\n"
                       "duracion_importado = int(proyecto.duration(duraciones_importado))\n"
                       "retraso = duracion_importado - duracion\n"
                       "o['retraso (semanas)'] = [0, retraso]\n"
                       "o['penalidad (k€)'] = o['retraso (semanas)'] * penalidad\n"
                       "o['coste total (k€)'] = o['coste en obra (k€)'] + o['penalidad (k€)']\n"
                       "o['coste total al precio sombra (k€)'] = o['coste total (k€)'] + o['emisiones (t CO₂)'] * precio_sombra / 1000\n"
                       "display(o[['coste en obra (k€)', 'retraso (semanas)', 'penalidad (k€)', 'coste total (k€)', 'emisiones (t CO₂)', 'coste total al precio sombra (k€)']].round(1).T)\n"
                       "Markdown(f'''{actividad_acero} tiene {holgura} semana(s) de holgura total: las {int(ofertas.loc[\"acería importada\", \"semanas de entrega\"])} de entrega retrasan el proyecto "
                       "**{retraso}** semana(s) ({retraso * penalidad} k€). Coste total: europea **{o.iloc[0, 14]:.1f} k€**, importada **{o.iloc[1, 14]:.1f} k€**; "
                       "al precio sombra, {o.iloc[0, 15]:.1f} frente a {o.iloc[1, 15]:.1f} k€.''')"))
    celdas += apartado(3)
    celdas.append(md(f"**S2.** La comparación correcta es el coste total puesto en obra con el ajuste en frontera, el "
                     f"transporte, el retraso y la externalidad al precio sombra: la **acería europea** "
                     f"({coste_eu + retraso * 0 + em_eu * precio / 1000:.1f} k€ frente a "
                     f"{tot_imp + em_imp * precio / 1000:.1f}). El {1 - p_imp / p_eu:.0%} del director de compras es la "
                     f"primera columna de una tabla de seis. **S6.** Las emisiones implícitas del acero comprado son "
                     f"**alcance 3** del proyecto: {Q * e_imp:.0f} t frente a {Q * e_eu:.0f}; ser de la acería no las "
                     f"saca del inventario, y el CBAM existe precisamente para que paguen el mismo carbono que las "
                     f"europeas. **Norma.** La importación ({Q} t) supera las 50 t anuales del umbral de minimis: la "
                     f"empresa, como declarante, compra los certificados ({cbam:.1f} k€ con el cálculo simplificado; la "
                     "liquidación real descuenta la asignación gratuita que aún reciben las acerías europeas y el precio "
                     f"del carbono pagado en origen). Y las {delta} semanas no «las absorbe la obra»: las absorbe la "
                     f"holgura de {X} ({e['H']} semana(s)), y lo que sobra va al plazo."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Las emisiones implícitas del acero comprado por el proyecto son, en su inventario…', ['alcance 3', 'alcance 1', 'alcance 2', 'de la acería, no del proyecto'],\n"
        "     'S6: materiales comprados.'),\n"
        "    ('Un importador de 120 t de acero al año, con el CBAM en su periodo definitivo…', ['debe adquirir certificados por las emisiones implícitas (supera las 50 t de minimis)', 'está exento', 'sólo declara, sin pagar', 'paga el IVA del carbono'],\n"
        "     'Reglamento (UE) 2023/956 modificado por el 2025/2083: umbral de 50 t por importador y año.'),\n"
        "    ('Un retraso de entrega de d semanas en una actividad con holgura total H retrasa el proyecto…', ['máx(0, d − H) semanas', 'd semanas', 'H semanas', 'nada si la actividad no es inicial'],\n"
        "     'Lección 05: la holgura total es lo que puede retrasarse sin mover el fin.'),\n"
        "]", "Sostenibilidad 20 — acero importado"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de compras: la tabla del apartado 2 entera, no la primera columna; si se quiere "
                     f"negociar, negociar con la acería europea con esa tabla en la mano, o pedir a la importada la "
                     f"entrega en plazo y el certificado de emisiones verificadas (si son menores que el valor por defecto, "
                     "el CBAM baja). Prueba del ciclo de vida: la diferencia de "
                     f"{em_imp - em_eu:.0f} t no desaparece por facturarla otro; prueba de publicidad: ¿explicaría al "
                     f"cliente las {retraso} semanas de retraso por un acero que, puesto en obra, salía más caro?"))
    return celdas, (f"{Q} t de acero: importado {coste_imp:.0f} k€ frente a europeo {coste_eu:.0f} k€, pero {tot_imp:.0f} k€ "
                    f"con CBAM y {retraso} semana(s) de retraso; alcance 3")


# ---------------------------------------------------------------------------
# Caso 11 — El caso de negocio que termina en la puesta en servicio
# ---------------------------------------------------------------------------
def _desmantelamiento(rng):
    quiero_negativo = bool(rng.random() < 0.4)
    for intento in range(5000):
        I = int(rng.integers(20, 60)) * 100
        N = int(rng.choice([15, 20, 25]))
        k = float(rng.choice([0.05, 0.06, 0.08]))
        anualidad = (1 - (1 + k) ** -N) / k
        objetivo_van = I * (float(rng.uniform(0.005, 0.03)) if quiero_negativo else float(rng.uniform(0.03, 0.12)))
        F = int(round((I + objetivo_van) / anualidad / 10)) * 10
        Dm = int(round(I * float(rng.choice([0.08, 0.10, 0.12, 0.15])) / 10)) * 10
        R = int(round(I * float(rng.choice([0.01, 0.02, 0.03])) / 10)) * 10
        van_sin = -I + F * anualidad
        va_desm = Dm / (1 + k) ** N
        va_res = R / (1 + k) ** N
        van_con = van_sin - va_desm + va_res
        if van_sin <= 0:
            continue
        if (van_con < 0) != quiero_negativo and intento < 4000:
            continue
        break
    cuota = Dm * k / ((1 + k) ** N - 1)
    tabla = pd.DataFrame({"concepto": ["inversión inicial (k€)", "flujo de caja neto anual (k€)", "años de explotación",
                                       "coste de desmantelamiento y restauración al final (k€)", "valor de los materiales recuperados al final (k€)",
                                       "coste de capital"],
                          "valor": [I, F, N, Dm, R, k]}).set_index("concepto")
    celdas = [md("# Enunciado"),
              md("El caso de negocio de una planta industrial (instalación en suelo rústico arrendado) presenta la "
                 "inversión y los flujos de caja de la explotación que se indican. El contrato de arrendamiento y la "
                 "autorización obligan, al final de la vida útil, a **desmantelar la instalación y restaurar el terreno**; "
                 "el equipo ha estimado ese coste y el valor de los materiales que se recuperarían."),
              code('#| tbl-cap: "Datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(tabla, indice="concepto")),
              _mensaje("Director financiero",
                       f"El caso de negocio termina en la puesta en servicio más los {N} años de explotación; el "
                       f"desmantelamiento es dentro de {N} años y no es de este proyecto ni de esta decisión. Quien "
                       "explote la planta entonces, que lo pague. Con el VAN tal como está, se aprueba."),
              _criterios(["S4", "S1"], constantes=["desmantelamiento"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule el VAN del caso de negocio tal como lo presenta el director financiero y el VAN incluyendo el "
        "desmantelamiento y la recuperación de materiales al final de la vida útil. ¿Cambia la decisión?",
        "(A) Calcule el valor actual de la obligación de desmantelamiento (la provisión que se reconoce al inicio) y "
        "la cantidad anual constante que, capitalizada al coste de capital, reuniría el coste del desmantelamiento al "
        "final. ¿Qué parte de la inversión «escondida» representa?",
        "(B) Según el criterio y la norma de referencia, ¿puede presentarse el caso de negocio sin el fin de vida? "
        "¿A quién se traslada el coste si no se provisiona, y en qué escalón de la pirámide queda cada opción?",
        "(C) Redacte el párrafo del caso de negocio y lo que diría al director financiero. Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"inversion, flujo_anual, anos, desmantelamiento, recuperado, k = {I}, {F}, {N}, {Dm}, {R}, {k}\n"
                       "def van(flujos, k):\n"
                       "    return float(sum(f / (1 + k) ** t for t, f in enumerate(flujos)))\n"
                       "flujos_sin = [-inversion] + [flujo_anual] * anos\n"
                       "flujos_con = list(flujos_sin); flujos_con[-1] += recuperado - desmantelamiento\n"
                       "van_sin, van_con = van(flujos_sin, k), van(flujos_con, k)"))
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Flujos y valor actual con y sin el fin de vida"\n#| label: tbl-flujos\n'
                       "detalle = pd.DataFrame({'flujo sin fin de vida': flujos_sin, 'flujo con fin de vida': flujos_con,\n"
                       "                        'factor 1/(1+k)^t': [1 / (1 + k) ** t for t in range(anos + 1)]}, index=pd.Index(range(anos + 1), name='año'))\n"
                       "detalle['VA sin'] = detalle['flujo sin fin de vida'] * detalle['factor 1/(1+k)^t']\n"
                       "detalle['VA con'] = detalle['flujo con fin de vida'] * detalle['factor 1/(1+k)^t']\n"
                       "detalle.loc['Total'] = [None, None, None, detalle['VA sin'].sum(), detalle['VA con'].sum()]\n"
                       "display(detalle.round(2))\n"
                       "Markdown(f'''VAN sin el fin de vida: **{van_sin:.1f} k€**; con desmantelamiento y materiales recuperados: **{van_con:.1f} k€**. ''' + "
                       "('La decisión cambia: el proyecto deja de crear valor.' if van_con < 0 else 'El proyecto sigue siendo viable, con menos margen.'))"))
    celdas += apartado(2)
    celdas.append(code("provision = desmantelamiento / (1 + k) ** anos\n"
                       "cuota_anual = desmantelamiento * k / ((1 + k) ** anos - 1)\n"
                       "Markdown(f'''Valor actual de la obligación: {desmantelamiento} / (1 + {k})^{anos} = **{provision:.1f} k€** ({provision / inversion:.1%} de la inversión), "
                       "que el PGC incorpora al coste del activo con su provisión. Cuota anual que reúne los {desmantelamiento} k€ al {k:.0%}: {desmantelamiento} · k / ((1 + k)^{anos} − 1) = **{cuota_anual:.1f} k€ al año** "
                       "({cuota_anual / flujo_anual:.1%} del flujo anual).''')"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma y S4.** La obligación de desmantelar nace con la inversión (el contrato y la autorización "
                     f"la imponen): el PGC incorpora al coste del inmovilizado el valor actual de esa obligación "
                     f"({va_desm:.1f} k€) y reconoce la provisión desde el primer día. Un caso de negocio sin el fin de vida "
                     f"está orientado a los entregables, no a los resultados; «dentro de {N} años» no lo saca del "
                     f"proyecto, porque la decisión que lo causa se toma hoy. "
                     + (f"Con el fin de vida el VAN es negativo ({van_con:.1f} k€): el proyecto **no debe aprobarse** tal "
                        "como está (o debe rediseñarse para abaratar el desmantelamiento o alargar la explotación). "
                        if van_con < 0 else
                        f"Con el fin de vida el VAN sigue siendo positivo ({van_con:.1f} k€): el proyecto se aprueba, pero "
                        "con la cifra correcta y la provisión. ")
                     + "**S1.** No provisionar traslada el coste a quien explote la planta en el año "
                     f"{N} —o, si entonces no hay nadie, al arrendador y a la comunidad, que se quedan con la instalación "
                     "abandonada—: no es compensar ni minimizar, es trasladar. Desmantelar y restaurar es el escalón de "
                     "**restaurar**; diseñar la planta para su desmontaje y reutilización (y abaratar los "
                     f"{Dm} k€) sería **minimizar**."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('La estimación del coste de desmantelamiento de una instalación, cuando la obligación nace de su uso…', ['forma parte del coste del activo por su valor actual, con la provisión correspondiente', 'se contabiliza cuando se desmantela', 'no se contabiliza si faltan más de 10 años', 'la paga quien explote la planta al final'],\n"
        "     'PGC, norma de registro y valoración 2.ª, apartado 1.'),\n"
        "    ('Un caso de negocio que termina en la puesta en servicio más la explotación, sin el fin de vida, está orientado…', ['a los entregables, no a los resultados', 'a los resultados', 'al proceso', 'a la sostenibilidad'],\n"
        "     'S4: orientación temporal de la figura 3-8 del PMBOK 8.'),\n"
        "    ('La cuota anual constante que reúne un importe D al cabo de N años al tipo k es…', ['D·k / ((1+k)^N − 1)', 'D / N', 'D / (1+k)^N', 'D·k'],\n"
        "     'Factor de fondo de amortización.'),\n"
        "]", "Sostenibilidad 20 — desmantelamiento"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(("Párrafo para el caso de negocio (modelo): «La inversión de "
                      f"{I} k€ genera {F} k€ anuales durante {N} años; al final, el desmantelamiento y la restauración "
                      f"del terreno cuestan {Dm} k€ (valor actual {va_desm:.0f} k€), que se provisionan desde el inicio, y "
                      f"los materiales recuperados valen {R} k€. VAN: {van_con:.0f} k€" +
                      (". No se recomienda aprobar el proyecto en estas condiciones; se proponen [alternativas]»"
                       if van_con < 0 else ". Se recomienda aprobar el proyecto con la provisión constituida»")
                      + f". Al director financiero: el VAN «tal como está» ({van_sin:.0f} k€) es el VAN de otro proyecto, uno "
                      "sin obligación de desmantelar; presentarlo así al comité es una verdad a medias (lección 19, C2). "
                      "Prueba del ciclo de vida: ¿quién paga, y con qué, en el año "
                      f"{N}?")))
    return celdas, (f"VAN {van_sin:.0f} k€ sin el fin de vida frente a {van_con:.0f} k€ con desmantelamiento ({Dm} k€ en el año {N}); "
                    "provisión según el PGC")


# ---------------------------------------------------------------------------
# Caso 12 — La ventana biológica de la DIA: desbroces fuera de la época de cría
# ---------------------------------------------------------------------------
VENTANAS_DIA = [("1 de marzo", "31 de mayo", 9, 22), ("1 de marzo", "30 de junio", 9, 26), ("1 de abril", "30 de junio", 13, 26)]


def _esperas_ventana(p, D, afectadas, v0, v1):
    """Esperas mínimas para que ninguna actividad afectada ocupe semanas de la ventana [v0, v1] (semana 1 = primera
    del proyecto): se procesa por inicio más temprano y se recalcula el calendario tras cada cambio."""
    espera = pd.Series(0, index=D.index, dtype=int)
    for _ in range(60):
        cal = _cal(p, D + espera)
        cambio = False
        for a in sorted(afectadas, key=lambda x: (int(cal.loc[x, "inicio_mas_temprano"]), x)):
            es, d = int(cal.loc[a, "inicio_mas_temprano"]), int(D[a])
            nueva = 0 if es + d <= v0 - 1 else max(0, v1 - es)
            if nueva != int(espera[a]):
                espera[a] = nueva
                cambio = True
                break
        if not cambio:
            break
    return espera


def _inicio_valido(cal, afectadas, v0, v1, x):
    """¿Ninguna actividad afectada toca la ventana si el proyecto empieza x semanas antes (x < 0: después)?"""
    for a in afectadas:
        es, ef = int(cal.loc[a, "inicio_mas_temprano"]) - x, int(cal.loc[a, "fin_mas_temprano"]) - x
        if not (ef <= v0 - 1 or es >= v1):
            return False
    return True


def _ventana_ambiental(rng):
    e = {}
    d0, d1, v0, v1 = VENTANAS_DIA[int(rng.integers(0, len(VENTANAS_DIA)))]

    def acepta(i):
        if not _base(i, criticas_1=False, T=(16, 26)):
            return False
        D, cal = i["D"], i["calendario"]
        afectadas = _elegir(rng, list(D.index), int(rng.integers(3, 5)))
        antes = [a for a in afectadas if int(cal.loc[a, "inicio_mas_temprano"]) + int(D[a]) <= v0 - 1]
        solapan = [a for a in afectadas
                   if max(0, min(int(cal.loc[a, "fin_mas_temprano"]), v1) - max(int(cal.loc[a, "inicio_mas_temprano"]), v0 - 1)) > 0]
        if not antes or len(solapan) < 2:
            return False
        espera = _esperas_ventana(i["proyecto"], D, afectadas, v0, v1)
        T_dia = _T(i["proyecto"], D + espera)
        retraso = T_dia - int(i["T"])
        if not (2 <= retraso <= 10):
            return False
        adelanto = next((x for x in range(0, 60) if _inicio_valido(cal, afectadas, v0, v1, x)), None)
        retraso_inicio = next((y for y in range(0, 60) if _inicio_valido(cal, afectadas, v0, v1, -y)), None)
        if adelanto is None or retraso_inicio is None or adelanto == 0:
            return False
        e.update(afectadas=afectadas, antes=antes, solapan=solapan, espera=espera, T_dia=T_dia, retraso=retraso,
                 adelanto=adelanto, retraso_inicio=retraso_inicio)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    penal = int(rng.choice([6, 8, 10, 12]))
    afeccion = pd.Series([1 if a in e["afectadas"] else 0 for a in D.index], index=D.index, name="afeccion")
    df = P.tabla_base(info, duration=D, afeccion=afeccion)
    solapan, antes, retraso = e["solapan"], e["antes"], e["retraso"]
    celdas = [md("# Enunciado"),
              md("Una empresa ejecuta la obra civil de una planta de generación en una zona de aves esteparias. La "
                 "**declaración de impacto ambiental (DIA)** del proyecto, incorporada a la autorización, impone la "
                 f"condición siguiente: «no se realizarán desbroces, talas, movimientos de tierra ni voladuras entre el "
                 f"**{d0}** y el **{d1}**, periodo reproductor de las especies esteparias del área». El proyecto empieza "
                 f"el 1 de enero (semana 1), de modo que la ventana ocupa las **semanas {v0} a {v1}**. El cuadro de "
                 "prelaciones, las duraciones (semanas) y qué actividades están afectadas por la condición (`afeccion` = 1: "
                 "desbroce, explanación, zanjas, cimentaciones con voladura) se adjuntan. El retraso de la entrega tiene "
                 f"una penalidad de {penal} k€ por semana."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              _mensaje("Director de obra",
                       f"El cronograma va con inicios más tempranos y así se ha presentado. Los trabajos de {solapan[0]} "
                       f"caen dentro de la ventana: los hacemos igual con un biólogo delante y, si aparece un nido, lo "
                       f"movemos. La ventana es una recomendación de la DIA, no está en el contrato con el cliente, y "
                       f"parar nos costaría {retraso} semanas, {retraso * penal} k€ de penalidad."),
              _criterios(["S12", "S11"], constantes=["dia_condiciones"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Con el calendario de inicios más tempranos, identifique qué actividades afectadas terminan antes de la "
        "ventana y cuáles la ocupan (semanas dentro de la ventana) y, para estas, la espera necesaria para empezar "
        f"después del {d1} y si cabe en su holgura total. Obtenga el cronograma con las esperas (recalculando el "
        "calendario después de cada una) y la duración del proyecto.",
        "(A) Calcule el retraso y la penalidad del plan con esperas. Obtenga el adelanto mínimo del inicio del "
        "proyecto con el que ninguna actividad afectada tocaría la ventana, y el retraso mínimo del inicio con el "
        "mismo efecto, y compare los tres planes.",
        "(B) Según el criterio y la norma de referencia, ¿puede ejecutarse el desbroce dentro de la ventana «con un "
        "biólogo delante»? ¿Es la ventana una restricción del cronograma o un riesgo que se acepta? ¿Qué actividades "
        "deben reprogramarse?",
        "(C) ¿Qué propone al director de obra y qué comunica al cliente y al órgano ambiental? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"ventana, penalidad = ({v0}, {v1}), {penal}\n" + SETUP +
                       "afectadas = sorted(datos_proyecto.index[datos_proyecto['afeccion'] == 1])\n"
                       "def semanas_en_ventana(es, d):\n"
                       "    '''Semanas de una actividad que empieza en es y dura d que caen en la ventana (semana 1 = primera del proyecto).'''\n"
                       "    return max(0, min(es + d, ventana[1]) - max(es, ventana[0] - 1))"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones del plan", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Actividades afectadas: fechas más tempranas, semanas en la ventana y espera necesaria"\n#| label: tbl-ventana\n'
                       "tabla = calendario.loc[afectadas, ['inicio_mas_temprano', 'fin_mas_temprano', 'H_total']].copy()\n"
                       "tabla['semanas en la ventana'] = [semanas_en_ventana(int(r['inicio_mas_temprano']), int(datos_proyecto.loc[a, 'duration'])) for a, r in tabla.iterrows()]\n"
                       "tabla['espera para empezar en la semana ' + str(ventana[1] + 1)] = [max(0, ventana[1] - int(r['inicio_mas_temprano'])) if r['semanas en la ventana'] > 0 else 0 for a, r in tabla.iterrows()]\n"
                       "tabla['cabe en la holgura'] = tabla.iloc[:, -1] <= tabla['H_total']\n"
                       "tabla"))
    celdas.append(code('#| tbl-cap: "Esperas del plan (recalculando el calendario tras cada una)"\n#| label: tbl-esperas\n'
                       "espera = pd.Series(0, index=datos_proyecto.index, dtype=int)\n"
                       "for _ in range(60):\n"
                       "    cal = proyecto.calendar(datos_proyecto['duration'] + espera)\n"
                       "    cambio = False\n"
                       "    for a in sorted(afectadas, key=lambda x: (int(cal.loc[x, 'inicio_mas_temprano']), x)):\n"
                       "        es, d = int(cal.loc[a, 'inicio_mas_temprano']), int(datos_proyecto.loc[a, 'duration'])\n"
                       "        nueva = 0 if es + d <= ventana[0] - 1 else max(0, ventana[1] - es)\n"
                       "        if nueva != int(espera[a]):\n"
                       "            espera[a] = nueva; cambio = True\n"
                       "            break\n"
                       "    if not cambio:\n"
                       "        break\n"
                       "duraciones_dia = datos_proyecto['duration'] + espera\n"
                       "calendario_dia = proyecto.calendar(duraciones_dia)\n"
                       "duracion_dia = int(proyecto.duration(duraciones_dia))\n"
                       "resumen = pd.DataFrame({'espera (semanas)': espera[afectadas], 'inicio real': calendario_dia.loc[afectadas, 'inicio_mas_temprano'] + espera[afectadas],\n"
                       "                        'fin': calendario_dia.loc[afectadas, 'fin_mas_temprano']})\n"
                       "display(resumen)\n"
                       "antes = [a for a in afectadas if espera[a] == 0 and semanas_en_ventana(int(calendario.loc[a, 'inicio_mas_temprano']), int(datos_proyecto.loc[a, 'duration'])) == 0]\n"
                       "Markdown(f'''Terminan antes de la ventana: {', '.join(antes)}. Esperan hasta la semana {ventana[1] + 1}: {', '.join(a for a in afectadas if espera[a] > 0)}. "
                       "Con las esperas el proyecto dura **{duracion_dia}** semanas frente a {duracion} del plan.''')"))
    celdas.append(code('#| fig-cap: "Gantt del plan con las esperas (la barra incluye la espera de cada actividad afectada)"\n#| label: fig-ganttDia\n'
                       "datos_dia = datos_proyecto.copy(); datos_dia['duration'] = duraciones_dia\n"
                       "_g, _dibujo = proyecto.copy().gantt(datos_dia, 'duration', holguras=True, tikz=True, params=dict(inner_text=None, **GANTT_PARAMS))\n"
                       "pinta_tikz(_dibujo)"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Tres formas de respetar la ventana"\n#| label: tbl-planes\n'
                       "retraso = duracion_dia - duracion\n"
                       "def inicio_valido(x):\n"
                       "    '''x > 0: el proyecto empieza x semanas antes; x < 0: después. ¿Ninguna afectada toca la ventana?'''\n"
                       "    for a in afectadas:\n"
                       "        es, ef = int(calendario.loc[a, 'inicio_mas_temprano']) - x, int(calendario.loc[a, 'fin_mas_temprano']) - x\n"
                       "        if not (ef <= ventana[0] - 1 or es >= ventana[1]):\n"
                       "            return False\n"
                       "    return True\n"
                       "adelanto = next(x for x in range(0, 60) if inicio_valido(x))\n"
                       "retraso_inicio = next(y for y in range(0, 60) if inicio_valido(-y))\n"
                       "planes = pd.DataFrame({'esperas selectivas': [duracion_dia, retraso, retraso * penalidad],\n"
                       "                       f'adelantar el inicio {adelanto} semanas': [duracion, -adelanto, 0],\n"
                       "                       f'retrasar el inicio {retraso_inicio} semanas': [duracion + retraso_inicio, retraso_inicio, retraso_inicio * penalidad]},\n"
                       "                      index=['fin respecto al inicio previsto (semana)', 'retraso de la entrega (semanas)', 'penalidad (k€)'])\n"
                       "display(planes)\n"
                       "Markdown(f'''Esperar sólo en las actividades afectadas retrasa la entrega **{retraso}** semanas ({retraso * penalidad} k€); empezar la obra "
                       "{adelanto} semanas antes (si la autorización y el contrato lo permiten) la adelanta; empezar {retraso_inicio} semanas después, "
                       "con todas las actividades afectadas tras la ventana, cuesta {retraso_inicio * penalidad} k€. Son los tres planes posibles; "
                       "«hacerlo igual» no es el cuarto.''')"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma.** La ventana es una condición de la DIA incorporada a la autorización: ejecutar los trabajos "
                     f"dentro de ella es incumplir la condición (infracción grave, Ley 21/2013, art. 55.3.c) con "
                     f"independencia de que aparezca un nido; y si aparece, molestar a la fauna en reproducción o mover el "
                     f"nido está prohibido con o sin biólogo (Ley 42/2007, art. 54.5): el biólogo puede documentar el daño, "
                     f"no autorizarlo. **S12.** La ventana no tiene probabilidad, tiene fecha: es una **restricción del "
                     f"cronograma** (inicio no antes de la semana {v1 + 1} o fin no después de la {v0 - 1}) de las actividades "
                     f"afectadas, no un riesgo que se acepta pagando {retraso * penal} k€. Las actividades que deben "
                     f"reprogramarse las decide el calendario, no la opinión: con inicios más tempranos ocupan la ventana "
                     f"**{', '.join(solapan)}**; {', '.join(antes)} terminan antes y no se tocan. «No está en el contrato con "
                     f"el cliente» es irrelevante: está en la autorización sin la cual no hay obra, y el cliente la conoce. "
                     f"**S11.** Lo que sí es un riesgo del proyecto es que el órgano ambiental amplíe la ventana, que un nido "
                     "aparezca fuera de ella o que el tiempo retrase lo que debía terminar antes: eso entra en el registro "
                     "con su probabilidad, su impacto y su respuesta (empezar antes, reordenar, reserva)."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Una condición de la DIA que prohíbe determinadas actividades en una época del año es, para el cronograma…', ['una restricción de fecha de las actividades afectadas', 'un riesgo que se acepta si la penalidad lo compensa', 'una recomendación', 'una reserva de gestión'],\n"
        "     'S12: tiene fecha, no probabilidad.'),\n"
        "    ('Incumplir las condiciones ambientales establecidas en la declaración de impacto ambiental es…', ['infracción grave (Ley 21/2013, art. 55.3.c)', 'una irregularidad sin sanción', 'legal si lo supervisa un biólogo', 'sólo un incumplimiento contractual con el cliente'],\n"
        "     'Art. 55.3.c) de la Ley 21/2013; además, Ley 42/2007, art. 54.5.'),\n"
        "    ('Una actividad afectada que, con inicios más tempranos, ocupa semanas de la ventana retrasa el proyecto…', ['sólo si la espera necesaria supera su holgura total (y el calendario se recalcula)', 'siempre, tantas semanas como la espera', 'nunca', 'sólo si es inicial'],\n"
        "     'Lección 05: la holgura total absorbe la espera; lo que sobra va al plazo.'),\n"
        "]", "Sostenibilidad 20 — ventana ambiental"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de obra: la tabla del apartado 2; de los tres planes, el que menos cuesta y el que la "
                     f"autorización permite, y la lista de lo que sí se puede hacer dentro de la ventana (todo lo que no "
                     f"tiene afección: {', '.join(a for a in D.index if a not in e['afectadas'])}). Al cliente: la condición "
                     f"de la DIA con su fecha, el plan elegido y su efecto en la entrega, y la petición de revisar la "
                     f"penalidad por una restricción que estaba en la autorización antes de firmar. Al órgano ambiental, si "
                     "hace falta: la consulta sobre el calendario, no el hecho consumado. Prueba de publicidad: ¿explicaría "
                     "al inspector del Seprona que el desbroce de abril «lo vigilaba un biólogo»?"))
    return celdas, (f"ventana {d0}–{d1} (semanas {v0}–{v1}): {len(solapan)} actividades esperan; {T} → {e['T_dia']} semanas "
                    f"(+{retraso}); adelantar {e['adelanto']} o retrasar {e['retraso_inicio']} el inicio")


# ---------------------------------------------------------------------------
# Caso 13 — Las tierras de la excavación: vertedero y préstamo, o reutilizar según el calendario
# ---------------------------------------------------------------------------
DENSIDAD_TIERRAS = 1.8                                   # t por m³
COSTE_VERTEDERO, COSTE_PRESTAMO = 4.0, 9.0               # € por m³: canon del vertedero de inertes (impuesto incluido), préstamo en cantera
COSTE_INTERNO, COSTE_ACOPIO = 1.2, 2.5                   # € por m³: transporte interno; doble manipulación si hay acopio
KM_VERTEDERO, KM_PRESTAMO = 25, 30


def _tierras_sobrantes(rng):
    e = {}
    quiero_retraso = bool(rng.random() < 0.5)

    def acepta(i):
        if not _base(i, criticas_1=False, T=(12, 22)):
            return False
        D, cal, preds = i["D"], i["calendario"], i["preds"]
        G = P.grafo(preds)
        pares = []
        for X in sorted(D.index):
            for Y in sorted(D.index):
                if X == Y or Y in nx.ancestors(G, X) or X in nx.ancestors(G, Y):
                    continue
                es_y, ef_x, ls_y = int(cal.loc[Y, "inicio_mas_temprano"]), int(cal.loc[X, "fin_mas_temprano"]), int(cal.loc[Y, "inicio_mas_tardio"])
                if es_y < ef_x and int(D[X]) >= 2 and int(D[Y]) >= 2:
                    pares.append((X, Y, ef_x, es_y, ls_y))
        if not pares:
            return False
        con_retraso = [p for p in pares if p[2] > p[4]]
        sin_retraso = [p for p in pares if p[2] <= p[4]]
        cands = con_retraso if quiero_retraso else sin_retraso
        if not cands:
            return False
        X, Y, ef_x, es_y, ls_y = cands[int(rng.integers(0, len(cands)))]
        retraso = max(0, ef_x - ls_y)
        if retraso > 4:
            return False
        e.update(X=X, Y=Y, ef_x=ef_x, es_y=es_y, ls_y=ls_y, retraso=retraso)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T, cal = info["D"], int(info["T"]), info["calendario"]
    X, Y, retraso = e["X"], e["Y"], e["retraso"]
    V_X = int(rng.integers(12, 36)) * 1000
    V_Y = int(round(V_X * float(rng.choice([0.7, 0.85, 1.0, 1.2])) / 1000)) * 1000
    penal = int(rng.choice([8, 10, 12, 15]))
    tierras = pd.Series(0, index=D.index, name="tierras_m3")
    tierras[X], tierras[Y] = V_X, -V_Y
    df = P.tabla_base(info, duration=D, tierras_m3=tierras)
    reutilizable = min(V_X, V_Y)
    sobrante, deficit = V_X - reutilizable, V_Y - reutilizable
    c_vert = COSTE_VERTEDERO + DENSIDAD_TIERRAS * KM_VERTEDERO * COSTE_CAMION
    c_prest = COSTE_PRESTAMO + DENSIDAD_TIERRAS * KM_PRESTAMO * COSTE_CAMION
    coste_propuesta = (V_X * c_vert + V_Y * c_prest) / 1000
    em_propuesta = DENSIDAD_TIERRAS * (V_X * KM_VERTEDERO + V_Y * KM_PRESTAMO) * FACTOR_CAMION / 1000 / 1000
    necesita_acopio = e["es_y"] > e["ef_x"]
    coste_reut = (reutilizable * (COSTE_INTERNO + (COSTE_ACOPIO if necesita_acopio else 0)) + sobrante * c_vert + deficit * c_prest) / 1000 + retraso * penal
    em_reut = DENSIDAD_TIERRAS * (sobrante * KM_VERTEDERO + deficit * KM_PRESTAMO) * FACTOR_CAMION / 1000 / 1000
    celdas = [md("# Enunciado"),
              md("En la obra de una variante de carretera, la actividad **" + X + "** es la excavación de un desmonte y la "
                 "actividad **" + Y + "** el terraplén de un relleno. El cuadro de prelaciones, las duraciones (semanas) y el "
                 "movimiento de tierras de cada actividad (`tierras_m3`: positivo, m³ que se excavan; negativo, m³ que "
                 "hacen falta) se adjuntan. El estudio geotécnico confirma que el material del desmonte es apto para el "
                 "relleno y no está contaminado."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"Costes (€ por m³): llevar tierras al vertedero de inertes, {COSTE_VERTEDERO} de canon (impuesto incluido) "
                 f"más el transporte a {KM_VERTEDERO} km; traer material de préstamo de la cantera, {COSTE_PRESTAMO} más el "
                 f"transporte a {KM_PRESTAMO} km; mover las tierras dentro de la obra, {COSTE_INTERNO}; si hay que acopiarlas "
                 f"y volver a cargarlas, {COSTE_ACOPIO} más. Transporte por carretera: {COSTE_CAMION} €/t·km y "
                 f"{FACTOR_CAMION} kg CO₂/t·km; densidad {DENSIDAD_TIERRAS} t/m³. Penalidad por retraso: {penal} k€ por semana."),
              _mensaje("Jefe de obra",
                       f"Las tierras de {X} al vertedero y el terraplén de {Y} con préstamo de cantera, como está en el plan: "
                       f"{Y} empieza antes de que {X} termine y no voy a tener la obra llena de montones. El gestor se lo "
                       "lleva, la cantera lo trae y cada actividad va a su ritmo. Son tierras: no es un residuo de verdad."),
              _criterios(["S13", "S8", "S1"], constantes=["suelos_excavados"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Haga el balance de tierras (m³ excavados, necesarios, reutilizables, sobrante y déficit) y calcule el coste y "
        "las emisiones del transporte de la propuesta (todo al vertedero y todo de préstamo) y de la alternativa de "
        "reutilizar en la obra lo que se pueda.",
        f"(A) Con el calendario, determine si {Y} puede empezar cuando termina {X} sin retrasar el proyecto (fin más "
        f"temprano de {X} frente a inicio más temprano y más tardío de {Y}), si hace falta acopio o retraso, y el coste "
        "total de cada opción con la penalidad. ¿Cuánto vale el «no tener montones»?",
        "(B) Según el criterio y la norma de referencia, ¿son residuo las tierras que se reutilizan en la obra? ¿Y las "
        "que van al vertedero? ¿En qué escalón de la jerarquía queda cada opción? ¿Puede reutilizarse sin acopio ni "
        "retraso con el calendario del plan?",
        "(C) ¿Qué propone al jefe de obra y qué comunica a la dirección facultativa? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"desmonte, terraplen, penalidad = {X!r}, {Y!r}, {penal}\n"
                       f"coste_vertedero, coste_prestamo, coste_interno, coste_acopio = {COSTE_VERTEDERO}, {COSTE_PRESTAMO}, {COSTE_INTERNO}, {COSTE_ACOPIO}\n"
                       f"km_vertedero, km_prestamo, coste_camion, factor_camion, densidad = {KM_VERTEDERO}, {KM_PRESTAMO}, {COSTE_CAMION}, {FACTOR_CAMION}, {DENSIDAD_TIERRAS}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Balance de tierras y coste del transporte de cada opción"\n#| label: tbl-balance\n'
                       "V_exc = float(datos_proyecto.loc[desmonte, 'tierras_m3']); V_nec = float(-datos_proyecto.loc[terraplen, 'tierras_m3'])\n"
                       "reutilizable = min(V_exc, V_nec); sobrante = V_exc - reutilizable; deficit = V_nec - reutilizable\n"
                       "c_vert = coste_vertedero + densidad * km_vertedero * coste_camion\n"
                       "c_prest = coste_prestamo + densidad * km_prestamo * coste_camion\n"
                       "e_vert = densidad * km_vertedero * factor_camion / 1000; e_prest = densidad * km_prestamo * factor_camion / 1000   # t CO₂ por m³\n"
                       "balance = pd.Series({'excavado (m³)': V_exc, 'necesario (m³)': V_nec, 'reutilizable (m³)': reutilizable, 'sobrante a vertedero (m³)': sobrante, 'déficit de préstamo (m³)': deficit})\n"
                       "display(balance.to_frame('m³').T)\n"
                       "opciones = pd.DataFrame({'propuesta: vertedero + préstamo': [V_exc * c_vert / 1000, V_nec * c_prest / 1000, 0.0, (V_exc * e_vert + V_nec * e_prest)],\n"
                       "                         'reutilizar en obra': [sobrante * c_vert / 1000, deficit * c_prest / 1000, reutilizable * coste_interno / 1000, (sobrante * e_vert + deficit * e_prest)]},\n"
                       "                        index=['vertedero (k€)', 'préstamo (k€)', 'transporte interno (k€)', 'emisiones del transporte (t CO₂)'])\n"
                       "opciones.loc['coste de tierras (k€)'] = opciones.iloc[:3].sum()\n"
                       "display(opciones.round(1))\n"
                       "Markdown(f'''Vertedero: {c_vert:.2f} €/m³ puesto allí; préstamo: {c_prest:.2f} €/m³ puesto en obra. La propuesta mueve {V_exc + V_nec:,.0f} m³ por carretera "
                       "({opciones.iloc[4, 0]:.1f} k€, {opciones.iloc[3, 0]:.1f} t); reutilizar deja en {sobrante + deficit:,.0f} m³ lo que sale o entra ({opciones.iloc[4, 1]:.1f} k€, {opciones.iloc[3, 1]:.1f} t).''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Fechas del desmonte y del terraplén"\n#| label: tbl-fechas\n'
                       "fechas = calendario.loc[[desmonte, terraplen], ['inicio_mas_temprano', 'fin_mas_temprano', 'inicio_mas_tardio', 'fin_mas_tardio', 'H_total']]\n"
                       "display(fechas)\n"
                       "ef_x = int(calendario.loc[desmonte, 'fin_mas_temprano']); es_y = int(calendario.loc[terraplen, 'inicio_mas_temprano']); ls_y = int(calendario.loc[terraplen, 'inicio_mas_tardio'])\n"
                       "desplazamiento = max(0, ef_x - es_y)\n"
                       "retraso = max(0, ef_x - ls_y)\n"
                       "necesita_acopio = es_y > ef_x\n"
                       "coste_reutilizar = opciones.loc['coste de tierras (k€)', 'reutilizar en obra'] + (reutilizable * coste_acopio / 1000 if necesita_acopio else 0) + retraso * penalidad\n"
                       "coste_propuesta = opciones.loc['coste de tierras (k€)', 'propuesta: vertedero + préstamo']\n"
                       "comparacion = pd.DataFrame({'propuesta': [coste_propuesta, 0, 0, coste_propuesta], 'reutilizar': [opciones.loc['coste de tierras (k€)', 'reutilizar en obra'], (reutilizable * coste_acopio / 1000 if necesita_acopio else 0), retraso * penalidad, coste_reutilizar]},\n"
                       "                           index=['tierras (k€)', 'acopio (k€)', 'penalidad por retraso (k€)', 'total (k€)'])\n"
                       "display(comparacion.round(1))\n"
                       "Markdown(f'''{desmonte} termina, como pronto, en la semana {ef_x}; {terraplen} puede empezar entre la semana {es_y} y la {ls_y}. ''' + "
                       "(f'''Empezar {terraplen} al terminar {desmonte} exige desplazarlo {desplazamiento} semana(s), ''' + ('que caben en su holgura: **sin retraso ni acopio**.' if retraso == 0 else f'''{retraso} más de las que caben en su holgura: el proyecto se retrasa **{retraso}** semana(s) ({retraso * penalidad} k€).''') if not necesita_acopio else "
                       "f'''{desmonte} termina antes de que {terraplen} pueda empezar: hacen falta {es_y - ef_x} semana(s) de acopio ({reutilizable * coste_acopio / 1000:.1f} k€), sin retraso.''') + "
                       "f''' Reutilizar cuesta **{coste_reutilizar:.1f} k€** frente a **{coste_propuesta:.1f}** de la propuesta: los «montones» valen {coste_propuesta - coste_reutilizar:.1f} k€.''')"))
    celdas.append(code('#| fig-cap: "Gantt con holguras: la ventana en la que puede moverse el terraplén"\n#| label: fig-gantt\n'
                       "_g, _dibujo = proyecto.copy().gantt(datos_proyecto, 'duration', holguras=True, tikz=True, params=dict(inner_text=None, **GANTT_PARAMS))\n"
                       "pinta_tikz(_dibujo)"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma y S13.** Las tierras que se reutilizan en su estado natural en la misma obra **no son residuo** "
                     f"(Ley 7/2022, art. 3.2.b); las que salen al vertedero **sí lo son** (y pagan canon e impuesto), y las "
                     f"{sobrante:,.0f} m³ sobrantes lo serán en cualquier caso salvo que se valoricen en otra obra conforme a la "
                     f"Orden APM/1007/2017. «No es un residuo de verdad» es cierto sólo para lo que se queda en la obra. **S8.** "
                     f"Reutilizar es **prevención**, el primer escalón; llevar a vertedero tierras aptas es **eliminación**, el "
                     f"último, y traer de cantera lo que ya se tenía consume un recurso natural además. **S1.** La propuesta "
                     f"ni compensa ni minimiza: traslada el impacto al vertedero, a la cantera y a la carretera. Y la "
                     f"pregunta del calendario tiene respuesta única: {X} termina en la semana {e['ef_x']} y {Y} puede empezar "
                     f"como tarde en la {e['ls_y']}; "
                     + ("se reutiliza **sin acopio ni retraso** desplazando el terraplén dentro de su holgura."
                        if (retraso == 0 and not necesita_acopio) else
                        (f"hace falta **acopio** de {e['es_y'] - e['ef_x']} semana(s), sin retraso." if necesita_acopio else
                         f"desplazar el terraplén supera su holgura en {retraso} semana(s): se reutiliza **con {retraso} semana(s) "
                         f"de retraso** ({retraso * penal} k€), que sigue siendo más barato que la propuesta."))
                     + " Lo que decide no es la opinión sobre los montones, sino las fechas."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Los suelos no contaminados excavados que se reutilizan en su estado natural en la misma obra…', ['no son residuo (Ley 7/2022, art. 3.2.b)', 'son residuo inerte', 'son residuo no peligroso', 'son residuo salvo que lo autorice el gestor'],\n"
        "     'S13: exclusión del ámbito de la ley de residuos.'),\n"
        "    ('Reutilizar las tierras de la excavación en los rellenos de la misma obra es, en la jerarquía de residuos…', ['prevención', 'reciclado', 'otra valorización', 'eliminación'],\n"
        "     'S8: no se genera el residuo.'),\n"
        "    ('El relleno puede empezar justo al terminar la excavación sin retrasar el proyecto si…', ['el fin más temprano de la excavación no supera el inicio más tardío del relleno', 'el relleno es crítico', 'la excavación tiene holgura', 'los volúmenes coinciden'],\n"
        "     'Lección 05: desplazar dentro de la holgura total no mueve el fin.'),\n"
        "]", "Sostenibilidad 20 — tierras sobrantes"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al jefe de obra: la tabla del apartado 2 y el plan de movimiento de tierras con fechas: {X} a su "
                     f"ritmo, {Y} desplazado "
                     + (f"{max(0, e['ef_x'] - e['es_y'])} semana(s) dentro de su holgura" if (retraso == 0 and not necesita_acopio)
                        else (f"con un acopio temporal de {e['es_y'] - e['ef_x']} semana(s) en una zona de la obra" if necesita_acopio
                              else f"hasta la semana {e['ef_x']} aceptando {retraso} semana(s) de retraso, o {X} acelerado para evitarlas (lección 10)"))
                     + f"; el sobrante ({sobrante:,.0f} m³) a vertedero o, mejor, ofrecido a otra obra con la orden de "
                     "valorización. A la dirección facultativa: el estudio de gestión de residuos actualizado con el balance "
                     "de tierras. Prueba del ciclo de vida: ¿quién paga los camiones de ida al vertedero y los de vuelta de "
                     "la cantera, y qué queda en cada sitio?"))
    return celdas, (f"{V_X:,} m³ de desmonte y {V_Y:,} m³ de terraplén: vertedero + préstamo {coste_propuesta:.0f} k€ frente a "
                    f"reutilizar {coste_reut:.0f} k€"
                    + (" sin retraso" if (retraso == 0 and not necesita_acopio) else (" con acopio" if necesita_acopio else f" con {retraso} semana(s) de retraso"))
                    + "; art. 3.2.b Ley 7/2022")


# ---------------------------------------------------------------------------
# Caso 14 — La cláusula social del contrato: inserción laboral que hay que planificar
# ---------------------------------------------------------------------------
COSTE_TUTORIA = 0.3          # k€ por operario-semana de inserción (tutoría, formación, menor rendimiento inicial)


def _gantt_cargas(info, df, etiqueta="resources"):
    with _silencio():
        g = info["proyecto"].copy()
        gantt_df, _ = g.gantt_cargas(df.copy(), "duration", etiqueta, report=False, tikz=False)
    return gantt_df


def _gantt_nivelado(info, df, etiqueta="resources"):
    with _silencio():
        g = info["proyecto"].copy()
        desp, _, datos, gantt_df, _ = g.nivelar(df.copy(), "duration", etiqueta, report=False)
    return desp, gantt_df


def _clausula_social(rng):
    e = {}

    def acepta(i):
        if not _base(i, criticas_1=False, T=(12, 20)):
            return False
        D = i["D"]
        rec = P.recursos(rng, list(D.index), hi=4)
        apta = pd.Series(rng.choice([0, 1], size=len(D), p=[0.45, 0.55]), index=D.index, name="apta")
        if apta.sum() < 3 or apta.sum() > len(D) - 2:
            return False
        df = P.tabla_base(i, duration=D, resources=rec, apta=apta)
        p = int(rng.choice([10, 15, 20]))
        q = int(rng.choice([1, 2]))
        try:
            g0 = _gantt_cargas(i, df)
            desp, g1 = _gantt_nivelado(i, df)
        except Exception:
            return False
        total = float(_carga_filas(g0, list(D.index)).sum())
        requerido = int(np.ceil(p / 100 * total))
        aptas = sorted(D.index[apta == 1])
        cap0 = float(_carga_filas(g0, aptas).clip(upper=q).sum())
        cap1 = float(_carga_filas(g1, aptas).clip(upper=q).sum())
        if not (cap0 < requerido <= cap1):
            return False
        e.update(rec=rec, apta=apta, df=df, p=p, q=q, total=total, requerido=requerido, cap0=cap0, cap1=cap1, aptas=aptas)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    p, q, df = e["p"], e["q"], e["df"]
    penal_punto = int(rng.choice([4, 5, 6]))
    deficit0 = e["requerido"] - e["cap0"]
    puntos0 = 100 * (e["requerido"] - e["cap0"]) / e["total"]
    celdas = [md("# Enunciado"),
              md("Una empresa constructora ha resultado adjudicataria de una obra pública. El pliego incluye, como "
                 "**condición especial de ejecución** de carácter social calificada de **obligación contractual "
                 f"esencial**, que **al menos el {p} % de las operario-semanas** de la obra las realicen personas en "
                 "situación de desempleo de larga duración contratadas para la obra a través del servicio público de "
                 f"empleo, con una penalidad de {penal_punto} k€ por cada punto porcentual no alcanzado y resolución del "
                 "contrato si no se alcanza la mitad. El cuadro de prelaciones, las duraciones (semanas), los operarios "
                 "de cada actividad y si la actividad admite personal de inserción sin cualificación específica "
                 "(`apta` = 1: peonaje, desbroce, acopios, limpieza, ayudas) se adjuntan."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"El servicio de empleo y el plan de formación permiten incorporar como máximo **{q} persona(s) de "
                 f"inserción por semana** (cada una necesita un tutor en la cuadrilla); cada operario-semana de inserción "
                 f"cuesta {COSTE_TUTORIA} k€ más que uno ordinario (tutoría, formación y rendimiento inicial)."),
              _mensaje("Director de producción",
                       "Eso se cubre en el papel: la subcontrata de pintura dice que su gente es de inserción y listo, o "
                       "contamos a los peones de siempre, que alguno estuvo en paro. Y si no, se lo pasamos a la "
                       "subcontrata en su contrato, que para eso está. No voy a parar la obra por una cláusula social."),
              _criterios(["S14", "S10"], constantes=["condiciones_ejecucion"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Con el diagrama de cargas de inicios más tempranos, calcule las operario-semanas totales de la obra, las "
        f"que exige la cláusula ({p} %) y la capacidad máxima de operario-semanas de inserción (semana a semana, el "
        f"mínimo entre {q} y la carga de las actividades aptas). ¿Cumple el plan? Nivele las cargas dentro de las "
        "holguras y repita el cálculo.",
        "(A) Calcule el coste de cumplir la cláusula (operario-semanas de inserción por su sobrecoste) y el coste de "
        "no cumplirla con el plan de inicios más tempranos (puntos de déficit por la penalidad). ¿Cuánto cuesta la "
        "cláusula frente a la penalidad?",
        "(B) Según el criterio y la norma de referencia, ¿es la condición una obligación o una recomendación? "
        "¿La cumple el plan con inicios más tempranos? ¿Y tras nivelar? ¿Vale trasladarla a la subcontrata o "
        "«cubrirla en el papel»?",
        "(C) ¿Qué propone al director de producción y qué comunica al responsable del contrato? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"porcentaje, maximo_insercion, coste_tutoria, penalidad_punto = {p}, {q}, {COSTE_TUTORIA}, {penal_punto}\n" + SETUP +
                       "aptas = sorted(datos_proyecto.index[datos_proyecto['apta'] == 1])\n"
                       "def carga_por_semana(gantt_df, filas):\n"
                       "    '''Carga semanal de las filas indicadas de un Gantt de cargas (celdas vacías = 0).'''\n"
                       "    cols = [c for c in gantt_df.data.columns if c != 'H_total']\n"
                       "    filas = [f for f in filas if f in gantt_df.data.index]\n"
                       "    return gantt_df.data.loc[filas, cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum()"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Gantt con inicios más tempranos y carga de operarios"\n#| label: fig-ganttCargas\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| tbl-cap: "Carga total, carga de las actividades aptas y capacidad de inserción, semana a semana (inicios más tempranos)"\n#| label: tbl-capacidad\n'
                       "carga_total = carga_por_semana(gantt_inicial, list(datos_proyecto.index))\n"
                       "carga_aptas = carga_por_semana(gantt_inicial, aptas)\n"
                       "capacidad = carga_aptas.clip(upper=maximo_insercion)\n"
                       "tabla = pd.DataFrame({'carga total': carga_total, 'carga de las aptas': carga_aptas, f'inserción posible (mín con {maximo_insercion})': capacidad})\n"
                       "tabla.loc['Total'] = tabla.sum()\n"
                       "display(tabla.astype(int).T)\n"
                       "total = float(carga_total.sum()); requerido = int(np.ceil(porcentaje / 100 * total)); capacidad_inicial = float(capacidad.sum())\n"
                       "Markdown(f'''Operario-semanas de la obra: **{total:.0f}**; la cláusula exige al menos {porcentaje} %: **{requerido}**. Con inicios más tempranos, las actividades aptas "
                       "admiten como máximo **{capacidad_inicial:.0f}** operario-semanas de inserción ({capacidad_inicial / total:.1%}): ''' + ('**cumple**.' if capacidad_inicial >= requerido else '**no cumple**: las actividades aptas se solapan y el límite semanal desperdicia su carga.'))"))
    celdas.append(code('#| tbl-cap: "Desplazamientos de la nivelación y capacidad de inserción tras nivelar"\n#| label: tbl-nivelacion\n'
                       "proyecto_nivelado = proyecto.copy()\n"
                       "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo_niv = proyecto_nivelado.nivelar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='resources')\n"
                       "display(desplazamientos[desplazamientos['desplazamientos'] > 0])\n"
                       "carga_aptas_niv = carga_por_semana(gantt_nivelado, aptas)\n"
                       "capacidad_niv = carga_aptas_niv.clip(upper=maximo_insercion)\n"
                       "tabla_niv = pd.DataFrame({'carga de las aptas (nivelada)': carga_aptas_niv, f'inserción posible (mín con {maximo_insercion})': capacidad_niv})\n"
                       "tabla_niv.loc['Total'] = tabla_niv.sum()\n"
                       "display(tabla_niv.astype(int).T)\n"
                       "capacidad_nivelada = float(capacidad_niv.sum())\n"
                       "Markdown(f'''Tras nivelar (misma duración, {duracion} semanas) la capacidad de inserción es **{capacidad_nivelada:.0f}** operario-semanas ({capacidad_nivelada / total:.1%}): ''' + ('**cumple** la cláusula.' if capacidad_nivelada >= requerido else 'sigue sin cumplir.'))"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas tras la nivelación"\n#| label: fig-cargasNivelado\n'
                       "plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=[int(carga_total.max())])"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Coste de cumplir frente a coste de no cumplir"\n#| label: tbl-coste\n'
                       "deficit_inicial = max(0, requerido - capacidad_inicial)\n"
                       "puntos_deficit = 100 * deficit_inicial / total\n"
                       "costes = pd.Series({'cumplir tras nivelar: sobrecoste de tutoría (k€)': requerido * coste_tutoria,\n"
                       "                    'plan inicial sin nivelar: déficit (operario-semanas)': deficit_inicial,\n"
                       "                    'plan inicial sin nivelar: déficit (puntos porcentuales)': puntos_deficit,\n"
                       "                    'plan inicial sin nivelar: penalidad (k€)': puntos_deficit * penalidad_punto})\n"
                       "display(costes.round(2).to_frame('valor'))\n"
                       "Markdown(f'''Cumplir cuesta **{requerido * coste_tutoria:.1f} k€** de tutoría y una nivelación que no alarga la obra; no cumplir con el plan inicial deja "
                       "{deficit_inicial:.0f} operario-semanas de déficit ({puntos_deficit:.1f} puntos), **{puntos_deficit * penalidad_punto:.1f} k€** de penalidad''' + "
                       "(' y, por debajo de la mitad, causa de resolución.' if capacidad_inicial < requerido / 2 else '.'))"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma y S14.** La condición especial de ejecución es una **obligación del contrato** (LCSP, art. "
                     f"202): el pliego la ha calificado de esencial, de modo que su incumplimiento lleva la penalidad y, por "
                     f"debajo de la mitad, la resolución (art. 202.3). Se exige igualmente a los subcontratistas (art. 202.4): "
                     f"pasarla a la subcontrata **no la traslada**, la extiende; y «cubrirla en el papel» con personal que no "
                     f"cumple la definición del pliego es una afirmación falsa en la certificación (lección 19, C1), además de "
                     f"incumplir. Si el plan la cumple es un cálculo: con inicios más tempranos la capacidad es de "
                     f"{e['cap0']:.0f} operario-semanas frente a las {e['requerido']} exigidas (**no cumple**, {puntos0:.1f} "
                     f"puntos de déficit); tras nivelar, {e['cap1']:.0f} (**cumple**), con la misma duración. **S10.** El plan "
                     "no puede apoyarse en una contratación que la norma y el contrato no permiten: la opción «en el papel» "
                     "no está en la tabla, y la de nivelar sí."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Una condición especial de ejecución de tipo social fijada en el pliego…', ['es una obligación del contrato, con penalidad o resolución si el pliego la califica de esencial', 'es una recomendación', 'sólo obliga si el contratista la aceptó expresamente en su oferta', 'se sustituye por una compensación económica'],\n"
        "     'S14: LCSP art. 202.1 y 202.3.'),\n"
        "    ('Las condiciones especiales de ejecución, respecto a los subcontratistas…', ['se exigen igualmente a todos los que participen en la ejecución (art. 202.4)', 'no les afectan', 'sólo si el contrato principal lo dice', 'sólo a los de primer nivel'],\n"
        "     'Art. 202.4 de la LCSP.'),\n"
        "    ('La capacidad máxima de operario-semanas de inserción con q personas por semana es…', ['la suma, semana a semana, del mínimo entre q y la carga de las actividades aptas', 'q por la duración del proyecto', 'q por el número de actividades aptas', 'el total de operario-semanas de las actividades aptas'],\n"
        "     'El límite semanal desperdicia la carga que se solapa: nivelar la reparte.'),\n"
        "]", "Sostenibilidad 20 — cláusula social"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de producción: la tabla del apartado 1 nivelada, con las semanas y actividades en las que "
                     f"entran las {q} persona(s) de inserción, el calendario de incorporación con el servicio de empleo "
                     f"(hay que pedirlas con antelación) y el coste ({e['requerido'] * COSTE_TUTORIA:.1f} k€) frente a la "
                     f"penalidad ({puntos0 * penal_punto:.1f} k€). Al responsable del contrato: el plan de cumplimiento "
                     "con la documentación que se aportará (contratos, certificados del servicio de empleo, partes de "
                     "trabajo), y a las subcontratas, la cláusula en su contrato porque la ley lo exige, no para "
                     "deshacerse de ella. Prueba de publicidad: ¿firmaría la certificación con la lista de «inserción» de "
                     "la subcontrata de pintura?"))
    return celdas, (f"cláusula del {p} %: {e['requerido']} de {e['total']:.0f} operario-semanas; capacidad {e['cap0']:.0f} con "
                    f"inicios más tempranos y {e['cap1']:.0f} tras nivelar (máx. {q}/semana); art. 202 LCSP")


# ---------------------------------------------------------------------------
# Caso 15 — El hormigón con menos clínker que tarda una semana más en desencofrar
# ---------------------------------------------------------------------------
def _hormigon_bajo_carbono(rng):
    e = {}

    def acepta(i):
        if not _base(i):
            return False
        D, cal = i["D"], i["calendario"]
        n_h = int(rng.integers(3, 5))
        horm = _elegir(rng, list(D.index), n_h)
        if not any(cal.loc[a, "H_total"] == 0 for a in horm):
            return False
        m3 = pd.Series(0, index=D.index, name="hormigon_m3")
        for a in horm:
            m3[a] = int(rng.integers(15, 61)) * 10
        D_B = D.copy()
        for a in horm:
            D_B[a] += 1
        T_B = _T(i["proyecto"], D_B)
        r = T_B - int(i["T"])
        if not (1 <= r <= 3):
            return False
        tope, cn, ct, cur = P.costes_reduccion(rng, i["D"], n_inf=0)
        cur = cur.astype(float)
        tope_B = tope.copy()
        for a in horm:
            tope_B[a] += 1
        try:
            pasos, _, coste_crash = _ackoff(i["proyecto"], D_B.astype(float), tope_B, cur, r)
        except Exception:
            return False
        if len(pasos) < r:
            return False
        pa, fa = int(rng.choice([90, 95])), float(rng.choice([0.30, 0.32]))
        pb, fb = pa + int(rng.choice([5, 8])), float(rng.choice([0.19, 0.21]))
        semana = int(rng.choice([6, 8, 10]))
        precio = int(rng.choice([100, 150, 250]))
        M = float(m3.sum())
        mat_A, mat_B = M * pa / 1000, M * pb / 1000
        em_A, em_B = M * fa, M * fb
        tot = {"A": mat_A + em_A * precio / 1000,
               "B con retraso": mat_B + r * semana + em_B * precio / 1000,
               "B comprimido": mat_B + coste_crash + em_B * precio / 1000}
        dinero = {"A": mat_A, "B con retraso": mat_B + r * semana, "B comprimido": mat_B + coste_crash}
        if not (dinero["A"] < min(dinero["B con retraso"], dinero["B comprimido"]) - 0.5 and tot["B comprimido"] < tot["A"] - 0.5):
            return False
        e.update(horm=horm, m3=m3, D_B=D_B, T_B=T_B, r=r, tope=tope, tope_B=tope_B, cur=cur, pasos=pasos, coste_crash=coste_crash,
                 pa=pa, fa=fa, pb=pb, fb=fb, semana=semana, precio=precio, tot=tot, dinero=dinero, M=M)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    r, precio, semana = e["r"], e["precio"], e["semana"]
    df = P.tabla_base(info, duration=D, cap_duration=e["tope"], ucr=e["cur"], hormigon_m3=e["m3"])
    hormigones = pd.DataFrame({"precio (€/m³)": [e["pa"], e["pb"]], "emisiones (t CO₂/m³)": [e["fa"], e["fb"]],
                               "semanas más por actividad de hormigonado": [0, 1]},
                              index=pd.Index(["A: CEM I (hormigón habitual)", "B: CEM III/A (bajo contenido en clínker)"], name="hormigon"))
    mejor = min(e["tot"], key=e["tot"].get)
    celdas = [md("# Enunciado"),
              md("La estructura de un edificio industrial se hormigona en las actividades con `hormigon_m3` > 0 del cuadro "
                 "adjunto (prelaciones, duraciones en semanas, duraciones mínimas alcanzables y coste unitario de "
                 "reducción `ucr` en k€ por semana). La obra se ejecuta en invierno. Hay dos hormigones que cumplen la "
                 "resistencia característica y la clase de exposición del pliego: **A**, con cemento CEM I, y **B**, con "
                 "cemento CEM III/A de bajo contenido en clínker, que adquiere la resistencia más despacio y, en estas "
                 "condiciones, alarga **una semana** cada actividad de hormigonado (desencofrado y puesta en carga)."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              code('#| tbl-cap: "Los dos hormigones"\n#| label: tbl-hormigones\n' + literal_dataframe(hormigones, nombre="hormigones", indice="hormigon")),
              md(f"El coste indirecto de la obra (medios auxiliares, instalaciones, equipo técnico) es de {semana} k€ por "
                 f"semana de duración. La organización aplica un precio sombra del carbono de {precio} €/t a sus decisiones."),
              _mensaje("Director de producción",
                       "Hormigón de siempre, CEM I: el otro tarda una semana más en desencofrar y nos come la ruta "
                       "crítica, y nunca hemos trabajado con CEM III. El carbono del cemento es del cementero, no "
                       "nuestro, y el pliego no dice nada del cemento."),
              _criterios(["S15", "S2", "S6", "S1"], constantes=["hormigon_prestaciones"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule el coste del hormigón y las emisiones de cada opción, y la duración del proyecto con cada una "
        "(con B, una semana más en cada actividad de hormigonado). ¿Cuántas semanas se retrasa el proyecto con B y "
        "cuánto cuestan al coste indirecto?",
        f"(A) Recupere con Ackoff-Sasieni las semanas de retraso de B acortando otras actividades (las de hormigonado "
        "no pueden bajar de su mínimo más la semana de curado) y obtenga el coste de la compresión. Compare el coste "
        "monetario y el coste total al precio sombra de las tres opciones: A, B con retraso y B comprimido.",
        "(B) Según los criterios de referencia, ¿puede rechazarse B «porque siempre se ha usado CEM I»? ¿Qué opción es "
        "la correcta al precio sombra? ¿De quién son las emisiones del cemento (alcance)? ¿En qué escalón de la "
        "pirámide queda cada opción?",
        "(C) ¿Qué propone al director de producción y a la dirección facultativa? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"coste_semana, precio_sombra = {semana}, {precio}\n" + SETUP +
                       "hormigonado = sorted(datos_proyecto.index[datos_proyecto['hormigon_m3'] > 0])\n"
                       "M = float(datos_proyecto['hormigon_m3'].sum())\n"
                       "duraciones_B = datos_proyecto['duration'].copy(); duraciones_B[hormigonado] += 1\n"
                       "minimos_B = datos_proyecto['cap_duration'].copy(); minimos_B[hormigonado] += 1\n"
                       "duracion_B = int(proyecto.duration(duraciones_B))\n"
                       "retraso = duracion_B - duracion"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el hormigón A", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Coste del hormigón, emisiones y duración con cada opción"\n#| label: tbl-opciones\n'
                       "h = hormigones.copy()\n"
                       "h['hormigón (k€)'] = M * h['precio (€/m³)'] / 1000\n"
                       "h['emisiones (t CO₂)'] = M * h['emisiones (t CO₂/m³)']\n"
                       "h['duración del proyecto (semanas)'] = [duracion, duracion_B]\n"
                       "h['retraso (semanas)'] = [0, retraso]\n"
                       "h['coste indirecto del retraso (k€)'] = h['retraso (semanas)'] * coste_semana\n"
                       "display(h.iloc[:, 3:].round(1))\n"
                       "camino_B = ' / '.join('–'.join(r) for r in proyecto.critical_path(duraciones_B).values())\n"
                       "Markdown(f'''{M:,.0f} m³ de hormigón: B cuesta {h.iloc[1, 3] - h.iloc[0, 3]:.1f} k€ más y emite **{h.iloc[0, 4] - h.iloc[1, 4]:.0f} t menos**. "
                       "Con B el proyecto dura {duracion_B} semanas (ruta crítica {camino_B}): **{retraso}** semana(s) de retraso, {retraso * coste_semana} k€ de coste indirecto.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Ackoff-Sasieni sobre el plan con B para recuperar el retraso"\n#| label: tbl-ackoff\n'
                       "tabla_c, pasos_c, duraciones_c, margen_c = proyecto.ackoff(durations=duraciones_B, min_durations=minimos_B,\n"
                       "                                                         costs=datos_proyecto['ucr'], reduction=retraso, report=False)\n"
                       "tabla_c"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso"\n#| label: tbl-ackoffPasos\n'
                       "detalle_ackoff(proyecto, duraciones_B, minimos_B, datos_proyecto['ucr'], pasos_c)"))
    celdas.append(code('#| tbl-cap: "Las tres opciones: coste monetario y coste total al precio sombra"\n#| label: tbl-comparacion\n'
                       "coste_compresion = float(sum(datos_proyecto.loc[a, 'ucr'] for paso in pasos_c.values() for a in paso))\n"
                       "opciones = pd.DataFrame({'A': [h.iloc[0, 3], 0.0, 0.0, h.iloc[0, 4]],\n"
                       "                         'B con retraso': [h.iloc[1, 3], retraso * coste_semana, 0.0, h.iloc[1, 4]],\n"
                       "                         'B comprimido': [h.iloc[1, 3], 0.0, coste_compresion, h.iloc[1, 4]]},\n"
                       "                        index=['hormigón (k€)', 'coste indirecto del retraso (k€)', 'compresión (k€)', 'emisiones (t CO₂)'])\n"
                       "opciones.loc['coste monetario (k€)'] = opciones.iloc[:3].sum()\n"
                       "opciones.loc['carbono al precio sombra (k€)'] = opciones.loc['emisiones (t CO₂)'] * precio_sombra / 1000\n"
                       "opciones.loc['coste total (k€)'] = opciones.loc['coste monetario (k€)'] + opciones.loc['carbono al precio sombra (k€)']\n"
                       "display(opciones.round(1))\n"
                       "mejor = opciones.loc['coste total (k€)'].idxmin(); mas_barata = opciones.loc['coste monetario (k€)'].idxmin()\n"
                       "Markdown(f'''Recuperar las {retraso} semana(s) cuesta **{coste_compresion:.0f} k€** (pasos: ''' + '; '.join('paso ' + str(k + 1) + ': ' + ', '.join(v) for k, v in sorted(pasos_c.items())) + "
                       "f'''). En dinero gana **{mas_barata}**; al precio sombra de {precio_sombra} €/t gana **{mejor}** ({opciones.loc['coste total (k€)', mejor]:.1f} k€ frente a {opciones.loc['coste total (k€)', 'A']:.1f} de A).''')"))
    celdas += apartado(3)
    celdas.append(md(f"**S15.** Los dos hormigones cumplen las prestaciones del pliego (resistencia característica y clase de "
                     f"exposición); el pliego, como dice el director, no fija el cemento, y «nunca hemos trabajado con CEM "
                     f"III» no es una prestación: B **no puede rechazarse por costumbre**. Lo que sí hay que calcular es su "
                     f"efecto en el cronograma ({r} semana(s)), y se ha calculado. **S2.** Con el precio sombra la opción "
                     f"correcta es la de menor coste total: **{mejor}** ({e['tot'][mejor]:.1f} k€ frente a {e['tot']['A']:.1f} "
                     f"de A); en dinero A es más barata ({e['dinero']['A']:.1f} frente a {e['dinero']['B comprimido']:.1f} de B "
                     f"comprimido), y esa diferencia es exactamente lo que la política de la organización decide pagar por "
                     f"{e['M'] * (e['fa'] - e['fb']):.0f} t. **S6.** Las emisiones de fabricación del cemento son **alcance 3** "
                     f"del proyecto (material comprado): ser «del cementero» no las saca del inventario, igual que el acero "
                     f"importado. **S1.** Sustituir el cemento **minimiza** ({e['M'] * (e['fa'] - e['fb']):.0f} t menos con la "
                     "misma estructura); **evitar** sería rediseñar para necesitar menos hormigón; A más créditos sería "
                     "compensar."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Un hormigón que cumple la resistencia característica y la clase de exposición del pliego…', ['no puede rechazarse por costumbre: se compara por coste total y ciclo de vida, con su efecto en el plazo calculado', 'puede rechazarse si el jefe de obra no lo conoce', 'sólo puede usarse si es más barato', 'sólo puede usarse en elementos no estructurales'],\n"
        "     'S15: especificación por prestaciones.'),\n"
        "    ('Las emisiones de fabricación del cemento del hormigón comprado son, para el proyecto…', ['alcance 3', 'alcance 1', 'alcance 2', 'del cementero, no del proyecto'],\n"
        "     'S6: materiales comprados.'),\n"
        "    ('Si el hormigón con menos clínker alarga una semana las actividades de hormigonado críticas, el proyecto…', ['se retrasa, salvo que se recupere comprimiendo otras actividades críticas (Ackoff-Sasieni)', 'no cambia', 'se retrasa tantas semanas como actividades de hormigonado', 'se acorta'],\n"
        "     'Lección 10: la semana extra va al plazo si la actividad es crítica, y se recupera con el coste unitario de otras.'),\n"
        "]", "Sostenibilidad 20 — hormigón bajo en carbono"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de producción: la tabla del apartado 2 con las tres opciones y el plan de compresión "
                     f"(qué actividades y cuánto cuesta), más lo que B exige en obra: ensayos de resistencia a edades "
                     f"tempranas para fijar el desencofrado con datos y no con la semana «por si acaso». A la dirección "
                     f"facultativa: la propuesta de B con la justificación de prestaciones del Código Estructural y la "
                     f"reducción de {e['M'] * (e['fa'] - e['fb']):.0f} t para la memoria. Prueba de publicidad: ¿pondría en la "
                     "memoria que se emitieron esas toneladas de más «porque siempre se ha usado CEM I»?"))
    return celdas, (f"{e['M']:,.0f} m³: B emite {e['M'] * (e['fa'] - e['fb']):.0f} t menos y retrasa {r} semana(s); "
                    f"recuperarlas cuesta {e['coste_crash']:.0f} k€; al precio sombra de {precio} €/t gana {mejor}")


# ---------------------------------------------------------------------------
# Caso 16 — Comprimir con turno de noche junto a un hospital
# ---------------------------------------------------------------------------
def _turno_de_noche(rng):
    e = {}

    def acepta(i):
        if not _base(i):
            return False
        D = i["D"]
        tope, cn, ct, cur = P.costes_reduccion(rng, D, n_inf=0)
        cur = cur.astype(float)
        ruidosa = pd.Series(rng.choice([0, 1], size=len(D)), index=D.index, name="ruidosa")
        noche = pd.Series([max(1.0, round(c * 0.6)) if c > 0 else 0.0 for c in cur], index=D.index, name="ucr_noche")
        reducibles = [a for a in D.index if tope[a] < D[a]]
        if ruidosa[reducibles].sum() < 2 or (1 - ruidosa[reducibles]).sum() < 1:
            return False
        r = int(rng.choice([2, 3]))
        cur_propuesta = pd.Series(np.minimum(cur, noche), index=D.index)
        cur_legal = pd.Series([noche[a] if ruidosa[a] == 0 else cur[a] for a in D.index], index=D.index)
        try:
            pasos1, _, coste1 = _ackoff(i["proyecto"], D.astype(float), tope, cur_propuesta, r)
            pasos2, _, coste2 = _ackoff(i["proyecto"], D.astype(float), tope, cur_legal, r)
        except Exception:
            return False
        if len(pasos1) < r or len(pasos2) < r:
            return False
        acts1 = sorted(a for acts in pasos1.values() for a in acts)
        acts2 = sorted(a for acts in pasos2.values() for a in acts)
        nocturnas_ruidosas = sorted(set(a for a in acts1 if ruidosa[a] == 1 and noche[a] < cur[a]))
        if not nocturnas_ruidosas or coste1 >= coste2 - 0.5:
            return False
        e.update(tope=tope, cur=cur, noche=noche, ruidosa=ruidosa, r=r, pasos1=pasos1, pasos2=pasos2, coste1=coste1, coste2=coste2,
                 acts1=acts1, acts2=acts2, nocturnas=nocturnas_ruidosas)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    r = e["r"]
    p_denuncia = float(rng.choice([0.5, 0.6, 0.8]))
    multa = int(rng.choice([6, 9, 12]))
    paralizacion = int(rng.choice([1, 2]))
    penal = int(rng.choice([8, 10, 12]))
    df = P.tabla_base(info, duration=D, cap_duration=e["tope"], ucr=e["cur"], ucr_noche=e["noche"], ruidosa=e["ruidosa"])
    esperado1 = e["coste1"] + p_denuncia * (multa + paralizacion * penal)
    celdas = [md("# Enunciado"),
              md(f"Una obra de edificación en el centro de la ciudad, con viviendas en las fachadas vecinas y un hospital "
                 f"enfrente, debe adelantar la entrega **{r} semanas**. El cuadro de prelaciones, las duraciones "
                 "(semanas), las duraciones mínimas, el coste unitario de reducción en jornada diurna (`ucr`, k€ por "
                 "semana: doblar cuadrillas y medios) y el coste unitario de reducción con **turno de noche** (`ucr_noche`, "
                 "más barato: los mismos medios, más horas) se adjuntan, junto con la indicación de si la actividad es "
                 "**ruidosa** (`ruidosa` = 1: demolición, perforación, hormigonado con bomba, maquinaria pesada)."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md("La ordenanza municipal de ruido limita los trabajos de obra ruidosos al horario de **8 a 22 h** y sólo "
                 "admite trabajos ruidosos nocturnos con **autorización expresa y motivada del ayuntamiento**, que en el "
                 "entorno del hospital no se concede salvo emergencia; el trabajo nocturno no ruidoso (interiores, "
                 "instalaciones, acabados) es libre. Según la experiencia de la empresa en obras similares, un turno de "
                 f"noche ruidoso sin autorización acaba en denuncia con probabilidad {p_denuncia:.0%}; si hay denuncia, la "
                 f"policía municipal paraliza el turno, la multa es de {multa} k€ y la obra pierde {paralizacion} semana(s) "
                 f"hasta reorganizarse, con una penalidad de {penal} k€ por semana de retraso en la entrega."),
              _mensaje("Jefe de obra",
                       f"Turno de noche en todo lo que haga falta, que sale un 40 % más barato que doblar cuadrillas: son "
                       f"sólo {r} semanas y el hospital tiene ventanas dobles. Pedir el permiso tarda seis semanas y no "
                       "llegamos; si viene la policía, se para esa noche y se sigue al día siguiente."),
              _criterios(["S12", "S10", "S1"], constantes=["ruido"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Obtenga con Ackoff-Sasieni el plan de reducción de {r} semanas más barato usando en cada actividad el "
        "menor de los dos costes unitarios (la propuesta del jefe de obra): actividades acortadas en cada paso y coste. "
        "¿Qué actividades ruidosas se harían de noche?",
        "(A) Repita el algoritmo con el coste unitario legal (turno de noche sólo en las actividades no ruidosas) y "
        "compare los dos planes. Calcule el coste esperado de la propuesta con la denuncia (probabilidad por multa y "
        "paralización). ¿Cuánto vale el «ahorro» del turno de noche?",
        "(B) Según el criterio y la norma de referencia, ¿es el turno de noche en actividades ruidosas una opción de "
        "la tabla? ¿Cuál es el plan de compresión correcto? ¿En qué escalón de la pirámide queda cada plan?",
        "(C) ¿Qué propone al jefe de obra y qué comunica al cliente y a los vecinos? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"semanas_pedidas, p_denuncia, multa, paralizacion, penalidad = {r}, {p_denuncia}, {multa}, {paralizacion}, {penal}\n" + SETUP +
                       "datos_proyecto['ucr_propuesta'] = np.minimum(datos_proyecto['ucr'], datos_proyecto['ucr_noche'])\n"
                       "datos_proyecto['ucr_legal'] = np.where(datos_proyecto['ruidosa'] == 1, datos_proyecto['ucr'], datos_proyecto['ucr_propuesta'])\n"
                       "def coste_plan(pasos, columna):\n"
                       "    return float(sum(datos_proyecto.loc[a, columna] for paso in pasos.values() for a in paso))"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con el camino crítico", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Ackoff-Sasieni con el menor coste unitario de cada actividad (propuesta)"\n#| label: tbl-ackoffPropuesta\n'
                       "tabla_1, pasos_1, duraciones_1, margen_1 = proyecto.ackoff(durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "                                                         costs=datos_proyecto['ucr_propuesta'], reduction=semanas_pedidas, report=False)\n"
                       "tabla_1"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso (propuesta)"\n#| label: tbl-ackoffPaso1\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr_propuesta'], pasos_1)"))
    celdas.append(code("acts_1 = [a for paso in pasos_1.values() for a in paso]\n"
                       "nocturnas_ruidosas = sorted(set(a for a in acts_1 if datos_proyecto.loc[a, 'ruidosa'] == 1 and datos_proyecto.loc[a, 'ucr_noche'] < datos_proyecto.loc[a, 'ucr']))\n"
                       "coste_1 = coste_plan(pasos_1, 'ucr_propuesta')\n"
                       "Markdown('Plan 1 (propuesta): ' + '; '.join('paso ' + str(k + 1) + ': ' + ', '.join(v) for k, v in sorted(pasos_1.items())) + "
                       "f'''. Coste **{coste_1:.0f} k€**. Actividades ruidosas que se harían de noche: **{', '.join(nocturnas_ruidosas)}**.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Ackoff-Sasieni con el coste unitario legal (noche sólo en las no ruidosas)"\n#| label: tbl-ackoffLegal\n'
                       "tabla_2, pasos_2, duraciones_2, margen_2 = proyecto.ackoff(durations=datos_proyecto['duration'], min_durations=datos_proyecto['cap_duration'],\n"
                       "                                                         costs=datos_proyecto['ucr_legal'], reduction=semanas_pedidas, report=False)\n"
                       "tabla_2"))
    celdas.append(code('#| tbl-cap: "Opciones consideradas en cada paso (legal)"\n#| label: tbl-ackoffPaso2\n'
                       "detalle_ackoff(proyecto, datos_proyecto['duration'], datos_proyecto['cap_duration'], datos_proyecto['ucr_legal'], pasos_2)"))
    celdas.append(code('#| tbl-cap: "Comparación de los dos planes"\n#| label: tbl-planes\n'
                       "coste_2 = coste_plan(pasos_2, 'ucr_legal')\n"
                       "esperado_1 = coste_1 + p_denuncia * (multa + paralizacion * penalidad)\n"
                       "comparacion = pd.DataFrame({'plan 1: propuesta (noche en todo)': [coste_1, p_denuncia * multa, p_denuncia * paralizacion * penalidad, esperado_1],\n"
                       "                            'plan 2: legal': [coste_2, 0.0, 0.0, coste_2]},\n"
                       "                           index=['coste de la compresión (k€)', 'multa esperada (k€)', 'paralización esperada (k€)', 'coste esperado (k€)'])\n"
                       "display(comparacion.round(1))\n"
                       "Markdown('Plan 2 (legal): ' + '; '.join('paso ' + str(k + 1) + ': ' + ', '.join(v) for k, v in sorted(pasos_2.items())) + "
                       "f'''. Coste **{coste_2:.0f} k€**, {coste_2 - coste_1:.0f} k€ más que la propuesta: ese es el «ahorro» del turno de noche. "
                       "Con la denuncia, el coste esperado de la propuesta es de {esperado_1:.1f} k€''' + (', más que el plan legal: mal negocio además de infracción.' if esperado_1 > coste_2 else ', todavía menor que el plan legal: la respuesta B no depende de esta cuenta.'))"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma y S12.** El horario de la ordenanza es una condición de la autorización de la obra: el turno de "
                     f"noche en actividades ruidosas junto a viviendas y un hospital, sin autorización expresa, es infracción "
                     f"(Ley 37/2003, arts. 6 y 28) y **no es una opción de la tabla**; sí lo es en las actividades no "
                     f"ruidosas. El plan correcto es el **plan 2**, con el coste unitario legal ({e['coste2']:.0f} k€ frente a "
                     f"{e['coste1']:.0f}): se excluyen las opciones prohibidas antes de ordenar, no se usan y se paga la multa "
                     f"como un coste. **S10.** El «ahorro» de {e['coste2'] - e['coste1']:.0f} k€ sale del sueño de los vecinos y "
                     f"de los pacientes del hospital, que no están en la mesa: una ventaja de plazo obtenida incumpliendo "
                     f"la norma no es una opción, y «si viene la policía se para esa noche» describe la infracción, no la "
                     f"evita. **S1.** El plan 1 ni compensa ni minimiza: traslada el impacto a la comunidad; el plan 2 "
                     f"**minimiza** (ruido sólo de día, noche en lo silencioso); **evitar** sería negociar con el cliente las "
                     f"{r} semanas (lección 19, C13) o pedir la autorización con tiempo para lo que de verdad lo necesite."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('El turno de noche en actividades ruidosas junto a viviendas y un hospital, sin autorización del ayuntamiento…', ['no es una opción del plan: la ordenanza lo prohíbe y es infracción (Ley 37/2003)', 'es una opción si se compensa a los vecinos', 'es legal si dura menos de un mes', 'es legal si la obra es pública'],\n"
        "     'S12: condiciones de la autorización como restricciones del plan.'),\n"
        "    ('Al aplicar Ackoff-Sasieni cuando algunas opciones de compresión están prohibidas para ciertas actividades…', ['se excluyen esas opciones (coste unitario legal) antes de ordenar', 'se usan y se añade la multa como coste', 'se reduce el número de semanas pedidas', 'no cambia nada'],\n"
        "     'El algoritmo ordena las opciones admisibles; la multa no convierte en admisible lo prohibido.'),\n"
        "    ('Trasladar el ruido nocturno a los vecinos para abaratar la compresión es, en la pirámide de la sostenibilidad…', ['trasladar el impacto: ni siquiera compensa', 'minimizar', 'restaurar', 'evitar'],\n"
        "     'S1: la pirámide empieza en compensar; la propuesta está por debajo.'),\n"
        "]", "Sostenibilidad 20 — turno de noche"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al jefe de obra: el plan 2 con sus pasos y su coste, y la petición de autorización para lo que sí "
                     f"pueda justificarse (un hormigonado continuo, por ejemplo) presentada hoy, no «cuando haga falta». Al "
                     f"cliente: el precio de cada semana con el plan legal ({e['coste2'] / r:.1f} k€ por semana de media) y "
                     "la alternativa de adelantar menos. A los vecinos y al hospital: el calendario de los trabajos "
                     "ruidosos y el horario, antes de empezar. Prueba de publicidad: ¿explicaría al gerente del hospital "
                     f"que el martillo de las 3 de la mañana ahorraba {e['coste2'] - e['coste1']:.0f} k€?"))
    return celdas, (f"adelantar {r} semanas: {e['coste1']:.0f} k€ con noche en todo ({', '.join(e['nocturnas'])} ruidosas) frente a "
                    f"{e['coste2']:.0f} k€ legal; esperado con denuncia {esperado1:.0f} k€")


# ---------------------------------------------------------------------------
# Caso 17 — Prefabricar o construir in situ: residuos, semanas de obra y camiones
# ---------------------------------------------------------------------------
COSTE_GESTION_RESIDUO = 45          # € por tonelada de residuo de obra (gestor e impuesto)
KM_GESTOR = 25


def _prefabricacion(rng):
    e = {}

    def acepta(i):
        if not _base(i, criticas_1=False, T=(14, 24)):
            return False
        D, cal = i["D"], i["calendario"]
        cands = [a for a in sorted(D.index) if int(D[a]) >= 3]
        if len(cands) < 3:
            return False
        pref = _elegir(rng, cands, int(rng.integers(3, 5)))
        if not any(cal.loc[a, "H_total"] == 0 for a in pref):
            return False
        red = pd.Series(0, index=D.index, name="reduccion_prefabricado")
        res = pd.Series(0, index=D.index, name="residuos_t")
        for a in pref:
            red[a] = int(rng.choice([1, 2])) if int(D[a]) >= 4 else 1
            res[a] = int(rng.integers(10, 41))
        D_p = D - red
        T_p = _T(i["proyecto"], D_p)
        ahorro = int(i["T"]) - T_p
        if not (2 <= ahorro <= 5):
            return False
        rec = P.recursos(rng, list(D.index), hi=4)
        e.update(pref=pref, red=red, res=res, D_p=D_p, T_p=T_p, ahorro=ahorro, rec=rec)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    pref, ahorro, T_p = e["pref"], e["ahorro"], e["T_p"]
    df = P.tabla_base(info, duration=D, resources=e["rec"], reduccion_prefabricado=e["red"], residuos_t=e["res"])
    for _ in range(1000):
        W = int(rng.integers(20, 50)) * 10                       # t de elementos prefabricados
        km = int(rng.integers(25, 46)) * 10
        extra = int(rng.choice([20, 30, 40, 50]))                # k€ más de material prefabricado
        semana = int(rng.choice([6, 8, 10]))                     # k€ por semana de obra (instalaciones, grúa, equipo)
        em_semana = float(rng.choice([2.0, 3.0, 4.0]))           # t CO₂ por semana de obra (maquinaria, grúa, suministros, electricidad)
        fraccion_pref = 0.2
        precio = int(rng.choice([100, 150, 250]))
        R = float(e["res"].sum())
        res_in, res_pref = R, R * fraccion_pref
        cost_in = T * semana + res_in * COSTE_GESTION_RESIDUO / 1000
        cost_pref = extra + T_p * semana + W * km * COSTE_CAMION / 1000 + res_pref * COSTE_GESTION_RESIDUO / 1000
        em_in = T * em_semana + res_in * KM_GESTOR * FACTOR_CAMION / 1000
        em_tr = W * km * FACTOR_CAMION / 1000
        em_pref = T_p * em_semana + em_tr + res_pref * KM_GESTOR * FACTOR_CAMION / 1000
        tot_in, tot_pref = cost_in + em_in * precio / 1000, cost_pref + em_pref * precio / 1000
        if tot_pref < tot_in - 3 and abs(cost_in - cost_pref) < 0.6 * max(cost_in, cost_pref):
            break
    while abs(tot_pref - tot_in) < 3:
        # si el muestreo no logró una diferencia clara a favor del prefabricado, se encarece el material hasta que
        # gane in situ con claridad: la respuesta B tiene que ser nítida en cualquiera de los dos sentidos
        extra += 10
        cost_pref += 10; tot_pref += 10
    gana_pref = tot_pref < tot_in
    celdas = [md("# Enunciado"),
              md("En un edificio de oficinas, las actividades de estructura y cerramiento del cuadro adjunto pueden "
                 "ejecutarse **in situ** (el plan) o con **elementos prefabricados** en una fábrica a "
                 f"**{km} km**. El cuadro recoge las prelaciones, las duraciones in situ (semanas), los operarios de cada "
                 "actividad, las semanas que se ahorran en cada actividad prefabricándola (`reduccion_prefabricado`) y "
                 "las toneladas de residuo que cada actividad genera en la obra si se ejecuta in situ (`residuos_t`); "
                 f"prefabricada, genera en la obra el {fraccion_pref:.0%} (el resto se produce y se recicla en la fábrica)."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"Datos: los elementos prefabricados pesan {W} t y llegan por carretera ({COSTE_CAMION} €/t·km, "
                 f"{FACTOR_CAMION} kg CO₂/t·km); el material prefabricado cuesta {extra} k€ más que el in situ; cada semana "
                 f"de obra cuesta {semana} k€ (instalaciones, grúa, equipo técnico) y emite {em_semana} t CO₂ (gasóleo y "
                 f"electricidad); el residuo de obra cuesta {COSTE_GESTION_RESIDUO} €/t de gestión y viaja {KM_GESTOR} km "
                 f"hasta el gestor. Precio sombra del carbono de la organización: {precio} €/t."),
              _mensaje("Director técnico",
                       f"In situ, como siempre. La prefabricación trae {W} toneladas de hormigón en camiones desde {km} km, "
                       f"que es lo contrario de sostenible, y nos cobran {extra} k€ más. Lo que ahorremos en semanas de "
                       "obra no es de este presupuesto, y los residuos los gestiona el gestor, para eso se le paga."),
              _criterios(["S2", "S6", "S8", "S4"], constantes=["factores", "alcances"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule la duración del proyecto in situ y prefabricado, las operario-semanas en obra de cada opción "
        "(diagrama de cargas) y las toneladas de residuo que se generan en la obra en cada caso.",
        "(A) Calcule el coste (material, semanas de obra, transporte, gestión de residuos) y las emisiones (semanas de "
        "obra, transporte de los prefabricados, transporte del residuo) de cada opción, y el coste total al precio "
        "sombra. ¿Pesan más los camiones de la fábrica o las semanas de obra?",
        "(B) Según los criterios de referencia, ¿es correcto decir que la prefabricación «es lo contrario de "
        "sostenible por los camiones»? Clasifique el transporte de los prefabricados por su alcance y el residuo "
        "evitado por su escalón en la jerarquía, y decida por el coste total. ¿«Lo que ahorremos en semanas no es de "
        "este presupuesto» cambia la cuenta?",
        "(C) ¿Qué propone al director técnico y a la dirección? Aplique la rúbrica de la lección."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"peso_prefabricados, km_fabrica, extra_material, coste_semana, emisiones_semana, fraccion_residuo_pref, precio_sombra = {W}, {km}, {extra}, {semana}, {em_semana}, {fraccion_pref}, {precio}\n"
                       f"coste_camion, factor_camion, coste_gestion_residuo, km_gestor = {COSTE_CAMION}, {FACTOR_CAMION}, {COSTE_GESTION_RESIDUO}, {KM_GESTOR}\n" + SETUP +
                       "prefabricadas = sorted(datos_proyecto.index[datos_proyecto['reduccion_prefabricado'] > 0])\n"
                       "datos_pref = datos_proyecto.copy()\n"
                       "datos_pref['duration'] = datos_proyecto['duration'] - datos_proyecto['reduccion_prefabricado']\n"
                       "duracion_pref = int(proyecto.duration(datos_pref['duration']))"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con las duraciones in situ", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Gantt y carga de operarios: ejecución in situ"\n#| label: fig-ganttInSitu\n'
                       "gantt_in, dibujo_in = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo_in)"))
    celdas.append(code('#| fig-cap: "Gantt y carga de operarios: con prefabricados"\n#| label: fig-ganttPref\n'
                       "gantt_pref, dibujo_pref = proyecto.copy().gantt_cargas(datos_pref.copy(), 'duration', 'resources', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo_pref)"))
    celdas.append(code("op_in = float(gantt_in.data.loc['Total'].drop('H_total').astype(float).sum())\n"
                       "op_pref = float(gantt_pref.data.loc['Total'].drop('H_total').astype(float).sum())\n"
                       "residuo_in = float(datos_proyecto['residuos_t'].sum()); residuo_pref = residuo_in * fraccion_residuo_pref\n"
                       "camino_pref = ' / '.join('–'.join(r) for r in proyecto.critical_path(datos_pref['duration']).values())\n"
                       "resumen = pd.DataFrame({'in situ': [duracion, op_in, residuo_in], 'prefabricado': [duracion_pref, op_pref, residuo_pref]},\n"
                       "                       index=['duración (semanas)', 'operario-semanas en obra', 'residuo en obra (t)'])\n"
                       "display(resumen.round(1))\n"
                       "Markdown(f'''Prefabricar acorta el proyecto de {duracion} a **{duracion_pref}** semanas (ruta crítica {camino_pref}), quita "
                       "{op_in - op_pref:.0f} operario-semanas de la obra y evita {residuo_in - residuo_pref:.0f} t de residuo en ella.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Coste y emisiones de cada opción"\n#| label: tbl-opciones\n'
                       "def opcion(semanas, residuo, transporte_t_km, extra):\n"
                       "    coste = pd.Series({'material prefabricado (k€)': extra, 'semanas de obra (k€)': semanas * coste_semana,\n"
                       "                       'transporte de prefabricados (k€)': transporte_t_km * coste_camion / 1000,\n"
                       "                       'gestión del residuo (k€)': residuo * coste_gestion_residuo / 1000})\n"
                       "    emisiones = pd.Series({'semanas de obra (t CO₂)': semanas * emisiones_semana,\n"
                       "                           'transporte de prefabricados, alcance 3 (t CO₂)': transporte_t_km * factor_camion / 1000,\n"
                       "                           'transporte del residuo, alcance 3 (t CO₂)': residuo * km_gestor * factor_camion / 1000})\n"
                       "    s = pd.concat([coste, emisiones])\n"
                       "    s['coste monetario (k€)'] = coste.sum(); s['emisiones (t CO₂)'] = emisiones.sum()\n"
                       "    s['coste total al precio sombra (k€)'] = s['coste monetario (k€)'] + s['emisiones (t CO₂)'] * precio_sombra / 1000\n"
                       "    return s\n"
                       "comparacion = pd.DataFrame({'in situ': opcion(duracion, residuo_in, 0, 0),\n"
                       "                            'prefabricado': opcion(duracion_pref, residuo_pref, peso_prefabricados * km_fabrica, extra_material)})\n"
                       "display(comparacion.round(2))\n"
                       "em_camiones = comparacion.loc['transporte de prefabricados, alcance 3 (t CO₂)', 'prefabricado']\n"
                       "em_semanas = comparacion.loc['semanas de obra (t CO₂)', 'in situ'] - comparacion.loc['semanas de obra (t CO₂)', 'prefabricado']\n"
                       "Markdown(f'''Los camiones de la fábrica emiten {em_camiones:.1f} t; las {duracion - duracion_pref} semanas de obra que se ahorran, {em_semanas:.1f} t. "
                       "En total, prefabricar ''' + ('emite menos' if comparacion.loc['emisiones (t CO₂)', 'prefabricado'] < comparacion.loc['emisiones (t CO₂)', 'in situ'] else 'emite más') + "
                       "f''' ({comparacion.loc['emisiones (t CO₂)', 'prefabricado']:.1f} frente a {comparacion.loc['emisiones (t CO₂)', 'in situ']:.1f} t) y cuesta ''' + "
                       "('**' + f'''{abs(comparacion.loc['coste total al precio sombra (k€)', 'in situ'] - comparacion.loc['coste total al precio sombra (k€)', 'prefabricado']):.1f} k€ ''' + "
                       "('menos' if comparacion.loc['coste total al precio sombra (k€)', 'prefabricado'] < comparacion.loc['coste total al precio sombra (k€)', 'in situ'] else 'más') + '**') + "
                       "f''' al precio sombra ({abs(comparacion.loc['coste monetario (k€)', 'in situ'] - comparacion.loc['coste monetario (k€)', 'prefabricado']):.1f} k€ ''' + "
                       "('menos' if comparacion.loc['coste monetario (k€)', 'prefabricado'] < comparacion.loc['coste monetario (k€)', 'in situ'] else 'más') + ' en dinero).')"))
    celdas += apartado(3)
    celdas.append(md(f"**S6.** El transporte de los prefabricados por un transportista contratado es **alcance 3** del "
                     f"proyecto, y cuenta: {em_tr:.1f} t. Pero también cuentan las semanas de obra que se ahorran "
                     f"({ahorro} × {em_semana} t) y el residuo que no se genera, y «lo contrario de sostenible» es una "
                     f"afirmación sobre una columna de la tabla, no sobre la tabla. **S8.** Las {R * (1 - fraccion_pref):.0f} t "
                     f"de residuo que no llegan a producirse en la obra son **prevención**, el primer escalón de la jerarquía; "
                     f"«los gestiona el gestor» es el último. **S2.** Con el precio sombra la decisión es el coste total: "
                     + (f"**prefabricado** ({tot_pref:.1f} k€ frente a {tot_in:.1f} in situ); "
                        + ("también es más barato en dinero. " if cost_pref < cost_in else
                           f"en dinero in situ es más barato por {cost_pref - cost_in:.1f} k€, y esa es la diferencia que la política de la organización decide pagar. ")
                        if gana_pref else
                        f"**in situ** ({tot_in:.1f} k€ frente a {tot_pref:.1f} prefabricado): con estas distancias y este precio sombra los "
                        f"camiones pesan más que las semanas, y la respuesta es la contraria a la de otros casos; lo que no cambia es "
                        "que se decide con la tabla entera, no con la columna de los camiones. ")
                     + f"**S4.** «Lo que ahorremos en semanas no es de este presupuesto» es la orientación al proceso: las "
                     f"{ahorro} semanas de instalaciones, grúa y equipo son consecuencia de la decisión que se toma ahora, "
                     "las pague quien las pague; sacarlas de la cuenta es la misma trampa que el consumo del equipo barato "
                     "(caso del ciclo de vida)."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('El transporte de los elementos prefabricados por un transportista contratado es, para el proyecto…', ['alcance 3', 'alcance 1', 'alcance 2', 'no se contabiliza por ser del proveedor'],\n"
        "     'S6: transporte contratado.'),\n"
        "    ('Generar menos residuo en la obra porque los elementos llegan fabricados es, en la jerarquía de residuos…', ['prevención, el primer escalón', 'reciclado', 'otra valorización', 'eliminación'],\n"
        "     'S8: el residuo que no se produce.'),\n"
        "    ('Entre dos sistemas constructivos, con un precio sombra del carbono fijado, se elige…', ['el de menor coste total: material, transporte, semanas de obra y residuos, más las emisiones al precio sombra', 'el de menos camiones', 'el de menor precio de material', 'el que siempre se ha usado'],\n"
        "     'S2: toda la tabla, no una columna.'),\n"
        "]", "Sostenibilidad 20 — prefabricación"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director técnico: la tabla del apartado 2 entera, con los camiones en su fila y las semanas en la "
                     f"suya; y las condiciones para que el prefabricado funcione (plazo de fábrica, grúa dimensionada, "
                     f"tolerancias). A la dirección: la decisión con el coste total y la reducción de residuo "
                     f"({R * (1 - fraccion_pref):.0f} t) para el plan de gestión y la memoria. Prueba del ciclo de vida: "
                     f"¿quién paga las {ahorro} semanas de obra que no se ahorran, y en qué presupuesto aparecen?"))
    return celdas, (f"prefabricar acorta {ahorro} semanas ({T} → {T_p}) y evita {R * (1 - fraccion_pref):.0f} t de residuo; "
                    f"{W} t a {km} km ({em_tr:.0f} t CO₂, alcance 3); total al precio sombra {tot_pref:.0f} frente a {tot_in:.0f} k€")


# ---------------------------------------------------------------------------
# Caso 18 — La cuota de agua del bando de sequía: nivelar, asignar con límite, camión cuba o «el hidrante de noche»
# ---------------------------------------------------------------------------
COSTE_CUBA, KM_CUBA = 14.0, 20            # € por m³ de agua traída en camión cuba (agua y porte); km del punto de carga
PRECIO_RED_AGUA = 2.5                      # € por m³ de la red (contador de obra)


def _restriccion_de_agua(rng):
    e = {}
    quiero_retraso = bool(rng.random() < 0.5)

    def acepta(i):
        if not _base(i, criticas_1=False, T=(12, 20)):
            return False
        D = i["D"]
        agua = pd.Series(rng.choice([0, 0, 10, 20, 30, 40, 60], size=len(D)), index=D.index, name="agua_m3").astype(int)
        if (agua > 0).sum() < 4:
            return False
        df = P.tabla_base(i, duration=D, agua_m3=agua)
        try:
            total = _cargas(i, df, "agua_m3")
            desp, total_niv = _nivelar(i, df, "agua_m3")
        except Exception:
            return False
        pico, pico_niv = int(total.max()), int(total_niv.max())
        if pico_niv >= pico or pico_niv < 40:
            return False
        if quiero_retraso:
            Q = pico_niv - int(rng.choice([10, 20]))
            if Q < int(agua.max()) or Q < 20:
                return False
            try:
                with _silencio():
                    _, g, _ = i["proyecto"].copy().asignar(data=df.copy(), duration_label="duration", resource_label="agua_m3",
                                                           maximo=[Q], report=False)
                T_L = len(g.data.columns.drop("H_total"))
            except Exception:
                return False
            if not (1 <= T_L - int(i["T"]) <= 3):
                return False
        else:
            Q, T_L = pico_niv, int(i["T"])
        exceso = (total - Q).clip(lower=0)
        exceso_niv = (total_niv - Q).clip(lower=0)
        if exceso.sum() < 20 or (quiero_retraso and exceso_niv.sum() < 10):
            return False
        e.update(agua=agua, df=df, total=total, total_niv=total_niv, pico=pico, pico_niv=pico_niv, Q=Q, T_L=T_L,
                 exceso=exceso, exceso_niv=exceso_niv)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    df, Q, T_L = e["df"], e["Q"], e["T_L"]
    penal = int(rng.choice([6, 8, 10]))
    multa, corte = int(rng.choice([6, 12, 30])), int(rng.choice([1, 2]))
    p_det = float(rng.choice([0.5, 0.7, 0.9]))
    exceso_ini, exceso_niv = float(e["exceso"].sum()), float(e["exceso_niv"].sum())
    retraso = T_L - T
    c_ini = exceso_ini * COSTE_CUBA / 1000
    c_niv = exceso_niv * COSTE_CUBA / 1000
    c_lim = retraso * penal
    esperado_hidrante = p_det * (multa + corte * penal)
    legales = {"plan inicial + cuba": c_ini, "nivelado + cuba": c_niv, f"asignación con límite de {Q} m³": c_lim}
    mejor = min(legales, key=legales.get)
    mejor_txt = (f"**nivelar**: el plan nivelado cabe en la cuota sin cuba ni retraso (0 k€)" if exceso_niv == 0
                 else f"**{mejor}** ({legales[mejor]:.1f} k€)")
    celdas = [md("# Enunciado"),
              md("Una obra de urbanización consume agua en el curado del hormigón, el riego de compactación de los "
                 "rellenos, el control del polvo y el lavado de maquinaria. El cuadro de prelaciones, las duraciones "
                 "(semanas) y el consumo semanal de agua de cada actividad (`agua_m3`, m³ por semana mientras está en "
                 "curso) se adjuntan. El agua se toma de la red municipal con un contador de obra "
                 f"({PRECIO_RED_AGUA} €/m³)."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"El organismo de cuenca ha declarado la situación excepcional por sequía y el ayuntamiento, por bando, "
                 f"limita el consumo de las obras a **{Q} m³ por semana** por contador. El agua que falte puede traerse "
                 f"en camión cuba desde un punto de carga autorizado a {KM_CUBA} km, a {COSTE_CUBA} €/m³ (agua y porte). "
                 f"El retraso de la entrega tiene una penalidad de {penal} k€ por semana. El reglamento del servicio "
                 f"sanciona la conexión no autorizada a un hidrante con {multa} k€ y el corte del suministro de obra "
                 f"({corte} semana(s) hasta restablecerlo); por la experiencia de la empresa, la probabilidad de que se "
                 f"detecte es del {p_det:.0%}."),
              _mensaje("Jefe de obra",
                       f"Con el plan de inicios más tempranos hay semanas en que necesitamos {e['pico']} m³ y nos dan {Q}. "
                       "No vamos a retrasar la obra por una sequía: el exceso lo sacamos del hidrante de la esquina por "
                       "la noche, que el contador de obra no lo ve, y si alguien pregunta es agua de la cuba. La cuba "
                       "de verdad, sólo si no queda otra."),
              _criterios(["S16", "S11", "S1"], constantes=["sequia"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Dibuje el diagrama de cargas de agua con inicios más tempranos y obtenga el pico y el exceso sobre los "
        f"{Q} m³ semanales, semana a semana. Nivele las cargas dentro de las holguras y repita el cálculo: ¿cabe el "
        "plan nivelado en la cuota sin alargar el proyecto?",
        f"(A) Calcule el coste de las opciones legales: traer en cuba el exceso del plan inicial, traer en cuba el "
        f"exceso del plan nivelado, y programar con asignación limitada a {Q} m³ por semana (duración, retraso y "
        "penalidad). Calcule también el coste esperado de la propuesta del hidrante. ¿Cuál es la opción legal más "
        "barata?",
        "(B) Según el criterio y la norma de referencia, ¿es la cuota un límite del plan o un riesgo que se acepta? "
        "¿Cabe el plan en la cuota tras nivelar, sin retraso? ¿Es el hidrante una opción de la tabla? ¿En qué "
        "escalón de la pirámide queda cada opción?",
        "(C) ¿Qué propone al jefe de obra y qué comunica al cliente y al ayuntamiento? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"cuota, coste_cuba, km_cuba, precio_red, penalidad, multa, corte, p_deteccion = {Q}, {COSTE_CUBA}, {KM_CUBA}, {PRECIO_RED_AGUA}, {penal}, {multa}, {corte}, {p_det}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del proyecto", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| fig-cap: "Gantt con inicios más tempranos y consumo de agua (m³ por semana)"\n#| label: fig-ganttCargas\n'
                       "gantt_inicial, dibujo = proyecto.copy().gantt_cargas(datos_proyecto.copy(), 'duration', 'agua_m3', report=False, tikz=True, params=GANTT_PARAMS)\n"
                       "pinta_tikz(dibujo)"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas de agua frente a la cuota del bando"\n#| label: fig-cargas\n'
                       "cargas = gantt_inicial.data.loc['Total'].drop('H_total').astype(float)\n"
                       "plot_cargas(gantt_df=gantt_inicial, limite_maximo_recursos=[cuota])"))
    celdas.append(code('#| tbl-cap: "Desplazamientos de la nivelación (suma de cuadrados mínima dentro de las holguras)"\n#| label: tbl-nivelacion\n'
                       "proyecto_nivelado = proyecto.copy()\n"
                       "desplazamientos, cuadrados, datos_nivelados, gantt_nivelado, dibujo_niv = proyecto_nivelado.nivelar(\n"
                       "    data=datos_proyecto.copy(), duration_label='duration', resource_label='agua_m3')\n"
                       "cargas_niveladas = gantt_nivelado.data.loc['Total'].drop('H_total').astype(float)\n"
                       "desplazamientos[desplazamientos['desplazamientos'] > 0]"))
    celdas.append(code('#| fig-cap: "Diagrama de cargas de agua tras la nivelación"\n#| label: fig-cargasNivelado\n'
                       "plot_cargas(gantt_df=gantt_nivelado, limite_maximo_recursos=[cuota])"))
    celdas.append(code('#| tbl-cap: "Exceso sobre la cuota, semana a semana, antes y después de nivelar (m³)"\n#| label: tbl-exceso\n'
                       "exceso = (cargas - cuota).clip(lower=0); exceso_niv = (cargas_niveladas - cuota).clip(lower=0)\n"
                       "tabla = pd.DataFrame({'consumo inicial': cargas, 'exceso inicial': exceso, 'consumo nivelado': cargas_niveladas, 'exceso nivelado': exceso_niv})\n"
                       "tabla.loc['Total'] = tabla.sum()\n"
                       "display(tabla.astype(int).T)\n"
                       "Markdown(f'''Pico de **{int(cargas.max())} m³** con inicios más tempranos y **{int(cargas_niveladas.max())} m³** tras nivelar, frente a la cuota de {cuota}; "
                       "el exceso pasa de {int(exceso.sum())} a **{int(exceso_niv.sum())} m³** con la misma duración ({duracion} semanas): ''' + "
                       "('el plan nivelado **cabe** en la cuota.' if exceso_niv.sum() == 0 else 'el plan nivelado **no cabe** del todo; quedan semanas por encima de la cuota.'))"))
    celdas += apartado(2)
    celdas.append(code("proyecto_limitado = proyecto.copy()\n"
                       "datos_asignados, gantt_asignado, dibujo_asig = proyecto_limitado.asignar(data=datos_proyecto.copy(), duration_label='duration',\n"
                       "                                                                      resource_label='agua_m3', maximo=[cuota], report=False, params=GANTT_PARAMS)\n"
                       "duracion_limitada = len(gantt_asignado.data.columns.drop('H_total'))\n"
                       "retraso = duracion_limitada - duracion\n"
                       "opciones = pd.DataFrame({'plan inicial + cuba': [float(exceso.sum()) * coste_cuba / 1000, 0, 0.0],\n"
                       "                         'nivelado + cuba': [float(exceso_niv.sum()) * coste_cuba / 1000, 0, 0.0],\n"
                       "                         f'asignación con límite de {cuota} m³': [0.0, retraso, retraso * penalidad],\n"
                       "                         'hidrante (propuesta)': [0.0, p_deteccion * corte, p_deteccion * (multa + corte * penalidad)]},\n"
                       "                        index=['agua en cuba (k€)', 'retraso (semanas; esperado en el hidrante)', 'penalidad o sanción (k€)'])\n"
                       "opciones.loc['coste (k€)'] = opciones.loc['agua en cuba (k€)'] + opciones.loc['penalidad o sanción (k€)']\n"
                       "display(opciones.round(1))\n"
                       "legales = opciones.loc['coste (k€)'].drop('hidrante (propuesta)')\n"
                       "Markdown(f'''Con la asignación limitada a {cuota} m³ el proyecto dura **{duracion_limitada}** semanas ({retraso} de retraso, {retraso * penalidad} k€). "
                       "La opción legal más barata es **{legales.idxmin()}** ({legales.min():.1f} k€). El hidrante «cuesta» {opciones.loc['coste (k€)', 'hidrante (propuesta)']:.1f} k€ esperados''' + "
                       "(', más que la opción legal: mal negocio además de infracción.' if opciones.loc['coste (k€)', 'hidrante (propuesta)'] > legales.min() else ', menos que la legal: la respuesta B no depende de esta cuenta.'))"))
    celdas.append(code('#| fig-cap: "Gantt con asignación limitada a la cuota"\n#| label: fig-ganttAsignado\npinta_tikz(dibujo_asig)'))
    celdas += apartado(3)
    celdas.append(md(f"**Norma y S16.** La cuota del bando es una limitación de uso del agua en situación excepcional "
                     f"(TRLA, arts. 55.2 y 58) y una condición del suministro de obra: es un **límite del plan**, con fecha "
                     f"y cifra, no un riesgo que se acepta pagando la penalidad. Si el plan cabe lo dice el diagrama de "
                     f"cargas: con inicios más tempranos el pico es de {e['pico']} m³ y el exceso de {exceso_ini:.0f} m³; "
                     f"tras nivelar, {e['pico_niv']} m³ "
                     + ("y **cabe** en la cuota sin retraso. " if exceso_niv == 0 else
                        f"y quedan {exceso_niv:.0f} m³ por encima de la cuota: **no cabe del todo**; o se traen en cuba o se "
                        f"programa con el límite, que retrasa {retraso} semana(s). ")
                     + f"El hidrante **no es una opción de la tabla**: una toma no autorizada es infracción (TRLA, art. "
                     f"116.3.b) y c); reglamento del servicio), y «si alguien pregunta es agua de la cuba» es una afirmación "
                     f"falsa (lección 19, C1). La opción correcta es la legal de menor coste: {mejor_txt}. "
                     f"**S11.** La sequía sí es, además, un riesgo del proyecto: que el bando "
                     f"baje la cuota o cierre el punto de carga de las cubas tiene probabilidad e impacto, y entra en el "
                     f"registro. **S1.** El hidrante traslada el impacto a los demás usuarios de la red (ni siquiera "
                     f"compensa); la cuba **compensa** (el consumo es el mismo, pagado y traído de otra fuente); nivelar y "
                     f"asignar **minimizan** el pico sin cambiar el volumen; reducir el consumo (curado con láminas o "
                     f"filmógenos, reutilizar el agua de lavado) **evita** parte del impacto."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('Una cuota semanal de agua fijada por el bando municipal de sequía es, para el cronograma de la obra…', ['un límite del plan, como el número de operarios: se nivela y, si no cabe, se asigna con límite o se compra el recurso fuera', 'un riesgo que se acepta si la penalidad por retraso es mayor que la sanción', 'una recomendación', 'una reserva de gestión'],\n"
        "     'S16: la cuota tiene cifra y fecha; cabe o no cabe, y lo dice el diagrama de cargas.'),\n"
        "    ('Tomar agua de un hidrante sin autorización durante las restricciones de sequía es…', ['una infracción (toma no autorizada; TRLA art. 116.3 y reglamento del servicio), no una opción del plan', 'una opción si se paga después el agua consumida', 'legal si el contador de obra no lo registra', 'legal de noche'],\n"
        "     'S16: obtener el recurso saltándose la limitación es infracción.'),\n"
        "    ('Nivelar las cargas de agua dentro de las holguras…', ['reduce el pico semanal sin cambiar el volumen total ni la duración', 'reduce el consumo total de agua', 'alarga el proyecto', 'elimina siempre el exceso sobre la cuota'],\n"
        "     'Lección 11: la nivelación mueve actividades no críticas; el volumen es el mismo.'),\n"
        "    ('Traer en camión cuba el agua que la cuota no permite es, en la pirámide de la sostenibilidad…', ['compensar: el mismo consumo, pagado y traído de otra fuente', 'evitar', 'minimizar', 'restaurar'],\n"
        "     'S1: el impacto no se reduce; se paga y se traslada de fuente.'),\n"
        "]", "Sostenibilidad 20 — cuota de agua"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al jefe de obra: la tabla del apartado 2 y el plan nivelado con el calendario de riegos y curados "
                     f"semana a semana; las medidas de reducción (curado con láminas, agua de lavado recirculada) para "
                     f"bajar el consumo, no sólo el pico; y la cuba o la asignación limitada para lo que no quepa, con su "
                     f"coste ({legales[mejor]:.1f} k€ en la opción elegida). Al cliente: la cuota del bando, el plan elegido y su efecto en la "
                     f"entrega. Al ayuntamiento, si hace falta: la solicitud motivada de una cuota mayor para las semanas de "
                     f"hormigonado, con el diagrama de cargas como justificación. Prueba de publicidad: ¿explicaría al "
                     f"concejal, en plena sequía, la manguera del hidrante «porque el contador no lo ve»?"))
    return celdas, (f"cuota de {Q} m³/semana: pico {e['pico']} → {e['pico_niv']} m³ nivelado, exceso {exceso_ini:.0f} → {exceso_niv:.0f} m³; "
                    + (f"asignar con límite retrasa {retraso} semana(s); " if retraso else "cabe tras nivelar; ")
                    + f"hidrante: {esperado_hidrante:.0f} k€ esperados")


# ---------------------------------------------------------------------------
# Caso 19 — La pista de obra por la ZEC: ahorrar semanas y camiones «y luego restauramos»
# ---------------------------------------------------------------------------
def _pista_de_obra(rng):
    e = {}

    def acepta(i):
        if not _base(i, criticas_1=False, T=(14, 24)):
            return False
        D, cal = i["D"], i["calendario"]
        cands = [a for a in sorted(D.index) if int(D[a]) >= 3]
        if len(cands) < 3:
            return False
        acarreo = _elegir(rng, cands, int(rng.integers(3, 5)))
        if not any(cal.loc[a, "H_total"] == 0 for a in acarreo):
            return False
        red = pd.Series(0, index=D.index, name="semanas_menos_pista_corta")
        ton = pd.Series(0, index=D.index, name="toneladas_acarreo")
        for a in acarreo:
            red[a] = 1
            ton[a] = int(rng.integers(5, 31)) * 100
        D_c = D - red
        T_c = _T(i["proyecto"], D_c)
        ahorro = int(i["T"]) - T_c
        if not (1 <= ahorro <= 3):
            return False
        # Si la modificación del proyecto exige una evaluación de W semanas antes de abrir la pista corta, las
        # actividades de acarreo no pueden empezar antes de la semana W: se modela con una actividad previa.
        W = int(rng.choice([16, 18, 20]))
        preds2 = {a: set(ps) for a, ps in i["preds"].items()}
        preds2["Z"] = set()
        for a in acarreo:
            preds2[a] = preds2[a] | {"Z"}
        D2 = pd.concat([D_c, pd.Series({"Z": W})]).astype(int)
        try:
            info2 = P.analizar(preds2, D2)
        except Exception:
            return False
        T_ev = int(info2["T"])
        if T_ev <= int(i["T"]):
            return False
        e.update(acarreo=acarreo, red=red, ton=ton, D_c=D_c, T_c=T_c, ahorro=ahorro, W=W, T_ev=T_ev)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    acarreo, ahorro, T_c, W, T_ev = e["acarreo"], e["ahorro"], e["T_c"], e["W"], e["T_ev"]
    km_c = int(rng.choice([3, 4, 5]))
    km_l = km_c + int(rng.choice([8, 12, 15]))
    semana = int(rng.choice([6, 8, 10]))
    em_semana = float(rng.choice([2.0, 3.0]))
    precio = int(rng.choice([100, 150, 250]))
    tn = float(e["ton"].sum())
    tr_c, tr_l = tn * km_c * COSTE_CAMION / 1000, tn * km_l * COSTE_CAMION / 1000
    em_c, em_l = tn * km_c * FACTOR_CAMION / 1000, tn * km_l * FACTOR_CAMION / 1000
    ahorro_dinero = (tr_l - tr_c) + ahorro * semana
    restaur = int(rng.choice([10, 15, 20, 25]))
    while restaur > ahorro_dinero - 3 and restaur > 5:
        restaur -= 5                                   # la propuesta tiene que ser tentadora: ahorra más de lo que cuesta restaurar
    comp = int(rng.choice([60, 90, 120]))
    df = P.tabla_base(info, duration=D, toneladas_acarreo=e["ton"], semanas_menos_pista_corta=e["red"])
    celdas = [md("# Enunciado"),
              md("La obra de una conducción en zona rural tiene su declaración de impacto ambiental (DIA) y su "
                 "autorización; el acceso de los camiones de materiales y tierras se hace por la **pista de obra "
                 f"evaluada**, de {km_l} km, que rodea una zona especial de conservación (ZEC) de la Red Natura 2000. "
                 "El cuadro de prelaciones, las duraciones (semanas), las toneladas que cada actividad mueve por la "
                 "pista (`toneladas_acarreo`) y las semanas que se ahorrarían en cada una con una **pista corta** "
                 f"de {km_c} km a través de la ZEC (`semanas_menos_pista_corta`) se adjuntan."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"Transporte: {COSTE_CAMION} €/t·km y {FACTOR_CAMION} kg CO₂/t·km. Cada semana de obra cuesta {semana} k€ "
                 f"(instalaciones, maquinaria, equipo técnico) y emite {em_semana} t CO₂. Restaurar la pista corta al "
                 f"terminar costaría {restaur} k€; la medida compensatoria que el órgano ambiental exigiría si la "
                 f"autorizara, unos {comp} k€. Modificar el proyecto evaluado para incluir la pista corta exigiría una "
                 f"evaluación de la modificación que la empresa estima en **{W} semanas** antes de poder abrirla. "
                 f"Precio sombra del carbono de la organización: {precio} €/t."),
              _mensaje("Director de obra",
                       f"Abrimos la pista corta por la ZEC: son {km_l - km_c} km menos por viaje y {ahorro} semana(s) "
                       f"menos de obra. Al acabar restauramos el trazado ({restaur} k€) y, si el órgano ambiental se pone "
                       f"serio, se paga la medida compensatoria; con los camiones y las semanas sale a cuenta. Lo de la "
                       "evaluación de la modificación lo dejamos para la documentación final."),
              _criterios(["S18", "S17", "S1", "S2"], constantes=["natura"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        "(A) Calcule la duración del proyecto con la pista evaluada y con la pista corta, el coste y las emisiones "
        "del transporte por cada una, y el ahorro de la pista corta (semanas al coste semanal más transporte) frente "
        "al coste de restaurarla. Compare el coste total de las dos opciones al precio sombra.",
        f"(A) Si la pista corta sólo pudiera abrirse tras una evaluación de la modificación de {W} semanas (las "
        "actividades de acarreo no empiezan antes), ¿cuánto dura el proyecto? Añada esa actividad previa al cuadro, "
        "recalcule el calendario y compare con el plan evaluado.",
        "(B) Según el criterio y la norma de referencia, ¿puede autorizarse la pista por la ZEC con la restauración "
        "y una medida compensatoria existiendo la pista evaluada? ¿Es la evaluación de la modificación un trámite "
        "«para la documentación final» o una actividad del cronograma? ¿En qué escalón de la pirámide queda cada "
        "opción?",
        "(C) ¿Qué propone al director de obra y qué comunica al cliente y al órgano ambiental? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"km_larga, km_corta, coste_semana, emisiones_semana, coste_restaurar, medida_compensatoria, semanas_evaluacion, precio_sombra = {km_l}, {km_c}, {semana}, {em_semana}, {restaur}, {comp}, {W}, {precio}\n"
                       f"coste_camion, factor_camion = {COSTE_CAMION}, {FACTOR_CAMION}\n" + SETUP +
                       "acarreo = sorted(datos_proyecto.index[datos_proyecto['toneladas_acarreo'] > 0])\n"
                       "duraciones_corta = datos_proyecto['duration'] - datos_proyecto['semanas_menos_pista_corta']\n"
                       "duracion_corta = int(proyecto.duration(duraciones_corta))\n"
                       "toneladas = float(datos_proyecto['toneladas_acarreo'].sum())"))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT con la pista evaluada", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Las dos pistas: duración, transporte, restauración y emisiones"\n#| label: tbl-pistas\n'
                       "def opcion(km, semanas, restaurar):\n"
                       "    s = pd.Series({'duración (semanas)': semanas, 'transporte (k€)': toneladas * km * coste_camion / 1000,\n"
                       "                   'semanas de obra (k€)': semanas * coste_semana, 'restauración (k€)': restaurar,\n"
                       "                   'emisiones del transporte, alcance 3 (t CO₂)': toneladas * km * factor_camion / 1000,\n"
                       "                   'emisiones de las semanas de obra (t CO₂)': semanas * emisiones_semana})\n"
                       "    s['coste monetario (k€)'] = s['transporte (k€)'] + s['semanas de obra (k€)'] + s['restauración (k€)']\n"
                       "    s['emisiones (t CO₂)'] = s['emisiones del transporte, alcance 3 (t CO₂)'] + s['emisiones de las semanas de obra (t CO₂)']\n"
                       "    s['coste total al precio sombra (k€)'] = s['coste monetario (k€)'] + s['emisiones (t CO₂)'] * precio_sombra / 1000\n"
                       "    return s\n"
                       "comparacion = pd.DataFrame({'pista evaluada': opcion(km_larga, duracion, 0), 'pista corta por la ZEC': opcion(km_corta, duracion_corta, coste_restaurar)})\n"
                       "display(comparacion.round(2))\n"
                       "camino_c = ' / '.join('–'.join(r) for r in proyecto.critical_path(duraciones_corta).values())\n"
                       "ahorro = comparacion.loc['coste monetario (k€)', 'pista evaluada'] - comparacion.loc['coste monetario (k€)', 'pista corta por la ZEC']\n"
                       "Markdown(f'''La pista corta acorta el proyecto de {duracion} a **{duracion_corta}** semanas (ruta crítica {camino_c}) y, restauración incluida, "
                       "ahorra **{ahorro:.1f} k€** en dinero y {comparacion.loc['emisiones (t CO₂)', 'pista evaluada'] - comparacion.loc['emisiones (t CO₂)', 'pista corta por la ZEC']:.1f} t: "
                       "al precio sombra, {comparacion.loc['coste total al precio sombra (k€)', 'pista evaluada'] - comparacion.loc['coste total al precio sombra (k€)', 'pista corta por la ZEC']:.1f} k€. "
                       "Ese es el tamaño de la tentación; la pregunta B es si está en la tabla.''')"))
    celdas += apartado(2)
    celdas.append(code('#| tbl-cap: "Calendario con la evaluación de la modificación (Z) antes de las actividades de acarreo"\n#| label: tbl-evaluacion\n'
                       "datos_ev = datos_proyecto[['predecessor']].copy()\n"
                       "datos_ev['duration'] = duraciones_corta\n"
                       "datos_ev.loc['Z'] = ['---', semanas_evaluacion]\n"
                       "for a in acarreo:\n"
                       "    datos_ev.loc[a, 'predecessor'] = 'Z' if datos_ev.loc[a, 'predecessor'] == '---' else datos_ev.loc[a, 'predecessor'] + ',Z'\n"
                       "datos_ev['duration'] = datos_ev['duration'].astype(int)\n"
                       "prelaciones_ev = PredecessorTable.from_dataframe_of_strings(datos_ev, activity='activity', predecessor='predecessor')\n"
                       "proyecto_ev = prelaciones_ev.create_project()\n"
                       "calendario_ev = proyecto_ev.calendar(datos_ev['duration'])\n"
                       "duracion_ev = int(proyecto_ev.duration(datos_ev['duration']))\n"
                       "display(calendario_ev)\n"
                       "camino_ev = ' / '.join('–'.join(r) for r in proyecto_ev.critical_path(datos_ev['duration']).values())\n"
                       "Markdown(f'''Con la evaluación de la modificación ({semanas_evaluacion} semanas) antes del primer acarreo, el proyecto con la pista corta dura **{duracion_ev}** semanas "
                       "(ruta crítica {camino_ev}): **{duracion_ev - duracion}** más que con la pista evaluada, {(duracion_ev - duracion) * coste_semana} k€ de coste indirecto. "
                       "La pista que ahorraba {duracion - duracion_corta} semana(s) cuesta {duracion_ev - duracion} si se hace como la ley manda; y si no, no se puede hacer.''')"))
    celdas.append(code('#| fig-cap: "Gantt con la evaluación de la modificación delante de los acarreos"\n#| label: fig-ganttEv\n'
                       "_g, _dibujo = proyecto_ev.copy().gantt(datos_ev, 'duration', holguras=True, tikz=True, params=dict(inner_text=None, **GANTT_PARAMS))\n"
                       "pinta_tikz(_dibujo)"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma y S18.** La pista corta afecta a un espacio de la Red Natura 2000: exige una adecuada "
                     f"evaluación de sus repercusiones (Ley 42/2007, art. 46.4) y sólo podría autorizarse si no perjudica "
                     f"la integridad del lugar; con conclusiones negativas, la medida compensatoria del art. 46.5 sólo cabe "
                     f"**a falta de soluciones alternativas** y por razones imperiosas de interés público de primer orden. "
                     f"La alternativa existe —es la pista evaluada, de {km_l} km, que ya está en el proyecto— y {ahorro} "
                     f"semana(s) de obra no son una razón imperiosa de interés público: la pista por la ZEC **no es "
                     f"autorizable**, ni con restauración ni con los {comp} k€ de compensación, y abrirla sin más es "
                     f"incumplir la DIA (Ley 21/2013, art. 55.3.c). **S17.** La evaluación de la modificación no es "
                     f"«documentación final»: es una actividad con duración ({W} semanas) y prelación (antes del primer "
                     f"acarreo), y el apartado 2 muestra que el proyecto duraría {T_ev} semanas en vez de {T}: la pista "
                     f"corta, hecha legalmente, es más lenta. **S1.** Pista corta más restauración es **restaurar**; "
                     f"con medida compensatoria, **compensar**; la pista evaluada **evita** el impacto, y aquí evitar no "
                     f"es sólo el escalón más alto: es el único que la norma admite. **S2.** El coste total al precio "
                     f"sombra del apartado 1 informa, pero no decide: la respuesta B no depende de que la pista corta "
                     f"ahorre {ahorro_dinero - restaur:.1f} k€."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('En un espacio de la Red Natura 2000, las medidas compensatorias de un proyecto con conclusiones negativas en su evaluación sólo caben…', ['a falta de soluciones alternativas y por razones imperiosas de interés público de primer orden', 'si el promotor restaura el lugar al terminar', 'si el ahorro del proyecto supera el coste de la medida', 'siempre que las apruebe la dirección facultativa'],\n"
        "     'S18: Ley 42/2007, art. 46.5.'),\n"
        "    ('Abrir una pista de obra no prevista en el proyecto evaluado, a través de una ZEC, sin evaluar la modificación es…', ['incumplir las condiciones de la DIA: infracción grave', 'correcto si se restaura al final', 'correcto si se documenta al cierre', 'una decisión del jefe de obra'],\n"
        "     'Ley 21/2013, arts. 7.2.c) y 55.3.c).'),\n"
        "    ('Un trámite que la norma exige antes de una actividad (evaluar una modificación, aprobar un plan de trabajo) es, para el cronograma…', ['una actividad con su duración y su prelación: se calcula su efecto en el plazo', 'documentación que se adjunta al cierre', 'un riesgo que se acepta', 'una holgura'],\n"
        "     'S17: omitirla no acorta el proyecto; lo deja sin autorización.'),\n"
        "    ('Elegir la pista evaluada que rodea la ZEC en lugar de cruzarla y restaurar después es, en la pirámide…', ['evitar', 'restaurar', 'compensar', 'minimizar'],\n"
        "     'S1: evitar el impacto es el escalón más deseable, y aquí el exigido.'),\n"
        "]", "Sostenibilidad 20 — pista de obra"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al director de obra: la tabla del apartado 1 con la tentación cuantificada y la del apartado 2 con lo "
                     f"que costaría hacerla legalmente ({T_ev - T} semanas más); y lo que sí puede hacerse por la pista "
                     f"evaluada: mejorar su firme para subir la velocidad de los camiones, cargar a plena capacidad, "
                     f"reordenar los acarreos con la nivelación (lección 11) o recuperar {ahorro} semana(s) con Ackoff en "
                     f"otras actividades críticas (lección 10), con su coste. Al cliente: por qué la pista corta no está en "
                     f"la tabla y cuánto cuesta cada semana por la vía legal. Al órgano ambiental: nada que consultar si se "
                     f"respeta la DIA; si se quisiera otra pista, la modificación con su evaluación, antes. Prueba de "
                     f"publicidad: ¿explicaría al Seprona y a la prensa local la pista de {km_c} km por la ZEC como «la "
                     f"restauramos al acabar»?"))
    return celdas, (f"pista corta por la ZEC: {ahorro} semana(s) y {ahorro_dinero:.0f} k€ de semanas y camiones frente a {restaur} k€ de "
                    f"restauración (neto {ahorro_dinero - restaur:.0f} k€); con la evaluación de la modificación ({W} semanas) el proyecto "
                    f"pasa de {T} a {T_ev}; art. 46.5 Ley 42/2007")


# ---------------------------------------------------------------------------
# Caso 20 — La cubierta de fibrocemento: el plan de trabajo de 45 días que no estaba en el cronograma
# ---------------------------------------------------------------------------
SEMANAS_PLAN_AMIANTO = 7       # 45 días del art. 12.2 del RD 396/2006 (6,4 semanas) redondeados a semanas enteras


def _amianto(rng):
    e = {}

    def acepta(i):
        if not _base(i, T=(12, 22)):
            return False
        D, cal = i["D"], i["calendario"]
        cands = [a for a in sorted(D.index) if cal.loc[a, "H_total"] == 0 and a not in i["iniciales"]
                 and 2 <= int(cal.loc[a, "inicio_mas_temprano"]) <= 6]
        if not cands:
            return False
        Y = str(rng.choice(cands))
        d_r = int(rng.choice([1, 2]))
        preds2 = {a: set(ps) for a, ps in i["preds"].items()}
        preds2["P"] = set()
        preds2["R"] = set(preds2[Y]) | {"P"}
        preds2[Y] = {"R"}
        D2 = pd.concat([D, pd.Series({"P": SEMANAS_PLAN_AMIANTO, "R": d_r})]).astype(int)
        try:
            info2 = P.analizar(preds2, D2)
        except Exception:
            return False
        T2 = int(info2["T"])
        retraso = T2 - int(i["T"])
        if not (2 <= retraso <= 8):
            return False
        es_y = int(cal.loc[Y, "inicio_mas_temprano"])
        adelanto = max(0, SEMANAS_PLAN_AMIANTO - es_y)        # semanas antes del inicio en que habría que presentar el plan
        e.update(Y=Y, d_r=d_r, T2=T2, retraso=retraso, es_y=es_y, adelanto=adelanto, preds2=preds2, D2=D2)
        return True

    info = P.buscar(rng, acepta, n=int(rng.choice([8, 9])), n_inicio=int(rng.choice([2, 3])), n_fin=int(rng.choice([1, 2])))
    D, T = info["D"], int(info["T"])
    Y, d_r, T2, retraso, adelanto = e["Y"], e["d_r"], e["T2"], e["retraso"], e["adelanto"]
    m2 = int(rng.integers(6, 25)) * 100
    c_rera = int(rng.choice([18, 24, 30]))       # k€ por la retirada por empresa inscrita (plan, EPI, embalaje, gestor)
    c_peones = int(rng.choice([3, 4, 5]))        # k€ de la propuesta (cuadrilla, contenedor)
    penal = int(rng.choice([6, 8, 10, 12]))
    p_insp = float(rng.choice([0.3, 0.5, 0.7]))
    multa = int(rng.choice([40, 80, 120]))
    paraliz = int(rng.choice([2, 3, 4]))
    esperado_peones = c_peones + p_insp * (multa + paraliz * penal)
    legal_sin = c_rera + retraso * penal
    legal_con = c_rera + d_r * penal
    df = P.tabla_base(info, duration=D)
    celdas = [md("# Enunciado"),
              md("Una obra de reforma de una nave industrial comprende las actividades del cuadro adjunto (duraciones en "
                 f"semanas). La actividad **{Y}** es la demolición de la cubierta y de la estructura ligera; la cubierta "
                 f"son **{m2} m² de placas de fibrocemento con amianto**, que el plan trata como parte de la demolición. "
                 "La obra empieza en la semana 1, al firmar el contrato; el jefe de obra prevé elegir entonces a los "
                 "subcontratistas, aunque la adjudicación se conoce varias semanas antes."),
              code('#| tbl-cap: "Cuadro de datos del enunciado"\n#| label: tbl-datosEnunciado\n' + literal_dataframe(df)),
              md(f"La retirada de la cubierta por una empresa inscrita en el RERA (plan de trabajo, equipos de protección, "
                 f"embalaje y gestor de residuos peligrosos) cuesta {c_rera} k€ y dura {d_r} semana(s); el plan de trabajo "
                 f"se presenta a la autoridad laboral cuando se contrata a la empresa y tarda **{SEMANAS_PLAN_AMIANTO} "
                 f"semanas** en quedar aprobado (45 días). Quitar las placas con la cuadrilla de peones un sábado y "
                 f"llevarlas en el contenedor de escombros costaría {c_peones} k€. La penalidad por retraso de la "
                 f"entrega es de {penal} k€ por semana. Por la experiencia de la empresa, una retirada sin plan ni empresa "
                 f"inscrita acaba en visita de la Inspección de Trabajo con probabilidad {p_insp:.0%}: paralización de la "
                 f"obra {paraliz} semanas y sanción de {multa} k€, además del riesgo para los operarios."),
              _mensaje("Jefe de obra",
                       f"La uralita la quitan los peones el sábado, con mascarilla y mojándola, y al contenedor; en "
                       f"{Y} ya está contada la demolición. Lo de la empresa de amianto son {c_rera} k€ y "
                       f"{SEMANAS_PLAN_AMIANTO} semanas de papeleo que no tenemos: el plan ya va justo."),
              _criterios(["S17", "S10", "S8", "S1"], constantes=["amianto"]),
              md("Responda a las preguntas: A (cálculo), B (criterio) y C (deliberación).")]
    celdas.append(preguntas([
        f"(A) Añada al cuadro las actividades que la norma exige: P, plan de trabajo aprobado ({SEMANAS_PLAN_AMIANTO} "
        f"semanas desde el inicio, sin predecesoras) y R, retirada de la cubierta por la empresa inscrita ({d_r} "
        f"semana(s), después de P y de las predecesoras de {Y}), de modo que {Y} empiece tras R. Calcule la nueva "
        "duración, la ruta crítica y el retraso respecto al plan.",
        f"(A) ¿Cuántas semanas antes del inicio habría que presentar el plan de trabajo para que el retraso fuera sólo "
        f"el de la retirada ({d_r} semana(s))? Calcule el coste de las opciones: retirada legal con el retraso, "
        "retirada legal con el plan presentado con antelación, y la propuesta de los peones con su coste esperado.",
        "(B) Según el criterio y la norma de referencia, ¿pueden los peones retirar la cubierta? ¿Es la aprobación "
        "del plan de trabajo una actividad del cronograma o «papeleo»? ¿Dónde va el residuo? ¿En qué escalón de la "
        "pirámide queda la propuesta?",
        "(C) ¿Qué propone al jefe de obra y qué comunica al cliente y a la dirección facultativa? Aplique la rúbrica."]))
    celdas.append(md("# Resolución"))
    celdas.append(code(f"demolicion, semanas_plan, semanas_retirada, coste_rera, coste_peones, penalidad, p_inspeccion, multa, paralizacion = "
                       f"{Y!r}, {SEMANAS_PLAN_AMIANTO}, {d_r}, {c_rera}, {c_peones}, {penal}, {p_insp}, {multa}, {paraliz}\n" + SETUP))
    celdas += fig_grafo(durations="datos_proyecto['duration']", cap="Grafo PERT del plan presentado", label="fig-pert")
    celdas += apartado(1)
    celdas.append(code('#| tbl-cap: "Cuadro con el plan de trabajo (P) y la retirada (R) delante de la demolición"\n#| label: tbl-datosLegal\n'
                       "datos_legal = datos_proyecto.copy()\n"
                       "pred_y = datos_legal.loc[demolicion, 'predecessor']\n"
                       "datos_legal.loc['P'] = ['---', semanas_plan]\n"
                       "datos_legal.loc['R'] = ['P' if pred_y == '---' else pred_y + ',P', semanas_retirada]\n"
                       "datos_legal.loc[demolicion, 'predecessor'] = 'R'\n"
                       "datos_legal['duration'] = datos_legal['duration'].astype(int)\n"
                       "display(datos_legal)\n"
                       "prelaciones_legal = PredecessorTable.from_dataframe_of_strings(datos_legal, activity='activity', predecessor='predecessor')\n"
                       "proyecto_legal = prelaciones_legal.create_project()\n"
                       "calendario_legal = proyecto_legal.calendar(datos_legal['duration'])\n"
                       "duracion_legal = int(proyecto_legal.duration(datos_legal['duration']))\n"
                       "retraso = duracion_legal - duracion"))
    celdas += fig_grafo(durations="datos_legal['duration']", cap="Grafo PERT con el plan de trabajo y la retirada", label="fig-pertLegal", obj="proyecto_legal")
    celdas.append(code('#| tbl-cap: "Calendario con las actividades exigidas por la norma"\n#| label: tbl-calendarioLegal\n'
                       "display(calendario_legal)\n"
                       "camino = ' / '.join('–'.join(r) for r in proyecto_legal.critical_path(datos_legal['duration']).values())\n"
                       "Markdown(f'''El proyecto pasa de {duracion} a **{duracion_legal}** semanas (ruta crítica {camino}): **{retraso}** semana(s) de retraso, "
                       "{retraso * penalidad} k€ de penalidad. El plan de trabajo (P) entra en la ruta crítica: {demolicion} no puede empezar hasta la semana "
                       "{int(calendario_legal.loc[demolicion, 'inicio_mas_temprano'])} en vez de la {int(calendario.loc[demolicion, 'inicio_mas_temprano'])}.''')"))
    celdas += apartado(2)
    celdas.append(code("es_y = int(calendario.loc[demolicion, 'inicio_mas_temprano'])\n"
                       "adelanto = max(0, semanas_plan - es_y)\n"
                       "opciones = pd.Series({'retirada legal, plan presentado al empezar': coste_rera + retraso * penalidad,\n"
                       "                      f'retirada legal, plan presentado {adelanto} semana(s) antes': coste_rera + semanas_retirada * penalidad,\n"
                       "                      'peones el sábado (propuesta): coste esperado': coste_peones + p_inspeccion * (multa + paralizacion * penalidad)})\n"
                       "display(opciones.round(1).to_frame('coste (k€)'))\n"
                       "Markdown(f'''{demolicion} empezaría en la semana {es_y}: presentando el plan **{adelanto} semana(s) antes del inicio** (en cuanto se adjudique, con la empresa "
                       "inscrita ya elegida) el retraso se reduce a la semana de la retirada, {semanas_retirada * penalidad} k€. La propuesta de los peones cuesta "
                       "{opciones.iloc[2]:.1f} k€ esperados''' + (', más que la vía legal con antelación: mal negocio además de infracción.' if opciones.iloc[2] > opciones.iloc[1] else ', menos que la vía legal: la respuesta B no depende de esta cuenta.'))"))
    celdas += apartado(3)
    celdas.append(md(f"**Norma y S17.** Los trabajos con riesgo de exposición al amianto sólo pueden hacerlos empresas "
                     f"inscritas en el RERA (RD 396/2006, art. 17.1) con un plan de trabajo aprobado por la autoridad "
                     f"laboral antes de empezar (arts. 11 y 12): los peones **no pueden** retirar la cubierta, con mascarilla "
                     f"o sin ella. El plan de trabajo no es papeleo: es una actividad con duración legal (45 días, art. 12.2, "
                     f"con silencio positivo) y prelación (antes de la retirada, que es antes de {Y}), y su efecto en el plazo "
                     f"es una cuenta: {retraso} semana(s) si se presenta al empezar, {d_r} si se presenta {adelanto} semana(s) "
                     f"antes. Que «en {Y} ya está contada la demolición» es la trampa: contaba una demolición que la ley no "
                     f"permite hacer así. **S8.** El residuo con amianto es **peligroso** (LER 17 06 05*): embalado y "
                     f"etiquetado, a gestor autorizado, sin mezclar con otros residuos (Ley 7/2022, arts. 20 y 21); «al "
                     f"contenedor de escombros» contamina el contenedor entero y es infracción. **S10.** El plan no puede "
                     f"apoyarse en exponer a la cuadrilla a un cancerígeno para ahorrar {c_rera - c_peones} k€: no es una opción de "
                     f"la tabla. **S1.** La propuesta traslada el impacto a los operarios (y, con el contenedor, al gestor "
                     f"y al vertedero): ni siquiera compensa. La vía legal con el plan presentado a tiempo **minimiza** el "
                     f"efecto en el plazo; **evitar** sería haber tenido en el cronograma P y R desde el principio."))
    celdas.append(_gift(
        "preguntas_test = [\n"
        "    ('La retirada de una cubierta de fibrocemento con amianto puede hacerla…', ['sólo una empresa inscrita en el RERA, con un plan de trabajo aprobado por la autoridad laboral antes de empezar', 'cualquier cuadrilla con mascarilla, mojando las placas', 'el contratista principal si la obra es pequeña', 'cualquier empresa si lo autoriza la dirección facultativa'],\n"
        "     'RD 396/2006, arts. 11, 12 y 17.'),\n"
        "    ('La autoridad laboral resuelve sobre el plan de trabajo con amianto…', ['en 45 días desde la entrada en su registro; sin resolución expresa, se entiende aprobado', 'en 10 días', 'no hace falta resolución: basta presentarlo', 'en el plazo que fije el contratista'],\n"
        "     'Art. 12.2 del RD 396/2006: para el cronograma son unas 7 semanas desde la presentación.'),\n"
        "    ('Si un trámite obligatorio de duración d debe terminar antes de una actividad crítica que empieza en la semana ES, el proyecto se retrasa…', ['máx(0, d − ES) semanas (más la duración de lo que el trámite habilita), salvo que el trámite se lance con antelación', 'd semanas siempre', 'nada, porque los trámites van en paralelo', 'ES semanas'],\n"
        "     'S17 y lección 05: el calendario dice cuánto y con qué antelación.'),\n"
        "    ('Los residuos con amianto de una obra son…', ['residuos peligrosos: embalados, etiquetados y entregados a gestor autorizado sin mezclar', 'residuos inertes', 'residuos de construcción ordinarios si están mojados', 'residuos municipales'],\n"
        "     'LER 17 06 05*; Ley 7/2022, arts. 20 y 21.'),\n"
        "]", "Sostenibilidad 20 — amianto"))
    celdas += apartado(4)
    celdas.append(_rubrica())
    celdas.append(md(f"Al jefe de obra: el cuadro del apartado 1 con P y R, la empresa inscrita elegida hoy y el plan de "
                     f"trabajo presentado hoy (cada semana que se tarde es una semana de retraso), y la lista de lo que sí "
                     f"se puede adelantar mientras se tramita (todo lo que no depende de {Y}). Al cliente: la cubierta de "
                     f"amianto que el proyecto no había contado como tal, el plazo legal y el efecto en la entrega "
                     f"({retraso} semanas, o {d_r} si se tramita con antelación), con la petición de revisar la penalidad por una "
                     f"obligación que estaba en la nave antes de firmar. A la dirección facultativa: el plan de trabajo y el "
                     f"estudio de gestión de residuos con el residuo peligroso. Prueba de publicidad: ¿explicaría a la "
                     f"Inspección de Trabajo, con un parte de exposición de la cuadrilla encima de la mesa, que la uralita «se "
                     f"quitó mojada»?"))
    return celdas, (f"{m2} m² de fibrocemento antes de {Y}: plan de trabajo de {SEMANAS_PLAN_AMIANTO} semanas y retirada de {d_r}; "
                    f"{T} → {T2} semanas (+{retraso}), o +{d_r} presentando el plan {adelanto} semana(s) antes; arts. 11, 12 y 17 RD 396/2006")


VARIANTES = [
    Variante("compresion-con-carbono", 3, "Ackoff con el coste monetario frente a Ackoff con el carbono al precio sombra; coste total y emisiones de cada plan; S2 y S1",
             _compresion_con_carbono),
    Variante("coste-ciclo-de-vida", 2, "dos equipos: precio de compra frente a coste del ciclo de vida (energía, mantenimiento, fin de vida, carbono); precio de equilibrio; S3, S2 y S4",
             _coste_ciclo_de_vida, grafo="pert"),
    Variante("residuos-de-obra", 2, "demolición selectiva frente a contenedor único: semanas, coste de gestión por fracciones e impuesto al vertido; art. 30 Ley 7/2022; S8 y S1",
             _residuos_de_obra),
    Variante("fraccionamiento-eia", 3, "fraccionar la planta para esquivar el anexo I: duración y P(hito) con la evaluación simplificada y con la ordinaria; art. 7.1.a Ley 21/2013; S9 y S11",
             _fraccionamiento_eia),
    Variante("huella-del-proyecto", 3, "valor ganado de carbono: PE, EE, AE, EPI y previsión al cierre frente al indicador del acta; alcances 1, 2 y 3 de las fuentes; S5 y S6",
             _huella_del_proyecto),
    Variante("neutro-en-carbono", 2, "compensar toda la huella con créditos baratos frente a la curva de coste marginal de reducción; afirmaciones prohibidas por la Directiva 2024/825; S7, S1 y S2",
             _neutro_en_carbono, grafo="pert"),
    Variante("ola-de-calor", 3, "pérdida esperada por avisos de calor en las actividades exteriores; P(plazo) con y sin el calor (Monte Carlo binomial); RD 486/1997 DA única; S10 y S11",
             _ola_de_calor),
    Variante("subcontratacion-en-cadena", 2, "pico de operarios, nivelación y asignación con límite frente a una cadena de cuatro subcontratas; art. 5 Ley 32/2006; S10 y S1",
             _subcontratacion_en_cadena),
    Variante("energia-en-obra", 2, "cargas eléctricas niveladas para caber en la acometida: coste y emisiones del grupo electrógeno frente a la red; alcances 1 y 2; S2, S1 y S6",
             _energia_en_obra),
    Variante("acero-importado", 3, "acero importado: precio, transporte, certificados CBAM, retraso frente a la holgura y emisiones de alcance 3; umbral de minimis; S2 y S6",
             _acero_importado),
    Variante("desmantelamiento", 2, "VAN con y sin el desmantelamiento al final de la vida útil; provisión según el PGC y cuota anual; S4 y S1",
             _desmantelamiento, grafo="pert"),
    Variante("ventana-ambiental", 2, "la DIA prohíbe desbroces en la época de cría: actividades afectadas que terminan antes o esperan, esperas dentro de la holgura, adelantar o retrasar el inicio; art. 55.3.c Ley 21/2013; S12 y S11",
             _ventana_ambiental),
    Variante("tierras-sobrantes", 2, "balance de tierras: vertedero y préstamo frente a reutilizar según las fechas del desmonte y del terraplén (holgura, acopio o retraso); art. 3.2.b Ley 7/2022; S13, S8 y S1",
             _tierras_sobrantes),
    Variante("clausula-social", 3, "condición especial de ejecución (porcentaje de operario-semanas de inserción): capacidad con inicios más tempranos y tras nivelar, coste frente a penalidad; art. 202 LCSP; S14 y S10",
             _clausula_social),
    Variante("hormigon-bajo-carbono", 3, "cemento con menos clínker que alarga una semana el hormigonado: emisiones, retraso, Ackoff para recuperarlo y coste total al precio sombra; S15, S2, S6 y S1",
             _hormigon_bajo_carbono),
    Variante("turno-de-noche", 2, "Ackoff con turno de noche en todas las actividades frente a sólo en las no ruidosas; coste esperado con denuncia; ordenanza y Ley 37/2003; S12, S10 y S1",
             _turno_de_noche),
    Variante("prefabricacion", 2, "in situ frente a prefabricado: duración, operario-semanas, residuo evitado (prevención), transporte de alcance 3 y coste total al precio sombra; S2, S6, S8 y S4",
             _prefabricacion),
    Variante("restriccion-de-agua", 2, "cuota semanal de agua del bando de sequía: cargas, nivelación, asignación con límite, camión cuba para el exceso frente al hidrante; TRLA arts. 55, 58 y 116; S16, S11 y S1",
             _restriccion_de_agua),
    Variante("pista-de-obra", 3, "pista corta por una ZEC frente a la pista evaluada: semanas, transporte, restauración y compensación; la evaluación de la modificación como actividad previa del cronograma; art. 46.5 Ley 42/2007; S18, S17, S1 y S2",
             _pista_de_obra),
    Variante("amianto", 2, "cubierta de fibrocemento antes de la demolición: plan de trabajo de 45 días y retirada por empresa RERA como actividades del cronograma, retraso y antelación necesaria, residuo peligroso; arts. 11, 12 y 17 RD 396/2006; S17, S10, S8 y S1",
             _amianto),
]


def generar(k, rng):
    return despachar(VARIANTES, k, rng)
