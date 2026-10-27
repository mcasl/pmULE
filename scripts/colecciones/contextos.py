"""
contextos.py — contextos narrativos para los enunciados: un tema, una unidad
de tiempo, un nombre de recurso y una lista de descripciones de actividades
en orden aproximadamente cronológico, que se asignan a A, B, C, … del proyecto.

Las plantillas no dependen de ellos: el generador los inyecta a posteriori en
una fracción de los cuadernos (ver `inyectar`).
"""
from __future__ import annotations

import re

CONTEXTOS = [
    dict(tema="la rehabilitación de una nave industrial para un taller", unidad="semanas", recurso="operarios",
         actividades=["Levantamiento topográfico y mediciones", "Proyecto de ejecución y licencia", "Desmontaje de instalaciones antiguas",
                      "Demolición de tabiquería interior", "Refuerzo de la estructura metálica", "Reparación de la cubierta",
                      "Solera y canalizaciones", "Cerramientos y carpintería exterior", "Instalación eléctrica",
                      "Instalación de aire comprimido", "Tabiquería de oficinas", "Pintura y acabados",
                      "Montaje de maquinaria", "Inspección final y puesta en marcha"]),
    dict(tema="el lanzamiento de una aplicación móvil", unidad="semanas", recurso="desarrolladores",
         actividades=["Estudio de requisitos", "Diseño de la arquitectura", "Diseño de interfaz y prototipo",
                      "Base de datos y modelo", "API del servidor", "Pantallas principales",
                      "Módulo de pagos", "Notificaciones y sincronización", "Pruebas de integración",
                      "Pruebas con usuarios", "Corrección de incidencias", "Documentación y material de ayuda",
                      "Publicación en las tiendas", "Campaña de lanzamiento"]),
    dict(tema="la organización de un congreso científico", unidad="semanas", recurso="personas del comité",
         actividades=["Definición del programa y fechas", "Reserva de la sede", "Convocatoria de comunicaciones",
                      "Página web e inscripciones", "Contratación de ponentes invitados", "Revisión de los trabajos recibidos",
                      "Elaboración del programa definitivo", "Contratación de catering y viajes", "Edición del libro de actas",
                      "Preparación de salas y audiovisuales", "Acreditaciones y material de congresistas", "Difusión en medios",
                      "Celebración del congreso", "Cierre económico y memoria"]),
    dict(tema="la instalación de una planta fotovoltaica sobre cubierta", unidad="días", recurso="instaladores",
         actividades=["Estudio de sombras y producción", "Cálculo estructural de la cubierta", "Tramitación de permisos",
                      "Acopio de módulos e inversores", "Montaje de la estructura soporte", "Colocación de módulos",
                      "Cableado de corriente continua", "Instalación de inversores", "Cuadro de protecciones y contador",
                      "Canalizaciones hasta el punto de conexión", "Sistema de monitorización", "Pruebas y medidas eléctricas",
                      "Legalización ante la distribuidora", "Puesta en servicio"]),
    dict(tema="la puesta en marcha de una línea de envasado", unidad="semanas", recurso="técnicos",
         actividades=["Especificación de la línea", "Adjudicación de los equipos", "Obra civil y bancadas",
                      "Recepción de la llenadora", "Recepción de la etiquetadora", "Montaje mecánico",
                      "Instalación neumática", "Cableado y armarios de control", "Programación de autómatas",
                      "Integración con el almacén", "Pruebas en vacío", "Pruebas con producto",
                      "Formación de operarios", "Validación y arranque de producción"]),
    dict(tema="la producción de un cortometraje", unidad="días", recurso="miembros del equipo",
         actividades=["Guion definitivo", "Casting", "Localizaciones y permisos",
                      "Diseño de vestuario y atrezo", "Plan de rodaje", "Ensayos con los actores",
                      "Rodaje de exteriores", "Rodaje en plató", "Grabación de sonido directo",
                      "Montaje", "Música y diseño sonoro", "Etalonaje y efectos",
                      "Subtitulado y copias", "Estreno y envío a festivales"]),
    dict(tema="la apertura de una clínica veterinaria", unidad="semanas", recurso="personas",
         actividades=["Estudio de mercado y local", "Proyecto de reforma y licencia", "Obra de reforma",
                      "Instalación de quirófano y radiología", "Mobiliario y equipamiento clínico", "Software de gestión",
                      "Selección de personal", "Acuerdos con proveedores", "Registro sanitario",
                      "Formación del personal", "Página web y redes", "Campaña de apertura",
                      "Jornada de puertas abiertas", "Inicio de la actividad"]),
]


def con_a(tema: str) -> str:
    """«a» + tema con la contracción «al» cuando el tema empieza por «el »."""
    return "al " + tema[3:] if tema.startswith("el ") else "a " + tema


def con_de(tema: str) -> str:
    """«de» + tema con la contracción «del» cuando el tema empieza por «el »."""
    return "del " + tema[3:] if tema.startswith("el ") else "de " + tema


def elegir(rng, n: int) -> dict:
    """Un contexto al azar con al menos n actividades; devuelve una copia con las n primeras."""
    candidatos = [c for c in CONTEXTOS if len(c["actividades"]) >= n]
    c = candidatos[int(rng.integers(0, len(candidatos)))]
    return dict(c, actividades=c["actividades"][:n])


def _actividades_de_la_tabla(celdas) -> tuple[int, list[str]]:
    """Busca la celda `datos_enunciado = pd.DataFrame([...` y devuelve los nombres de actividad."""
    for i, (tipo, texto) in enumerate(celdas):
        if tipo == "code" and "datos_enunciado = pd.DataFrame([" in texto:
            nombres = re.findall(r"^\s*\(\s*'([A-Za-z0-9]+)'\s*,", texto, re.M)
            return i, nombres
    return -1, []


_UNIDADES = ("semanas", "días", "meses", "horas", "jornadas")


def _unidad_del_enunciado(celdas) -> str | None:
    """Unidad de tiempo que usa el enunciado (la primera que aparece en sus textos y preguntas)."""
    texto = "\n".join(t for _, t in celdas)
    posiciones = {u: texto.find(u) for u in _UNIDADES if u in texto}
    return min(posiciones, key=posiciones.get) if posiciones else None


def inyectar(celdas: list, rng) -> list:
    """Inserta, tras la tabla de datos, una tabla con la descripción de cada actividad
    según un contexto narrativo. Si no encuentra la tabla devuelve las celdas intactas."""
    i, nombres = _actividades_de_la_tabla(celdas)
    if i < 0 or not nombres or len(nombres) > 14:
        return celdas
    ctx = elegir(rng, len(nombres))
    lineas = ["descripciones = pd.Series({"]
    for a, desc in zip(nombres, ctx["actividades"]):
        lineas.append(f"    {a!r}: {desc!r},")
    lineas.append("}, name='Descripción')")
    lineas.append("descripciones.to_frame()")
    # La unidad de tiempo es la que usa el enunciado (no la del contexto); si el cuadro no
    # lleva duraciones (colecciones de prelaciones/PERT) no se menciona ninguna.
    unidad = _unidad_del_enunciado(celdas[:i + 1])
    con_duraciones = re.search(r"['\"](duration|a|m|b|D|dur\w*)['\"]", celdas[i][1]) is not None
    coletilla = f" (las duraciones del cuadro están en {unidad})" if (unidad and con_duraciones) else ""
    md = ("md", f"El proyecto corresponde {con_a(ctx['tema'])}; las actividades son las siguientes{coletilla}.")
    code = ("code", '#| tbl-cap: "Descripción de las actividades"\n#| label: tbl-descripciones\n' + "\n".join(lineas))
    return celdas[:i + 1] + [md, code] + celdas[i + 1:]
