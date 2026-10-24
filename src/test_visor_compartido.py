"""Visores paso a paso con elementos SVG compartidos (``_svg_en_elementos``, ``_compartir_svgs``, ``_html_pasos``)."""
import base64
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

import pmule


def svg(glifo="glyph-0-0", mascara="mask-0", fuente="source-1", x=1, extra="", viewbox="0 0 10 10", pixel="AAAA"):
    """SVG con la forma de los de pdftocairo: glifo -> máscara -> imagen en <defs> y un cuerpo que los usa."""
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="10" height="10" '
            f'viewBox="{viewbox}">\n<defs>\n<g>\n<g id="{glifo}">\n<rect x="0" y="0" width="1" height="1" mask="url(#{mascara})"/>\n'
            f'</g>\n</g>\n<mask id="{mascara}">\n<use xlink:href="#{fuente}"/>\n</mask>\n'
            f'<image id="{fuente}" x="0" y="0" width="1" height="1" xlink:href="data:image/png;base64,{pixel}"/>\n</defs>\n'
            f'<g fill="rgb(0%, 0%, 0%)" fill-opacity="1">\n<use xlink:href="#{glifo}" x="{x}" y="2"/>\n</g>\n'
            f'<path d="M 0 0 L 1 1" stroke="red"/>\n{extra}</svg>\n')


def foto(s, pie=""):
    b64 = base64.b64encode(s.encode("utf-8")).decode("ascii")
    return ('<img alt="paso" style="display:block;margin:0 auto;width:100%;max-width:1100px;height:auto" '
            f'src="data:image/svg+xml;base64,{b64}">{pie}')


def referencias_resueltas(html):
    refs = set(re.findall(r'href="#([^"]+)"', html)) | set(re.findall(r'url\(#([^)]+)\)', html))
    return [r for r in refs if f'id="{r}"' not in html]


class TestSvgEnElementos(unittest.TestCase):
    def test_los_ids_dependen_del_contenido_y_no_del_nombre_original(self):
        a = pmule._svg_en_elementos(svg())
        b = pmule._svg_en_elementos(svg(glifo="glyph-9-4", mascara="mask-7", fuente="source-52"))
        self.assertEqual(a[1], b[1])                                   # mismos defs con los mismos ids
        self.assertEqual(a[2], b[2])
        self.assertEqual(a[0], "0 0 10 10")

    def test_contenido_distinto_da_ids_distintos(self):
        a = pmule._svg_en_elementos(svg(pixel="AAAA"))
        b = pmule._svg_en_elementos(svg(pixel="BBBB"))                  # otro mapa de bits: otro glifo
        self.assertNotEqual(set(a[1]), set(b[1]))

    def test_cuerpo_repetido_entre_fotogramas_se_guarda_una_vez(self):
        a = pmule._svg_en_elementos(svg(x=1))
        b = pmule._svg_en_elementos(svg(x=1, extra='<rect x="3" y="3" width="2" height="2"/>\n'))
        self.assertEqual(a[2], b[2][:len(a[2])])                        # los mismos elementos de primer nivel
        self.assertEqual(len(b[2]), len(a[2]) + 1)
        self.assertEqual(len({*a[1], *b[1]}), len(a[1]) + 1)            # solo el rectángulo nuevo añade un elemento

    def test_referencia_inexistente_deja_la_imagen_como_estaba(self):
        roto = svg().replace('xlink:href="#source-1"', 'xlink:href="#no-existe"')
        self.assertIsNone(pmule._svg_en_elementos(roto))

    def test_ids_en_el_cuerpo_no_se_tocan(self):
        self.assertIsNone(pmule._svg_en_elementos(svg(extra='<g id="clip1"><path d="M0 0"/></g>\n')))

    def test_sin_viewbox_o_con_atributos_raros_no_se_toca(self):
        self.assertIsNone(pmule._svg_en_elementos(svg().replace(' viewBox="0 0 10 10"', "")))
        self.assertIsNone(pmule._svg_en_elementos(svg().replace('width="10" height="10"', 'width="10" height="10" fill="red"')))

    def test_ids_anidados_no_se_tocan(self):
        anidado = svg().replace('<rect x="0" y="0" width="1" height="1" mask="url(#mask-0)"/>',
                                '<g id="dentro"><rect x="0" y="0" width="1" height="1"/></g>')
        self.assertIsNone(pmule._svg_en_elementos(anidado))


class TestCompartirSvgs(unittest.TestCase):
    def test_sustituye_las_imagenes_y_conserva_el_resto(self):
        fotos = [foto(svg(x=1), "<p>pie 1</p>"), foto(svg(x=2), "<p>pie 2</p>")]
        nuevas, comun = pmule._compartir_svgs(fotos, "pmpTEST")
        self.assertEqual(len(nuevas), 2)
        for f, pie in zip(nuevas, ("pie 1", "pie 2")):
            self.assertNotIn("base64", f)
            self.assertIn("<svg", f)
            self.assertIn(f"<p>{pie}</p>", f)
            self.assertIn('viewBox="0 0 10 10"', f)
            self.assertIn("max-width:1100px", f)                       # conserva el estilo de la imagen
        self.assertEqual(referencias_resueltas(comun + "".join(nuevas)), [])
        self.assertTrue(all("pmpTEST-" in m for m in re.findall(r'href="#([^"]+)"', "".join(nuevas))))

    def test_comparte_los_glifos_entre_fotogramas(self):
        fotos = [foto(svg(x=i)) for i in range(6)]
        _, comun = pmule._compartir_svgs(fotos, "pmpTEST")
        self.assertEqual(comun.count("<image "), 1)                    # un solo mapa de bits para los 6 fotogramas
        self.assertEqual(comun.count('<mask '), 1)

    def test_lo_que_no_es_svg_o_no_se_puede_descomponer_no_cambia(self):
        raro = svg(extra='<g id="clip1"/>\n')
        fotos = ["<p>sin imagen</p>", foto(raro), '<img alt="x" src="data:image/png;base64,AAAA">']
        nuevas, comun = pmule._compartir_svgs(fotos, "pmpTEST")
        self.assertEqual(nuevas, fotos)
        self.assertEqual(comun, "")

    def test_desactivado_por_variable_de_entorno(self):
        fotos = [foto(svg())]
        with mock.patch.dict(os.environ, {"PMULE_VISOR_COMPARTIDO": "0"}):
            self.assertEqual(pmule._compartir_svgs(fotos, "pmpTEST"), (fotos, ""))


class TestHtmlPasosCompartido(unittest.TestCase):
    def fotos(self):
        return [foto(svg(x=i), f"<p>paso {i}</p>") for i in range(8)]

    def test_visor_reproducible_pequeno_y_sin_referencias_rotas(self):
        a = pmule._html_pasos(self.fotos(), inicio=0)
        b = pmule._html_pasos(self.fotos(), inicio=0)
        self.assertEqual(a, b)
        with mock.patch.dict(os.environ, {"PMULE_VISOR_COMPARTIDO": "0"}):
            viejo = pmule._html_pasos(self.fotos(), inicio=0)
        self.assertLess(len(a), len(viejo))
        self.assertEqual(referencias_resueltas(a), [])
        self.assertEqual(a.count('class="pmule-paso"'), 8)
        self.assertEqual(a.count("<svg class=\"pmule-fig\""), 8)
        self.assertIn('img,svg.pmule-fig', a)                          # la pantalla completa también amplía los SVG
        self.assertEqual(a.count("base64,AAAA"), 1)                    # el mapa de bits del glifo, una sola vez
        self.assertNotIn("data:image/svg+xml", a)

    def test_el_id_del_visor_no_cambia_con_la_compartición(self):
        a = pmule._html_pasos(self.fotos(), inicio=0)
        with mock.patch.dict(os.environ, {"PMULE_VISOR_COMPARTIDO": "0"}):
            viejo = pmule._html_pasos(self.fotos(), inicio=0)
        self.assertEqual(re.search(r'id="(pmp[0-9a-f]{10})" class="pmule-pasos"', a).group(1),
                         re.search(r'id="(pmp[0-9a-f]{10})" class="pmule-pasos"', viejo).group(1))

    def test_dos_visores_distintos_no_comparten_ids(self):
        a = pmule._html_pasos([foto(svg(x=1))], inicio=0)
        b = pmule._html_pasos([foto(svg(x=2))], inicio=0)
        ids = lambda h: set(re.findall(r'<(?:g|mask|image)[^>]* id="([^"]+)"', h))
        self.assertFalse(ids(a) & ids(b))


@unittest.skipUnless(shutil.which("pdflatex") and shutil.which("pdftocairo"), "hace falta pdflatex y pdftocairo")
class TestVisorCompartidoConLatexReal(unittest.TestCase):
    def test_fotogramas_reales(self):
        cods = [rf"\begin{{tikzpicture}}\node[draw] {{Actividad {i} $T_{i}$}}; \draw (0,0) -- ({i},1);\end{{tikzpicture}}"
                for i in range(6)]
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with mock.patch.dict(os.environ, {"PMULE_TIKZ_CACHE": d, "PMULE_TIKZ_JOBS": "1"}):
            svgs = pmule._svgs_de_tikz(cods)
        fotos = [foto(s) for s in svgs]
        nuevo = pmule._html_pasos(fotos, inicio=0)
        with mock.patch.dict(os.environ, {"PMULE_VISOR_COMPARTIDO": "0"}):
            viejo = pmule._html_pasos(fotos, inicio=0)
        self.assertEqual(nuevo.count("<svg class=\"pmule-fig\""), 6)    # ninguno quedó sin descomponer
        self.assertEqual(referencias_resueltas(nuevo), [])
        self.assertLess(len(nuevo), len(viejo))


@unittest.skipUnless(shutil.which("pdflatex") and shutil.which("pdftocairo"), "hace falta pdflatex y pdftocairo")
class TestOtrosVisoresCompartidos(unittest.TestCase):
    """Matriz de rutas y calendario interactivo: sus imágenes se buscan por clase desde el JavaScript."""

    def setUp(self):
        from test_pert_pasos import project
        self.pg = project()
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = mock.patch.dict(os.environ, {"PMULE_TIKZ_CACHE": d})
        p.start()
        self.addCleanup(p.stop)

    def comprueba(self, html, clase, viejo):
        marcos = re.findall(r'<svg class="pmule-fig %s".*?</svg>' % clase, html, flags=re.S)
        self.assertGreater(len(marcos), 2)
        self.assertEqual(len(marcos), viejo.count(f'class="{clase}"'))          # una por imagen original
        self.assertNotIn("data:image/svg+xml", html)
        self.assertEqual(referencias_resueltas(html), [])
        self.assertIn('style="display:block;', marcos[0])                         # el primero visible
        self.assertIn('style="display:none;', marcos[1])
        self.assertLess(len(html), len(viejo))

    def test_rutas_html(self):
        html = self.pg.rutas_html("pert")
        with mock.patch.dict(os.environ, {"PMULE_VISOR_COMPARTIDO": "0"}):
            viejo = self.pg.rutas_html("pert")
        self.comprueba(html, "pmule-rut-img", viejo)

    def test_calendario_html(self):
        from test_pert_pasos import DUR
        html = self.pg.calendario_html(DUR)
        with mock.patch.dict(os.environ, {"PMULE_VISOR_COMPARTIDO": "0"}):
            viejo = self.pg.calendario_html(DUR)
        self.comprueba(html, "pmule-cal-img", viejo)


class TestRutaResaltadaCoincideConLaFila(unittest.TestCase):
    """El grafo de cada fotograma del visor de la matriz de rutas resalta las actividades de la fila
    que se muestra (``path_matrix(dummies=True)`` ordena las rutas de otro modo que ``dummies=False``)."""

    def reales_resaltadas(self, pg):
        capturas = []
        original = type(pg)._pert_fotograma

        def espia(self_, *a, **kw):
            capturas.append(kw.get("estilo_arcos"))
            return original(self_, *a, **kw)
        with mock.patch.object(type(pg), "_pert_fotograma", espia):
            pg.plot_tikz_rutas("pert")
        arco_de = {(f"{u}", f"{v}"): d.get("activity", "") for u, v, d in pg.pert_graph.edges(data=True)}
        return [frozenset(arco_de[e] for e, s in ea.items() if s == "actual") for ea in capturas[1:]]

    def test_pert_resalta_las_actividades_de_la_fila(self):
        from test_pert_pasos import project
        pg = project()
        R = pg.path_matrix(dummies=False).astype(bool)
        Rd = pg.path_matrix(dummies=True).astype(bool)
        filas = [frozenset(c for c in R.columns if R.iloc[i][c]) for i in range(len(R))]
        filas_d = [frozenset(c for c in R.columns if c in Rd.columns and Rd.iloc[i][c]) for i in range(len(Rd))]
        self.assertNotEqual(filas, filas_d)                    # el desorden que causaba el fallo
        resaltadas = self.reales_resaltadas(pg)
        self.assertEqual(len(resaltadas), len(R))
        for i, (esperadas, vistas) in enumerate(zip(filas, resaltadas)):
            self.assertEqual({a for a in vistas if a in R.columns}, set(esperadas), f"fila {i}")


if __name__ == "__main__":
    unittest.main()
