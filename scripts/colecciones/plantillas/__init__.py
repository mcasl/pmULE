"""Registro de plantillas. Cada módulo expone META (slug, prefijo, titulo, descripcion),
VARIANTES (lista de cuaderno.Variante) y generar(k, rng) -> (celdas, subtitulo, variante)."""
from . import (p01_prelaciones, p02_pert, p03_roy, p04_zaderenko, p05_holguras,
               p06_caminos, p07_estimaciones, p08_probabilidad, p09_montecarlo,
               p10_reduccion, p11_recursos, p12_valor_ganado, p13_formula,
               p14_comprension, p15_avanzado, p16_gantt, p17_viabilidad, p18_casos)

PLANTILLAS = [p01_prelaciones, p02_pert, p03_roy, p04_zaderenko, p05_holguras,
              p06_caminos, p07_estimaciones, p08_probabilidad, p09_montecarlo,
              p10_reduccion, p11_recursos, p12_valor_ganado, p13_formula,
              p14_comprension, p15_avanzado, p16_gantt, p17_viabilidad, p18_casos]
