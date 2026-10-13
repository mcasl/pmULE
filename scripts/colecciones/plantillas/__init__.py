"""Registro de plantillas. Cada módulo expone META (slug, prefijo, titulo,
descripcion, n_defecto) y generar(k, rng) -> (celdas, subtitulo)."""
from . import (p01_prelaciones, p02_pert, p03_roy, p04_zaderenko, p05_holguras,
               p06_caminos, p07_estimaciones, p08_probabilidad, p09_montecarlo,
               p10_reduccion, p11_recursos, p12_valor_ganado)

PLANTILLAS = [p01_prelaciones, p02_pert, p03_roy, p04_zaderenko, p05_holguras,
              p06_caminos, p07_estimaciones, p08_probabilidad, p09_montecarlo,
              p10_reduccion, p11_recursos, p12_valor_ganado]
