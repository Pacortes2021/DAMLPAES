"""
09_carrera_stats_recientes.py — Actualiza el corte/cupos de referencia a la cohorte MÁS RECIENTE.
DAML 2026 · Grupo 5.

La tarjeta de la app y la referencia del modelo (`CORTE_ANTERIOR`/`CUPOS_ANTERIOR`) usaban el corte 2025.
Para un estudiante que hoy postula al proceso siguiente, el corte vigente es el del último año cerrado
(2026). Este script regenera carrera_stats.json con el corte 2026 (admisión REGULAR, ESTADO_PREF=24,
mín. ponderado de los seleccionados) y los cupos 2026, con respaldo en 2025 para las pocas carreras sin
datos 2026. Guarda el año usado en cada entrada para rotular correctamente.

Misma definición que scripts/00_build_dataset.py::cortes_y_cupos (regular). No reentrena modelos.
Salida: data/processed/carrera_stats.json  {cod: {corte, cupos, anio}}
Uso:  python3 scripts/09_carrera_stats_recientes.py
"""
from __future__ import annotations
import os, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MA = os.path.join(ROOT, "data/processed/master_admision.parquet")
OUT = os.path.join(ROOT, "data/processed/carrera_stats.json")
PRIMARIO, FALLBACK = 2026, 2025


def cortes_cupos(ma: pd.DataFrame, anio: int) -> dict:
    """Corte (mín ponderado de seleccionados REGULAR, ESTADO 24) y cupos (n) de `anio`."""
    sub = ma[(ma["anio"] == anio) & (ma["ESTADO_PREF"] == 24) & (ma["TIPO_PREF"] == "REGULAR")].copy()
    sub["_p"] = pd.to_numeric(sub["PTJE_PREF"], errors="coerce")
    g = sub.dropna(subset=["_p"]).groupby("COD_CARRERA_PREF")["_p"]
    return {int(c): {"corte": round(float(v.min()), 1), "cupos": int(v.size)} for c, v in g}


ma = pd.read_parquet(MA)
prim = cortes_cupos(ma, PRIMARIO)
fall = cortes_cupos(ma, FALLBACK)

stats, n_prim, n_fall = {}, 0, 0
for cod, d in prim.items():
    stats[str(cod)] = {**d, "anio": PRIMARIO}; n_prim += 1
for cod, d in fall.items():                       # respaldo para carreras sin datos del año primario
    if str(cod) not in stats:
        stats[str(cod)] = {**d, "anio": FALLBACK}; n_fall += 1

json.dump(stats, open(OUT, "w"), ensure_ascii=False)
print(f"✅ carrera_stats.json — {len(stats)} carreras · {n_prim} con corte {PRIMARIO} · {n_fall} respaldo {FALLBACK}")
print("   Ind UdeC (13072):", stats.get("13072"))
