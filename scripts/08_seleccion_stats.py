"""
08_seleccion_stats.py — Distribución del PONDERADO de los SELECCIONADOS por carrera. DAML 2026 · Grupo 5.

Para cada carrera, resume con qué puntaje ponderado entró realmente la gente (no solo el corte = mínimo).
Usa SOLO la cohorte más reciente (escala PAES nueva 100–1000); NO mezcla años viejos (escala 850).
Resumen de 5 números para dibujar un boxplot sin exponer datos individuales.

Salida: data/processed/seleccion_stats.json  {cod: {n, p05, p25, p50, p75, p95, min, max, anio}}
Uso:  python3 scripts/08_seleccion_stats.py
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data/processed/dataset_modelo_acceso.parquet")
OUT = os.path.join(ROOT, "data/processed/seleccion_stats.json")
MIN_N = 15   # bajo esto la distribución no es confiable → se omite

df = pd.read_parquet(DS)
anio = int(df["cohorte"].max())                      # cohorte más reciente (escala nueva)
sel = df[(df["cohorte"] == anio) & (df["ACCESO_1PREF"] == 1)].dropna(subset=["PTJE_PREF"])
print(f"Seleccionados {anio}: {len(sel):,} filas")

out = {}
for cod, sub in sel.groupby("COD_CARRERA_PREF"):
    v = sub["PTJE_PREF"].to_numpy()
    if len(v) < MIN_N:
        continue
    p05, p25, p50, p75, p95 = (float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95))
    out[str(int(cod))] = {"n": int(len(v)), "p05": round(p05, 1), "p25": round(p25, 1),
                          "p50": round(p50, 1), "p75": round(p75, 1), "p95": round(p95, 1),
                          "min": round(float(v.min()), 1), "max": round(float(v.max()), 1), "anio": anio}

json.dump(out, open(OUT, "w"), ensure_ascii=False)
print(f"✅ seleccion_stats.json — {len(out)} carreras con ≥{MIN_N} seleccionados ({anio})")
# control: UdeC Arquitectura (13080)
if "13080" in out:
    print("   UdeC Arquitectura (13080):", out["13080"])
