"""
06_titulacion_stats.py — Estadísticas de TITULACIÓN por carrera/área. DAML 2026 · Grupo 5.

Lee el archivo CRUDO del SIES (titulados 2024, encoding latin-1) y agrega contexto de
titulación: % de mujeres (brecha de género), edad promedio y volumen, por carrera genérica
y por área de conocimiento. Es DESCRIPTIVO (institucional/agregado), no predicción individual.

Se construye desde el crudo (data/raw/TITULADO_*_web_*.csv) — NO desde el master institucional,
que tenía el encoding corrupto. Trazable y reproducible.

Salida: data/processed/titulacion_stats.json  {por_carrera, por_area, anio}

Uso:  python3 scripts/06_titulacion_stats.py
"""
from __future__ import annotations
import os, sys, glob, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from areas import area_de, _norm                       # clasificador carrera→área (mismo que el dashboard)

RAW = glob.glob(os.path.join(ROOT, "data/raw/TITULADO_*_web_*.csv"))
if not RAW:
    raise SystemExit("No se encontró el crudo de titulados (data/raw/TITULADO_*_web_*.csv).")

print(f"Leyendo crudo SIES: {os.path.relpath(RAW[0], ROOT)}")
df = pd.read_csv(RAW[0], sep=";", encoding="latin-1")

C_TOT, C_MUJ = "TOTAL TITULACIONES", "TITULACIONES MUJERES POR PROGRAMA"
C_GEN, C_EDAD = "ÁREA CARRERA GENÉRICA", "PROMEDIO EDAD PROGRAMA "
for c in (C_TOT, C_MUJ, C_EDAD):
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
df = df.dropna(subset=[C_TOT])
df = df[df[C_TOT] > 0]

df["_carrera"] = df[C_GEN].map(_norm)                  # nombre genérico normalizado (sin tildes, mayús)
df["_area"] = df[C_GEN].map(area_de)                   # área (10 grandes áreas)


def _agg(col: str, min_n: int) -> dict:
    out = {}
    for key, sub in df.groupby(col):
        if key is None or (isinstance(key, float) and key != key):
            continue
        n = float(sub[C_TOT].sum())
        if n < min_n:
            continue
        muj = float(sub[C_MUJ].sum())
        edad = float((sub[C_EDAD] * sub[C_TOT]).sum() / n) if sub[C_EDAD].notna().any() else None
        out[str(key)] = {"n": int(n), "pct_muj": round(100 * muj / n, 1),
                         "edad": round(edad, 1) if edad else None}
    return out


stats = {"anio": 2024, "por_carrera": _agg("_carrera", 30), "por_area": _agg("_area", 100)}
json.dump(stats, open(os.path.join(ROOT, "data/processed/titulacion_stats.json"), "w"), ensure_ascii=False, indent=0)
print(f"✅ titulacion_stats.json — {len(stats['por_carrera'])} carreras genéricas, {len(stats['por_area'])} áreas (titulados 2024)")
# muestra
for a, r in list(stats["por_area"].items()):
    print(f"   {a:30} n={r['n']:>6,}  mujeres={r['pct_muj']:>4.0f}%  edad={r['edad']}")
