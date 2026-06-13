"""
06_titulacion_stats.py — Estadísticas de TITULACIÓN por carrera/área/institución. DAML 2026 · Grupo 5.

Lee el crudo del SIES (titulados 2024, latin-1) y agrega, por carrera genérica, por área y por
(carrera × institución): n° titulados, % mujeres/hombres, edad PROMEDIO y edad MEDIANA (esta última
calculada desde los rangos de edad del SIES, más robusta a la cola de titulados mayores).

Descriptivo institucional/agregado (no individual). Desde el crudo, no del master corrupto.
Salida: data/processed/titulacion_stats.json  {anio, por_carrera, por_area, por_carrera_inst}

Uso:  python3 scripts/06_titulacion_stats.py
"""
from __future__ import annotations
import os, sys, glob, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from areas import area_de, _norm

RAW = glob.glob(os.path.join(ROOT, "data/raw/TITULADO_*_web_*.csv"))
if not RAW:
    raise SystemExit("No se encontró el crudo de titulados (data/raw/TITULADO_*_web_*.csv).")

print(f"Leyendo crudo SIES: {os.path.relpath(RAW[0], ROOT)}")
df = pd.read_csv(RAW[0], sep=";", encoding="latin-1")

C_TOT, C_MUJ, C_HOM = "TOTAL TITULACIONES", "TITULACIONES MUJERES POR PROGRAMA", "TITULACIONES HOMBRES POR PROGRAMA"
C_GEN, C_INST, C_EDAD = "ÁREA CARRERA GENÉRICA", "NOMBRE INSTITUCIÓN", "PROMEDIO EDAD PROGRAMA "
# rangos de edad → para la mediana. (lo, hi) de cada tramo; 40+ se trata como 40–45.
BRK = [("RANGO DE EDAD 15 A 19 AÑOS", 15, 20), ("RANGO DE EDAD 20 A 24 AÑOS", 20, 25),
       ("RANGO DE EDAD 25 A 29 AÑOS", 25, 30), ("RANGO DE EDAD 30 A 34 AÑOS", 30, 35),
       ("RANGO DE EDAD 35 A 39 AÑOS", 35, 40), ("RANGO DE EDAD 40 Y MÁS AÑOS", 40, 45)]
NUMCOLS = [C_TOT, C_MUJ, C_HOM, C_EDAD] + [b[0] for b in BRK]
for c in NUMCOLS:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
df = df.dropna(subset=[C_TOT])
df = df[df[C_TOT] > 0]
df["_carrera"] = df[C_GEN].map(_norm)
df["_area"] = df[C_GEN].map(area_de)
df["_inst"] = df[C_INST].map(_norm)


def _mediana_edad(sub) -> float | None:
    """Mediana de edad por interpolación lineal dentro del tramo que cruza el 50%."""
    counts = [sub[col].sum() for col, _, _ in BRK]
    total = sum(counts)
    if total <= 0:
        return None
    half, cum = total / 2, 0.0
    for (_, lo, hi), c in zip(BRK, counts):
        if cum + c >= half and c > 0:
            return round(lo + (half - cum) / c * (hi - lo), 1)
        cum += c
    return float(BRK[-1][1])


def _stats(sub, min_n) -> dict | None:
    n = float(sub[C_TOT].sum())
    if n < min_n:
        return None
    muj, hom = float(sub[C_MUJ].sum()), float(sub[C_HOM].sum())
    prom = float((sub[C_EDAD] * sub[C_TOT]).sum() / n) if sub[C_EDAD].notna().any() else None
    return {"n": int(n), "pct_muj": round(100 * muj / n, 1), "pct_hom": round(100 * hom / n, 1),
            "edad_prom": round(prom, 1) if prom else None, "edad_mediana": _mediana_edad(sub)}


por_carrera = {k: s for k, sub in df.groupby("_carrera") if (s := _stats(sub, 30))}
por_area = {k: s for k, sub in df.groupby("_area") if k and k == k and (s := _stats(sub, 100))}
por_carrera_inst: dict = {}
for (car, inst), sub in df.groupby(["_carrera", "_inst"]):
    s = _stats(sub, 5)
    if s:
        por_carrera_inst.setdefault(car, {})[inst] = s

out = {"anio": 2024, "por_carrera": por_carrera, "por_area": por_area, "por_carrera_inst": por_carrera_inst}
json.dump(out, open(os.path.join(ROOT, "data/processed/titulacion_stats.json"), "w"), ensure_ascii=False)
print(f"✅ titulacion_stats.json — {len(por_carrera)} carreras, {len(por_area)} áreas, "
      f"{sum(len(v) for v in por_carrera_inst.values())} pares carrera×institución")
ici = por_carrera.get("INGENIERIA CIVIL INDUSTRIAL", {})
print(f"   Ing Civil Industrial (todas): {ici}")
udec = por_carrera_inst.get("INGENIERIA CIVIL INDUSTRIAL", {}).get("UNIVERSIDAD DE CONCEPCION")
print(f"   Ing Civil Industrial (UdeC):  {udec}")
