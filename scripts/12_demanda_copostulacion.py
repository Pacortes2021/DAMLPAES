"""
12_demanda_copostulacion.py — Demanda histórica y co-postulación por carrera. DAML 2026 · Grupo 5.

Dos artefactos descriptivos desde las postulaciones completas:

1) DEMANDA (data/processed/demanda_hist.json): n° de postulantes que pusieron la carrera como
   1ª PREFERENCIA (regular) por año, 2018–2026. Los CONTEOS no dependen de la escala de puntajes,
   así que la serie completa es comparable aunque la escala PAES cambiara en 2024.
   {cod: {anio: n}}

2) CO-POSTULACIÓN (data/processed/copostulacion.json): para quienes pusieron la carrera de 1ª pref
   (cohorte más reciente), ¿a qué OTRAS carreras postularon en su cartola? Top-N con % sobre los
   postulantes de 1ª pref. Revela las carreras que compiten por los mismos estudiantes.
   {cod: {"anio": a, "n1": n, "top": [{"cod","n","pct"}, ...]}}

Uso:  python3 scripts/12_demanda_copostulacion.py
"""
from __future__ import annotations
import os, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
TOP_N = 6
MIN_N1 = 20     # bajo esto la co-postulación no es representativa → se omite

# ---------------------------------------------------------------- 1 · demanda 2018–2026
print("1. Demanda histórica (postulantes 1ª pref regular por año)...")
mh = pd.read_parquet(P("data/processed/master_admision_2018_2026.parquet"),
                     columns=["ID_aux", "anio", "ORDEN_PREF", "COD_CARRERA_PREF", "TIPO_PREF"])
p1 = mh[(mh["ORDEN_PREF"] == 1) & (mh["TIPO_PREF"] == "REGULAR")]
dem = p1.groupby(["COD_CARRERA_PREF", "anio"])["ID_aux"].nunique()
demanda: dict = {}
for (cod, anio), n in dem.items():
    demanda.setdefault(str(int(cod)), {})[str(int(anio))] = int(n)
json.dump(demanda, open(P("data/processed/demanda_hist.json"), "w"), ensure_ascii=False)
print(f"   ✅ demanda_hist.json — {len(demanda):,} carreras, años "
      f"{p1['anio'].min()}–{p1['anio'].max()}")

# ---------------------------------------------------------------- 2 · co-postulación (año reciente)
print("2. Co-postulación (cohorte más reciente)...")
ma = pd.read_parquet(P("data/processed/master_admision.parquet"),
                     columns=["ID_aux", "anio", "ORDEN_PREF", "COD_CARRERA_PREF", "TIPO_PREF"])
anio = int(ma["anio"].max())
ma = ma[(ma["anio"] == anio) & (ma["TIPO_PREF"] == "REGULAR")]
primera = ma[ma["ORDEN_PREF"] == 1][["ID_aux", "COD_CARRERA_PREF"]].rename(columns={"COD_CARRERA_PREF": "COD1"})
primera = primera.drop_duplicates("ID_aux")                      # 1ª pref única por estudiante
otras = ma[ma["ORDEN_PREF"] > 1][["ID_aux", "COD_CARRERA_PREF"]].drop_duplicates()
par = otras.merge(primera, on="ID_aux")
par = par[par["COD_CARRERA_PREF"] != par["COD1"]]
n1 = primera.groupby("COD1")["ID_aux"].nunique()
cnt = par.groupby(["COD1", "COD_CARRERA_PREF"])["ID_aux"].nunique()

copost: dict = {}
for cod1, sub in cnt.groupby(level=0):
    total = int(n1.get(cod1, 0))
    if total < MIN_N1:
        continue
    top = sub.sort_values(ascending=False).head(TOP_N)
    copost[str(int(cod1))] = {
        "anio": anio, "n1": total,
        "top": [{"cod": int(c2), "n": int(v), "pct": round(100 * v / total, 1)}
                for (_, c2), v in top.items()],
    }
json.dump(copost, open(P("data/processed/copostulacion.json"), "w"), ensure_ascii=False)
print(f"   ✅ copostulacion.json — {len(copost):,} carreras (n1≥{MIN_N1}), top {TOP_N}, año {anio}")

# control: UdeC Arquitectura (13080)
ej = copost.get("13080")
if ej:
    print(f"   UdeC Arq: n1={ej['n1']} · top1={ej['top'][0]}")
print("   demanda UdeC Arq:", demanda.get("13080"))
