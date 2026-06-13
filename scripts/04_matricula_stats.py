"""
04_matricula_stats.py — Estadísticas de MATRÍCULA EFECTIVA por carrera. DAML 2026 · Grupo 5.

Métrica DESCRIPTIVA (no es target del modelo): cruza admisión ↔ matrícula por `ID_aux`
para responder, por carrera, "de los seleccionados en 1ª preferencia, ¿cuántos se
matricularon efectivamente en esa carrera?" — y cuántos se matricularon en total
(incluye lista de espera / otras preferencias).

Cohorte: proceso 2026 (el más reciente; selección 2026 → matrícula 2026).

Entradas:
  data/processed/dataset_modelo_acceso.parquet         (ID_aux, COD_CARRERA_PREF, ACCESO_1PREF, cohorte)
  data/processed/master_matricula_individual_2025_2026.parquet  (ID_aux, anio, COD_CARRERA)
Salida:
  data/processed/matricula_stats.json   {cod: {anio, n_sel, n_sel_matric, tasa, n_matric_total}}

Uso:  python3 scripts/04_matricula_stats.py
"""
from __future__ import annotations
import os, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
ANIO = 2026

print("Cargando admisión y matrícula...")
adm = pd.read_parquet(P("data/processed/dataset_modelo_acceso.parquet"),
                      columns=["ID_aux", "COD_CARRERA_PREF", "ACCESO_1PREF", "cohorte"])
mat = pd.read_parquet(P("data/processed/master_matricula_individual_2025_2026.parquet"),
                      columns=["ID_aux", "anio", "COD_CARRERA"])

adm = adm[adm.cohorte == ANIO].dropna(subset=["COD_CARRERA_PREF"])
mat = mat[mat.anio == ANIO].dropna(subset=["COD_CARRERA"])
adm["COD_CARRERA_PREF"] = adm["COD_CARRERA_PREF"].astype(int)
mat["COD_CARRERA"] = mat["COD_CARRERA"].astype(int)

# matriculados (ID_aux) por carrera
matric_ids = mat.groupby("COD_CARRERA")["ID_aux"].apply(set)
matric_total = mat.groupby("COD_CARRERA")["ID_aux"].nunique()

# postulantes y seleccionados en 1ª preferencia por carrera
postula = adm.groupby("COD_CARRERA_PREF")["ID_aux"].nunique()          # pusieron la carrera como 1ª pref
sel = adm[adm.ACCESO_1PREF == 1]
sel_ids = sel.groupby("COD_CARRERA_PREF")["ID_aux"].apply(set)

stats = {}
codigos = set(matric_total.index) | set(sel_ids.index) | set(postula.index)
for cod in codigos:
    s = sel_ids.get(cod, set())
    m = matric_ids.get(cod, set())
    n_sel = len(s)
    n_sel_matric = len(s & m)
    stats[str(cod)] = {
        "anio": ANIO,
        "n_postula": int(postula.get(cod, 0)),
        "n_sel": n_sel,
        "n_sel_matric": n_sel_matric,
        "tasa": round(n_sel_matric / n_sel, 4) if n_sel else None,
        "n_matric_total": int(matric_total.get(cod, 0)),
    }

with open(P("data/processed/matricula_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=0)

# resumen
con_tasa = [v for v in stats.values() if v["tasa"] is not None and v["n_sel"] >= 20]
prom = sum(v["tasa"] for v in con_tasa) / len(con_tasa)
print(f"✅ {len(stats):,} carreras. Tasa media de matrícula efectiva (n_sel≥20): {prom:.1%}")
print("   Guardado en data/processed/matricula_stats.json")
