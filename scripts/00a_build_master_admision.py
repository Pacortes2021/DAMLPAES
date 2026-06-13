"""
00a_build_master_admision.py — Reconstruye master_admision DESDE LOS CRUDOS. DAML 2026 · Grupo 5.

Reemplaza el antiguo master_admision_2018_2026.parquet (hecho fuera del repo, sin script y
con filtrados no documentados) por uno **reproducible y trazable**: concatena los ArchivoD
(postulaciones) del DEMRE tal cual, agregando la columna `anio`. SIN filtrar — todo el
filtrado (1ª pref, REGULAR, join con perfil académico) ocurre después, explícito, en 00.

Insumos (crudos DEMRE, descargados del portal de transparencia):
  data/raw/.../ArchivoD_Adm2024.csv · ArchivoD_Adm2025.csv · ArchivoD_Adm2026REG.csv
Salida:
  data/processed/master_admision.parquet   (cohortes que el modelo usa: 2024, 2025, 2026)

Uso:  python3 scripts/00a_build_master_admision.py
"""
from __future__ import annotations
import os, glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)

# ArchivoD por año (el del DEMRE trae exactamente: ID_aux, ORDEN_PREF, COD_CARRERA_PREF,
# ESTADO_PREF, TIPO_PREF, PTJE_PREF). Se busca por NOMBRE DE ARCHIVO (sin tildes) de forma
# recursiva, para evitar problemas de normalización Unicode de las carpetas con tilde en macOS.
def _find(anio: int) -> str | None:
    hits = glob.glob(P("data/raw", "**", f"ArchivoD_Adm{anio}*.csv"), recursive=True)
    return hits[0] if hits else None

FUENTES = {a: _find(a) for a in (2024, 2025, 2026)}

COLS = ["ID_aux", "ORDEN_PREF", "COD_CARRERA_PREF", "ESTADO_PREF", "TIPO_PREF", "PTJE_PREF"]

partes = []
print("Leyendo ArchivoD crudos (sin filtrar):")
for anio, ruta in FUENTES.items():
    if not ruta or not os.path.exists(ruta):
        print(f"  ⚠️  {anio}: NO encontrado (se omite). Descarga el ArchivoD {anio} del portal DEMRE.")
        continue
    # decimal="," → el ArchivoD trae PTJE_PREF con coma decimal (ej. "855,2")
    df = pd.read_csv(ruta, sep=";", encoding="latin-1", usecols=COLS, decimal=",")
    df["anio"] = anio
    sel = int(((df.ORDEN_PREF == 1) & (df.TIPO_PREF == "REGULAR") & (df.ESTADO_PREF == 24)).sum())
    print(f"  {anio}: {len(df):,} filas | seleccionados 1ª pref regular = {sel:,}  ({os.path.relpath(ruta, ROOT)})")
    partes.append(df)

if not partes:
    raise SystemExit("No se encontró ningún ArchivoD. Revisa data/raw/.")

master = pd.concat(partes, ignore_index=True)
out = P("data/processed/master_admision.parquet")
master.to_parquet(out)
print(f"\n✅ master_admision.parquet  ({len(master):,} filas, años {sorted(master.anio.unique())})")
print("   Trazable: cada fila viene 1:1 de un ArchivoD del DEMRE, sin filtrado oculto.")
