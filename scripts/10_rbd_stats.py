"""
10_rbd_stats.py — Features de COLEGIO (RBD) para afinar la estimación de puntaje PAES. DAML 2026 · Grupo 5.

Para cada establecimiento (RBD) calcula la MEDIA histórica de puntaje PAES por prueba (pool de los ArchivoC
recientes, misma escala 100–1000) y la cruza con el Directorio Oficial del MINEDUC para tener el NOMBRE del
colegio. Genera también medias por COMUNA y global, usadas como respaldo cuando no se conoce el colegio.

Salida: data/processed/rbd_stats.json
  {"pruebas":[...], "anios":[...],
   "colegios": {RBD: {"nom","comuna","com_cod","reg_cod","n","m":{prueba:media}}},   # n>=MIN
   "comuna":   {com_cod: {prueba:media}},
   "global":   {prueba:media}}

Uso:  python3 scripts/10_rbd_stats.py
"""
from __future__ import annotations
import os, json, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data/raw/directorio_ee_2024.csv")
# ArchivoC recientes (misma escala): 2025 y 2026, ambos en el repo
ARCHIVOC = {
    2025: glob.glob(os.path.join(ROOT, "data/raw/**/ArchivoC_Adm2025.csv"), recursive=True),
    2026: glob.glob(os.path.join(ROOT, "data/raw/ArchivoC_Adm2026REG.csv")),
}
PRUEBAS = ["CLEC", "MATE1", "MATE2", "HCSOC", "CIEN"]
MIN_N = 10   # colegios con <10 rendiciones (pool) → no entran al selector; usan respaldo de comuna


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def cargar_archivoc(anio, path):
    cols = ["RBD", "CODIGO_COMUNA", "CODIGO_REGION"] + [f"{p}_REG_ACTUAL" for p in PRUEBAS]
    df = pd.read_csv(path, sep=";", encoding="latin-1", usecols=lambda c: c in cols, low_memory=False)
    df = df.rename(columns={f"{p}_REG_ACTUAL": p for p in PRUEBAS})
    for p in PRUEBAS:
        df[p] = _num(df[p]).where(lambda x: x.between(100, 1000))
    df["RBD"] = _num(df["RBD"])
    return df


print("Cargando ArchivoC 2025/2026 ...")
acs = []
for anio, paths in ARCHIVOC.items():
    if not paths:
        print(f"  ⚠️ sin ArchivoC {anio}"); continue
    acs.append(cargar_archivoc(anio, paths[0])); print(f"  {anio}: {len(acs[-1]):,} filas")
ac = pd.concat(acs, ignore_index=True)

# medias por prueba: por RBD, por comuna y global
glob_m = {p: round(float(ac[p].mean()), 1) for p in PRUEBAS}
com_m = {int(c): {p: round(float(v), 1) for p, v in row.items() if v == v}
         for c, row in ac.groupby("CODIGO_COMUNA")[PRUEBAS].mean().iterrows() if c == c}
g = ac.dropna(subset=["RBD"]).groupby("RBD")
rbd_m = g[PRUEBAS].mean()
rbd_n = g.size()

# directorio MINEDUC → nombre del colegio
dire = pd.read_csv(DIR, sep=";", encoding="latin-1",
                   usecols=["RBD", "NOM_RBD", "COD_COM_RBD", "NOM_COM_RBD", "COD_REG_RBD"], low_memory=False)
dire = dire.drop_duplicates("RBD").set_index("RBD")

colegios = {}
for rbd, fila in rbd_m.iterrows():
    n = int(rbd_n.loc[rbd])
    if n < MIN_N:
        continue
    medias = {p: round(float(v), 1) for p, v in fila.items() if v == v}
    if not medias:
        continue
    info = dire.loc[rbd] if rbd in dire.index else None
    nom = str(info["NOM_RBD"]).title() if info is not None else f"RBD {int(rbd)}"
    com = str(info["NOM_COM_RBD"]).title() if info is not None else ""
    com_cod = int(info["COD_COM_RBD"]) if info is not None and info["COD_COM_RBD"] == info["COD_COM_RBD"] else None
    reg_cod = int(info["COD_REG_RBD"]) if info is not None and info["COD_REG_RBD"] == info["COD_REG_RBD"] else None
    colegios[str(int(rbd))] = {"nom": nom, "comuna": com, "com_cod": com_cod, "reg_cod": reg_cod,
                               "n": n, "m": medias}

out = {"pruebas": PRUEBAS, "anios": list(ARCHIVOC), "colegios": colegios, "comuna": com_m, "global": glob_m}
json.dump(out, open(os.path.join(ROOT, "data/processed/rbd_stats.json"), "w"), ensure_ascii=False)
con_nombre = sum(1 for c in colegios.values() if not c["nom"].startswith("RBD "))
print(f"✅ rbd_stats.json — {len(colegios)} colegios (n≥{MIN_N}), {con_nombre} con nombre · "
      f"{len(com_m)} comunas · global {glob_m}")
# control
ej = next(iter(colegios.items()))
print("   ejemplo:", ej[0], ej[1]["nom"], ej[1]["comuna"], "| CLEC medio:", ej[1]["m"].get("CLEC"))
