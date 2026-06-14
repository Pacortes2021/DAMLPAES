"""
07_oferta_detalle.py — Detalle institucional de cada carrera DEMRE desde la matrícula SIES.
DAML 2026 · Grupo 5.

Cruza el catálogo DEMRE (institución + carrera + región) con master_matricula_institucional
(SIES, PREGRADO, último año disponible) y extrae, por código DEMRE:
  - nivel de la carrera (Técnico de nivel superior / Profesional con o sin licenciatura)
  - jornada (Diurna / Vespertina / A Distancia / …)
  - duración formal (semestres → años)
  - región / comuna / sede de la universidad
  - composición de matriculados por ESTABLECIMIENTO DE ORIGEN (TES: municipal, particular
    subvencionado, particular pagado, corp. administración delegada, servicio local) → %

NO incluye arancel: no está en los datos DEMRE/SIES de este repositorio.

El cruce es por NOMBRE (los códigos DEMRE y SIES son sistemas distintos): institución normalizada,
región del código DEMRE, y nombre de carrera (exacto → sin sufijo → subcadena). Descriptivo/agregado.

Uso:  python3 scripts/07_oferta_detalle.py
Salida: data/processed/oferta_detalle.json  {cod_demre: {...}}
"""
from __future__ import annotations
import os, sys, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from areas import _norm   # NFKD sin tildes + MAYÚSCULAS + strip

SIES = os.path.join(ROOT, "data/processed/master_matricula_institucional.parquet")
CAT = os.path.join(ROOT, "data/processed/catalogo_carreras.parquet")
LAB = os.path.join(ROOT, "data/processed/labels.json")
OUT = os.path.join(ROOT, "data/processed/oferta_detalle.json")

# código región DEMRE → nombre región SIES normalizado (NFKD sin tildes)
REG_SIES = {
    "1": "TARAPACA", "2": "ANTOFAGASTA", "3": "ATACAMA", "4": "COQUIMBO", "5": "VALPARAISO",
    "6": "LIB. GRAL. B. O'HIGGINS", "7": "MAULE", "8": "BIOBIO", "9": "LA ARAUCANIA",
    "10": "LOS LAGOS", "11": "AYSEN", "12": "MAGALLANES", "13": "METROPOLITANA",
    "14": "LOS RIOS", "15": "ARICA Y PARINACOTA", "16": "NUBLE",
}
TES_COLS = [("municipal", "TES MUNICIPAL"), ("part_subv", "TES PARTICULAR SUBVENCIONADO"),
            ("part_pagado", "TES PARTICULAR PAGADO"), ("corp_ad", "TES CORP. DE ADMINISTRACIÓN DELEGADA"),
            ("sle", "TES SERVICIO LOCAL EDUCACION")]
NIVEL_LBL = {"Técnico de Nivel Superior": "Técnico de nivel superior",
             "Profesional Con Licenciatura": "Profesional (con licenciatura)",
             "Profesional Sin Licenciatura": "Profesional (sin licenciatura)"}


def _f(v) -> float:
    try:
        f = float(v)
        return f if f == f else 0.0
    except (TypeError, ValueError):
        return 0.0


def carrera_score(demre_n: str, sies_n: str) -> int:
    """Calce de nombre de carrera: 3=exacto, 2=sin sufijo (..)/- , 1=subcadena, 0=no."""
    if demre_n == sies_n:
        return 3
    db = demre_n.split(" (")[0].split(" - ")[0].strip()
    sb = sies_n.split(" (")[0].split(" - ")[0].strip()
    if db and db == sb:
        return 2
    if (sb and sb in demre_n) or (db and db in sies_n):
        return 1
    return 0


def main():
    cat = pd.read_parquet(CAT)
    L = json.load(open(LAB))
    df = pd.read_parquet(SIES)
    df = df[(df["NIVEL GLOBAL"] == "Pregrado") & (df["anio"] == df["anio"].max())].copy()
    anio = int(df["anio"].max())
    df["_inst"] = df["NOMBRE INSTITUCIÓN"].map(_norm)
    df["_carr"] = df["NOMBRE CARRERA"].map(_norm)
    df["_reg"] = df["REGIÓN"].map(_norm)
    df["_primer"] = pd.to_numeric(df["TOTAL MATRÍCULA PRIMER AÑO"], errors="coerce").fillna(0.0)
    df["_dur"] = pd.to_numeric(df["DURACIÓN ESTUDIO CARRERA"], errors="coerce")
    por_inst = {k: sub for k, sub in df.groupby("_inst")}

    out, hit = {}, 0
    for _, c in cat.iterrows():
        cod = int(c["CODIGO_CARRERA"])
        inst_n = _norm(c["NOMBRE_UNIVERSIDAD"])
        carr_n = _norm(c["NOMBRE_CARRERA"])
        reg_code = str(pd.array([c["REGION_CASA_MATRIZ"]], dtype="Int64")[0])
        # 1) institución: exacta, si no, una contiene a la otra
        sub = por_inst.get(inst_n)
        if sub is None:
            cands_inst = [k for k in por_inst if inst_n in k or k in inst_n]
            sub = pd.concat([por_inst[k] for k in cands_inst]) if cands_inst else None
        if sub is None or sub.empty:
            continue
        # 2) región del código DEMRE (si no hay filas en esa región, se ignora el filtro)
        reg_sies = REG_SIES.get(reg_code)
        sub_r = sub[sub["_reg"] == reg_sies] if reg_sies is not None else sub
        if sub_r.empty:
            sub_r = sub
        # 3) calce de nombre de carrera; nos quedamos con el mejor puntaje
        sub_r = sub_r.assign(_sc=[carrera_score(carr_n, x) for x in sub_r["_carr"]])
        cand = sub_r[sub_r["_sc"] > 0]
        if cand.empty:
            continue
        cand = cand[cand["_sc"] == cand["_sc"].max()]
        # fila principal = plan vigente (más matrícula de 1er año; desempata por TOTAL TES)
        cand = cand.assign(_tt=pd.to_numeric(cand["TOTAL TES"], errors="coerce").fillna(0.0))
        prin = cand.sort_values(["_primer", "_tt"], ascending=False).iloc[0]
        # TES agregado sobre todas las filas calzadas (composición por colegio de origen)
        tes = {k: sum(_f(v) for v in cand[col]) for k, col in TES_COLS}
        tes_total = sum(tes.values())
        nivel_raw = str(prin["CARRERA CLASIFICACIÓN NIVEL 1"])
        jornadas = sorted({str(j) for j in cand["JORNADA"].dropna().unique() if str(j) != "nan"})
        dur_sem = _f(prin["_dur"]) or None
        rec = {
            "anio": anio,
            "nivel": NIVEL_LBL.get(nivel_raw, nivel_raw),
            "jornada": str(prin["JORNADA"]) if prin["JORNADA"] == prin["JORNADA"] else None,
            "jornadas": jornadas,
            "dur_sem": dur_sem,
            "dur_anios": round(dur_sem / 2, 1) if dur_sem else None,
            "region": str(prin["REGIÓN"]) if prin["REGIÓN"] == prin["REGIÓN"] else None,
            "comuna": str(prin["COMUNA"]).title() if prin["COMUNA"] == prin["COMUNA"] else None,
            "sede": str(prin["NOMBRE SEDE"]).title() if prin["NOMBRE SEDE"] == prin["NOMBRE SEDE"] else None,
            "tes": {k: round(100 * v / tes_total, 1) for k, v in tes.items()} if tes_total > 0 else None,
            "tes_n": int(tes_total),
            "match": int(prin["_sc"]),
        }
        out[str(cod)] = rec
        hit += 1

    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    print(f"✅ oferta_detalle.json — {hit}/{len(cat)} carreras DEMRE calzadas ({hit/len(cat):.0%}), SIES {anio}")
    # muestra de control
    for cod in list(out)[:1]:
        print("   ejemplo:", cod, json.dumps(out[cod], ensure_ascii=False)[:300])
    # UdeC Arquitectura si está
    udec_arq = cat[(cat["NOMBRE_UNIVERSIDAD"].str.contains("CONCEPCION", case=False, na=False)) &
                   (cat["NOMBRE_CARRERA"].str.upper() == "ARQUITECTURA")]
    if not udec_arq.empty:
        k = str(int(udec_arq.iloc[0]["CODIGO_CARRERA"]))
        print("   UdeC Arquitectura:", json.dumps(out.get(k, "s/c"), ensure_ascii=False))


if __name__ == "__main__":
    main()
