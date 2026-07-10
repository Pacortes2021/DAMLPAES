"""
11_retrain_score_rbd.py — Reentrena los modelos de PUNTAJE agregando la feature de COLEGIO (RBD_HIST).
DAML 2026 · Grupo 5.

RBD_HIST = media histórica PAES del establecimiento del alumno (de rbd_stats.json), por prueba, con
respaldo a la media de la comuna y luego global cuando el colegio no se conoce. El RBD se une al set de
modelado por ID_aux desde el ArchivoC del año. Misma config que 02_build_score_model.py (cuantiles,
categorical_features, lr=0.08), validación temporal 2025→2026. Sobrescribe los modelos de puntaje.

Salida: models/modelo_puntaje_{prueba}_q{10,50,90}.joblib + modelo_puntaje_meta.json (con RBD_HIST)
Uso:  python3 scripts/11_retrain_score_rbd.py
"""
from __future__ import annotations
import os, json, glob, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
RS = 42
NUM = ["PTJE_NEM", "PTJE_RANKING", "PROMEDIO_NOTAS", "PORC_SUP_NOTAS"]
NUM_RBD = NUM + ["RBD_HIST"]
CAT = ["GRUPO_DEPENDENCIA", "CODIGO_REGION", "CODIGO_COMUNA", "RAMA_EDUCACIONAL"]
PRUEBAS = ["CLEC", "MATE1", "MATE2", "HCSOC", "CIEN"]
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
ARCHIVOC = {2025: glob.glob(P("data/raw/**/ArchivoC_Adm2025.csv"), recursive=True),
            2026: glob.glob(P("data/raw/ArchivoC_Adm2026REG.csv"))}

print("1. Cargando dataset + rbd_stats + RBD por ID_aux ...")
d = pd.read_parquet(P("data/processed/dataset_modelo_acceso.parquet"))
rbd_stats = json.load(open(P("data/processed/rbd_stats.json")))

# RBD por estudiante (ID_aux → RBD) desde el ArchivoC de cada cohorte
id2rbd = {}
for anio, paths in ARCHIVOC.items():
    ac = pd.read_csv(paths[0], sep=";", encoding="latin-1", usecols=["ID_aux", "RBD"], low_memory=False)
    ac["RBD"] = pd.to_numeric(ac["RBD"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    id2rbd[anio] = dict(zip(ac["ID_aux"], ac["RBD"]))
d["RBD"] = [id2rbd.get(a, {}).get(i, np.nan) for a, i in zip(d["cohorte"], d["ID_aux"])]
print(f"   RBD asignado: {d['RBD'].notna().mean():.1%} de las filas")


def rbd_hist_col(df, prueba):
    """Media del colegio para la prueba; respaldo: media de comuna → global. (igual que en inferencia)"""
    esc = {int(k): v["m"][prueba] for k, v in rbd_stats["colegios"].items() if prueba in v["m"]}
    com = {int(k): v[prueba] for k, v in rbd_stats["comuna"].items() if prueba in v}
    g = rbd_stats["global"][prueba]
    s = df["RBD"].map(esc)
    s = s.fillna(df["CODIGO_COMUNA"].map(com))
    return s.fillna(g)


def fit_cat_mappings(df, max_cat=254):
    maps = {}
    for c in CAT:
        s = df[c].astype("string").fillna("__NA__")
        vc = s.value_counts()
        if len(vc) > max_cat:
            s = s.where(s.isin(set(vc.index[: max_cat - 1])), "__OTRAS__")
        cats = sorted(s.unique().tolist())
        if "__OTRAS__" not in cats:
            cats.append("__OTRAS__")
        maps[c] = {v: i for i, v in enumerate(cats)}
    return maps


def apply_cat(df, maps):
    out = df.copy()
    for c in CAT:
        s = out[c].astype("string").fillna("__NA__")
        out[c] = s.map(maps[c]).fillna(maps[c]["__OTRAS__"]).astype("int32")
    return out


maps = fit_cat_mappings(d[d.cohorte == 2025])
cat_idx = [(NUM_RBD + CAT).index(c) for c in CAT]
metrics = {}

for prueba in PRUEBAS:
    print(f"2. Entrenando {prueba} (con RBD_HIST) ...")
    sub = d.dropna(subset=[prueba, "PTJE_NEM", "PTJE_RANKING"]).copy()
    sub["RBD_HIST"] = rbd_hist_col(sub, prueba)
    tr, te = sub[sub.cohorte == 2025], sub[sub.cohorte == 2026]
    Xtr = apply_cat(tr, maps)[NUM_RBD + CAT]; ytr = tr[prueba].values
    Xte = apply_cat(te, maps)[NUM_RBD + CAT]; yte = te[prueba].values
    for c in NUM_RBD:
        Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce"); Xte[c] = pd.to_numeric(Xte[c], errors="coerce")

    preds = {}
    for name, q in QUANTILES.items():
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, max_depth=6,
                                          learning_rate=0.08, categorical_features=cat_idx, random_state=RS)
        m.fit(Xtr, ytr)
        joblib.dump(m, P("models", f"modelo_puntaje_{prueba}_{name}.joblib"))
        preds[name] = m.predict(Xte)

    p10 = np.minimum.reduce(list(preds.values()))
    p90 = np.maximum.reduce(list(preds.values()))
    cob = float(((yte >= p10) & (yte <= p90)).mean())
    mae = float(mean_absolute_error(yte, preds["q50"]))
    metrics[prueba] = {"cobertura_p10_p90": cob, "mae_p50": mae, "ancho_banda": float(np.mean(p90 - p10))}
    print(f"   {prueba}: cobertura={cob:.1%} | MAE={mae:.1f} | banda={metrics[prueba]['ancho_banda']:.0f}")

meta = {
    "modelo": "Puntaje PAES por prueba — regresión por cuantiles + feature de colegio (RBD_HIST)",
    "validacion": "temporal: entrena 2025 → evalúa 2026",
    "features_num": NUM_RBD, "features_cat": CAT, "cat_mappings": maps,
    "usa_rbd": True, "pruebas": PRUEBAS, "quantiles": QUANTILES, "metrics": metrics,
}
json.dump(meta, open(P("models/modelo_puntaje_meta.json"), "w"), indent=2, ensure_ascii=False)
print("✅ Modelos de puntaje reentrenados con feature de colegio.")
