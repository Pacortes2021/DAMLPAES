"""
02_build_score_model.py — Modelos de PUNTAJE PAES probable (regresión por cuantiles, POR PRUEBA).
DAML 2026 · Grupo 5.

Para CADA prueba (CLEC y MATE1) entrena 3 modelos (P10/P50/P90) que predicen el puntaje
a partir de ORIGEN + NOTAS (información disponible ANTES de rendir). Devuelve una banda de
incertidumbre honesta por prueba. Es el componente "determinantes" de la pestaña Pre-PAES
(origen → puntaje), y NO una fórmula: es un modelo entrenado.

Validación temporal: entrena 2025 → evalúa cobertura en 2026.
Salida: models/modelo_puntaje_{CLEC,MATE1}_q{10,50,90}.joblib + modelo_puntaje_meta.json

Uso:  python3 scripts/02_build_score_model.py
"""
from __future__ import annotations
import os, json, joblib
import numpy as np, pandas as pd, polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
RS = 42
NUM = ["PTJE_NEM", "PTJE_RANKING", "PROMEDIO_NOTAS", "PORC_SUP_NOTAS"]
CAT = ["GRUPO_DEPENDENCIA", "CODIGO_REGION", "CODIGO_COMUNA", "RAMA_EDUCACIONAL"]
PRUEBAS = ["CLEC", "MATE1", "MATE2", "HCSOC", "CIEN"]   # todas las pruebas PAES
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}

print("1. Cargando dataset...")
d = pl.read_parquet(P("data/processed/dataset_modelo_acceso.parquet")).to_pandas()


def fit_cat_mappings(df, cat_cols, max_cat=254):
    maps = {}
    for c in cat_cols:
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


maps = fit_cat_mappings(d[d.cohorte == 2025], CAT)
cat_idx = [(NUM + CAT).index(c) for c in CAT]
metrics = {}

for prueba in PRUEBAS:
    print(f"2. Entrenando cuantiles para {prueba} ...")
    sub = d.dropna(subset=[prueba, "PTJE_NEM", "PTJE_RANKING"])
    tr, te = sub[sub.cohorte == 2025], sub[sub.cohorte == 2026]
    Xtr = apply_cat(tr, maps)[NUM + CAT]; ytr = tr[prueba].values
    Xte = apply_cat(te, maps)[NUM + CAT]; yte = te[prueba].values
    for c in NUM:
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
    print(f"   {prueba}: cobertura P10–P90={cob:.1%} | MAE P50={mae:.0f} | banda={metrics[prueba]['ancho_banda']:.0f} pts")

print("3. Guardando metadata...")
meta = {
    "modelo": "Puntaje PAES por prueba (CLEC, MATE1) — regresión por cuantiles",
    "validacion": "temporal: entrena 2025 → evalúa 2026",
    "features_num": NUM, "features_cat": CAT, "cat_mappings": maps,
    "pruebas": PRUEBAS, "quantiles": QUANTILES, "metrics": metrics,
}
json.dump(meta, open(P("models/modelo_puntaje_meta.json"), "w"), indent=2, ensure_ascii=False)
print("✅ Modelos de puntaje por prueba (cuantiles) generados.")
