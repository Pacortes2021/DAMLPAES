"""
01_build_models.py — Modelos de ACCESO a 1ª preferencia con VALIDACIÓN TEMPORAL.
DAML 2026 · Grupo 5.

Consume data/processed/dataset_modelo_acceso.parquet (cohortes 2025 + 2026).
Ver docs/METODOLOGIA.md (decisiones D1..D7).

Para cada modelo (PRE-PAES / POST-PAES):
  1) VALIDACIÓN TEMPORAL honesta: entrena 2025 → evalúa 2026 (out-of-time).
  2) MODELO FINAL para el dashboard: entrena con 2025+2026 (más señal), calibrado.
  Se reportan las métricas TEMPORALES (la cifra honesta) en la metadata.

Salidas:
  - models/modelo_acceso_{pre,post}.joblib  + *_meta.json
  - reports/figures/calibracion_{pre,post}.png

Uso:  python3 scripts/01_build_models.py
"""
from __future__ import annotations
import os, json, joblib
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
RS = 42
os.makedirs(P("models"), exist_ok=True)
os.makedirs(P("reports", "figures"), exist_ok=True)

# ----------------------------------------------------------------------------- feature sets
NUM_PRE  = ["PTJE_NEM", "PTJE_RANKING", "PROMEDIO_NOTAS", "PORC_SUP_NOTAS",
            "CORTE_ANTERIOR", "CUPOS_ANTERIOR", "ES_CARRERA_NUEVA"]
CAT      = ["GRUPO_DEPENDENCIA", "CODIGO_REGION", "CODIGO_COMUNA", "RAMA_EDUCACIONAL"]
NUM_POST = NUM_PRE + ["CLEC", "MATE1", "MATE2", "HCSOC", "CIEN",
                      "TIENE_MATE2", "TIENE_HCSOC", "TIENE_CIEN", "PTJE_PREF", "MARGEN_CORTE"]

print("Cargando dataset de modelado...")
data = pl.read_parquet(P("data/processed/dataset_modelo_acceso.parquet")).to_pandas()
data["ACCESO_1PREF"] = data["ACCESO_1PREF"].astype(int)
print(f"  Total {len(data):,} | 2025={int((data.cohorte==2025).sum()):,} | 2026={int((data.cohorte==2026).sum()):,}")


def fit_cat_mappings(df: pd.DataFrame, cat_cols, max_cat=254):
    """Mapeo categoría→código (ordinal). Reduce cardinalidad >max_cat a '__OTRAS__'. Reusable en dashboard."""
    maps = {}
    for c in cat_cols:
        s = df[c].astype("string").fillna("__NA__")
        vc = s.value_counts()
        if len(vc) > max_cat:
            keep = set(vc.index[: max_cat - 1])
            s = s.where(s.isin(keep), "__OTRAS__")
        cats = sorted(s.unique().tolist())
        if "__OTRAS__" not in cats:
            cats.append("__OTRAS__")          # asegurar bucket para desconocidos
        maps[c] = {v: i for i, v in enumerate(cats)}
    return maps


def apply_cat(df: pd.DataFrame, cat_cols, maps):
    out = df.copy()
    for c in cat_cols:
        s = out[c].astype("string").fillna("__NA__")
        otras = maps[c].get("__OTRAS__")
        out[c] = s.map(maps[c]).fillna(otras).astype("int32")
    return out


def make_clf(cat_idx):
    base = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.08,
        categorical_features=cat_idx, random_state=RS,   # D3: sin class_weight; D6: NaN nativo
    )
    return CalibratedClassifierCV(base, method="isotonic", cv=5)   # D4: calibración


def prep(df, num, cat, maps):
    X = apply_cat(df, cat, maps)[num + cat]
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def train_model(name, num, cat):
    print(f"\n{'='*62}\nMODELO {name.upper()}\n{'='*62}")
    cols = num + cat
    cat_idx = [cols.index(c) for c in cat]

    tr = data[data.cohorte == 2025]
    te = data[data.cohorte == 2026]

    # ---------- (1) Validación TEMPORAL: entrena 2025 → evalúa 2026 ----------
    maps_t = fit_cat_mappings(tr, cat)
    Xtr, ytr = prep(tr, num, cat, maps_t), tr["ACCESO_1PREF"].values
    Xte, yte = prep(te, num, cat, maps_t), te["ACCESO_1PREF"].values

    clf_t = make_clf(cat_idx).fit(Xtr, ytr)
    base_t = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                            categorical_features=cat_idx, random_state=RS).fit(Xtr, ytr)
    p   = clf_t.predict_proba(Xte)[:, 1]
    p0  = base_t.predict_proba(Xte)[:, 1]
    auc, pr = roc_auc_score(yte, p), average_precision_score(yte, p)
    brier, brier0 = brier_score_loss(yte, p), brier_score_loss(yte, p0)
    print(f"  [TEMPORAL 2025→2026]  AUC={auc:.4f}  PR-AUC={pr:.4f}  "
          f"Brier={brier:.4f} (sin calibrar {brier0:.4f})")
    print("  " + classification_report(yte, (p >= .5).astype(int),
          target_names=["Lista espera", "Seleccionado"]).replace("\n", "\n  "))

    # curva de calibración temporal
    fig, ax = plt.subplots(figsize=(6, 6))
    for lab, prob in [("Sin calibrar", p0), ("Calibrado", p)]:
        ft, mp = calibration_curve(yte, prob, n_bins=10, strategy="quantile")
        ax.plot(mp, ft, "o-", label=lab)
    ax.plot([0, 1], [0, 1], "k--", alpha=.5, label="Perfecto")
    ax.set(xlabel="Probabilidad predicha", ylabel="Frecuencia real",
           title=f"Calibración temporal (2025→2026) — {name}")
    ax.legend(); fig.tight_layout()
    fig.savefig(P("reports/figures", f"calibracion_{name}.png"), dpi=110); plt.close(fig)

    # importancia por permutación (sobre test 2026)
    samp = min(8000, len(Xte))
    pi = permutation_importance(clf_t, Xte.iloc[:samp], yte[:samp],
                                scoring="roc_auc", n_repeats=5, random_state=RS, n_jobs=-1)
    imp = sorted(zip(cols, pi.importances_mean), key=lambda t: t[1], reverse=True)
    print("  Top features:", ", ".join(f"{k}({v:.3f})" for k, v in imp[:6]))

    # ---------- (2) Modelo FINAL para el dashboard: 2025+2026 ----------
    maps_f = fit_cat_mappings(data, cat)
    Xall, yall = prep(data, num, cat, maps_f), data["ACCESO_1PREF"].values
    clf_final = make_clf(cat_idx).fit(Xall, yall)

    joblib.dump(clf_final, P("models", f"modelo_acceso_{name}.joblib"))
    meta = {
        "modelo": f"HistGradientBoosting calibrado — {name.upper()}",
        "target": "ACCESO_1PREF (ESTADO_PREF==24, 1ª pref, REGULAR)",
        "validacion": "temporal: entrena 2025 → evalúa 2026",
        "features_num": num, "features_cat": cat, "cat_mappings": maps_f,
        "metrics_temporal": {"auc_roc": float(auc), "pr_auc": float(pr),
                             "brier": float(brier), "brier_sin_calibrar": float(brier0)},
        "feature_importance": [{"feature": k, "auc_drop": float(v)} for k, v in imp],
        "n_train_2025": int(len(tr)), "n_test_2026": int(len(te)), "n_final": int(len(data)),
    }
    with open(P("models", f"modelo_acceso_{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


m_pre  = train_model("pre",  NUM_PRE,  CAT)
m_post = train_model("post", NUM_POST, CAT)

print(f"\n{'='*62}\nRESUMEN (validación temporal 2025→2026)\n{'='*62}")
print(f"PRE-PAES : AUC={m_pre['metrics_temporal']['auc_roc']:.3f}  Brier={m_pre['metrics_temporal']['brier']:.3f}")
print(f"POST-PAES: AUC={m_post['metrics_temporal']['auc_roc']:.3f}  Brier={m_post['metrics_temporal']['brier']:.3f}")
print("\n✅ Modelos finales (2025+2026) + métricas temporales generados.")
