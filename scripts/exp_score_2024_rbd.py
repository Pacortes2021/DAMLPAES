"""
exp_score_2024_rbd.py — EXPERIMENTO (no toca producción). DAML 2026 · Grupo 5.

Mide si el modelo de PUNTAJE (NEM+región → PAES por cuantiles) mejora al:
  (A) sumar la cohorte 2024 al entrenamiento, y/o
  (B) agregar una feature de HISTORIAL DEL COLEGIO (media PAES del RBD el año previo, sin fuga).

Test fijo = cohorte 2026. Métricas por prueba: MAE de la mediana, cobertura P10–P90, ancho de banda.
Mismo modelo/config que producción (HistGradientBoostingRegressor quantile, max_iter=300, depth=6).
"""
from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    2024: "/Users/pabloignaciocortesvielma/Desktop/2024/ArchivoC_Adm2024.csv",
    2025: os.path.join(ROOT, "data/raw/PROCESO-DE-ADMISIÓN-2025-RENDICIÓN-19-01-2025T23-39-20/ArchivoC_Adm2025.csv"),
    2026: os.path.join(ROOT, "data/raw/ArchivoC_Adm2026REG.csv"),
}
NUM = ["PTJE_NEM", "PTJE_RANKING", "PROMEDIO_NOTAS", "PORC_SUP_NOTAS"]
CAT = ["GRUPO_DEPENDENCIA", "CODIGO_REGION", "CODIGO_COMUNA", "RAMA_EDUCACIONAL"]
PRUEBAS = ["CLEC", "MATE1"]
QS = {"q10": 0.10, "q50": 0.50, "q90": 0.90}


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def cargar(anio):
    cols = ["RBD"] + NUM + CAT + [f"{p}_REG_ACTUAL" for p in PRUEBAS]
    df = pd.read_csv(FILES[anio], sep=";", encoding="latin-1",
                     usecols=lambda c: c in cols, low_memory=False)
    for c in NUM + [f"{p}_REG_ACTUAL" for p in PRUEBAS]:
        df[c] = _num(df[c])
    for p in PRUEBAS:
        df = df.rename(columns={f"{p}_REG_ACTUAL": p})
    df["RBD"] = df["RBD"].astype(str)
    df["anio"] = anio
    return df


print("Cargando ArchivoC 2024/2025/2026 ...")
D = {y: cargar(y) for y in (2024, 2025, 2026)}
for y in D:
    print(f"  {y}: {len(D[y]):,} filas")

# mapeos categóricos desde 2025 (consistente con producción)
maps = {}
for c in CAT:
    vc = D[2025][c].astype(str).value_counts()
    keys = list(vc.index[:254])
    maps[c] = {k: i for i, k in enumerate(keys)}
    maps[c]["__OTRAS__"] = len(keys)


def encode(df, extra_num=None):
    X = pd.DataFrame(index=df.index)
    for c in NUM:
        X[c] = df[c]
    if extra_num:
        for c in extra_num:
            X[c] = df[c]
    for c in CAT:
        m = maps[c]
        X[c] = df[c].astype(str).map(m).fillna(m["__OTRAS__"]).astype("int32")
    return X


def rbd_hist(train_years, prueba):
    """Media de la prueba por RBD en el año PREVIO a cada cohorte (sin fuga)."""
    tabla = {y: D[y].groupby("RBD")[prueba].mean() for y in (2024, 2025)}  # históricos disponibles
    glob = {y: D[y][prueba].mean() for y in (2024, 2025)}

    def feat(df, anio):
        prev = anio - 1
        if prev in tabla:
            return df["RBD"].map(tabla[prev]).fillna(glob[prev])
        return pd.Series(np.nan, index=df.index)   # 2024 no tiene 2023 → NaN (HGBR lo maneja)
    return feat


def entrena_evalua(prueba, train_anios, usar_rbd):
    # datos de entrenamiento (filtrar target válido)
    trs = []
    for y in train_anios:
        d = D[y][D[y][prueba].between(100, 1000)].copy()
        trs.append(d)
    tr = pd.concat(trs)
    te = D[2026][D[2026][prueba].between(100, 1000)].copy()

    extra = None
    if usar_rbd:
        feat = rbd_hist(train_anios, prueba)
        for d, y in [(tr, None)]:
            pass
        tr = tr.assign(RBD_HIST=pd.concat([feat(D[y][D[y][prueba].between(100,1000)], y) for y in train_anios]).values)
        te = te.assign(RBD_HIST=feat(te, 2026).values)
        extra = ["RBD_HIST"]

    Xtr, Xte = encode(tr, extra), encode(te, extra)
    ytr, yte = tr[prueba].values, te[prueba].values
    preds = {}
    for qn, q in QS.items():
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                          max_depth=6, learning_rate=0.05, random_state=0)
        m.fit(Xtr, ytr)
        preds[qn] = m.predict(Xte)
    p10 = np.minimum.reduce([preds["q10"], preds["q50"], preds["q90"]])
    p90 = np.maximum.reduce([preds["q10"], preds["q50"], preds["q90"]])
    p50 = np.clip(preds["q50"], p10, p90)
    return {"mae": float(np.mean(np.abs(p50 - yte))),
            "cob": float(np.mean((yte >= p10) & (yte <= p90))),
            "banda": float(np.mean(p90 - p10)), "n_tr": len(tr), "n_te": len(te)}


CONFIGS = [
    ("Base (2025)",        [2025], False),
    ("+2024 (2024+2025)",  [2024, 2025], False),
    ("+RBD (2025+colegio)",[2025], True),
    ("+2024 +RBD",         [2024, 2025], True),
]
print("\n" + "=" * 72)
for prueba in PRUEBAS:
    print(f"\n### {prueba}  (test = 2026)")
    print(f"{'config':<22} {'n_train':>9} {'MAE':>7} {'cobertura':>10} {'ancho_banda':>12}")
    base = None
    for nombre, anios, rbd in CONFIGS:
        r = entrena_evalua(prueba, anios, rbd)
        if base is None:
            base = r
        dmae = r["mae"] - base["mae"]
        dban = r["banda"] - base["banda"]
        print(f"{nombre:<22} {r['n_tr']:>9,} {r['mae']:>7.1f} {r['cob']:>9.1%} {r['banda']:>11.1f}"
              + ("" if nombre.startswith("Base") else f"   ΔMAE={dmae:+.1f} Δbanda={dban:+.1f}"))
print("\nListo. (experimento, no modifica producción)")
