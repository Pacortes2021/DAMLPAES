"""
inference.py — Lógica de predicción de ACCESO a 1ª preferencia (sin dependencia de Streamlit).
DAML 2026 · Grupo 5.

Encapsula: carga de artefactos, cálculo del puntaje ponderado (PTJE_PREF) y predicción
PRE-PAES / POST-PAES con los modelos calibrados. Reusable y testeable.
"""
from __future__ import annotations
import os, json, joblib
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_P = lambda *a: os.path.join(ROOT, *a)

# Pesos de la oferta → prueba PAES correspondiente
WEIGHT_MAP = {
    "%_NOTAS": "PTJE_NEM", "%_Ranking": "PTJE_RANKING", "%_LENG": "CLEC",
    "%_MATE1": "MATE1", "%_MATE2": "MATE2", "%_HYCS": "HCSOC", "%_CIEN": "CIEN",
}


@dataclass
class Artifacts:
    pre: object
    post: object
    meta_pre: dict
    meta_post: dict
    stats: dict                      # {cod_carrera: {"corte":x, "cupos":y}}
    catalogo: pd.DataFrame
    labels: dict
    conv: dict                       # curvas de equivalencia NEM↔notas, ranking↔%sup
    score_models: dict               # {prueba: {q10,q50,q90}} regresores de puntaje por prueba
    meta_score: dict
    territorio: dict                  # tasas históricas de acceso por región/comuna
    matricula: dict                   # matrícula efectiva por carrera (descriptivo, no es target)


def _load_json_opt(path, default):
    """Carga un JSON si existe; si no, devuelve default (artefacto opcional)."""
    return json.load(open(path)) if os.path.exists(path) else default


def load_artifacts() -> Artifacts:
    meta_score = json.load(open(_P("models/modelo_puntaje_meta.json")))
    return Artifacts(
        pre=joblib.load(_P("models/modelo_acceso_pre.joblib")),
        post=joblib.load(_P("models/modelo_acceso_post.joblib")),
        meta_pre=json.load(open(_P("models/modelo_acceso_pre_meta.json"))),
        meta_post=json.load(open(_P("models/modelo_acceso_post_meta.json"))),
        stats=json.load(open(_P("data/processed/carrera_stats.json"))),
        catalogo=pd.read_parquet(_P("data/processed/catalogo_carreras.parquet")),
        labels=json.load(open(_P("data/processed/labels.json"))),
        conv=json.load(open(_P("data/processed/conversiones.json"))),
        score_models={t: {q: joblib.load(_P("models", f"modelo_puntaje_{t}_{q}.joblib"))
                          for q in ("q10", "q50", "q90")}
                      for t in meta_score["pruebas"]},
        meta_score=meta_score,
        territorio=json.load(open(_P("data/processed/territorio_stats.json"))),
        matricula=_load_json_opt(_P("data/processed/matricula_stats.json"), {}),
    )


def _interp(x, curve):
    """source → target (curve['x'] creciente)."""
    return float(np.interp(x, curve["x"], curve["y"]))


def _interp_inv(y, curve):
    """target → source (ordena por y para invertir)."""
    xs, ys = np.array(curve["x"]), np.array(curve["y"])
    o = np.argsort(ys)
    return float(np.interp(y, ys[o], xs[o]))


def completar_equivalencias(perfil: "Perfil", conv: dict) -> None:
    """Rellena el miembro faltante de cada par equivalente (NEM↔notas, ranking↔%superior)."""
    if perfil.nem is None and perfil.promedio_notas is not None:
        perfil.nem = _interp(perfil.promedio_notas, conv["notas_a_nem"])
    elif perfil.promedio_notas is None and perfil.nem is not None:
        perfil.promedio_notas = _interp_inv(perfil.nem, conv["notas_a_nem"])
    if perfil.ranking is None and perfil.porc_sup is not None:
        perfil.ranking = _interp(perfil.porc_sup, conv["psup_a_rank"])
    elif perfil.porc_sup is None and perfil.ranking is not None:
        perfil.porc_sup = _interp_inv(perfil.ranking, conv["psup_a_rank"])


@dataclass
class Perfil:
    cod_carrera: int
    # Pre-PAES (siempre)
    nem: float | None = None
    ranking: float | None = None
    promedio_notas: float | None = None
    porc_sup: float | None = None
    region: str = ""
    comuna: str = ""
    dependencia: str = ""
    rama: str = ""
    # Post-PAES (opcionales)
    clec: float | None = None
    mate1: float | None = None
    mate2: float | None = None
    hcsoc: float | None = None
    cien: float | None = None
    ptje_pref_override: float | None = None   # si el alumno ya conoce su ponderado


def ponderado(perfil: Perfil, carrera_row: pd.Series) -> tuple[float | None, bool]:
    """Puntaje ponderado del alumno para la carrera.

    Devuelve (ponderado, prueba_especial). El electivo usa el mejor entre HCSOC/CIEN.
    Se divide por la SUMA REAL de ponderaciones académicas (promedio ponderado), de modo que:
      - carreras normales (pesos suman 100): da el ponderado oficial;
      - carreras con prueba especial (pesos académicos <100): aproxima el ponderado suponiendo
        que la prueba especial rinde como el promedio académico (se marca con la bandera).
    """
    if perfil.ptje_pref_override is not None:
        return perfil.ptje_pref_override, False
    scores = {"PTJE_NEM": perfil.nem, "PTJE_RANKING": perfil.ranking, "CLEC": perfil.clec,
              "MATE1": perfil.mate1, "MATE2": perfil.mate2, "HCSOC": perfil.hcsoc, "CIEN": perfil.cien}
    num = 0.0
    w_acad = 0.0
    elect_contrib, elect_w = [], []
    for wcol, scol in WEIGHT_MAP.items():
        w = carrera_row.get(wcol)
        if w is None or pd.isna(w) or float(w) == 0:
            continue
        w = float(w)
        s = scores.get(scol)
        s = 0.0 if (s is None or pd.isna(s)) else float(s)
        if scol in ("HCSOC", "CIEN"):
            elect_contrib.append(w * s); elect_w.append(w)     # electivo: mejor de los dos
        else:
            num += w * s; w_acad += w
    if elect_contrib:
        num += max(elect_contrib); w_acad += max(elect_w)
    if w_acad == 0:
        return None, False
    prueba_especial = w_acad < 99.0
    return num / w_acad, prueba_especial


def _num(v):
    """Coerción escalar a float (NaN si no es numérico). Equivale a pd.to_numeric(errors='coerce')."""
    try:
        f = float(v)
        return f if f == f else np.nan
    except (TypeError, ValueError):
        return np.nan


def _fila(perfil: Perfil, meta: dict) -> dict:
    """Diccionario de features (num + cat ya codificadas) para una postulación."""
    num, cat, maps = meta["features_num"], meta["features_cat"], meta["cat_mappings"]
    st = perfil._stats
    val = {
        "PTJE_NEM": perfil.nem, "PTJE_RANKING": perfil.ranking,
        "PROMEDIO_NOTAS": perfil.promedio_notas, "PORC_SUP_NOTAS": perfil.porc_sup,
        "CORTE_ANTERIOR": st["corte"] if st else np.nan,
        "CUPOS_ANTERIOR": st["cupos"] if st else np.nan,
        "ES_CARRERA_NUEVA": 0 if st else 1,
        "CLEC": perfil.clec, "MATE1": perfil.mate1, "MATE2": perfil.mate2,
        "HCSOC": perfil.hcsoc, "CIEN": perfil.cien,
        "TIENE_MATE2": int(perfil.mate2 is not None), "TIENE_HCSOC": int(perfil.hcsoc is not None),
        "TIENE_CIEN": int(perfil.cien is not None),
        "PTJE_PREF": perfil._ptje, "MARGEN_CORTE": perfil._margen,
        "GRUPO_DEPENDENCIA": perfil.dependencia, "CODIGO_REGION": perfil.region,
        "CODIGO_COMUNA": perfil.comuna, "RAMA_EDUCACIONAL": perfil.rama,
    }
    fila = {c: _num(val.get(c)) for c in num}
    for c in cat:
        m = maps[c]
        fila[c] = m.get(str(val.get(c)), m.get("__OTRAS__"))
    return fila


def _row(perfil: Perfil, meta: dict) -> pd.DataFrame:
    """Construye la fila de features (DataFrame de 1 fila) en el orden que espera el modelo."""
    num, cat = meta["features_num"], meta["features_cat"]
    X = pd.DataFrame([_fila(perfil, meta)])[num + cat]
    for c in cat:
        X[c] = X[c].astype("int32")
    return X


def predecir(art: Artifacts, perfil: Perfil) -> dict:
    """Devuelve probabilidades PRE/POST + datos de contexto (corte, margen, ponderado)."""
    completar_equivalencias(perfil, art.conv)        # rellena NEM↔notas, ranking↔%sup
    cod = int(perfil.cod_carrera)
    carrera_row = art.catalogo.set_index("CODIGO_CARRERA").loc[cod]
    st = art.stats.get(str(cod))
    perfil._stats = st
    perfil._ptje, prueba_especial = ponderado(perfil, carrera_row)
    corte = st["corte"] if st else None
    perfil._margen = (perfil._ptje - corte) if (perfil._ptje is not None and corte is not None) else np.nan

    out = {
        "ponderado": perfil._ptje, "corte": corte,
        "cupos": st["cupos"] if st else None,
        "margen": perfil._margen, "carrera_nueva": st is None,
        "prueba_especial": prueba_especial,
        "p_pre": None, "p_post": None,
    }
    # PRE-PAES: requiere al menos NEM y ranking
    if perfil.nem is not None and perfil.ranking is not None:
        out["p_pre"] = float(art.pre.predict_proba(_row(perfil, art.meta_pre))[0, 1])
    # POST-PAES: requiere CLEC y MATE1 (núcleo PAES)
    if perfil.clec is not None and perfil.mate1 is not None:
        out["p_post"] = float(art.post.predict_proba(_row(perfil, art.meta_post))[0, 1])
    return out


def predecir_puntaje(art: Artifacts, perfil: Perfil) -> dict:
    """Puntaje PAES probable POR PRUEBA (CLEC, MATE1), cada uno con banda P10/P50/P90.

    Devuelve {"CLEC": {"p10","p50","p90"}, "MATE1": {...}, "nivel": {...}}.
    No es una fórmula: cada valor sale de un modelo de regresión por cuantiles
    entrenado con origen + notas.
    """
    completar_equivalencias(perfil, art.conv)
    meta = art.meta_score
    num, cat, maps = meta["features_num"], meta["features_cat"], meta["cat_mappings"]
    val = {"PTJE_NEM": perfil.nem, "PTJE_RANKING": perfil.ranking,
           "PROMEDIO_NOTAS": perfil.promedio_notas, "PORC_SUP_NOTAS": perfil.porc_sup,
           "GRUPO_DEPENDENCIA": perfil.dependencia, "CODIGO_REGION": perfil.region,
           "CODIGO_COMUNA": perfil.comuna, "RAMA_EDUCACIONAL": perfil.rama}
    fila = {}
    for c in num:
        fila[c] = pd.to_numeric(pd.Series([val.get(c)]), errors="coerce").iloc[0]
    for c in cat:
        fila[c] = maps[c].get(str(val.get(c)), maps[c].get("__OTRAS__"))
    X = pd.DataFrame([fila])[num + cat]
    for c in cat:
        X[c] = X[c].astype("int32")

    out = {}
    for prueba, qmodels in art.score_models.items():
        q = {k: float(m.predict(X)[0]) for k, m in qmodels.items()}
        p10, p90 = min(q.values()), max(q.values())
        out[prueba] = {"p10": p10, "p50": min(max(q["q50"], p10), p90), "p90": p90}
    # nivel agregado = promedio de las medianas (resumen)
    out["nivel"] = {k: (out["CLEC"][k] + out["MATE1"][k]) / 2 for k in ("p10", "p50", "p90")}
    return out


def rankear(art: Artifacts, perfil_base: Perfil, codigos, modo: str = "post") -> list[dict]:
    """Ordena una lista de carreras candidatas por probabilidad de acceso para el perfil dado.

    Para cada código calcula el ponderado del alumno con las ponderaciones de ESA carrera
    (y su margen al corte) y predice P(acceso) en un solo batch. Devuelve dicts con
    {cod, p, ponderado, corte, margen, cupos, carrera_nueva} ordenados por p descendente.
    `modo`: "post" (usa puntajes PAES) o "pre" (solo notas + contexto).
    """
    completar_equivalencias(perfil_base, art.conv)
    meta = art.meta_post if modo == "post" else art.meta_pre
    model = art.post if modo == "post" else art.pre
    num, cat = meta["features_num"], meta["features_cat"]
    cat_idx = art.catalogo.set_index("CODIGO_CARRERA")
    filas, info = [], []
    for cod in codigos:
        cod = int(cod)
        if cod not in cat_idx.index:
            continue
        carrera_row = cat_idx.loc[cod]
        if isinstance(carrera_row, pd.DataFrame):       # código duplicado (defensivo): 1ª fila
            carrera_row = carrera_row.iloc[0]
        st = art.stats.get(str(cod))
        p = replace(perfil_base, cod_carrera=cod)
        p._stats = st
        pond, _ = ponderado(p, carrera_row)
        corte = st["corte"] if st else None
        p._ptje = pond
        p._margen = (pond - corte) if (pond is not None and corte is not None) else np.nan
        filas.append(_fila(p, meta))
        info.append({"cod": cod, "ponderado": pond, "corte": corte,
                     "margen": (None if p._margen != p._margen else float(p._margen)),
                     "cupos": st["cupos"] if st else None, "carrera_nueva": st is None})
    if not filas:
        return []
    X = pd.DataFrame(filas)[num + cat]
    for c in cat:
        X[c] = X[c].astype("int32")
    probs = model.predict_proba(X)[:, 1]
    for d, pr in zip(info, probs):
        d["p"] = float(pr)
    info.sort(key=lambda d: d["p"], reverse=True)
    return info
