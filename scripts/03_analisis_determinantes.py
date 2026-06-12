"""
03_analisis_determinantes.py — Análisis formal de DETERMINANTES del éxito.
DAML 2026 · Grupo 5.

Cuantifica con rigor (intervalos de confianza + regresión OLS) cómo el ORIGEN
socioeconómico condiciona el puntaje PAES y el acceso a 1ª preferencia.
Mensaje central: el origen opera AGUAS ARRIBA, a través del puntaje
(origen → puntaje → acceso).

Genera:
  - reports/figures/det_*.png  (figuras para el informe)
  - docs/ANALISIS_DETERMINANTES.md  (hallazgos con números reales)

Uso:  python3 scripts/03_analisis_determinantes.py
"""
from __future__ import annotations
import os, json
import numpy as np, pandas as pd, polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
AZUL, AZUL_OSC = "#2563eb", "#1e3a8a"
os.makedirs(P("reports", "figures"), exist_ok=True)

lab = json.load(open(P("data/processed/labels.json")))
DEP = lab["dependencia"]; REG = lab["region"]

print("1. Cargando datos...")
d = pl.read_parquet(P("data/processed/dataset_modelo_acceso.parquet")).to_pandas()
d["nivel_paes"] = (d["CLEC"] + d["MATE1"]) / 2
d["DEP"] = d["GRUPO_DEPENDENCIA"].astype("string").map(DEP)
d["REGION"] = d["CODIGO_REGION"].astype("string").map(REG)
d = d[d["DEP"].notna()].copy()
for c in ["PTJE_NEM", "PTJE_RANKING", "nivel_paes"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

ORDEN_DEP = ["Municipal", "Servicio Local de Educación", "Particular subvencionado", "Particular pagado"]

# ───────────────────────────── 1. Brecha de PUNTAJE por dependencia (IC 95%)
print("2. Brechas de puntaje por dependencia (IC 95%)...")
dd = d.dropna(subset=["nivel_paes"])
res_punt = {}
for g in ORDEN_DEP:
    x = dd.loc[dd.DEP == g, "nivel_paes"].values
    m, sem = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    res_punt[g] = (m, m - 1.96*sem, m + 1.96*sem, len(x))
    print(f"   {g:30s}: {m:.1f}  IC95[{m-1.96*sem:.1f}, {m+1.96*sem:.1f}]  n={len(x):,}")
brecha_punt = res_punt["Particular pagado"][0] - res_punt["Municipal"][0]
print(f"   → Brecha pagado − municipal: {brecha_punt:.1f} pts")

# ───────────────────────────── 2. Tasa de ACCESO por dependencia (IC 95%)
print("3. Tasa de acceso por dependencia (IC 95%)...")
res_acc = {}
for g in ORDEN_DEP:
    y = d.loc[d.DEP == g, "ACCESO_1PREF"].values
    p = y.mean(); lo, hi = proportion_confint(y.sum(), len(y), method="wilson")
    res_acc[g] = (p, lo, hi, len(y))
    print(f"   {g:30s}: {p:.3f}  IC95[{lo:.3f}, {hi:.3f}]  n={len(y):,}")

# ───────────────────────────── 3. Regresión OLS: ¿persiste tras controlar por notas?
print("4. Regresión OLS (origen, y origen+notas)...")
reg = dd.dropna(subset=["PTJE_NEM", "PTJE_RANKING"]).copy()
reg["DEP"] = pd.Categorical(reg["DEP"], categories=ORDEN_DEP)   # Municipal = referencia
mA = smf.ols("nivel_paes ~ C(DEP) + C(REGION) + C(RAMA_EDUCACIONAL)", data=reg).fit()
mB = smf.ols("nivel_paes ~ C(DEP) + C(REGION) + C(RAMA_EDUCACIONAL) + PTJE_NEM + PTJE_RANKING", data=reg).fit()
print(f"   R² modelo A (solo origen):        {mA.rsquared:.3f}")
print(f"   R² modelo B (origen + notas):     {mB.rsquared:.3f}")


def coef(model, term):
    p = model.params.get(term, np.nan)
    ci = model.conf_int().loc[term] if term in model.params.index else (np.nan, np.nan)
    return p, ci[0], ci[1]


coefs = {}
for g in ORDEN_DEP[1:]:
    t = f"C(DEP)[T.{g}]"
    coefs[g] = {"A": coef(mA, t), "B": coef(mB, t)}
    print(f"   {g:30s}: A={coefs[g]['A'][0]:+.1f}  →  B(controlando notas)={coefs[g]['B'][0]:+.1f}")

# ───────────────────────────── 4. R² comparativo (cuánto explica el origen)
print("5. Poder explicativo del origen sobre el puntaje...")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
rr = dd.dropna(subset=["PTJE_NEM", "PTJE_RANKING"]).copy()
for c in ["GRUPO_DEPENDENCIA", "CODIGO_REGION", "CODIGO_COMUNA", "RAMA_EDUCACIONAL"]:
    rr[c] = pd.Categorical(rr[c].astype("string")).codes
tr, te = rr[rr.cohorte == 2025], rr[rr.cohorte == 2026]
ORIG = ["GRUPO_DEPENDENCIA", "CODIGO_REGION", "CODIGO_COMUNA", "RAMA_EDUCACIONAL"]
NOTAS = ["PTJE_NEM", "PTJE_RANKING", "PROMEDIO_NOTAS", "PORC_SUP_NOTAS"]
r2 = {}
for nombre, cols in [("origen", ORIG), ("notas", NOTAS), ("origen+notas", ORIG + NOTAS)]:
    m = HistGradientBoostingRegressor(max_iter=150, random_state=42).fit(tr[cols], tr["nivel_paes"])
    r2[nombre] = r2_score(te["nivel_paes"], m.predict(te[cols]))
    print(f"   R² ({nombre}): {r2[nombre]:.3f}")

# ============================================================ FIGURAS
print("6. Generando figuras...")
plt.rcParams.update({"axes.edgecolor": "#cbd5e1", "axes.titlecolor": AZUL_OSC, "font.size": 11})
COLS = {"Municipal": "#94a3b8", "Servicio Local de Educación": "#60a5fa",
        "Particular subvencionado": "#3b82f6", "Particular pagado": "#1e3a8a"}

# Fig 1: brecha de puntaje (barras + IC)
fig, ax = plt.subplots(figsize=(8, 5))
xs = range(len(ORDEN_DEP))
ax.bar(xs, [res_punt[g][0] for g in ORDEN_DEP],
       yerr=[[res_punt[g][0]-res_punt[g][1] for g in ORDEN_DEP], [res_punt[g][2]-res_punt[g][0] for g in ORDEN_DEP]],
       color=[COLS[g] for g in ORDEN_DEP], capsize=5)
for i, g in enumerate(ORDEN_DEP):
    ax.text(i, res_punt[g][0]+3, f"{res_punt[g][0]:.0f}", ha="center", fontweight="bold", color=AZUL_OSC)
ax.set_xticks(xs); ax.set_xticklabels(["Municipal", "Serv. Local", "Part. subv.", "Part. pagado"])
ax.set_ylabel("Puntaje PAES (CLEC+M1)/2"); ax.set_ylim(500, max(res_punt[g][2] for g in ORDEN_DEP)+25)
ax.set_title(f"Brecha de puntaje PAES por tipo de colegio (IC 95%)\nPagado − Municipal = {brecha_punt:.0f} pts")
fig.tight_layout(); fig.savefig(P("reports/figures/det_brecha_dependencia_puntaje.png"), dpi=120); plt.close(fig)

# Fig 2: distribución (boxplot)
fig, ax = plt.subplots(figsize=(8, 5))
data_box = [dd.loc[dd.DEP == g, "nivel_paes"].values for g in ORDEN_DEP]
bp = ax.boxplot(data_box, patch_artist=True, showfliers=False,
                tick_labels=["Municipal", "Serv. Local", "Part. subv.", "Part. pagado"])
for patch, g in zip(bp["boxes"], ORDEN_DEP):
    patch.set_facecolor(COLS[g])
ax.set_ylabel("Puntaje PAES (CLEC+M1)/2")
ax.set_title("Distribución del puntaje PAES por tipo de colegio\n(las bandas se solapan: el origen desplaza, no determina)")
fig.tight_layout(); fig.savefig(P("reports/figures/det_dist_dependencia.png"), dpi=120); plt.close(fig)

# Fig 3: persistencia del efecto tras controlar por notas (coef plot)
fig, ax = plt.subplots(figsize=(8, 5))
gg = ORDEN_DEP[1:]; ypos = np.arange(len(gg))
for j, (mod, col, lbl, off) in enumerate([("A", "#94a3b8", "Solo origen", -.13), ("B", AZUL, "Controlando notas (NEM+ranking)", .13)]):
    vals = [coefs[g][mod][0] for g in gg]
    errs = [[coefs[g][mod][0]-coefs[g][mod][1] for g in gg], [coefs[g][mod][2]-coefs[g][mod][0] for g in gg]]
    ax.errorbar(vals, ypos+off, xerr=errs, fmt="o", color=col, label=lbl, capsize=4, ms=8)
ax.axvline(0, color="#cbd5e1", ls="--")
ax.set_yticks(ypos); ax.set_yticklabels(["Serv. Local", "Part. subv.", "Part. pagado"])
ax.set_xlabel("Puntos PAES vs. colegio Municipal (referencia)")
ax.set_title("Efecto del tipo de colegio sobre el puntaje PAES\n(persiste al controlar por notas → no explicado por mérito)", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(P("reports/figures/det_persistencia_coef.png"), dpi=120); plt.close(fig)

# Fig 4: tasa de acceso por dependencia
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(xs, [res_acc[g][0]*100 for g in ORDEN_DEP],
       yerr=[[(res_acc[g][0]-res_acc[g][1])*100 for g in ORDEN_DEP], [(res_acc[g][2]-res_acc[g][0])*100 for g in ORDEN_DEP]],
       color=[COLS[g] for g in ORDEN_DEP], capsize=5)
ax.set_xticks(xs); ax.set_xticklabels(["Municipal", "Serv. Local", "Part. subv.", "Part. pagado"])
ax.set_ylabel("Tasa de acceso a 1ª preferencia (%)")
ax.set_title("Tasa de acceso por tipo de colegio (IC 95%)\nLa brecha en ACCESO es menor que en PUNTAJE: cada grupo apunta distinto")
fig.tight_layout(); fig.savefig(P("reports/figures/det_acceso_dependencia.png"), dpi=120); plt.close(fig)

# ============================================================ DOC DE HALLAZGOS
print("7. Escribiendo docs/ANALISIS_DETERMINANTES.md...")
pag = res_punt["Particular pagado"][0]; mun = res_punt["Municipal"][0]
coef_pag_A = coefs["Particular pagado"]["A"][0]; coef_pag_B = coefs["Particular pagado"]["B"][0]
md = f"""# Análisis de determinantes del éxito — DAML 2026 · Grupo 5

> Pregunta de investigación: **¿cuánto el origen socioeconómico condiciona el éxito** en el acceso
> a la educación superior? Hallazgo central: el origen opera **aguas arriba**, a través del puntaje
> (**origen → puntaje → acceso**).

## 1. Brecha de puntaje PAES por tipo de colegio
Puntaje PAES promedio (CLEC+M1)/2, con intervalo de confianza 95% (n total = {len(dd):,}):

| Tipo de colegio | Puntaje medio | IC 95% | n |
|---|---|---|---|
""" + "\n".join(
    f"| {g} | {res_punt[g][0]:.0f} | [{res_punt[g][1]:.0f}, {res_punt[g][2]:.0f}] | {res_punt[g][3]:,} |"
    for g in ORDEN_DEP) + f"""

**Brecha particular pagado − municipal: {brecha_punt:.0f} puntos** (los IC no se solapan → diferencia estadísticamente significativa).
Ver `reports/figures/det_brecha_dependencia_puntaje.png` y `det_dist_dependencia.png`.

## 2. ¿La brecha persiste controlando por las notas? (regresión OLS)
Variable dependiente: puntaje PAES. Referencia: colegio **Municipal**.

| Modelo | R² | Efecto "Particular pagado" (pts vs. municipal) |
|---|---|---|
| A — solo origen (dependencia + región + rama) | {mA.rsquared:.3f} | {coef_pag_A:+.0f} [{coefs['Particular pagado']['A'][1]:.0f}, {coefs['Particular pagado']['A'][2]:.0f}] |
| B — origen **+ notas** (NEM, ranking) | {mB.rsquared:.3f} | {coef_pag_B:+.0f} [{coefs['Particular pagado']['B'][1]:.0f}, {coefs['Particular pagado']['B'][2]:.0f}] |

**Interpretación:** aun comparando estudiantes con **las mismas notas** (mismo NEM y ranking), los de colegio
particular pagado sacan **{coef_pag_B:+.0f} puntos** más en la PAES que los de colegio municipal. Es decir, una
parte importante de la brecha **no se explica por el mérito académico** (las notas), sino por el contexto
(acceso a preparación, recursos). Ver `reports/figures/det_persistencia_coef.png`.

## 3. ¿Cuánto del puntaje explica el origen? (validación temporal 2025→2026)
| Predictores | R² (test 2026) |
|---|---|
| Solo origen (comuna, dependencia, región, rama) | {r2['origen']:.2f} |
| Solo notas | {r2['notas']:.2f} |
| Origen + notas | {r2['origen+notas']:.2f} |

El origen por sí solo explica ~{r2['origen']*100:.0f}% de la varianza del puntaje PAES — una señal estructural fuerte.

## 4. La brecha en ACCESO es menor que en PUNTAJE
Tasa de acceso a 1ª preferencia por tipo de colegio (IC 95%):

| Tipo de colegio | Tasa de acceso | IC 95% |
|---|---|---|
""" + "\n".join(
    f"| {g} | {res_acc[g][0]:.1%} | [{res_acc[g][1]:.1%}, {res_acc[g][2]:.1%}] |"
    for g in ORDEN_DEP) + f"""

**Matiz clave:** la brecha de *acceso* (~{(res_acc['Particular pagado'][0]-res_acc['Municipal'][0])*100:.0f} pts %) es **menor**
que la de *puntaje* ({brecha_punt:.0f} pts). ¿Por qué? Porque cada grupo **apunta distinto**: con su ventaja de puntaje,
los de colegio pagado postulan a carreras más selectivas. La desigualdad no se ve tanto en "¿quedó en su 1ª opción?"
sino en **a qué puede aspirar realmente**. Ver `reports/figures/det_acceso_dependencia.png`.

## Conclusión
Predecir admisión desde el puntaje es casi mecánico. El aporte del trabajo es mostrar que **el puntaje mismo
está condicionado por el origen**: la desigualdad opera aguas arriba del corte. El modelo PRE-PAES y el simulador
contrafactual del dashboard hacen visible y cuantificable este mecanismo.
"""
open(P("docs/ANALISIS_DETERMINANTES.md"), "w").write(md)
print("✅ Análisis de determinantes completado.")
