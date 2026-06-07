"""
app.py — Dashboard de ACCESO a 1ª preferencia universitaria. DAML 2026 · Grupo 5.

Inputs en la página principal. Dos pestañas:
  1) ANTES de la PAES  → origen + notas → puntaje PAES probable POR PRUEBA (banda) y P(acceso)
  2) DESPUÉS de la PAES → puntajes reales → P(acceso); con el puntaje el origen ya no cambia el resultado.

Ejecutar:  python3 -m streamlit run src/app.py
"""
from __future__ import annotations
from dataclasses import replace
import streamlit as st
import plotly.graph_objects as go
from inference import load_artifacts, Perfil, predecir, predecir_puntaje

st.set_page_config(page_title="¿Quedo en mi 1ª preferencia?", page_icon="🎓", layout="wide")
AZUL, AZUL_OSC = "#2563eb", "#1e3a8a"
st.markdown(f"""
<style>
  .main .block-container {{max-width:1200px;padding-top:1rem;}}
  .stApp {{background:#f5f8ff;}}
  .hero {{background:linear-gradient(120deg,{AZUL_OSC} 0%,{AZUL} 70%,#3b82f6 100%);
          padding:24px 30px;border-radius:16px;color:white;margin-bottom:16px;box-shadow:0 8px 24px rgba(37,99,235,.25);}}
  .hero h1 {{color:white;font-size:1.8rem;margin:0 0 6px 0;}}
  .hero p {{color:#e0ecff;margin:0;font-size:.95rem;}}
  .sec h3 {{margin:14px 0 8px 0;color:{AZUL_OSC};font-size:1.18rem;}}
  .stats {{display:flex;gap:12px;flex-wrap:wrap;margin:4px 0;}}
  .stat {{flex:1;min-width:90px;background:white;border:1px solid #e2e8f5;border-top:4px solid {AZUL};
          border-radius:12px;padding:12px 14px;box-shadow:0 2px 8px rgba(30,58,138,.06);}}
  .stat .v {{font-size:1.45rem;font-weight:800;color:{AZUL_OSC};line-height:1;}}
  .stat .l {{font-size:.72rem;color:#64748b;margin-top:5px;text-transform:uppercase;letter-spacing:.03em;}}
  .nota {{background:#dbeafe;border-left:5px solid {AZUL};padding:13px 16px;border-radius:10px;font-size:.92rem;color:#1e3a8a;margin-top:10px;}}
  .warn {{background:#fff6e6;border-left:5px solid #f0a020;padding:13px 16px;border-radius:10px;font-size:.9rem;color:#7c5a00;margin-top:10px;}}
  [data-testid="stMetric"] {{background:white;border:1px solid #e2e8f5;border-radius:12px;padding:10px 14px;box-shadow:0 2px 8px rgba(30,58,138,.06);}}
  [data-testid="stMetricValue"] {{color:{AZUL_OSC};}}
  div[role="radiogroup"] label {{background:#eef3ff;border:1px solid #d6e2ff;border-radius:8px;padding:3px 10px;margin-right:5px;}}
  .stTabs [data-baseweb="tab"] {{font-size:1.02rem;font-weight:600;}}
  .stTabs [aria-selected="true"] {{color:{AZUL}!important;}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_artifacts():
    return load_artifacts()

art = get_artifacts()
L = art.labels
opt = lambda d: sorted(d.keys(), key=lambda k: d[k])
PESOS = [("Notas", "%_NOTAS"), ("Ranking", "%_Ranking"), ("C. Lectora", "%_LENG"),
         ("Matem. M1", "%_MATE1"), ("Matem. M2", "%_MATE2"), ("Historia", "%_HYCS"), ("Ciencias", "%_CIEN")]


# ----------------------------------------------------------------- figuras
def gauge(p, titulo):
    color = "#16a34a" if p >= .66 else "#f59e0b" if p >= .33 else "#dc2626"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p*100,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        title={"text": titulo, "font": {"size": 14, "color": AZUL_OSC}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": .78},
               "borderwidth": 1, "bordercolor": "#e2e8f5",
               "steps": [{"range": [0, 33], "color": "#fee2e2"}, {"range": [33, 66], "color": "#fef3c7"},
                         {"range": [66, 100], "color": "#dcfce7"}]}))
    fig.update_layout(height=250, margin=dict(l=22, r=22, t=48, b=6), paper_bgcolor="white")
    return fig


def fig_radar(row):
    vals = [float(row[c]) if row[c] == row[c] else 0.0 for _, c in PESOS]
    cats = [n for n, _ in PESOS]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
        fillcolor="rgba(37,99,235,.22)", line=dict(color=AZUL, width=2),
        text=[f"{v:.0f}%" for v in vals] + [""], hovertemplate="%{theta}: %{r:.0f}%<extra></extra>"))
    fig.update_layout(height=330, margin=dict(l=40, r=40, t=46, b=20),
        title=dict(text="⚖️ Ponderaciones que exige la carrera", font=dict(size=14, color=AZUL_OSC)),
        polar=dict(radialaxis=dict(range=[0, max(vals)*1.15 if max(vals) else 40], ticksuffix="%",
                                   tickfont=dict(size=9)), angularaxis=dict(tickfont=dict(size=11, color=AZUL_OSC))),
        paper_bgcolor="white")
    return fig


TEST_LABEL = {"CLEC": "Comp. Lectora", "MATE1": "Matem. M1", "MATE2": "Matem. M2 (electiva)",
              "HCSOC": "Historia (electiva)", "CIEN": "Ciencias (electiva)"}


def fig_bandas(r):
    tests = [t for t in ["CLEC", "MATE1", "MATE2", "HCSOC", "CIEN"] if t in r]
    fig = go.Figure()
    for i, t in enumerate(tests):
        y = len(tests) - i                       # primera prueba arriba
        b = r[t]
        oblig = t in ("CLEC", "MATE1")
        col = "#93c5fd" if oblig else "#c7dbff"
        fig.add_trace(go.Scatter(x=[b["p10"], b["p90"]], y=[y, y], mode="lines",
                      line=dict(color=col, width=18), showlegend=False))
        fig.add_trace(go.Scatter(x=[b["p50"]], y=[y], mode="markers",
                      marker=dict(color=AZUL_OSC if oblig else AZUL, size=15), showlegend=False))
        fig.add_annotation(x=b["p10"], y=y, text=f"{b['p10']:.0f}", showarrow=False, xshift=-20, font=dict(color="#64748b", size=10))
        fig.add_annotation(x=b["p90"], y=y, text=f"{b['p90']:.0f}", showarrow=False, xshift=20, font=dict(color="#64748b", size=10))
        fig.add_annotation(x=b["p50"], y=y, text=f"<b>{b['p50']:.0f}</b>", showarrow=False, yshift=15, font=dict(color=AZUL_OSC, size=12))
    fig.update_layout(height=max(250, 54*len(tests)+60), margin=dict(l=10, r=20, t=44, b=10),
        title=dict(text="🎯 Tu puntaje PAES probable por prueba (predicho por el modelo)", font=dict(size=14, color=AZUL_OSC)),
        xaxis=dict(range=[150, 1000], title="Puntaje", showgrid=True, gridcolor="#eef"),
        yaxis=dict(tickvals=list(range(1, len(tests)+1)),
                   ticktext=[TEST_LABEL[t] for t in reversed(tests)], range=[.4, len(tests)+.6]),
        plot_bgcolor="white", paper_bgcolor="white")
    return fig


def fig_cf(dprob, titulo, color):
    items = sorted(dprob.items(), key=lambda t: t[1])
    fig = go.Figure(go.Bar(x=[v*100 for _, v in items], y=[k for k, _ in items], orientation="h",
        marker_color=color, text=[f"{v:.0%}" for _, v in items], textposition="outside", textfont=dict(color=AZUL_OSC)))
    fig.update_layout(height=max(170, 42*len(items)+50), margin=dict(l=8, r=30, t=34, b=8),
        title=dict(text=titulo, font=dict(size=13, color=AZUL_OSC)),
        xaxis=dict(range=[0, 108], ticksuffix="%", showgrid=False), plot_bgcolor="white", paper_bgcolor="white")
    return fig


def cf_dependencia(base: Perfil, modo: str) -> dict:
    out = {}
    for code, lbl in L["dependencia"].items():
        r = predecir(art, replace(base, dependencia=code))
        v = r["p_pre"] if modo == "pre" else r["p_post"]
        if v is not None:
            out[lbl] = v
    return out


# ----------------------------------------------------------------- hero
st.markdown("""
<div class='hero'><h1>🎓 ¿Quedaré en mi primera preferencia?</h1>
<p>Determinantes del acceso a 1ª preferencia universitaria (modalidad regular).
Modelos validados temporalmente (entrena 2025 → testea 2026) · DAML 2026 · Grupo 5</p></div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- 1 · CARRERA (página principal)
cat = art.catalogo.copy()
cat["reg_nom"] = cat["REGION_CASA_MATRIZ"].astype("Int64").astype(str).map(L["region"]).fillna("")
# desambiguación: nombre — universidad · región (cód) — los duplicados son códigos distintos sin sede en los datos
cat["display"] = (cat["NOMBRE_CARRERA"].fillna("¿?") + " — " + cat["NOMBRE_UNIVERSIDAD"].fillna("¿?")
                  + " · " + cat["reg_nom"] + "  (cód " + cat["CODIGO_CARRERA"].astype(str) + ")")
cat = cat.sort_values("display")

st.markdown("<div class='sec'><h3>1 · Elige la carrera</h3></div>", unsafe_allow_html=True)
sel = st.selectbox("Carrera (escribe para buscar; si se repite, distínguelas por región/código/corte)",
                   cat["display"].tolist())
row = cat[cat["display"] == sel].iloc[0]
cod = int(row["CODIGO_CARRERA"])
st_info = art.stats.get(str(cod))

cL, cR = st.columns([1, 1])
with cL:
    corte_txt = f"{st_info['corte']:.0f}" if st_info else "s/d"
    cupos_txt = f"{st_info['cupos']}" if st_info else "s/d"
    vac_txt = f"{int(row['VACANTES_1SEM'])}" if row['VACANTES_1SEM'] == row['VACANTES_1SEM'] else "s/d"
    st.markdown(f"<div class='stats'><div class='stat'><div class='v'>{corte_txt}</div><div class='l'>Corte 2025</div></div>"
                f"<div class='stat'><div class='v'>{cupos_txt}</div><div class='l'>Sel. 2025</div></div>"
                f"<div class='stat'><div class='v'>{vac_txt}</div><div class='l'>Vacantes 2026</div></div></div>",
                unsafe_allow_html=True)
    st.caption(f"📍 {row['NOMBRE_UNIVERSIDAD']} · Región {row['reg_nom']} · código {cod}")
    if st_info is None:
        st.markdown("<div class='warn'>⚠️ Carrera sin corte histórico 2025 (nueva/sin datos): mayor incertidumbre.</div>",
                    unsafe_allow_html=True)
with cR:
    st.plotly_chart(fig_radar(row), use_container_width=True, key="radar_top")

# ----------------------------------------------------------------- 2 · PERFIL (página principal)
st.markdown("<div class='sec'><h3>2 · Tu perfil</h3></div>", unsafe_allow_html=True)
with st.container(border=True):
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**📘 Rendimiento de Enseñanza Media**")
        if st.radio("Notas", ["Puntaje NEM", "Promedio de notas"], horizontal=True,
                    label_visibility="collapsed", key="m_nem") == "Puntaje NEM":
            nem = st.number_input("Puntaje NEM (100–1000)", 100, 1000, 650, 5); promedio = None
        else:
            promedio = st.number_input("Promedio de notas (1.0–7.0)", 1.0, 7.0, 6.0, 0.1); nem = None
        if st.radio("Ranking", ["Puntaje Ranking", "% superior del curso"], horizontal=True,
                    label_visibility="collapsed", key="m_rk") == "Puntaje Ranking":
            ranking = st.number_input("Puntaje Ranking (100–1000)", 100, 1000, 680, 5); porc_sup = None
        else:
            porc_sup = st.number_input("% superior del curso (menor = mejor; ej. 5 = top 5%)", 1, 100, 30, 1); ranking = None
    with p2:
        st.markdown("**🏫 Contexto del establecimiento**")
        region = st.selectbox("Región", opt(L["region"]), format_func=lambda k: L["region"].get(k, k),
                              index=opt(L["region"]).index("13") if "13" in L["region"] else 0)
        # comuna en cascada: solo comunas de la región elegida
        comunas_reg = [c for c in opt(L["comuna"]) if L["comuna_region"].get(c) == region]
        if not comunas_reg:
            comunas_reg = opt(L["comuna"])
        comuna = st.selectbox("Comuna", comunas_reg, format_func=lambda k: L["comuna"].get(k, k))
        dependencia = st.selectbox("Dependencia del colegio", opt(L["dependencia"]),
                                   format_func=lambda k: L["dependencia"].get(k, k))
        rama = st.selectbox("Rama educacional", opt(L["rama"]), format_func=lambda k: L["rama"].get(k, k))

# tasa histórica de acceso por territorio (contexto; robustez si la comuna tiene pocos datos)
tr_reg = art.territorio["region"].get(str(region))
tr_com = art.territorio["comuna"].get(str(comuna))
if tr_reg:
    reg_txt = f"<b>{L['region'].get(region, region)}: {tr_reg['tasa']:.0%}</b> (n={tr_reg['n']:,})"
    if tr_com:
        pocos = tr_com["n"] < 100
        com_txt = (f"Comuna {L['comuna'].get(comuna, comuna)}: <b>{tr_com['tasa']:.0%}</b> (n={tr_com['n']:,})"
                   + (" ⚠️ pocos datos, considera la regional" if pocos else ""))
    else:
        com_txt = f"Comuna {L['comuna'].get(comuna, comuna)}: sin datos suficientes → usa la regional"
    st.markdown(f"<div class='nota'>📍 <b>Tasa histórica de acceso a 1ª preferencia</b> (todas las carreras) — "
                f"{reg_txt} · {com_txt}</div>", unsafe_allow_html=True)

perfil_base = Perfil(cod_carrera=cod, nem=nem, ranking=ranking, promedio_notas=promedio, porc_sup=porc_sup,
                     region=region, comuna=comuna, dependencia=dependencia, rama=rama)

# ----------------------------------------------------------------- 3 · RESULTADOS (pestañas)
st.markdown("<div class='sec'><h3>3 · Resultados</h3></div>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["🔮 Antes de la PAES", "✅ Después de la PAES"])

with tab1:
    st.caption("Estimación **antes de rendir**: el origen y las notas predicen qué puntaje PAES es probable "
               "que obtengas, y con eso tu probabilidad de acceso.")
    res = predecir(art, perfil_base)
    banda = predecir_puntaje(art, perfil_base)
    c1, c2 = st.columns([1, 1.25])
    with c1:
        if res["p_pre"] is not None:
            st.plotly_chart(gauge(res["p_pre"], "Probabilidad de acceso (PRE-PAES)"), use_container_width=True, key="g_pre")
        else:
            st.info("Completa NEM/notas y ranking/%superior.")
    with c2:
        st.plotly_chart(fig_bandas(banda), use_container_width=True, key="bandas")
        st.caption("📊 **Cómo leer la banda:** de cada 100 estudiantes con tu mismo perfil (notas + contexto), "
                   "**80 sacan un puntaje dentro de la banda**; el punto central es la mediana (la mitad saca más, "
                   "la mitad menos). El extremo izquierdo (P10) es un escenario bajo y el derecho (P90) uno alto. "
                   "La banda es ancha porque el puntaje no está determinado por tu perfil — el origen lo *desplaza*, no lo fija. "
                   "Las pruebas electivas se muestran como referencia *si las rindes*.")

    st.markdown("<div class='sec'><h3>🔬 El efecto del origen (mismas notas, distinto colegio)</h3></div>",
                unsafe_allow_html=True)
    cfp = cf_dependencia(perfil_base, "pre")
    if cfp:
        st.plotly_chart(fig_cf(cfp, "Probabilidad de acceso según tipo de colegio — mismo perfil académico", AZUL),
                        use_container_width=True, key="cf_pre")
        gap = (max(cfp.values()) - min(cfp.values())) * 100
        st.markdown(f"<div class='nota'>Con <b>las mismas notas</b>, cambiar el tipo de colegio mueve la probabilidad "
                    f"<b>~{gap:.0f} pts porcentuales</b>: el contexto predice el puntaje que probablemente obtendrás "
                    f"→ <b>determinante estructural del éxito</b>.</div>", unsafe_allow_html=True)

with tab2:
    st.caption("Estimación **con tus puntajes reales**: aquí decide el puntaje vs el corte; el origen deja de importar.")
    with st.container(border=True):
        st.markdown("**✏️ Tus puntajes PAES**  ·  obligatorias: Lectora y Matemática M1")
        q1, q2, q3, q4, q5 = st.columns(5)
        clec = q1.number_input("C. Lectora", 100, 1000, 650, 5)
        mate1 = q2.number_input("Matemática M1", 100, 1000, 650, 5)
        mate2 = q3.number_input("Matem. M2", 0, 1000, 0, 5, help="0 si no rendiste")
        hcsoc = q4.number_input("Historia", 0, 1000, 0, 5, help="0 si no rendiste")
        cien = q5.number_input("Ciencias", 0, 1000, 0, 5, help="0 si no rendiste")
    perfil_post = replace(perfil_base, clec=clec, mate1=mate1,
                          mate2=mate2 if mate2 >= 100 else None, hcsoc=hcsoc if hcsoc >= 100 else None,
                          cien=cien if cien >= 100 else None)
    res2 = predecir(art, perfil_post)

    c1, c2 = st.columns([1, 1.25])
    with c1:
        if res2["p_post"] is not None:
            st.plotly_chart(gauge(res2["p_post"], "Probabilidad de acceso (POST-PAES)"), use_container_width=True, key="g_post")
        else:
            st.info("Ingresa al menos C. Lectora y Matemática M1.")
    with c2:
        if res2["ponderado"] is not None and res2["corte"] is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Tu ponderado", f"{res2['ponderado']:.0f}")
            m2.metric("Corte 2025", f"{res2['corte']:.0f}")
            m3.metric("Margen", f"{res2['margen']:+.0f}", delta=f"{res2['margen']:+.0f}")
        st.plotly_chart(fig_radar(row), use_container_width=True, key="radar_post")
    if res2.get("prueba_especial"):
        st.markdown("<div class='warn'>🎭 Carrera con <b>prueba especial</b>: el ponderado es aproximado.</div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='sec'><h3>🔬 Con tu puntaje, ¿importa el origen?</h3></div>", unsafe_allow_html=True)
    cfp2 = cf_dependencia(perfil_post, "post")
    if cfp2:
        st.plotly_chart(fig_cf(cfp2, "Probabilidad según tipo de colegio — con TUS puntajes fijos", "#16a34a"),
                        use_container_width=True, key="cf_post")
        gap2 = (max(cfp2.values()) - min(cfp2.values())) * 100
        st.markdown(f"<div class='nota'>Con tus puntajes puestos, cambiar el colegio mueve la probabilidad solo "
                    f"<b>~{gap2:.1f} pts</b> (≈0). <b>El puntaje ya lo explica todo.</b> El origen importaba antes "
                    f"porque actuaba <i>a través</i> del puntaje: <b>origen → puntaje → acceso</b>.</div>",
                    unsafe_allow_html=True)

# ----------------------------------------------------------------- info modelos
with st.expander("ℹ️ Sobre los modelos y los datos"):
    mt, mp = art.meta_post["metrics_temporal"], art.meta_pre["metrics_temporal"]
    sc = art.meta_score["metrics"]
    st.markdown(f"""
**Validación temporal (entrena 2025 → testea 2026):**
- Acceso POST-PAES — AUC **{mt['auc_roc']:.3f}** · Acceso PRE-PAES — AUC **{mp['auc_roc']:.3f}**
- Puntaje probable por prueba (cuantiles): cobertura P10–P90 entre **{min(v['cobertura_p10_p90'] for v in sc.values()):.0%} y {max(v['cobertura_p10_p90'] for v in sc.values()):.0%}** (objetivo 80%)

**Cómo se predice el puntaje (percentiles, en simple):** no es una fórmula. Para cada prueba entrenamos un
modelo de *regresión por cuantiles* (gradient boosting) que aprende, a partir de **notas + contexto**, no un
único número sino **tres percentiles** de lo que sacan estudiantes parecidos:
- **P10** = solo el 10% saca *menos* (escenario bajo)
- **P50 / mediana** = la mitad saca menos y la mitad más (lo típico)
- **P90** = solo el 10% saca *más* (escenario alto)

Así, entre P10 y P90 cae el **80% de los estudiantes con tu perfil**. Validamos que sea cierto: en 2026, ~80%
de los puntajes reales cayeron dentro de su banda. Probabilidades de acceso **calibradas**.

**Limitaciones:** género no disponible a nivel individuo en los datos DEMRE; las "carreras repetidas" son
códigos distintos sin sede/jornada en la oferta (se distinguen por región, código y corte).
Target: seleccionado/a (`ESTADO_PREF=24`) en 1ª preferencia, modalidad regular.
""")
