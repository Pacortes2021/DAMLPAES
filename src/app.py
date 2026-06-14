"""
app.py — Dashboard de ACCESO a 1ª preferencia universitaria. DAML 2026 · Grupo 5.

Inputs en la página principal. Dos pestañas:
  1) ANTES de la PAES  → origen + notas → puntaje PAES probable POR PRUEBA (banda) y P(acceso)
  2) DESPUÉS de la PAES → puntajes reales → P(acceso); con el puntaje el origen ya no cambia el resultado.

Ejecutar:  python3 -m streamlit run src/app.py
"""
from __future__ import annotations
from dataclasses import replace
import json, os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from inference import load_artifacts, Perfil, predecir, predecir_puntaje, rankear
from areas import area_de, _norm

st.set_page_config(page_title="¿Quedo en mi 1ª preferencia?", page_icon="🎓", layout="wide")
AZUL, AZUL_OSC = "#2563eb", "#1e3a8a"
st.markdown(f"""
<style>
  .main .block-container {{max-width:1200px;padding-top:1rem;}}
  .stApp {{background:#f5f8ff;}}
  section[data-testid="stSidebar"] {{width:370px!important;}}
  section[data-testid="stSidebar"] > div {{width:370px!important;}}
  .sbfull {{font-size:.82rem;color:{AZUL_OSC};background:#eef3ff;border:1px solid #d6e2ff;
            border-radius:8px;padding:6px 9px;margin:-4px 0 4px 0;line-height:1.25;word-break:break-word;}}
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
  .pchips {{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 2px 0;}}
  .pchip {{background:white;border:1px solid #d6e2ff;border-radius:999px;padding:5px 13px;font-size:.86rem;
           color:{AZUL_OSC};box-shadow:0 1px 4px rgba(30,58,138,.06);white-space:nowrap;}}
  .pchip b {{font-size:1rem;color:{AZUL};}}
  .pchip.ob {{background:{AZUL};border-color:{AZUL};color:white;}}
  .pchip.ob b {{color:white;}}
  .nota {{background:#dbeafe;border-left:5px solid {AZUL};padding:13px 16px;border-radius:10px;font-size:.92rem;color:#1e3a8a;margin-top:10px;}}
  .warn {{background:#fff6e6;border-left:5px solid #f0a020;padding:13px 16px;border-radius:10px;font-size:.9rem;color:#7c5a00;margin-top:10px;}}
  [data-testid="stMetric"] {{background:white;border:1px solid #e2e8f5;border-radius:12px;padding:10px 14px;box-shadow:0 2px 8px rgba(30,58,138,.06);}}
  [data-testid="stMetricValue"] {{color:{AZUL_OSC};}}
  div[role="radiogroup"] label {{background:#eef3ff;border:1px solid #d6e2ff;border-radius:8px;padding:3px 10px;margin-right:5px;}}
  .stTabs [data-baseweb="tab"] {{font-size:1.02rem;font-weight:600;}}
  .stTabs [aria-selected="true"] {{color:{AZUL}!important;}}
</style>
""", unsafe_allow_html=True)


_GEO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "geo")


@st.cache_data
def get_geo():
    geo = json.load(open(os.path.join(_GEO, "regiones.geojson")))
    cent = json.load(open(os.path.join(_GEO, "comuna_centroides.json")))
    return geo, cent


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
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=16),
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


def fig_corte_trend(hist: dict):
    """Mini-línea con el corte de la carrera en 2024–2026 (tendencia). None si <2 años."""
    yrs = [y for y in ("2024", "2025", "2026") if hist.get(y) is not None]
    vals = [hist[y] for y in yrs]
    if len(vals) < 2:
        return None
    delta = vals[-1] - vals[0]
    col = "#16a34a" if delta < -3 else "#dc2626" if delta > 3 else AZUL   # bajó=más fácil, subió=más difícil
    fig = go.Figure(go.Scatter(x=yrs, y=vals, mode="lines+markers+text",
        text=[f"{v:.0f}" for v in vals], textposition="top center", textfont=dict(size=11, color=AZUL_OSC),
        line=dict(color=col, width=3), marker=dict(size=9, color=col)))
    fig.update_layout(height=200, margin=dict(l=10, r=14, t=42, b=6),
        title=dict(text=f"📈 Corte de la carrera · {delta:+.0f} pts en 3 años", font=dict(size=13, color=AZUL_OSC)),
        yaxis=dict(title="Corte (ponderado)", showgrid=True, gridcolor="#eef"),
        xaxis=dict(showgrid=False), plot_bgcolor="white", paper_bgcolor="white",
        yaxis_range=[min(vals) - 25, max(vals) + 30])
    return fig


def fig_seleccion(s: dict, user_pond, corte, anio: int, es_real: bool):
    """Boxplot horizontal del ponderado de los seleccionados (5 números) + el corte + tu ponderado."""
    fig = go.Figure()
    fig.add_trace(go.Box(q1=[s["p25"]], median=[s["p50"]], q3=[s["p75"]],
        lowerfence=[s["p05"]], upperfence=[s["p95"]], orientation="h", y=["sel"], name="",
        fillcolor="rgba(37,99,235,.18)", line=dict(color=AZUL, width=2),
        whiskerwidth=.6, showlegend=False, hoverinfo="skip"))
    if corte:                                              # corte del MISMO año de la caja (piso de admisión)
        fig.add_vline(x=corte, line=dict(color="#dc2626", width=2, dash="dot"),
                      annotation_text=f"corte {anio}: {corte:.0f}", annotation_position="top left",
                      annotation_font=dict(color="#dc2626", size=11))
    if user_pond is not None:                              # tú = diamante verde
        fig.add_trace(go.Scatter(x=[user_pond], y=["sel"], mode="markers",
            marker=dict(symbol="diamond", size=17, color="#16a34a", line=dict(width=2, color="white")),
            showlegend=False, hovertemplate=f"Tu ponderado: {user_pond:.0f}<extra></extra>"))
        fig.add_annotation(x=user_pond, y="sel", text=f"<b>{user_pond:.0f}</b> · tú",
                           showarrow=True, arrowhead=2, ay=-34, font=dict(color="#15803d", size=12))
    lo = min(s["p05"], corte or 9e9, user_pond if user_pond is not None else 9e9) - 25
    hi = max(s["p95"], user_pond if user_pond is not None else 0) + 25
    fig.update_layout(height=240, margin=dict(l=10, r=18, t=46, b=10),
        title=dict(text=f"📊 ¿Con qué puntaje entró la gente? · seleccionados {anio} (n={s['n']})",
                   font=dict(size=13, color=AZUL_OSC)),
        xaxis=dict(title="Puntaje ponderado", range=[lo, hi], showgrid=True, gridcolor="#eef"),
        yaxis=dict(showticklabels=False), plot_bgcolor="white", paper_bgcolor="white")
    return fig


def fig_pie_genero(s, titulo):
    """Torta de proporción por género de los titulados."""
    pm, ph = s["pct_muj"], s["pct_hom"]
    resto = max(0.0, 100 - pm - ph)
    labels, vals, cols = ["Mujeres", "Hombres"], [pm, ph], ["#ec4899", "#2563eb"]
    if resto > 0.5:
        labels.append("Otro/NB"); vals.append(resto); cols.append("#94a3b8")
    fig = go.Figure(go.Pie(labels=labels, values=vals, marker_colors=cols, hole=.45, sort=False,
        textinfo="label+percent", textfont=dict(size=11, color="white"),
        hovertemplate="%{label}: %{percent}<extra></extra>"))
    fig.update_layout(height=215, margin=dict(l=6, r=6, t=40, b=6), showlegend=False,
        title=dict(text=titulo, font=dict(size=12, color=AZUL_OSC)), paper_bgcolor="white")
    return fig


TES_LBL = {"municipal": "Municipal", "part_subv": "Part. subvencionado", "part_pagado": "Part. pagado",
           "corp_ad": "Adm. delegada", "sle": "Servicio local"}
TES_COL = {"municipal": "#2563eb", "part_subv": "#16a34a", "part_pagado": "#f59e0b",
           "corp_ad": "#8b5cf6", "sle": "#0ea5e9"}


def fig_tes(tes: dict, n: int):
    """Barra horizontal: % de matriculados de 1er año por tipo de establecimiento de ORIGEN (SIES)."""
    items = [(k, tes.get(k, 0.0)) for k in TES_LBL if tes.get(k, 0.0) > 0]
    items.sort(key=lambda t: t[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[TES_LBL[k] for k, _ in items], orientation="h",
        marker_color=[TES_COL[k] for k, _ in items], text=[f"{v:.0f}%" for _, v in items],
        textposition="outside", textfont=dict(color=AZUL_OSC, size=12)))
    fig.update_layout(height=max(170, 40 * len(items) + 60), margin=dict(l=8, r=30, t=40, b=8),
        title=dict(text=f"🏫 Origen escolar · matrícula total n={n:,}".replace(",", "."),
                   font=dict(size=13, color=AZUL_OSC)),
        xaxis=dict(range=[0, max(v for _, v in items) * 1.18], ticksuffix="%", showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white")
    return fig


def ficha_oferta_html(of: dict) -> str:
    """Chips con la ficha institucional de la carrera (nivel, jornada, duración, sede)."""
    if not of:
        return ""
    g = lambda x: f"{x:g}"
    chips = []
    if of.get("nivel"):
        chips.append(f"<span class='pchip ob'>🎓 {of['nivel']}</span>")
    if of.get("jornada"):
        otras = [j for j in of.get("jornadas", []) if j != of["jornada"]]
        chips.append(f"<span class='pchip'>🕗 {of['jornada']}" + (f" +{len(otras)}" if otras else "") + "</span>")
    if of.get("dur_sem"):
        a = of.get("dur_anios")
        chips.append(f"<span class='pchip'>⏳ {g(of['dur_sem'])} sem" + (f" · {g(a)} años" if a else "") + "</span>")
    loc = " · ".join(x for x in [of.get("comuna"), of.get("region")] if x)
    if loc:
        chips.append(f"<span class='pchip'>📍 {loc}</span>")
    if of.get("sede"):
        chips.append(f"<span class='pchip'>🏛️ {of['sede']}</span>")
    return "<div class='pchips' style='margin-top:8px'>" + "".join(chips) + "</div>"


def tit_fila(etiqueta, s) -> dict:
    return {" ": etiqueta, "Titulados": s["n"], "% Mujeres": s["pct_muj"], "% Hombres": s["pct_hom"],
            "Edad prom.": s["edad_prom"], "Edad mediana": s["edad_mediana"]}


def match_titulacion(nombre: str, por_carrera: dict) -> str | None:
    """Calza el nombre DEMRE (con sufijos de campus/menciones) con la carrera genérica SIES:
    exacto → sin sufijo '(...)'/'- ...' → clave SIES más larga contenida en el nombre."""
    if nombre in por_carrera:
        return nombre
    base = nombre.split(" (")[0].split(" - ")[0].strip()
    if base in por_carrera:
        return base
    cands = [k for k in por_carrera if k in nombre]
    return max(cands, key=len) if cands else None


def vac_total_de(row) -> float:
    """Vacantes 2026 totales (regular 1er+2º sem + admisión especial PACE/CDP/género)."""
    g = lambda c: (lambda x: 0.0 if (x is None or x != x) else float(x))(row.get(c))
    return g("VACANTES_1SEM") + g("VACANTES_2SEM") + g("CAR_VACANTES_PACE") + g("CDP_VACANTES_ESPECIALES") + g("VACANTES_GENERO")


def fig_radar_multi(rows, labels):
    """Radar que superpone las ponderaciones de 2-3 carreras para compararlas."""
    cats = [n for n, _ in PESOS]
    palette = [AZUL, "#16a34a", "#f59e0b"]
    fig = go.Figure()
    mx = 0
    for i, (r, lab) in enumerate(zip(rows, labels)):
        vals = [float(r[c]) if r[c] == r[c] else 0.0 for _, c in PESOS]
        mx = max(mx, max(vals))
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
            name=lab[:26], line=dict(color=palette[i % len(palette)], width=2), opacity=.5,
            hovertemplate="%{theta}: %{r:.0f}%<extra></extra>"))
    fig.update_layout(height=400, margin=dict(l=40, r=40, t=44, b=46),
        title=dict(text="⚖️ Ponderaciones comparadas", font=dict(size=14, color=AZUL_OSC)),
        polar=dict(radialaxis=dict(range=[0, mx * 1.15 if mx else 40], ticksuffix="%", tickfont=dict(size=9)),
                   angularaxis=dict(tickfont=dict(size=10, color=AZUL_OSC))),
        legend=dict(orientation="h", y=-0.12, font=dict(size=10)), paper_bgcolor="white")
    return fig


def fig_mapa(territorio, geo, cent, region_sel, comuna_sel, L):
    """Coroplético de Chile por tasa de acceso (región) + tu región resaltada + tu comuna marcada."""
    codes = [f["properties"]["codregion"] for f in geo["features"]]
    z, txt = [], []
    for c in codes:
        s = territorio["region"].get(str(c))
        nom = L["region"].get(str(c), str(c))
        z.append(s["tasa"] * 100 if s else None)
        txt.append(f"<b>{nom}</b><br>Acceso 1ª pref: {s['tasa']:.0%}<br>n={s['n']:,}" if s else f"<b>{nom}</b><br>s/d")
    zz = [v for v in z if v is not None]
    fig = go.Figure(go.Choropleth(
        geojson=geo, locations=codes, featureidkey="properties.codregion", z=z,
        colorscale="Blues", zmin=min(zz), zmax=max(zz), marker_line_color="white", marker_line_width=.5,
        colorbar=dict(title="% acceso", thickness=12, len=.55, x=.0, xanchor="left"),
        text=txt, hovertemplate="%{text}<extra></extra>"))
    if region_sel:                                      # resaltar tu región (borde naranjo)
        fig.add_trace(go.Choropleth(geojson=geo, locations=[int(region_sel)],
            featureidkey="properties.codregion", z=[0], showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker_line_color="#f59e0b", marker_line_width=3, hoverinfo="skip"))
    cc = cent.get(str(comuna_sel))                      # marcar tu comuna (punto rojo)
    if cc:
        fig.add_trace(go.Scattergeo(lon=[cc["lon"]], lat=[cc["lat"]], mode="markers",
            marker=dict(size=11, color="#dc2626", line=dict(width=2, color="white")),
            hovertemplate=f"📍 Tu comuna: <b>{L['comuna'].get(str(comuna_sel),'')}</b><extra></extra>"))
    # encuadre fijo a Chile CONTINENTAL (excluye Isla de Pascua ~-109° y la Antártica, que distorsionan)
    fig.update_geos(visible=False, bgcolor="rgba(0,0,0,0)", projection_type="mercator",
                    lonaxis_range=[-76.5, -66.0], lataxis_range=[-56.0, -17.3])
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=8, b=0), paper_bgcolor="white", showlegend=False)
    return fig


def fig_mapa_puntaje(territorio, geo, cent, comuna_sel, L):
    """Mapa de Chile (regiones en gris de fondo) con cada comuna como punto coloreado por PUNTAJE PAES
    promedio. Resalta tu región (borde naranjo) y marca tu comuna (📍). Así se ve 'el mapa', no puntos
    flotando: no tenemos polígonos de comuna (solo centroides), por eso la comuna va como punto."""
    codes = [f["properties"]["codregion"] for f in geo["features"]]
    fig = go.Figure()
    # base: silueta de Chile (regiones gris claro) para dar contexto geográfico
    fig.add_trace(go.Choropleth(geojson=geo, locations=codes, featureidkey="properties.codregion",
        z=[1] * len(codes), colorscale=[[0, "#e9eef7"], [1, "#e9eef7"]], showscale=False,
        marker_line_color="white", marker_line_width=.6, hoverinfo="skip"))
    reg_sel = L.get("comuna_region", {}).get(str(comuna_sel))      # región de tu comuna → resaltar
    if reg_sel:
        fig.add_trace(go.Choropleth(geojson=geo, locations=[int(reg_sel)],
            featureidkey="properties.codregion", z=[0], showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker_line_color="#f59e0b", marker_line_width=2.5, hoverinfo="skip"))
    # comunas como puntos coloreados por puntaje
    lons, lats, vals, txt, sizes = [], [], [], [], []
    for code, c in cent.items():
        s = territorio["comuna"].get(code)
        if s and s.get("puntaje") and s["n"] >= 30:
            lons.append(c["lon"]); lats.append(c["lat"]); vals.append(s["puntaje"])
            sizes.append(7 + min(13, s["n"] ** 0.5 / 10))
            txt.append(f"<b>{L['comuna'].get(code, code)}</b><br>Puntaje PAES: {s['puntaje']:.0f}<br>n={s['n']:,}")
    fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode="markers",
        marker=dict(size=sizes, color=vals, colorscale="RdYlGn", cmin=min(vals), cmax=max(vals),
                    showscale=True, colorbar=dict(title="Puntaje", thickness=12, len=.55, x=0, xanchor="left"),
                    line=dict(width=.4, color="#475569"), opacity=.9),
        text=txt, hovertemplate="%{text}<extra></extra>"))
    cc = cent.get(str(comuna_sel))
    if cc:
        fig.add_trace(go.Scattergeo(lon=[cc["lon"]], lat=[cc["lat"]], mode="markers+text",
            marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=3, color="#1e3a8a")),
            text=["📍"], textposition="middle center", textfont=dict(size=15),
            hovertemplate=f"📍 Tu comuna: <b>{L['comuna'].get(str(comuna_sel),'')}</b><extra></extra>"))
    fig.update_geos(visible=False, bgcolor="rgba(0,0,0,0)", projection_type="mercator",
                    lonaxis_range=[-76.5, -66.0], lataxis_range=[-56.0, -17.3])
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=8, b=0), paper_bgcolor="white", showlegend=False)
    return fig


def fig_barras_region(territorio, region_sel, L):
    """Ranking horizontal de tasa de acceso por región; tu región en naranjo."""
    items = sorted(((L["region"].get(str(c), str(c)), s["tasa"] * 100, str(c))
                    for c, s in territorio["region"].items()), key=lambda x: x[1])
    cols = ["#f59e0b" if c == str(region_sel) else "#bcd4f6" for _, _, c in items]
    fig = go.Figure(go.Bar(x=[v for _, v, _ in items], y=[n.replace("Region ", "") for n, _, _ in items],
        orientation="h", marker_color=cols, text=[f"{v:.0f}%" for _, v, _ in items],
        textposition="outside", textfont=dict(size=10, color=AZUL_OSC)))
    fig.update_layout(height=640, margin=dict(l=8, r=24, t=30, b=8),
        title=dict(text="Acceso a 1ª preferencia por región", font=dict(size=13, color=AZUL_OSC)),
        xaxis=dict(range=[0, max(v for _, v, _ in items) * 1.18], ticksuffix="%", showgrid=True, gridcolor="#eef"),
        yaxis=dict(tickfont=dict(size=10)), plot_bgcolor="white", paper_bgcolor="white")
    return fig


def tabla_rank(rows, idx, incluir_carrera: bool, incluir_margen: bool = True,
               modo_orden: str = "alcanzo", n: int = 15, umbral: float = 0.5) -> pd.DataFrame:
    """DataFrame para st.dataframe a partir del ranking.

    modo_orden:
      - "alcanzo": lo MEJOR que alcanzas → entre las que tienes ≥`umbral` de prob., ordena por
        corte (selectividad) descendente. Si casi ninguna llega al umbral, cae a las más probables.
      - "prob": las más probables primero (P desc, desempata por corte).
    Las carreras sin corte histórico (nuevas) se marcan 🆕 y van al final en modo "alcanzo".
    """
    # carreras con corte (ranking confiable) primero; las nuevas (🆕) al final, su P es solo orientativa
    con = [d for d in rows if d["corte"] is not None]
    sin = sorted([d for d in rows if d["corte"] is None], key=lambda d: d["p"], reverse=True)
    if modo_orden == "alcanzo":
        cand = [d for d in con if d["p"] >= umbral]
        base = (sorted(cand, key=lambda d: d["corte"], reverse=True) if len(cand) >= 3
                else sorted(con, key=lambda d: d["p"], reverse=True))     # fallback: las más probables
    else:
        base = sorted(con, key=lambda d: (d["p"], d["corte"]), reverse=True)
    rr = (base + sin)[:n]
    data = []
    for d in rr:
        c = idx.loc[d["cod"]]
        nueva = d["corte"] is None
        fila = {}
        if incluir_carrera:
            fila["Carrera"] = str(c["NOMBRE_CARRERA"]).title()
        fila["Universidad"] = ("🆕 " if nueva else "") + str(c["NOMBRE_UNIVERSIDAD"]).title()
        fila["Región"] = str(c["reg_nom"])
        fila["P(acceso)"] = d["p"] * 100
        fila["Corte"] = d["corte"]
        if incluir_margen:
            fila["Tu margen"] = d["margen"]
        data.append(fila)
    return pd.DataFrame(data)


def mostrar_tabla(df: pd.DataFrame):
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "P(acceso)": st.column_config.ProgressColumn("Prob. acceso", min_value=0, max_value=100,
                                                     format="%.0f%%", help="Probabilidad calibrada (POST-PAES)"),
        "Corte": st.column_config.NumberColumn("Corte", format="%.0f", help="Corte regular más reciente (mín. ponderado de los seleccionados)"),
        "Tu margen": st.column_config.NumberColumn("Tu margen", format="%+.0f",
                                                   help="Tu ponderado menos el corte más reciente"),
    })


def historia_carrera(mtr: dict, v2: float, vac_total: float, vac_esp: float = 0.0) -> str:
    """Narrativa en lenguaje natural del embudo de la carrera (postulan → quedan → se matriculan
    → total), año 2026. Devuelve HTML (.nota) o '' si no hay datos suficientes."""
    if not mtr:
        return ""
    np_ = mtr.get("n_postula", 0) or 0
    ns = mtr.get("n_sel", 0) or 0
    nsm = mtr.get("n_sel_matric", 0) or 0
    tasa = mtr.get("tasa")
    ntot = mtr.get("n_matric_total", 0) or 0
    anio = mtr.get("anio", 2026)
    m = lambda n: f"{int(n):,}".replace(",", ".")   # miles a la chilena, solo en números
    partes = []
    if np_ > 0 and ns > 0:
        partes.append(f"De los <b>{m(np_)}</b> estudiantes que la pusieron como <b>1ª preferencia</b> en {anio}, "
                      f"quedaron seleccionados <b>{m(ns)}</b> (<b>{ns/np_:.0%}</b> de los postulantes)")
    elif ns > 0:
        partes.append(f"En {anio} quedaron seleccionados <b>{m(ns)}</b> en 1ª preferencia")
    if ns > 0 and tasa is not None:
        partes.append(f"de ellos, el <b>{tasa:.0%}</b> se matriculó en esta carrera")
    if ntot > 0:
        ctx = ""
        if ntot > nsm:
            extras = ["otras preferencias"] + (["2º semestre"] if v2 > 0 else []) \
                + (["vías especiales como PACE"] if vac_esp > 0 else [])
            ctx = " (sumando a quienes entraron por " + ", ".join(extras) + ")"
        cola = f"en total la carrera matriculó a <b>{m(ntot)}</b> personas{ctx}"
        if vac_total > 0:
            cola += f", sobre <b>{m(vac_total)}</b> vacantes en total"
        partes.append(cola)
    if not partes:
        return ""
    texto = "; ".join(partes) + "."
    texto = texto[0].upper() + texto[1:]
    poco = " <span style='color:#94a3b8;font-size:.85em'>· pocos casos, dato referencial</span>" if 0 < ns < 20 else ""
    return f"<div class='nota'>📖 <b>La historia de esta carrera ({anio}):</b> {texto}{poco}</div>"


def ponderaciones_html(row):
    """Fila de chips con la ponderación (%) de cada prueba. Obligatorias (Notas, Ranking,
    C. Lectora, Matem. M1) resaltadas. Historia y Ciencias son electivos ALTERNATIVOS:
    cuenta el mejor (igual que el modelo en ponderado()), no se suman ambos."""
    obligatorias = {"%_NOTAS", "%_Ranking", "%_LENG", "%_MATE1"}
    base = [("Notas", "%_NOTAS"), ("Ranking", "%_Ranking"), ("C. Lectora", "%_LENG"),
            ("Matem. M1", "%_MATE1"), ("Matem. M2", "%_MATE2")]
    chips, total = [], 0.0
    for lbl, col in base:
        w = row.get(col)
        if w is None or w != w or float(w) == 0:
            continue
        total += float(w)
        cls = "pchip ob" if col in obligatorias else "pchip"
        chips.append(f"<span class='{cls}'><b>{float(w):.0f}%</b> {lbl}</span>")
    # electivo Historia/Ciencias: se computa el mejor de los dos, no la suma
    hy = row.get("%_HYCS"); ci = row.get("%_CIEN")
    hy = 0.0 if (hy is None or hy != hy) else float(hy)
    ci = 0.0 if (ci is None or ci != ci) else float(ci)
    if hy > 0 or ci > 0:
        we = max(hy, ci); total += we
        etq = ("Historia o Ciencias" if hy > 0 and ci > 0 else "Historia" if hy > 0 else "Ciencias")
        chips.append(f"<span class='pchip'><b>{we:.0f}%</b> {etq}</span>")
    extra = "" if total >= 99 else f"<span class='pchip'>+ prueba especial ({100-total:.0f}%)</span>"
    return "<div class='pchips'>" + "".join(chips) + extra + "</div>"


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
<p>Acceso a 1ª preferencia universitaria <b>antes</b> y <b>después</b> de la PAES.
Modelos validados temporalmente (entrena 2025 → testea 2026) · DAML 2026 · Grupo 5</p></div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- tutorial (panel inline, NO modal)
# Se usaba @st.dialog, pero Streamlit re-invoca el diálogo por su cuenta en cada rerun y no expone un
# callback de "cerrado": eso lo hacía reaparecer solo con cualquier clic. Un panel inline gobernado por
# un booleano (show_tut) es 100% predecible: se muestra/oculta sin ciclo de modal.
TUT_PASOS = [
    ("1 · Elige tu carrera", "En la **barra lateral** (izquierda ⬅️) elige la **carrera** y la **universidad** "
     "que te interesan. Escribe para buscar; los nombres están en MAYÚSCULAS."),
    ("2 · Completa tu perfil", "Más abajo en la barra lateral, ingresa tus **notas** (NEM y ranking) y de **dónde "
     "vienes** (región, comuna, colegio). Si ya rendiste la PAES, agrega tus **puntajes**."),
    ("3 · Mira tu resultado", "En **🎯 Mi resultado** verás tu **probabilidad de acceso**, tu **ponderado vs el "
     "corte** y **cuánto te falta**. Puedes descargar el resumen."),
    ("4 · Explora", "Arriba cambias de modo: **📋 La carrera** (ponderaciones, titulación, cortes), "
     "**🔎 ¿Dónde quedo?** (dónde tienes chance), **⚖️ Comparar** y **🗺️ Mapa**."),
]


def render_tutorial():
    paso = st.session_state.get("tut_paso", 0)
    titulo, texto = TUT_PASOS[paso]
    with st.container(border=True):
        hc1, hc2 = st.columns([8, 1])
        hc1.markdown(f"#### 📖 Cómo usar este dashboard")
        if hc2.button("✕", key="tut_x", help="Cerrar tutorial"):
            st.session_state["show_tut"] = False; st.rerun()
        st.markdown(f"**{titulo}**")
        st.markdown(texto)
        st.progress((paso + 1) / len(TUT_PASOS))
        b1, _, b3 = st.columns([1, 3, 1])
        if paso > 0 and b1.button("← Atrás", key="tut_prev", use_container_width=True):
            st.session_state["tut_paso"] = paso - 1; st.rerun()
        if paso < len(TUT_PASOS) - 1:
            if b3.button("Siguiente →", key="tut_next", use_container_width=True, type="primary"):
                st.session_state["tut_paso"] = paso + 1; st.rerun()
        elif b3.button("¡Listo! 🎉", key="tut_done", use_container_width=True, type="primary"):
            st.session_state["show_tut"] = False; st.session_state["tut_paso"] = 0; st.rerun()


if "tut_seen" not in st.session_state:           # se abre automáticamente solo la 1ª vez de la sesión
    st.session_state["tut_seen"] = True
    st.session_state["tut_paso"] = 0
    st.session_state["show_tut"] = True

cat = art.catalogo.copy()
cat["reg_nom"] = cat["REGION_CASA_MATRIZ"].astype("Int64").astype(str).map(L["region"]).fillna("")
# normalización a MAYÚSCULAS SIN TILDES: unifica "Ingeniería"/"INGENIERIA"/"INGENIERÍA"
cat["CARRERA_U"] = cat["NOMBRE_CARRERA"].fillna("¿?").map(_norm)
cat["UNIV_U"] = cat["NOMBRE_UNIVERSIDAD"].fillna("¿?").map(_norm)
# desambiguación de universidad: universidad · región (cód) — cada opción mapea a UN código
cat["univ_display"] = (cat["UNIV_U"] + " · " + cat["reg_nom"].str.upper()
                       + "  (cód " + cat["CODIGO_CARRERA"].astype(str) + ")")
cat["area"] = cat["CARRERA_U"].map(area_de)
cat["comp_label"] = (cat["NOMBRE_CARRERA"].fillna("¿?").str.title() + " — " + cat["UNIV_U"].str.title()
                     + " · " + cat["reg_nom"] + " (cód " + cat["CODIGO_CARRERA"].astype(str) + ")")
cat_idx = cat.set_index("CODIGO_CARRERA")

# ----------------------------------------------------------------- INPUTS (barra lateral)
with st.sidebar:
    st.markdown(f"<div style='font-weight:800;color:{AZUL_OSC};font-size:1.1rem'>🎓 Tus datos</div>", unsafe_allow_html=True)
    if st.button("📖 Tutorial · cómo usar", use_container_width=True):
        st.session_state["tut_paso"] = 0; st.session_state["show_tut"] = True; st.rerun()
    st.markdown("**1 · Tu carrera**")
    carreras = sorted(cat["CARRERA_U"].unique())
    idx0 = carreras.index("ARQUITECTURA") if "ARQUITECTURA" in carreras else 0
    carrera_sel = st.selectbox("Carrera (escribe para buscar)", carreras, index=idx0)
    st.markdown(f"<div class='sbfull'>📚 {carrera_sel}</div>", unsafe_allow_html=True)
    subset = cat[cat["CARRERA_U"] == carrera_sel].sort_values("univ_display")
    uni_sel = st.selectbox(f"Universidad · {len(subset)} opción(es)", subset["univ_display"].tolist())
    st.markdown(f"<div class='sbfull'>🏛️ {uni_sel}</div>", unsafe_allow_html=True)
    st.markdown("**2 · Tu perfil**")
    if st.radio("Notas", ["Puntaje NEM", "Promedio de notas"], horizontal=True, label_visibility="collapsed", key="m_nem") == "Puntaje NEM":
        nem = st.number_input("Puntaje NEM (100–1000)", 100, 1000, 650, 5); promedio = None
    else:
        promedio = st.number_input("Promedio de notas (1.0–7.0)", 1.0, 7.0, 6.0, 0.1); nem = None
    if st.radio("Ranking", ["Puntaje Ranking", "% superior del curso"], horizontal=True, label_visibility="collapsed", key="m_rk") == "Puntaje Ranking":
        ranking = st.number_input("Puntaje Ranking (100–1000)", 100, 1000, 680, 5); porc_sup = None
    else:
        porc_sup = st.number_input("% superior del curso (menor = mejor)", 1, 100, 30, 1); ranking = None
    region = st.selectbox("Región", opt(L["region"]), format_func=lambda k: L["region"].get(k, k),
                          index=opt(L["region"]).index("13") if "13" in L["region"] else 0)
    comunas_reg = [c for c in opt(L["comuna"]) if L["comuna_region"].get(c) == region] or opt(L["comuna"])
    comuna = st.selectbox("Comuna", comunas_reg, format_func=lambda k: L["comuna"].get(k, k))
    dependencia = st.selectbox("Dependencia del colegio", opt(L["dependencia"]), format_func=lambda k: L["dependencia"].get(k, k))
    rama = st.selectbox("Rama educacional", opt(L["rama"]), format_func=lambda k: L["rama"].get(k, k))
    # colegio (opcional): afina la estimación con el historial PAES del establecimiento.
    # Se filtra por comuna Y por la dependencia elegida, para no mezclar (ej.: "Municipal" + un colegio
    # particular), combinación que el modelo nunca vio y que sería incoherente.
    _cols = art.rbd_stats.get("colegios", {})
    try:
        _ccod = int(comuna)
    except (TypeError, ValueError):
        _ccod = None
    _ops = sorted([(r, c["nom"]) for r, c in _cols.items()
                   if c.get("com_cod") == _ccod and c.get("dep") == dependencia], key=lambda t: t[1])
    _nm = {r: n for r, n in _ops}
    # key dinámico: al cambiar región/comuna/dependencia el selector se RESETEA (evita arrastrar un
    # colegio de otra comuna/dependencia, bug de estado de Streamlit que metía un RBD incoherente).
    rbd_sel = st.selectbox("🏫 Tu colegio (opcional)", [None] + [r for r, _ in _ops],
                           format_func=lambda r: "— No especificar —" if r is None else _nm.get(r, r),
                           key=f"colegio_{region}_{comuna}_{dependencia}",
                           help="Lista filtrada por tu comuna y dependencia. Afina la estimación PRE-PAES con el historial de tu colegio. Opcional.")
    if rbd_sel is not None and rbd_sel not in _nm:   # red de seguridad: solo un RBD del filtro actual
        rbd_sel = None
    if not _ops:
        st.caption(f"Sin colegios **{L['dependencia'].get(dependencia, '')}** con historial en esta comuna. "
                   "Cambia la dependencia si no ves el tuyo; si no, se usa el promedio comunal.")
    st.markdown("**3 · Tus puntajes PAES** · *si ya rendiste*")
    s_clec = st.number_input("C. Lectora", 0, 1000, 0, 5, key="s_clec", help="Déjalo en 0 si aún no rindes")
    s_mate1 = st.number_input("Matemática M1", 0, 1000, 0, 5, key="s_mate1")
    s_mate2 = st.number_input("Matem. M2", 0, 1000, 0, 5, key="s_mate2")
    s_hcsoc = st.number_input("Historia", 0, 1000, 0, 5, key="s_hcsoc")
    s_cien = st.number_input("Ciencias", 0, 1000, 0, 5, key="s_cien")

row = subset[subset["univ_display"] == uni_sel].iloc[0]
cod = int(row["CODIGO_CARRERA"])
st_info = art.stats.get(str(cod))
perfil_base = Perfil(cod_carrera=cod, nem=nem, ranking=ranking, promedio_notas=promedio, porc_sup=porc_sup,
                     region=region, comuna=comuna, dependencia=dependencia, rama=rama, rbd=rbd_sel)
es_post = s_clec >= 100 and s_mate1 >= 100
perfil_post = replace(perfil_base, clec=s_clec if s_clec >= 100 else None, mate1=s_mate1 if s_mate1 >= 100 else None,
                      mate2=s_mate2 if s_mate2 >= 100 else None, hcsoc=s_hcsoc if s_hcsoc >= 100 else None,
                      cien=s_cien if s_cien >= 100 else None)
perfil_exp = perfil_post if es_post else perfil_base      # para los modos de exploración
modo_modelo = "post" if es_post else "pre"
_vac = lambda c: (lambda x: 0.0 if (x is None or x != x) else float(x))(row.get(c))
v1, v2 = _vac("VACANTES_1SEM"), _vac("VACANTES_2SEM")
vac_esp = _vac("CAR_VACANTES_PACE") + _vac("CDP_VACANTES_ESPECIALES") + _vac("VACANTES_GENERO")
vac_total = v1 + v2 + vac_esp

if st.session_state.get("show_tut"):                 # panel inline, sobre las pestañas (sin modal)
    render_tutorial()

# ----------------------------------------------------------------- modo: MI RESULTADO
def render_resultado():
    _vres = predecir(art, perfil_base)
    _vbanda = predecir_puntaje(art, perfil_base)

    def _esc(q):
        pp = replace(perfil_base, clec=_vbanda["CLEC"][q], mate1=_vbanda["MATE1"][q],
                     mate2=_vbanda.get("MATE2", {}).get(q), hcsoc=_vbanda.get("HCSOC", {}).get(q),
                     cien=_vbanda.get("CIEN", {}).get(q))
        r = predecir(art, pp)
        return r["ponderado"], r["p_post"]

    if _vres["p_pre"] is None:
        st.info("Completa **NEM/notas y ranking** en la barra lateral ⬅️ para ver tu resultado.")
        return
    st.markdown(f"#### {carrera_sel.title()} · {str(row['UNIV_U']).title()}")
    _corte = _vres["corte"]
    _canio = (st_info or {}).get("anio", 2026)             # año del corte de referencia (último cerrado)
    _es_real = es_post                                     # ¿hay puntajes PAES reales? → un solo modelo/gráfico

    if _es_real:                                           # POST: puntajes PAES reales
        _res2 = predecir(art, perfil_post)
        _p, _pond = _res2["p_post"], _res2["ponderado"]
        _gtit = "Probabilidad de acceso (con tu PAES)"
        st.caption("✅ Resultado con tus **puntajes PAES reales**. Con el puntaje, el origen ya no cambia el resultado.")
    else:                                                  # PRE: estimación desde notas + contexto
        _p, _pond = _vres["p_pre"], _esc("p50")[0]
        _gtit = "Probabilidad de acceso (estimada)"
        _coltxt = " (afinada con el historial de tu colegio 🏫)" if perfil_base.rbd else ""
        st.caption(f"🔮 **Estimación antes de la PAES**, a partir de tus notas y contexto{_coltxt}. "
                   "Ingresa tus puntajes PAES en la barra lateral ⬅️ para ver el resultado real.")
    _gap = (_corte - _pond) if (_corte and _pond is not None) else None

    vc1, vc2 = st.columns([1, 1.35])
    vc1.plotly_chart(gauge(_p, _gtit), use_container_width=True, key="ver_gauge")
    with vc2:
        st.markdown("<br>", unsafe_allow_html=True)
        _vm = st.columns(3)
        _vm[0].metric("Tu ponderado" if _es_real else "🎯 Ponderado estimado", f"{_pond:.0f}" if _pond is not None else "s/d")
        _vm[1].metric(f"Corte {_canio}", f"{_corte:.0f}" if _corte else "s/d")
        if _gap is not None:
            _vm[2].metric("Tu margen" if _es_real else "Margen estimado", f"{-_gap:+.0f}", delta=f"{-_gap:+.0f}")
        if _gap is not None and _gap > 0:
            if _es_real:
                st.markdown(f"<div class='nota'>📐 <b>¿Cuánto te falta?</b> Tu ponderado (<b>{_pond:.0f}</b>) está "
                            f"<b>{_gap:.0f} pts bajo el corte</b> ({_corte:.0f}).</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='nota'>📐 <b>¿Cuánto te falta?</b> Tu ponderado estimado (~{_pond:.0f}) está "
                            f"<b>{_gap:.0f} pts bajo el corte</b> ({_corte:.0f}) → necesitarías subir "
                            f"<b>~{_gap:.0f} pts en cada prueba</b>.</div>", unsafe_allow_html=True)
        elif _gap is not None:
            if _es_real:
                st.markdown(f"<div class='nota'>✅ <b>Sobre el corte:</b> tu ponderado (<b>{_pond:.0f}</b>) supera el "
                            f"corte ({_corte:.0f}) por <b>{-_gap:.0f} pts</b>.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='nota'>✅ <b>Vas bien:</b> tu ponderado estimado (~{_pond:.0f}) supera el corte "
                            f"({_corte:.0f}) por ~{-_gap:.0f} pts. Igual depende de cómo rindas.</div>", unsafe_allow_html=True)
        if not _es_real:
            _, _plo = _esc("p10"); _, _phi = _esc("p90")
            if _plo is not None and _phi is not None:
                st.markdown(f"<div class='warn'>🎲 <b>Depende de la PAES:</b> bajo (P10) ~<b>{_plo:.0%}</b>; "
                            f"alto (P90) ~<b>{_phi:.0%}</b>. La prueba aún no está jugada.</div>", unsafe_allow_html=True)

    # distribución del ponderado de los seleccionados (boxplot) + dónde caes tú
    _sel = art.seleccion.get(str(cod))
    if _sel and _pond is not None:
        _corte_box = (art.cortes_hist.get(str(cod)) or {}).get(str(_sel["anio"]))   # corte del MISMO año de la caja
        st.plotly_chart(fig_seleccion(_sel, _pond, _corte_box, _sel["anio"], _es_real), use_container_width=True, key="box_sel")
        _pos = ("**por sobre la mediana**" if _pond >= _sel["p50"] else
                "**dentro del 50% central**" if _pond >= _sel["p25"] else
                "**bajo el 25% que entró más bajo**")
        st.caption(f"📊 **Cómo leerlo:** la **caja azul** abarca al 50% central de quienes entraron en {_sel['anio']} "
                   f"(del p25 al p75); la **línea** del medio es la **mediana** ({_sel['p50']:.0f} → la mitad entró con menos "
                   f"y la mitad con más); los **bigotes** llegan del p5 al p95. La línea **roja** es el **corte {_sel['anio']}** "
                   f"(el mínimo con que entró alguien ese año), por eso coincide con el piso de la caja. El **diamante verde "
                   f"eres tú** ({'real' if _es_real else 'estimado'}): caes {_pos}.")

    _tkey = match_titulacion(carrera_sel, art.titulacion.get("por_carrera", {}))
    _tq = art.titulacion.get("por_carrera", {}).get(_tkey)
    _modo = "POST-PAES (puntajes reales)" if _es_real else "PRE-PAES (estimación por notas)"
    _lineas = [f"MI RESULTADO — {carrera_sel.title()} · {str(row['UNIV_U']).title()}",
               f"Región/comuna: {L['region'].get(region, region)} / {L['comuna'].get(comuna, comuna)}", "",
               f"Probabilidad de acceso [{_modo}]: {_p:.0%}",
               (f"Tu ponderado: {_pond:.0f}" if _es_real else f"Ponderado estimado: ~{_pond:.0f}")
               if _pond is not None else "Ponderado: s/d"]
    if _corte and _pond is not None:
        _lineas.append(f"Corte {_canio}: {_corte:.0f}  ·  margen: {(_pond-_corte):+.0f}")
    if _sel:
        _lineas.append(f"Seleccionados {_sel['anio']}: entraron entre {_sel['p05']:.0f} y {_sel['p95']:.0f} "
                       f"(mediana {_sel['p50']:.0f})")
    if _tq:
        _lineas.append(f"Titulación: {_tq['pct_muj']:.0f}% mujeres · edad mediana {_tq['edad_mediana']:.0f} años")
    _lineas += ["", "Estimación del dashboard DAML 2026 · Grupo 5 — no es garantía."]
    st.download_button("📄 Descargar mi resumen", data="\n".join(_lineas), file_name="mi_resultado_PAES.txt", key="dl_resumen")

    with st.expander("🔬 Ver detalle: efecto del origen" + ("" if _es_real else " y tu puntaje PAES probable")):
        if not _es_real:
            st.plotly_chart(fig_bandas(_vbanda), use_container_width=True, key="bandas")
            st.caption("De cada 100 estudiantes con tu perfil, ~80 sacan un puntaje dentro de la banda; el punto es la mediana.")
        cfp = cf_dependencia(perfil_base, "pre")
        if cfp:
            st.plotly_chart(fig_cf(cfp, "Probabilidad según tipo de colegio — mismo perfil académico", AZUL),
                            use_container_width=True, key="cf_pre")
            gap = (max(cfp.values()) - min(cfp.values())) * 100
            st.markdown(f"<div class='nota'>Con <b>las mismas notas</b>, cambiar el colegio mueve la probabilidad "
                        f"<b>~{gap:.0f} pts</b>: el contexto predice el puntaje → <b>determinante estructural</b>.</div>",
                        unsafe_allow_html=True)
tab_res, tab_car, tab3, tab_comp, tab4 = st.tabs(
    ["🎯 Mi resultado", "📋 La carrera", "🔎 ¿Dónde quedo?", "⚖️ Comparar", "🗺️ Mapa"])

with tab_res:
    render_resultado()

with tab_car:
    st.markdown(f"#### {carrera_sel.title()} · {str(row['UNIV_U']).title()}")
    of = art.oferta.get(str(cod))                         # ficha institucional SIES (nivel/jornada/duración/sede)
    if of:
        st.markdown(ficha_oferta_html(of), unsafe_allow_html=True)
    cL, cR = st.columns([1, 1])
    with cL:
        _canio = (st_info or {}).get("anio", 2026)         # año del corte/ingreso de referencia
        corte_txt = f"{st_info['corte']:.0f}" if st_info else "s/d"
        cupos_txt = f"{st_info['cupos']}" if st_info else "s/d"
        vac_txt = f"{int(vac_total)}" if vac_total > 0 else "s/d"
        st.markdown(f"<div class='stats'><div class='stat'><div class='v'>{corte_txt}</div><div class='l'>Corte {_canio}</div></div>"
                    f"<div class='stat'><div class='v'>{cupos_txt}</div><div class='l'>Ingresaron {_canio}</div></div>"
                    f"<div class='stat'><div class='v'>{vac_txt}</div><div class='l'>Vacantes 2026</div></div></div>",
                    unsafe_allow_html=True)
        desg = ([f"{int(v1)} (1er sem)"] if v1 > 0 else []) + ([f"{int(v2)} (2º sem)"] if v2 > 0 else []) \
            + ([f"{int(vac_esp)} (admisión especial: PACE/otros)"] if vac_esp > 0 else [])
        st.caption(f"📍 {row['reg_nom']} · código {cod}"
                   + (" · 🗓️ vacantes = " + " + ".join(desg) if len(desg) > 1 else ""))
        if st_info is None:
            st.markdown("<div class='warn'>⚠️ Carrera sin corte reciente (nueva/sin datos): mayor incertidumbre.</div>",
                        unsafe_allow_html=True)
        mtr = art.matricula.get(str(cod))
        historia = historia_carrera(mtr, v2, vac_total, vac_esp) if mtr else ""
        if historia:
            st.markdown(historia, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px'><b style='color:#1e3a8a'>⚖️ Ponderación por prueba (%)</b><br>"
                    "<span style='color:#64748b;font-size:.82rem'>en azul, las 4 obligatorias · Historia/Ciencias es electivo (cuenta el mejor)</span></div>"
                    + ponderaciones_html(row), unsafe_allow_html=True)
    with cR:
        st.plotly_chart(fig_radar(row), use_container_width=True, key="radar_top")
        _ch = art.cortes_hist.get(str(cod))
        if _ch:
            _ftrend = fig_corte_trend(_ch)
            if _ftrend is not None:
                st.plotly_chart(_ftrend, use_container_width=True, key="corte_trend")
    if of and of.get("tes"):                              # origen escolar de los matriculados (TES, SIES)
        st.markdown("<div class='sec'><h3>🏫 ¿De qué colegios vienen sus matriculados?</h3></div>", unsafe_allow_html=True)
        oc1, oc2 = st.columns([1.3, 1])
        oc1.plotly_chart(fig_tes(of["tes"], of.get("tes_n", 0)), use_container_width=True, key="tes_bar")
        with oc2:
            st.markdown("<br>", unsafe_allow_html=True)
            t = of["tes"]
            muni_sub = t.get("municipal", 0) + t.get("part_subv", 0) + t.get("sle", 0) + t.get("corp_ad", 0)
            _nt, _np = of.get("n_total"), of.get("n_primer")
            _mt = (f"<b>{_nt:,}</b> matriculados (todos los años de la carrera)".replace(",", ".")
                   + (f" · <b>{_np:,}</b> ingreso de 1er año".replace(",", ".") if _np else "")) if _nt else ""
            st.markdown(f"<div class='nota'>De la matrícula de <b>{str(row['UNIV_U']).title()}</b> en esta carrera "
                        f"(SIES {of.get('anio')}), <b>{muni_sub:.0f}%</b> viene de colegios <b>públicos o "
                        f"subvencionados</b> y <b>{t.get('part_pagado', 0):.0f}%</b> de <b>particulares pagados</b>."
                        + (f"<br><span style='font-size:.85em;color:#475569'>{_mt}</span>" if _mt else "") + "</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='warn'>💲 <b>Arancel:</b> no disponible en los datos DEMRE/SIES de este proyecto.</div>",
                        unsafe_allow_html=True)
        st.caption("El desglose por **establecimiento de origen** es sobre la **matrícula total** de la carrera (todas las "
                   "cohortes que siguen estudiando), no solo la cohorte que entró el último año: el SIES no publica este "
                   "desglose separado para el ingreso de 1er año. Es contexto socioeconómico, no predicción.")
    _tnorm = match_titulacion(carrera_sel, art.titulacion.get("por_carrera", {}))
    _tt = art.titulacion.get("por_carrera", {}).get(_tnorm) if _tnorm else None
    _tlabel = "Todas las universidades"
    if not _tt:
        _tarea = area_de(carrera_sel)
        _tt = art.titulacion.get("por_area", {}).get(_tarea)
        _tlabel = f"Tu área: {_tarea}" if _tarea else None
    _tu = art.titulacion.get("por_carrera_inst", {}).get(_tnorm, {}).get(str(row["UNIV_U"])) if _tt else None
    if _tt and _tlabel:
        st.markdown("<div class='sec'><h3>🎓 Titulación de la carrera (SIES 2024)</h3></div>", unsafe_allow_html=True)
        _filas = [tit_fila(_tlabel, _tt)] + ([tit_fila(str(row["UNIV_U"]).title(), _tu)] if _tu else [])
        st.dataframe(pd.DataFrame(_filas), hide_index=True, width="stretch", column_config={
            "Titulados": st.column_config.NumberColumn(format="%d"),
            "% Mujeres": st.column_config.NumberColumn(format="%.0f%%"),
            "% Hombres": st.column_config.NumberColumn(format="%.0f%%"),
            "Edad prom.": st.column_config.NumberColumn(format="%.0f años", help="Promedio (lo infla la cola de titulados mayores)"),
            "Edad mediana": st.column_config.NumberColumn(format="%.0f años", help="Más representativa")})
        _pcols = st.columns(2 if _tu else 1)
        _pcols[0].plotly_chart(fig_pie_genero(_tt, _tlabel), use_container_width=True, key="pie_tot")
        if _tu:
            _pcols[1].plotly_chart(fig_pie_genero(_tu, str(row["UNIV_U"]).title()), use_container_width=True, key="pie_uni")
        st.caption("💡 La **mediana** es más representativa que el promedio (la cola de titulados mayores lo infla). SIES, agregado nacional.")

with tab3:
    st.caption("Te muestro **dónde tienes más chance de quedar**: la misma carrera en todas las "
               "universidades, y **carreras afines de tu área**. Probabilidades **calibradas**. "
               + ("Usando tus **puntajes PAES**." if es_post else "Usando tus **notas** (antes de la PAES; más incierto)."))
    perfil_rec = perfil_exp
    o1, o2 = st.columns([2, 1])
    orden = o1.radio("Ordenar por", ["🏅 Lo mejor que alcanzo", "🎯 Más probable"], horizontal=True, key="r_orden")
    modo_orden = "alcanzo" if orden.startswith("🏅") else "prob"
    solo_reg = o2.checkbox(f"Solo en {L['region'].get(region, region)}", value=False, key="r_reg")

    area_sel = area_de(carrera_sel)
    sub_misma = cat[cat["CARRERA_U"] == carrera_sel]
    sub_area = cat[(cat["area"] == area_sel) & (cat["CARRERA_U"] != carrera_sel)] if area_sel else cat.iloc[0:0]
    if solo_reg:
        rint = int(region)
        sub_misma = sub_misma[sub_misma["REGION_CASA_MATRIZ"].astype("Int64") == rint]
        sub_area = sub_area[sub_area["REGION_CASA_MATRIZ"].astype("Int64") == rint]

    st.markdown(f"<div class='sec'><h3>📍 {carrera_sel.title()} — dónde tienes más chance</h3></div>", unsafe_allow_html=True)
    r_misma = rankear(art, perfil_rec, sub_misma["CODIGO_CARRERA"].tolist(), modo_modelo)
    if r_misma:
        mostrar_tabla(tabla_rank(r_misma, cat_idx, incluir_carrera=False, incluir_margen=es_post, modo_orden=modo_orden))
    else:
        st.info("No hay universidades para mostrar con ese filtro.")

    if area_sel:
        st.markdown(f"<div class='sec'><h3>🧭 Otras carreras de tu área: {area_sel}</h3></div>", unsafe_allow_html=True)
        r_area = rankear(art, perfil_rec, sub_area["CODIGO_CARRERA"].tolist(), modo_modelo)
        if r_area:
            mostrar_tabla(tabla_rank(r_area, cat_idx, incluir_carrera=True, incluir_margen=es_post, modo_orden=modo_orden))
        else:
            st.info("No hay carreras afines para mostrar con ese filtro.")
    else:
        st.caption("No pude clasificar el área de esta carrera; muestro solo la misma carrera en otras universidades.")
    st.caption("🏅 *Lo mejor que alcanzo* = entre las que tienes ≥50% de probabilidad, ordenadas de más a menos "
               "selectiva. 🆕 = carrera nueva (sin corte histórico). La tabla es ordenable por cualquier columna.")

with tab_comp:
    st.caption("Compara **2 o 3 programas** lado a lado para decidir entre tus finalistas: "
               "probabilidad de acceso, corte, tu margen, vacantes, matrícula efectiva y ponderaciones.")
    sel_comp = st.multiselect("Programas a comparar (elige 2 o 3)", cat["comp_label"].tolist(),
                              default=[row["comp_label"]], max_selections=3, key="comp_sel")
    st.caption("Con tus **puntajes PAES** de la barra lateral." if es_post
               else "🔮 Con tus **notas** de la barra lateral (antes de la PAES).")
    c_post = es_post
    perfil_comp = perfil_exp

    if len(sel_comp) < 2:
        st.info("Elige al menos **2 programas** para comparar (puedes buscar otras carreras/universidades en el selector de arriba).")
    else:
        cols = st.columns(len(sel_comp))
        rows_comp = []
        for col, disp in zip(cols, sel_comp):
            rc = cat[cat["comp_label"] == disp].iloc[0]
            rows_comp.append(rc)
            codc = int(rc["CODIGO_CARRERA"])
            res = predecir(art, replace(perfil_comp, cod_carrera=codc))
            p = res["p_post"] if c_post else res["p_pre"]
            with col:
                st.markdown(f"<div style='min-height:48px'><b style='color:#1e3a8a'>{str(rc['NOMBRE_CARRERA']).title()}</b><br>"
                            f"<span style='color:#64748b;font-size:.82rem'>{str(rc['UNIV_U']).title()}</span></div>", unsafe_allow_html=True)
                if p is not None:
                    st.plotly_chart(gauge(p, "Prob. de acceso"), use_container_width=True, key=f"cg_{codc}")
                else:
                    st.info("Faltan datos para estimar.")
                _canio_c = (art.stats.get(str(codc)) or {}).get("anio", 2026)
                st.metric(f"Corte {_canio_c}", f"{res['corte']:.0f}" if res["corte"] else "s/d")
                if c_post and res["margen"] is not None and res["margen"] == res["margen"]:
                    st.metric("Tu margen", f"{res['margen']:+.0f}")
                st.metric("Vacantes 2026", f"{int(vac_total_de(rc))}" if vac_total_de(rc) else "s/d")
                mtr_c = art.matricula.get(str(codc))
                if mtr_c and mtr_c.get("tasa") is not None:
                    st.metric("Matríc. efectiva", f"{mtr_c['tasa']:.0%}")
                st.caption(f"📍 {rc['reg_nom']} · {rc['area'] or 'área s/c'}")
        st.plotly_chart(fig_radar_multi(rows_comp, [str(r["UNIV_U"]).title() for r in rows_comp]),
                        use_container_width=True, key="comp_radar")

with tab4:
    metrica = st.radio("Colorear el mapa por", ["🎯 Tasa de acceso (región)", "📈 Puntaje PAES (comuna)"],
                       horizontal=True, key="mapa_metrica")
    es_puntaje = metrica.startswith("📈")
    geo, cent = get_geo()
    if es_puntaje:
        st.caption("**Puntaje PAES promedio (CLEC+M1)/2 por comuna** — revela la **brecha territorial**: el origen "
                   "condiciona el puntaje (origen → puntaje → acceso). Tu comuna va con borde azul 🔵.")
    else:
        st.caption("**Tasa histórica de acceso a 1ª preferencia** por región. Tu **región** en naranjo 🟠 y tu "
                   "**comuna** en rojo 🔴.")
    mc1, mc2 = st.columns([1, 1.15])
    with mc1:
        if es_puntaje:
            st.plotly_chart(fig_mapa_puntaje(art.territorio, geo, cent, comuna, L), use_container_width=True, key="mapa_pje")
        else:
            st.plotly_chart(fig_mapa(art.territorio, geo, cent, region, comuna, L), use_container_width=True, key="mapa")
    with mc2:
        st.plotly_chart(fig_barras_region(art.territorio, region, L), use_container_width=True, key="barras_reg")
    tr_reg = art.territorio["region"].get(str(region))
    tr_com = art.territorio["comuna"].get(str(comuna))
    cols = st.columns(3)
    if es_puntaje and tr_com and tr_com.get("puntaje"):
        cols[0].metric(f"🔵 {L['comuna'].get(comuna, comuna)} (puntaje)", f"{tr_com['puntaje']:.0f}")
        pjes = [s["puntaje"] for s in art.territorio["comuna"].values() if s.get("puntaje")]
        cols[1].metric("📊 Mediana nacional (comunas)", f"{sorted(pjes)[len(pjes)//2]:.0f}")
        cols[2].metric("↔️ Brecha máx-mín comuna", f"{max(pjes)-min(pjes):.0f} pts")
    else:
        if tr_reg:
            cols[0].metric(f"🟠 {L['region'].get(region, region).replace('Region ', '')}", f"{tr_reg['tasa']:.0%}", help=f"n={tr_reg['n']:,}")
        if tr_com:
            cols[1].metric(f"🔴 {L['comuna'].get(comuna, comuna)}", f"{tr_com['tasa']:.0%}", help=f"n={tr_com['n']:,}")
        prom = sum(s["tasa"] * s["n"] for s in art.territorio["region"].values()) / sum(s["n"] for s in art.territorio["region"].values())
        cols[2].metric("📊 Promedio nacional", f"{prom:.0%}")

# ----------------------------------------------------------------- info modelos
with st.expander("ℹ️ Sobre los modelos y los datos"):
    mt, mp = art.meta_post["metrics_temporal"], art.meta_pre["metrics_temporal"]
    sc = art.meta_score["metrics"]
    st.markdown(f"""
**Validación temporal (entrena 2025 → testea 2026):**
- Acceso POST-PAES — AUC **{mt['auc_roc']:.3f}** · Acceso PRE-PAES — AUC **{mp['auc_roc']:.3f}**
- Puntaje probable por prueba (cuantiles): cobertura P10–P90 entre **{min(v['cobertura_p10_p90'] for v in sc.values()):.0%} y {max(v['cobertura_p10_p90'] for v in sc.values()):.0%}** (objetivo 80%)

**Ficha de la carrera (descriptivo, SIES, matrícula):** nivel (técnico/profesional), jornada, duración
formal, región/comuna de la sede y composición de matriculados por **establecimiento de origen** (municipal,
particular subvencionado/pagado, etc.). Se cruza la oferta DEMRE con la matrícula SIES por institución +
carrera + región (los códigos difieren entre sistemas). **El arancel no está** en estos datos. Contexto, no predicción.

**Titulación (descriptivo, SIES 2024):** % de mujeres y edad promedio de titulación por carrera/área,
desde el archivo crudo de titulados del SIES (agregado nacional, no individual). Se asigna por nombre de
carrera genérica con respaldo por área. Es contexto ("¿cómo es titularse de esto?"), no predicción.

**Matrícula efectiva (descriptivo, no es target):** cruzando admisión ↔ matrícula 2026 por estudiante
(`ID_aux`), muestra qué % de los **seleccionados** en 1ª preferencia **efectivamente se matriculó** en esa
carrera. El modelo predice la *selección* (hacer match con la carrera); matricularse o no es decisión del
postulante, por eso esta cifra se reporta como contexto, no como variable objetivo. Los matriculados totales
pueden superar a los seleccionados (ingresos vía lista de espera u otras preferencias).

**Cómo se predice el puntaje (percentiles, en simple):** no es una fórmula. Para cada prueba entrenamos un
modelo de *regresión por cuantiles* (gradient boosting) que aprende, a partir de **notas + contexto** (y el
**historial PAES de tu colegio** si lo indicas, vía el Directorio MINEDUC), no un único número sino **tres
percentiles** de lo que sacan estudiantes parecidos:
- **P10** = solo el 10% saca *menos* (escenario bajo)
- **P50 / mediana** = la mitad saca menos y la mitad más (lo típico)
- **P90** = solo el 10% saca *más* (escenario alto)

Así, entre P10 y P90 cae el **80% de los estudiantes con tu perfil**. Validamos que sea cierto: en 2026, ~80%
de los puntajes reales cayeron dentro de su banda. Probabilidades de acceso **calibradas**.

**Limitaciones:** género no disponible a nivel individuo en los datos DEMRE; las "carreras repetidas" son
códigos distintos sin sede/jornada en la oferta (se distinguen por región, código y corte).
Target: seleccionado/a (`ESTADO_PREF=24`) en 1ª preferencia, modalidad regular.
""")
