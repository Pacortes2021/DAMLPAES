"""
05_geo_data.py — Datos geográficos para el mapa territorial. DAML 2026 · Grupo 5.

Genera (descargando una sola vez de una fuente pública) los insumos del coroplético:
  - data/geo/regiones.geojson          polígonos de las 16 regiones (prop. `codregion` = 1..16)
  - data/geo/comuna_centroides.json    {cod_comuna: {lat, lon}} para marcar la comuna del usuario

Fuente: https://github.com/caracena/chile-geojson (regiones.json + N.geojson por región).
Los polígonos de comuna (~19 MB) NO se versionan: solo se usan aquí para calcular centroides
(JSON ~15 KB). Códigos compatibles con labels.json (CUT, sin cero a la izquierda).

Uso:  python3 scripts/05_geo_data.py
"""
from __future__ import annotations
import os, json, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO = os.path.join(ROOT, "data", "geo")
os.makedirs(GEO, exist_ok=True)
BASE = "https://raw.githubusercontent.com/caracena/chile-geojson/master/{}"


def _fetch(name: str) -> dict:
    with urllib.request.urlopen(BASE.format(name), timeout=90) as r:
        return json.loads(r.read())


def _coords(geom: dict) -> list:
    out = []
    def rec(x):
        if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], (int, float)):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                rec(y)
    rec(geom["coordinates"])
    return out


print("1. Descargando polígonos de regiones...")
reg = _fetch("regiones.json")
json.dump(reg, open(os.path.join(GEO, "regiones.geojson"), "w"), ensure_ascii=False)
print(f"   ✅ regiones.geojson ({len(reg['features'])} regiones)")

print("2. Calculando centroides de comunas (descarga temporal por región)...")
cent = {}
for r in range(1, 17):
    g = _fetch(f"{r}.geojson")
    for f in g["features"]:
        pts = _coords(f["geometry"])
        if not pts:
            continue
        lon = sum(p[0] for p in pts) / len(pts)
        lat = sum(p[1] for p in pts) / len(pts)
        cent[str(f["properties"]["cod_comuna"])] = {"lat": round(lat, 4), "lon": round(lon, 4)}
json.dump(cent, open(os.path.join(GEO, "comuna_centroides.json"), "w"), ensure_ascii=False)
print(f"   ✅ comuna_centroides.json ({len(cent)} comunas)")
