"""
Tests de los módulos del dashboard. DAML 2026 · Grupo 5.

Cubre: clasificador de áreas (src/areas.py), ranking de carreras (inference.rankear)
y la validez de los artefactos descriptivos (matrícula efectiva, titulación, cortes).

Ejecutar:  python3 tests/test_modulos.py     (o:  pytest tests/test_modulos.py)
"""
from __future__ import annotations
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
_P = lambda *a: os.path.join(ROOT, *a)

from areas import area_de, _norm, AREAS                       # noqa: E402


# --------------------------------------------------------------------- áreas
def test_norm_quita_tildes():
    assert _norm("Ingeniería Civil") == "INGENIERIA CIVIL"
    assert _norm("Pedagogía en Matemática") == "PEDAGOGIA EN MATEMATICA"


def test_area_mapea_casos_clave():
    assert area_de("MEDICINA VETERINARIA") == "Agropecuaria y Veterinaria"   # antes que Salud
    assert area_de("MEDICINA") == "Salud"
    assert area_de("INGENIERIA COMERCIAL") == "Administración y Negocios"     # antes que Tecnología
    assert area_de("INGENIERIA CIVIL INDUSTRIAL") == "Ingeniería y Tecnología"
    assert area_de("DERECHO") == "Derecho"
    assert area_de("PEDAGOGIA EN HISTORIA") == "Educación"                    # antes que Cs. Sociales
    assert area_de("PROGRAMA INEXISTENTE XYZ") is None


def test_area_cobertura_catalogo():
    import pandas as pd
    cat = _P("data/processed/catalogo_carreras.parquet")
    if not os.path.exists(cat):
        return
    nombres = pd.read_parquet(cat)["NOMBRE_CARRERA"].dropna().unique()
    cob = sum(area_de(n) is not None for n in nombres) / len(nombres)
    assert cob >= 0.97, f"cobertura de áreas {cob:.0%} < 97%"


def test_areas_son_validas():
    import pandas as pd
    cat = _P("data/processed/catalogo_carreras.parquet")
    if not os.path.exists(cat):
        return
    for n in pd.read_parquet(cat)["NOMBRE_CARRERA"].dropna().unique():
        a = area_de(n)
        assert a is None or a in AREAS


# --------------------------------------------------------------------- rankear
def _artifacts():
    try:
        from inference import load_artifacts
        return load_artifacts()
    except Exception:
        return None


def test_rankear_ordena_y_valida():
    art = _artifacts()
    if art is None:
        return  # sin modelos locales (entorno sin deps): se omite
    from inference import Perfil, rankear
    import pandas as pd
    cat = art.catalogo
    cods = cat["CODIGO_CARRERA"].head(40).tolist()
    p = Perfil(cod_carrera=int(cods[0]), nem=720, ranking=740, region="13", comuna="13101",
               dependencia="3", rama="H1", clec=700, mate1=710, cien=700)
    r = rankear(art, p, cods, "post")
    assert len(r) >= 1
    ps = [d["p"] for d in r]
    assert ps == sorted(ps, reverse=True)            # ordenado por probabilidad desc
    assert all(0.0 <= d["p"] <= 1.0 for d in r)      # probabilidades válidas
    assert all("ponderado" in d and "corte" in d for d in r)


# --------------------------------------------------------------------- artefactos
def test_matricula_stats_valido():
    f = _P("data/processed/matricula_stats.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    assert len(d) > 1000
    v = next(iter(d.values()))
    assert {"n_postula", "n_sel", "tasa", "n_matric_total"} <= set(v)
    assert all(s["tasa"] is None or 0 <= s["tasa"] <= 1 for s in d.values())


def test_titulacion_stats_valido():
    f = _P("data/processed/titulacion_stats.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    assert {"por_carrera", "por_area", "por_carrera_inst"} <= set(d)
    ici = d["por_carrera"].get("INGENIERIA CIVIL INDUSTRIAL")
    if ici:
        # mediana de edad debe ser razonable y <= promedio (cola de mayores infla el promedio)
        assert 20 <= ici["edad_mediana"] <= 35
        assert ici["edad_mediana"] <= ici["edad_prom"] + 0.1
        assert 0 <= ici["pct_muj"] <= 100


def test_oferta_detalle_valido():
    f = _P("data/processed/oferta_detalle.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    assert len(d) > 1000
    for v in list(d.values())[:50]:
        if v.get("dur_sem"):                              # duración en semestres; años = sem/2
            assert 1 <= v["dur_sem"] <= 16
            assert abs(v["dur_anios"] - v["dur_sem"] / 2) < 0.06
        if v.get("tes"):                                  # composición por origen suma ~100%
            assert abs(sum(v["tes"].values()) - 100) <= 1.5
            assert all(0 <= x <= 100 for x in v["tes"].values())


def test_cortes_historicos_valido():
    f = _P("data/processed/cortes_historicos.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    assert len(d) > 1000
    for v in list(d.values())[:50]:
        for anio, corte in v.items():
            assert anio in ("2024", "2025", "2026")
            assert 100 <= corte <= 1000


# --------------------------------------------------------------------- runner
def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn(); print(f"  ✅ {fn.__name__}"); ok += 1
        except AssertionError as e:
            print(f"  ❌ {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  💥 {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(fns)} tests pasaron.")
    return ok == len(fns)


if __name__ == "__main__":
    print("Tests de módulos\n" + "=" * 40)
    sys.exit(0 if _run() else 1)
