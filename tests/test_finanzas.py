"""
Tests del módulo de retorno económico (src/finanzas.py). DAML 2026 · Grupo 5.

Cubre: normalización, prioridad del matching por palabras clave, cobertura sobre el
catálogo real, y la coherencia de los indicadores financieros (payback, ROI, VAN,
break-even, gratuidad y casos de premium negativo).

Ejecutar:
    python3 tests/test_finanzas.py          # modo script (asserts + resumen)
    pytest tests/test_finanzas.py           # si pytest está disponible
"""
from __future__ import annotations
import os, sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from finanzas import (perfil_financiero, indicadores, _norm, PERFILES, AREAS,  # noqa: E402
                      SUELDO_SIN_TITULO)


# --------------------------------------------------------------------- normalización
def test_norm_quita_tildes_y_mayusculas():
    assert _norm("Pedagogía en Matemática") == "PEDAGOGIA EN MATEMATICA"
    assert _norm("INGENIERÍA CIVIL") == "INGENIERIA CIVIL"
    assert _norm("  Diseño  ") == "DISENO"


# --------------------------------------------------------------------- matching prioridad
def test_veterinaria_no_se_confunde_con_medicina():
    assert perfil_financiero("MEDICINA VETERINARIA").area == "Agropecuaria"
    assert perfil_financiero("MEDICINA").area == "Salud"
    assert perfil_financiero("MEDICINA").empleabilidad == 97


def test_tecnologia_medica_no_es_medicina():
    p = perfil_financiero("TECNOLOGÍA MÉDICA")
    assert p.ingreso == PERFILES["tecnologia_med"]["ingreso"]


def test_ingenieria_civil_desambigua_mencion():
    assert perfil_financiero("INGENIERÍA CIVIL EN COMPUTACIÓN").ingreso == PERFILES["ing_civil_comp"]["ingreso"]
    assert perfil_financiero("INGENIERÍA CIVIL INDUSTRIAL").ingreso == PERFILES["ing_civil_ind"]["ingreso"]
    assert perfil_financiero("INGENIERÍA CIVIL MECÁNICA").ingreso == PERFILES["ing_civil"]["ingreso"]
    assert perfil_financiero("INGENIERÍA COMERCIAL").ingreso == PERFILES["ing_comercial"]["ingreso"]


def test_carrera_desconocida_cae_en_default():
    p = perfil_financiero("PROGRAMA EXÓTICO INEXISTENTE 9000")
    assert p.match == "default"
    assert p.empleabilidad == 68


def test_educacion_diferencial_y_basica_son_pedagogia():
    # No contienen la palabra "PEDAGOGIA" pero son educación.
    assert perfil_financiero("EDUCACIÓN DIFERENCIAL").area == "Educación"
    assert perfil_financiero("EDUCACIÓN BÁSICA").area == "Educación"


def test_pedagogia_en_historia_es_educacion_no_cs_sociales():
    # Orden: Educación debe vencer a la regla que captura "HISTORIA".
    assert perfil_financiero("PEDAGOGÍA EN HISTORIA").area == "Educación"
    assert perfil_financiero("LICENCIATURA EN HISTORIA").area == "Ciencias Sociales"


def test_traduccion_e_idiomas_son_humanidades():
    assert perfil_financiero("TRADUCCIÓN INGLÉS-ESPAÑOL").area == "Humanidades"
    assert perfil_financiero("LICENCIATURA EN FILOSOFÍA").area == "Humanidades"


def test_terapia_ocupacional_es_salud():
    assert perfil_financiero("TERAPIA OCUPACIONAL").area == "Salud"


def test_area_hint_se_usa_si_no_hay_match_de_carrera():
    p = perfil_financiero("ALGO RARO", area_hint="Salud")
    assert p.match == "area"
    assert p.area == "Salud"
    assert p.empleabilidad == AREAS["Salud"]["empleabilidad"]


# --------------------------------------------------------------------- cobertura catálogo
def test_cobertura_catalogo_real():
    """Al menos 85% de las carreras del catálogo deben matchear un perfil específico."""
    import pandas as pd
    cat = os.path.join(ROOT, "data/processed/catalogo_carreras.parquet")
    if not os.path.exists(cat):
        return  # sin datos locales: se omite (no falla)
    nombres = pd.read_parquet(cat)["NOMBRE_CARRERA"].dropna().unique()
    esp = sum(perfil_financiero(n).match == "carrera" for n in nombres)
    cob = esp / len(nombres)
    assert cob >= 0.95, f"cobertura {cob:.0%} < 95%"


# --------------------------------------------------------------------- indicadores
def test_indicadores_coherentes_carrera_rentable():
    ind = indicadores(perfil_financiero("MEDICINA"))
    assert ind["inversion_arancel"] > 0
    assert ind["premium_anual"] > 0                       # medicina rinde > no estudiar
    assert ind["payback_anios"] > 0 and ind["payback_anios"] < 25
    assert ind["break_even_anio"] is not None
    assert ind["van"] > 0                                  # proyecto con VAN positivo


def test_gratuidad_anula_arancel_y_dispara_roi():
    ind = indicadores(perfil_financiero("PEDAGOGÍA EN HISTORIA"), gratuidad=True)
    assert ind["arancel_efectivo"] == 0
    assert ind["inversion_arancel"] == 0
    assert ind["roi_10"] == float("inf")                  # sin costo de arancel
    assert ind["payback_anios"] == 0                      # recupera de inmediato


def test_premium_negativo_no_tiene_break_even():
    """Carrera de baja empleabilidad/ingreso: el premium puede ser negativo → payback infinito."""
    ind = indicadores(perfil_financiero("ACTUACIÓN TEATRAL"))
    if ind["premium_anual"] <= 0:
        assert ind["payback_anios"] == float("inf")
        assert ind["break_even_anio"] is None


def test_arancel_editable_cambia_inversion():
    base = perfil_financiero("DERECHO")
    barato = indicadores(base, arancel=2_000_000)
    caro = indicadores(base, arancel=6_000_000)
    assert caro["inversion_arancel"] > barato["inversion_arancel"]
    assert caro["payback_anios"] > barato["payback_anios"]   # más caro → tarda más en recuperarse


def test_flujo_acumulado_monotono_tras_titularse():
    """Después de titularse (premium>0), el flujo acumulado debe crecer monótonamente."""
    ind = indicadores(perfil_financiero("INGENIERÍA CIVIL INDUSTRIAL"))
    dur = int(PERFILES["ing_civil_ind"]["duracion"])
    tramo = ind["flujo_acum"][dur:]
    assert all(b >= a for a, b in zip(tramo, tramo[1:]))


def test_break_even_posterior_a_la_duracion():
    p = perfil_financiero("ENFERMERÍA")
    ind = indicadores(p)
    if ind["break_even_anio"] is not None:
        assert ind["break_even_anio"] >= p.duracion


# --------------------------------------------------------------------- runner script
def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"  ❌ {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  💥 {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(fns)} tests pasaron.")
    return ok == len(fns)


if __name__ == "__main__":
    print("Tests finanzas.py\n" + "=" * 40)
    sys.exit(0 if _run() else 1)
