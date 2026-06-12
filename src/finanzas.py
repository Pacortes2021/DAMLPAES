"""
finanzas.py — Retorno económico de la carrera (ROI educativo). DAML 2026 · Grupo 5.

Estima el retorno de la inversión en educación superior a partir de:
  - empleabilidad al 1er año (probabilidad de estar trabajando),
  - ingreso mensual mediano al 4° año de titulación,
  - arancel anual y duración típica de la carrera.

Las cifras son **referenciales agregadas** del Servicio de Información de Educación
Superior (SIES — mifuturo.cl, reportes "Empleabilidad e Ingresos"). No son por programa
exacto: se asignan por familia de carrera / área de conocimiento mediante palabras clave.
Sirven para una estimación de orden de magnitud en el MVP, no para asesoría financiera.

Sin dependencia de Streamlit: lógica pura, reusable y testeable.
"""
from __future__ import annotations
import unicodedata
from dataclasses import dataclass

# Supuestos macro (referenciales, Chile ~2024). Editables desde la UI.
SUELDO_SIN_TITULO = 550_000      # ingreso mensual mediano de un trabajador sin educación superior
MESES = 12                        # meses trabajados al año (sin gratificaciones, conservador)


# --------------------------------------------------------------------------- perfiles
# Cada perfil: empleabilidad (% 1er año) · ingreso (mediana mensual bruta 4° año, CLP)
#              arancel (anual, CLP) · duracion (años)
PERFILES: dict[str, dict] = {
    # — Salud —
    "medicina":      {"area": "Salud", "empleabilidad": 97, "ingreso": 2_900_000, "arancel": 7_200_000, "duracion": 7},
    "odontologia":   {"area": "Salud", "empleabilidad": 83, "ingreso": 1_500_000, "arancel": 6_500_000, "duracion": 6},
    "enfermeria":    {"area": "Salud", "empleabilidad": 90, "ingreso": 1_400_000, "arancel": 4_200_000, "duracion": 5},
    "obstetricia":   {"area": "Salud", "empleabilidad": 88, "ingreso": 1_450_000, "arancel": 4_200_000, "duracion": 5},
    "kinesiologia":  {"area": "Salud", "empleabilidad": 74, "ingreso": 1_150_000, "arancel": 4_300_000, "duracion": 5},
    "nutricion":     {"area": "Salud", "empleabilidad": 66, "ingreso":   950_000, "arancel": 3_900_000, "duracion": 5},
    "tecnologia_med":{"area": "Salud", "empleabilidad": 80, "ingreso": 1_300_000, "arancel": 4_200_000, "duracion": 5},
    "fonoaudiologia":{"area": "Salud", "empleabilidad": 70, "ingreso": 1_100_000, "arancel": 4_100_000, "duracion": 5},
    "quim_farmacia": {"area": "Salud", "empleabilidad": 85, "ingreso": 1_600_000, "arancel": 5_000_000, "duracion": 6},
    "veterinaria":   {"area": "Agropecuaria", "empleabilidad": 70, "ingreso": 1_050_000, "arancel": 4_800_000, "duracion": 6},
    # — Tecnología / Ingeniería —
    "ing_civil_comp":{"area": "Tecnología", "empleabilidad": 88, "ingreso": 2_000_000, "arancel": 4_900_000, "duracion": 6},
    "ing_civil_ind": {"area": "Tecnología", "empleabilidad": 85, "ingreso": 1_900_000, "arancel": 4_900_000, "duracion": 6},
    "ing_civil":     {"area": "Tecnología", "empleabilidad": 83, "ingreso": 1_750_000, "arancel": 4_800_000, "duracion": 6},
    "ing_comercial": {"area": "Administración y Comercio", "empleabilidad": 82, "ingreso": 1_650_000, "arancel": 4_500_000, "duracion": 5},
    "ing_ejecucion": {"area": "Tecnología", "empleabilidad": 78, "ingreso": 1_300_000, "arancel": 4_000_000, "duracion": 5},
    "geologia":      {"area": "Ciencias Básicas", "empleabilidad": 78, "ingreso": 1_900_000, "arancel": 4_800_000, "duracion": 6},
    "construccion":  {"area": "Tecnología", "empleabilidad": 80, "ingreso": 1_350_000, "arancel": 4_000_000, "duracion": 5},
    # — Derecho / Adm / Cs. Sociales —
    "derecho":       {"area": "Derecho", "empleabilidad": 72, "ingreso": 1_250_000, "arancel": 4_200_000, "duracion": 5},
    "contador":      {"area": "Administración y Comercio", "empleabilidad": 80, "ingreso": 1_200_000, "arancel": 3_800_000, "duracion": 5},
    "psicologia":    {"area": "Ciencias Sociales", "empleabilidad": 70, "ingreso": 1_000_000, "arancel": 3_900_000, "duracion": 5},
    "trabajo_social":{"area": "Ciencias Sociales", "empleabilidad": 75, "ingreso": 1_000_000, "arancel": 3_400_000, "duracion": 5},
    "sociologia":    {"area": "Ciencias Sociales", "empleabilidad": 62, "ingreso":   950_000, "arancel": 3_600_000, "duracion": 5},
    "periodismo":    {"area": "Ciencias Sociales", "empleabilidad": 58, "ingreso":   900_000, "arancel": 3_900_000, "duracion": 5},
    # — Educación —
    "pedagogia":     {"area": "Educación", "empleabilidad": 80, "ingreso": 1_100_000, "arancel": 3_200_000, "duracion": 5},
    "parvularia":    {"area": "Educación", "empleabilidad": 78, "ingreso":   950_000, "arancel": 3_000_000, "duracion": 5},
    # — Arte y Arquitectura / Humanidades —
    "arquitectura":  {"area": "Arte y Arquitectura", "empleabilidad": 62, "ingreso": 1_100_000, "arancel": 4_600_000, "duracion": 6},
    "diseno":        {"area": "Arte y Arquitectura", "empleabilidad": 60, "ingreso":   900_000, "arancel": 3_900_000, "duracion": 5},
    "arte":          {"area": "Arte y Arquitectura", "empleabilidad": 55, "ingreso":   800_000, "arancel": 3_800_000, "duracion": 5},
    # — Agro / Ciencias —
    "agronomia":     {"area": "Agropecuaria", "empleabilidad": 72, "ingreso": 1_150_000, "arancel": 4_200_000, "duracion": 5},
    "ciencias":      {"area": "Ciencias Básicas", "empleabilidad": 60, "ingreso": 1_000_000, "arancel": 4_000_000, "duracion": 5},
    "terapia_ocup":  {"area": "Salud", "empleabilidad": 68, "ingreso": 1_050_000, "arancel": 3_900_000, "duracion": 5},
    "publicidad":    {"area": "Administración y Comercio", "empleabilidad": 62, "ingreso":   950_000, "arancel": 3_800_000, "duracion": 5},
    "humanidades":   {"area": "Humanidades", "empleabilidad": 56, "ingreso":   820_000, "arancel": 3_400_000, "duracion": 5},
}

# Promedios por área (fallback cuando no hay match específico). Fuente: SIES, agregados por área.
AREAS: dict[str, dict] = {
    "Administración y Comercio": {"empleabilidad": 76, "ingreso": 1_100_000, "arancel": 3_800_000, "duracion": 5},
    "Agropecuaria":              {"empleabilidad": 72, "ingreso": 1_100_000, "arancel": 4_000_000, "duracion": 5},
    "Arte y Arquitectura":       {"empleabilidad": 58, "ingreso":   900_000, "arancel": 4_000_000, "duracion": 5},
    "Ciencias Básicas":          {"empleabilidad": 62, "ingreso": 1_050_000, "arancel": 4_000_000, "duracion": 5},
    "Ciencias Sociales":         {"empleabilidad": 66, "ingreso": 1_000_000, "arancel": 3_700_000, "duracion": 5},
    "Derecho":                   {"empleabilidad": 72, "ingreso": 1_200_000, "arancel": 4_100_000, "duracion": 5},
    "Educación":                 {"empleabilidad": 80, "ingreso": 1_080_000, "arancel": 3_200_000, "duracion": 5},
    "Humanidades":               {"empleabilidad": 58, "ingreso":   850_000, "arancel": 3_500_000, "duracion": 5},
    "Salud":                     {"empleabilidad": 80, "ingreso": 1_300_000, "arancel": 4_300_000, "duracion": 5},
    "Tecnología":                {"empleabilidad": 78, "ingreso": 1_450_000, "arancel": 4_400_000, "duracion": 5},
}
DEFAULT = {"area": "General", "empleabilidad": 68, "ingreso": 1_050_000, "arancel": 3_900_000, "duracion": 5}

# Reglas de matching (orden = prioridad; el primero que calza gana). Se evalúan sobre el
# nombre NORMALIZADO (mayúsculas, sin tildes). Las más específicas/excluyentes van primero.
_REGLAS: list[tuple[list[str], str]] = [
    # — Salud (específicas; veterinaria y tecnología médica antes que "MEDICINA") —
    (["VETERINAR"], "veterinaria"),
    (["TECNOLOGIA MEDIC", "TECNOLOGO MEDIC"], "tecnologia_med"),
    (["TERAPIA OCUPACIONAL"], "terapia_ocup"),
    (["MEDICINA"], "medicina"),
    (["ODONTOLOG"], "odontologia"),
    (["ENFERMER"], "enfermeria"),
    (["OBSTETRICIA", "MATRON"], "obstetricia"),
    (["KINESIOLOG"], "kinesiologia"),
    (["NUTRICION", "DIETETICA"], "nutricion"),
    (["FONOAUDIOLOG"], "fonoaudiologia"),
    (["QUIMICA Y FARMACIA", "QUIMICO FARMAC", "FARMACIA"], "quim_farmacia"),
    # — Tecnología / Ingeniería (menciones específicas antes que la genérica) —
    (["ICA EN COMPUTAC", "CIVIL EN COMPUTAC", "CIVIL INFORMAT", "CIVIL EN INFORMAT",
      "INGENIERIA EN COMPUTAC", "INGENIERIA INFORMAT", "CIENCIA DE LA COMPUTAC",
      "COMPUTER SCIENCE", "DESARROLLO DE VIDEOJUEGO", "VIDEOJUEGOS Y SIMULACION"], "ing_civil_comp"),
    (["CIVIL INDUSTRIAL"], "ing_civil_ind"),
    (["INGENIERIA COMERCIAL"], "ing_comercial"),
    (["INGENIERIA CIVIL", "ING. CIVIL", "ING CIVIL"], "ing_civil"),
    (["GEOLOG"], "geologia"),
    (["CONSTRUCCION CIVIL", "INGENIERIA EN CONSTRUCCION", "INGENIERIA CONSTRUCCION",
      "TECNOLOGIA EN CONSTRUCC"], "construccion"),
    (["INGENIERIA DE EJECUCION", "INGENIERIA EN EJECUCION", "BIOINGENIER",
      "INGENIERIA", "PLAN COMUN"], "ing_ejecucion"),   # catch-all de ingenierías (civil/comercial ya filtradas)
    # — Derecho / Negocios / Administración —
    (["DERECHO"], "derecho"),
    (["CONTADOR", "AUDITORIA", "AUDITOR"], "contador"),
    (["PUBLICIDAD", "MARKETING"], "publicidad"),
    (["NEGOCIOS", "BUSINESS", "INTERNATIONAL MANAGEMENT", "CONTROL DE GESTION"], "ing_comercial"),
    (["TURISMO", "HOTELER", "GASTRONOM"], "publicidad"),
    (["ADMINISTRACION", "GESTION DE PERSONAS", "GESTION PUBLICA", "GESTION Y "], "contador"),
    # — Ciencias Sociales —
    (["PSICOLOG"], "psicologia"),
    (["TRABAJO SOCIAL", "SERVICIO SOCIAL"], "trabajo_social"),
    (["SOCIOLOG"], "sociologia"),
    (["PERIODISMO", "COMUNICACION"], "periodismo"),
    # — Educación (antes que Cs. Sociales para que "PEDAGOGÍA EN HISTORIA" sea pedagogía;
    #   parvularia antes que la genérica; diferencial/básica no llevan la palabra "PEDAGOGIA") —
    (["PARVULAR"], "parvularia"),
    (["PEDAGOGIA", "PROFESOR", "EDUCACION DIFERENCIAL", "EDUCACION BASICA", "ED DIFERENCIAL",
      "EDUCACION GENERAL BASICA", "EDUCACION FISICA", "EDUCACION DE ", "EDUCACION EN ",
      "DEPORT", "ENTRENADOR"], "pedagogia"),
    (["ARQUEOLOG", "ANTROPOLOG", "BIBLIOTECOLOG", "GESTION DE INFORMACION", "ARCHIVISTICA",
      "ESTUDIOS INTERNACIONALES", "INTERNATIONAL STUDIES", "CIENCIA POLITICA", "GEOGRAFIA",
      "HISTORIA", "CIENCIAS SOCIALES", "COLLEGE"], "psicologia"),
    # — Arte y Arquitectura —
    (["ARQUITECTURA"], "arquitectura"),
    (["DISENO"], "diseno"),
    (["ACTUACION", "TEATRAL", "TEATRO", "ARTES VISUALES", "ARTES MUSICAL", "MUSICA", "ARTE", "CINE",
      "DANZA", "ANIMACION", "AUDIOVISUAL", "ILUSTRACION", "FOTOGRAFIA", "OFICIOS CREATIVOS",
      "DIBUJANTE", "MULTIMEDIA", "CREACION"], "arte"),
    # — Humanidades / Idiomas —
    (["TRADUCCION", "INTERPRETACION", "INTERPRETE", "LINGUISTICA", "LITERATURA", "LETRAS",
      "FILOSOFIA", "TEOLOGIA", "HUMANIDADES", "IDIOMAS", "INGLES", "HISPANIC"], "humanidades"),
    # — Agro / Ciencias —
    (["AGRONOM"], "agronomia"),
    (["TECNOLOGIA EN", "TECNOLOGO EN", "AUTOMATIZACION", "MANTENIMIENTO INDUSTRIAL",
      "PROCESOS PRODUCTIVOS"], "ing_ejecucion"),
    (["BIOQUIMICA", "BIOLOGIA", "QUIMICA", "QUIMICO", "ANALISTA QUIMICO", "FISICA", "MATEMATICA",
      "ASTRONOM", "BIOTECNOLOG", "ECOLOGIA", "ESTADISTICA", "CIENCIA DE DATOS",
      "BACHILLERATO EN CIENCIAS", "LICENCIATURA EN CIENCIAS", "BACHILLER EN CIENCIAS"], "ciencias"),
]


def _norm(s: str) -> str:
    """Mayúsculas sin tildes para matching robusto."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.upper().strip()


@dataclass
class PerfilFinanciero:
    nombre: str
    area: str
    empleabilidad: float       # %
    ingreso: float             # CLP/mes (mediana 4° año)
    arancel: float             # CLP/año
    duracion: float            # años
    match: str                 # "carrera" | "area" | "default"


def perfil_financiero(nombre_carrera: str, area_hint: str | None = None) -> PerfilFinanciero:
    """Asigna el perfil financiero a partir del nombre de la carrera (palabras clave)."""
    n = _norm(nombre_carrera)
    for claves, key in _REGLAS:
        if any(c in n for c in claves):
            p = PERFILES[key]
            return PerfilFinanciero(nombre_carrera, p["area"], p["empleabilidad"],
                                    p["ingreso"], p["arancel"], p["duracion"], "carrera")
    if area_hint and area_hint in AREAS:
        p = AREAS[area_hint]
        return PerfilFinanciero(nombre_carrera, area_hint, p["empleabilidad"],
                                p["ingreso"], p["arancel"], p["duracion"], "area")
    return PerfilFinanciero(nombre_carrera, DEFAULT["area"], DEFAULT["empleabilidad"],
                            DEFAULT["ingreso"], DEFAULT["arancel"], DEFAULT["duracion"], "default")


def indicadores(perfil: PerfilFinanciero, *, arancel: float | None = None,
                gratuidad: bool = False, sueldo_base: float = SUELDO_SIN_TITULO,
                anios_proyeccion: int = 20, tasa_descuento: float = 0.06) -> dict:
    """Calcula indicadores de retorno de la inversión educativa.

    - inversion_arancel  : arancel desembolsado en toda la carrera (0 con gratuidad).
    - costo_oportunidad  : ingreso que se deja de percibir mientras se estudia.
    - premium_anual      : ganancia anual extra vs. no estudiar (ajustada por empleabilidad).
    - payback_anios      : años de trabajo para recuperar el arancel desembolsado.
    - roi_10             : retorno sobre el arancel a 10 años de egresado (%).
    - van                : valor actual neto del proyecto educativo (flujos descontados).
    - flujo              : serie de flujo de caja acumulado (para graficar el break-even).
    """
    ar = 0.0 if gratuidad else (perfil.arancel if arancel is None else float(arancel))
    dur = perfil.duracion
    empl = perfil.empleabilidad / 100.0

    inversion_arancel = ar * dur
    costo_oportunidad = sueldo_base * MESES * dur
    ingreso_anual_titulado = perfil.ingreso * MESES
    ingreso_anual_esperado = ingreso_anual_titulado * empl           # ajustado por prob. de empleo
    premium_anual = ingreso_anual_esperado - sueldo_base * MESES     # ganancia vs no estudiar

    payback = (inversion_arancel / premium_anual) if premium_anual > 0 else float("inf")
    roi_10 = ((premium_anual * 10 - inversion_arancel) / inversion_arancel * 100
              if inversion_arancel > 0 else float("inf"))

    # Flujo de caja acumulado: años de estudio (pago arancel) → años de trabajo (premium).
    serie_t, serie_acum = [], []
    acum = 0.0
    for t in range(0, int(dur) + anios_proyeccion + 1):
        if t < dur:
            acum -= ar                       # estudiando: desembolsa arancel
        else:
            acum += premium_anual            # trabajando: gana el premium
        serie_t.append(t); serie_acum.append(acum)

    # VAN: arancel descontado durante estudios + premium descontado durante vida laboral.
    van = 0.0
    for t in range(0, int(dur) + anios_proyeccion):
        flujo = -ar if t < dur else premium_anual
        van += flujo / ((1 + tasa_descuento) ** t)

    # break-even: primer año con flujo acumulado >= 0 (desde que empieza a trabajar)
    be = next((serie_t[i] for i, v in enumerate(serie_acum) if v >= 0 and serie_t[i] >= dur), None)

    return {
        "arancel_efectivo": ar,
        "inversion_arancel": inversion_arancel,
        "costo_oportunidad": costo_oportunidad,
        "inversion_total": inversion_arancel + costo_oportunidad,
        "ingreso_anual_esperado": ingreso_anual_esperado,
        "premium_anual": premium_anual,
        "payback_anios": payback,
        "roi_10": roi_10,
        "van": van,
        "break_even_anio": be,
        "flujo_t": serie_t,
        "flujo_acum": serie_acum,
    }
