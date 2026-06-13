"""
areas.py — Clasificador de carrera → área de conocimiento (por palabras clave). DAML 2026 · Grupo 5.

Se usa en el recomendador inverso para mostrar "carreras afines de tu área" y NO sugerir
programas de un área que no interesa. Matching ordenado (lo específico vence a lo genérico:
p.ej. veterinaria antes que medicina; ingeniería comercial → Negocios antes que el catch-all
de ingenierías → Tecnología). Cobertura ~100% del catálogo de admisión.
"""
from __future__ import annotations
import unicodedata

AREAS = ["Salud", "Ingeniería y Tecnología", "Ciencias Básicas", "Administración y Negocios",
         "Derecho", "Educación", "Ciencias Sociales", "Arte y Arquitectura",
         "Humanidades", "Agropecuaria y Veterinaria"]

# (palabras clave, área) — orden = prioridad; el primero que calza gana.
_REGLAS: list[tuple[list[str], str]] = [
    (["VETERINAR"], "Agropecuaria y Veterinaria"),
    (["TECNOLOGIA MEDIC", "TECNOLOGO MEDIC", "TERAPIA OCUPACIONAL", "MEDICINA", "ODONTOLOG",
      "ENFERMER", "OBSTETRICIA", "MATRON", "KINESIOLOG", "NUTRICION", "DIETETICA",
      "FONOAUDIOLOG", "QUIMICA Y FARMACIA", "QUIMICO FARMAC", "FARMACIA", "SALUD"], "Salud"),
    (["INGENIERIA COMERCIAL"], "Administración y Negocios"),
    (["ICA EN COMPUTAC", "CIVIL EN COMPUTAC", "CIVIL INFORMAT", "CIENCIA DE LA COMPUTAC",
      "COMPUTER SCIENCE", "DESARROLLO DE VIDEOJUEGO", "VIDEOJUEGOS Y SIMULACION",
      "INGENIERIA CIVIL", "ING. CIVIL", "ING CIVIL", "CIVIL INDUSTRIAL", "GEOLOG",
      "CONSTRUCCION CIVIL", "INGENIERIA EN CONSTRUCCION", "INGENIERIA CONSTRUCCION",
      "TECNOLOGIA EN CONSTRUCC", "INGENIERIA DE EJECUCION", "INGENIERIA EN EJECUCION",
      "BIOINGENIER", "INGENIERIA", "PLAN COMUN", "TECNOLOGIA EN", "TECNOLOGO EN",
      "AUTOMATIZACION", "MANTENIMIENTO INDUSTRIAL", "TELECOMUNIC", "INFORMATICA"], "Ingeniería y Tecnología"),
    (["DERECHO"], "Derecho"),
    (["CONTADOR", "AUDITORIA", "AUDITOR", "PUBLICIDAD", "MARKETING", "NEGOCIOS", "BUSINESS",
      "INTERNATIONAL MANAGEMENT", "CONTROL DE GESTION", "TURISMO", "HOTELER", "GASTRONOM",
      "ADMINISTRACION", "GESTION DE PERSONAS", "GESTION PUBLICA", "COMERCIO"], "Administración y Negocios"),
    (["PARVULAR", "PEDAGOGIA", "PROFESOR", "EDUCACION DIFERENCIAL", "EDUCACION BASICA",
      "ED DIFERENCIAL", "EDUCACION GENERAL BASICA", "EDUCACION FISICA", "EDUCACION DE ",
      "EDUCACION EN ", "DEPORT", "ENTRENADOR"], "Educación"),
    (["PSICOLOG", "TRABAJO SOCIAL", "SERVICIO SOCIAL", "SOCIOLOG", "PERIODISMO", "COMUNICACION",
      "ARQUEOLOG", "ANTROPOLOG", "BIBLIOTECOLOG", "GESTION DE INFORMACION", "ARCHIVISTICA",
      "ESTUDIOS INTERNACIONALES", "INTERNATIONAL STUDIES", "CIENCIA POLITICA", "GEOGRAFIA",
      "HISTORIA", "CIENCIAS SOCIALES", "COLLEGE"], "Ciencias Sociales"),
    (["ARQUITECTURA", "DISENO", "ACTUACION", "TEATRAL", "TEATRO", "ARTES VISUALES",
      "ARTES MUSICAL", "MUSICA", "ARTE", "CINE", "DANZA", "ANIMACION", "AUDIOVISUAL",
      "ILUSTRACION", "FOTOGRAFIA", "OFICIOS CREATIVOS", "DIBUJANTE", "MULTIMEDIA", "CREACION"], "Arte y Arquitectura"),
    (["TRADUCCION", "INTERPRETACION", "INTERPRETE", "LINGUISTICA", "LITERATURA", "LETRAS",
      "FILOSOFIA", "TEOLOGIA", "HUMANIDADES", "IDIOMAS", "INGLES", "HISPANIC"], "Humanidades"),
    (["AGRONOM", "AGROPECUAR", "FORESTAL", "ACUICULTURA", "RECURSOS NATURALES"], "Agropecuaria y Veterinaria"),
    (["BIOQUIMICA", "BIOLOGIA", "QUIMICA", "QUIMICO", "ANALISTA QUIMICO", "FISICA", "MATEMATICA",
      "ASTRONOM", "BIOTECNOLOG", "ECOLOGIA", "ESTADISTICA", "CIENCIA DE DATOS",
      "BACHILLERATO EN CIENCIAS", "LICENCIATURA EN CIENCIAS", "BACHILLER EN CIENCIAS"], "Ciencias Básicas"),
]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.upper().strip()


def area_de(nombre_carrera: str) -> str | None:
    """Área de conocimiento de la carrera (o None si no se reconoce)."""
    n = _norm(nombre_carrera)
    for claves, area in _REGLAS:
        if any(c in n for c in claves):
            return area
    return None
