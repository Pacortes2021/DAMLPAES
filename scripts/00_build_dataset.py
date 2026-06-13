"""
00_build_dataset.py — ETL limpio y documentado para el modelo de ACCESO.
DAML 2026 · Grupo 5.

Construye una tabla analítica tidy con DOS cohortes PAES (2025 y 2026) a partir de
los archivos individuales que ya existen en data/raw/, habilitando validación
TEMPORAL (entrenar 2025 → testear 2026). Ver docs/METODOLOGIA.md.

Fuentes:
  - ArchivoC_Adm2025.csv / ArchivoC_Adm2026REG.csv  → perfil académico individual
  - master_admision_2018_2026.parquet               → preferencias/estados (capa validada)
  - OfertaAcadémica_Admisión2026.csv                → catálogo de carreras (para dashboard)

Salidas:
  - data/processed/dataset_modelo_acceso.parquet    → tabla de modelado (cohortes 2025+2026)
  - data/processed/cortes_historicos.json           → cortes 2025 (margen en vivo del dashboard)
  - data/processed/catalogo_carreras.parquet        → carreras (código, nombre, universidad, vacantes)

Uso:  python3 scripts/00_build_dataset.py
"""
from __future__ import annotations
import os, json
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)

ARCHIVO_C = {
    2025: P("data/raw/PROCESO-DE-ADMISIÓN-2025-RENDICIÓN-19-01-2025T23-39-20/ArchivoC_Adm2025.csv"),
    2026: P("data/raw/ArchivoC_Adm2026REG.csv"),
}
MASTER = P("data/processed/master_admision.parquet")   # reconstruido desde crudos por 00a (trazable)
OFERTA = P("data/raw/OfertaAcadémica_Admisión2026.csv")

PAES = ["CLEC", "MATE1", "MATE2", "HCSOC", "CIEN"]          # pruebas (se coalescen 3 fuentes)
NULOS = ["NA", "", " ", "0"]                                 # '0' = sin puntaje en estos archivos

# ----------------------------------------------------------------------------- helpers de limpieza
def _num(col: str) -> pl.Expr:
    """String con coma decimal → Float64 (sin validación de rango)."""
    return pl.col(col).str.replace_all(",", ".").cast(pl.Float64, strict=False)

def _score(col: str) -> pl.Expr:
    """Puntaje PAES/NEM/Ranking: válido solo en [100, 1000]; fuera de rango → null."""
    e = _num(col)
    return pl.when((e >= 100) & (e <= 1000)).then(e).otherwise(None)

def _nota(col: str) -> pl.Expr:
    """Promedio de notas: válido en [1.0, 7.0]; fuera → null."""
    e = _num(col)
    return pl.when((e >= 1.0) & (e <= 7.0)).then(e).otherwise(None)


def build_perfil(anio: int) -> pl.DataFrame:
    """Lee y limpia ArchivoC del año `anio`, con rescate de puntajes (invierno/anterior)."""
    path = ARCHIVO_C[anio]
    lf = pl.scan_csv(path, separator=";", encoding="utf8-lossy", null_values=NULOS,
                     infer_schema_length=0)                  # todo string, casteamos nosotros
    cols = lf.collect_schema().names()

    # Coalesce por prueba: REG_ACTUAL → INV_ACTUAL → REG_ANTERIOR (rescate de puntajes)
    score_exprs = []
    for t in PAES:
        fuentes = [c for c in (f"{t}_REG_ACTUAL", f"{t}_INV_ACTUAL", f"{t}_REG_ANTERIOR") if c in cols]
        if fuentes:
            score_exprs.append(pl.coalesce([_score(c) for c in fuentes]).alias(t))
        else:
            score_exprs.append(pl.lit(None, dtype=pl.Float64).alias(t))

    out = (
        lf.select(
            pl.col("ID_aux"),
            pl.col("GRUPO_DEPENDENCIA").cast(pl.Utf8),
            pl.col("RAMA_EDUCACIONAL").cast(pl.Utf8),
            pl.col("CODIGO_REGION").cast(pl.Utf8),
            pl.col("CODIGO_COMUNA").cast(pl.Utf8),
            _nota("PROMEDIO_NOTAS").alias("PROMEDIO_NOTAS"),
            _num("PORC_SUP_NOTAS").alias("PORC_SUP_NOTAS"),
            _score("PTJE_NEM").alias("PTJE_NEM"),
            _score("PTJE_RANKING").alias("PTJE_RANKING"),
            *score_exprs,
        )
        .with_columns(
            pl.col("MATE2").is_not_null().cast(pl.Int8).alias("TIENE_MATE2"),
            pl.col("HCSOC").is_not_null().cast(pl.Int8).alias("TIENE_HCSOC"),
            pl.col("CIEN").is_not_null().cast(pl.Int8).alias("TIENE_CIEN"),
            pl.lit(anio).alias("cohorte"),
        )
        .collect()
    )
    print(f"   ArchivoC {anio}: {out.height:,} filas | "
          f"NEM nulos={out['PTJE_NEM'].null_count():,} | CLEC nulos={out['CLEC'].null_count():,}")
    return out


def cortes_y_cupos(master: pl.DataFrame, anio_previo: int) -> pl.DataFrame:
    """Corte (mín puntaje de seleccionados) y cupos (n seleccionados) del año previo, REGULAR."""
    sel = master.filter((pl.col("anio") == anio_previo) & (pl.col("ESTADO_PREF") == 24)
                        & (pl.col("TIPO_PREF") == "REGULAR"))
    return sel.group_by("COD_CARRERA_PREF").agg(
        pl.col("PTJE_PREF").min().alias("CORTE_ANTERIOR"),
        pl.len().alias("CUPOS_ANTERIOR"),
    )


def build_cohorte(master: pl.DataFrame, anio: int) -> pl.DataFrame:
    """Tabla de modelado para la cohorte `anio`: 1ª preferencia REGULAR + perfil + dificultad carrera."""
    # Población del modelo = postulaciones VÁLIDAMENTE RANKEADAS en 1ª pref regular:
    #   ESTADO_PREF 24 (seleccionado, target=1) vs 25 (lista de espera, target=0).
    # Se excluyen los estados sin ponderado (no evaluados / no cumplen requisitos), que no
    # compitieron de verdad. Filtro EXPLÍCITO aquí (antes venía oculto en el master antiguo).
    pref = (
        master.filter((pl.col("anio") == anio) & (pl.col("ORDEN_PREF") == 1)
                      & (pl.col("TIPO_PREF") == "REGULAR") & (pl.col("ESTADO_PREF").is_in([24, 25])))
        .select(
            "ID_aux", "COD_CARRERA_PREF", "PTJE_PREF",
            (pl.col("ESTADO_PREF") == 24).cast(pl.Int8).alias("ACCESO_1PREF"),
        )
    )
    perfil = build_perfil(anio)
    cortes = cortes_y_cupos(master, anio - 1)

    df = (
        pref.join(perfil, on="ID_aux", how="inner")
            .join(cortes, on="COD_CARRERA_PREF", how="left")
    )
    df = df.with_columns(pl.col("CORTE_ANTERIOR").is_null().cast(pl.Int8).alias("ES_CARRERA_NUEVA"))
    df = df.with_columns((pl.col("PTJE_PREF") - pl.col("CORTE_ANTERIOR")).alias("MARGEN_CORTE"))
    print(f"   Cohorte {anio}: {df.height:,} postulaciones | tasa acceso={df['ACCESO_1PREF'].mean():.3f} "
          f"| carreras nuevas={df['ES_CARRERA_NUEVA'].sum():,}")
    return df


def build_conversiones():
    """Curvas de equivalencia (mediana por tramo) para permitir ingresar NEM↔notas y ranking↔%superior."""
    import pandas as pd
    d = pl.read_parquet(P("data/processed/dataset_modelo_acceso.parquet")) \
          .select(["PROMEDIO_NOTAS", "PTJE_NEM", "PORC_SUP_NOTAS", "PTJE_RANKING"]).drop_nulls().to_pandas()

    def curve(src, tgt, nbins=40):
        q = pd.qcut(d[src], nbins, duplicates="drop")
        g = d.groupby(q, observed=True).agg(x=(src, "median"), y=(tgt, "median")).dropna().sort_values("x")
        return {"x": [float(v) for v in g["x"]], "y": [float(v) for v in g["y"]]}

    conv = {"notas_a_nem": curve("PROMEDIO_NOTAS", "PTJE_NEM"),
            "psup_a_rank": curve("PORC_SUP_NOTAS", "PTJE_RANKING")}
    json.dump(conv, open(P("data/processed/conversiones.json"), "w"))
    print(f"   ✅ conversiones.json  (notas↔NEM: {len(conv['notas_a_nem']['x'])} pts, "
          f"%sup↔ranking: {len(conv['psup_a_rank']['x'])} pts)")


def build_labels():
    """Etiquetas legibles (región, comuna, dependencia, rama) desde el libro de códigos."""
    import re, pandas as pd
    xls = pd.ExcelFile(P("data/raw/Libro_CódigosADM2026_ArchivoC.xlsx"))
    cr = pd.read_excel(xls, "Anexo - ComunasRegiones"); cr.columns = [c.strip() for c in cr.columns]
    reg = {str(int(r["COD REG."])): str(r["REGION NOMBRE"]).strip().title()
           for _, r in cr.dropna(subset=["COD REG."]).iterrows()}
    com = {str(int(r["COD.COMUNA"])): str(r["COM NOMBRE"]).strip().title()
           for _, r in cr.dropna(subset=["COD.COMUNA"]).iterrows()}
    # comuna → región (para el filtro en cascada del dashboard)
    com_reg = {str(int(r["COD.COMUNA"])): str(int(r["COD REG."]))
               for _, r in cr.dropna(subset=["COD.COMUNA", "COD REG."]).iterrows()}
    ri = pd.read_excel(xls, "Rinden"); dep, rama, section = {}, {}, None
    for _, row in ri.iterrows():
        s = " ".join(str(x) for x in row.values if pd.notna(x)).strip()
        if "GRUPO_DEPENDENCIA" in s: section = "dep"; continue
        if "RAMA_EDUCACIONAL" in s: section = "rama"; continue
        m = re.match(r"^([0-9]+)[.\)]\s*(.+)", s) or re.match(r"^([HT][0-9])[.\)]\s*(.+)", s)
        if m and section:
            (dep if section == "dep" else rama)[m.group(1).strip()] = m.group(2).strip()
        elif s and not m and section and len(s) > 40:
            section = None
    json.dump({"region": reg, "comuna": com, "comuna_region": com_reg, "dependencia": dep, "rama": rama},
              open(P("data/processed/labels.json"), "w"), ensure_ascii=False, indent=1)
    print(f"   ✅ labels.json  (regiones={len(reg)}, comunas={len(com)}, dep={len(dep)}, rama={len(rama)})")


def build_territorio_stats():
    """Tasa histórica de acceso a 1ª preferencia por región y por comuna (contexto territorial)."""
    d = pl.read_parquet(P("data/processed/dataset_modelo_acceso.parquet"))
    stats = {}
    for col, name in [("CODIGO_REGION", "region"), ("CODIGO_COMUNA", "comuna")]:
        g = (d.filter(pl.col(col).is_not_null())
             .group_by(col).agg(pl.col("ACCESO_1PREF").mean().alias("tasa"), pl.len().alias("n")))
        stats[name] = {str(r[col]): {"tasa": float(r["tasa"]), "n": int(r["n"])}
                       for r in g.iter_rows(named=True)}
    json.dump(stats, open(P("data/processed/territorio_stats.json"), "w"))
    print(f"   ✅ territorio_stats.json  (regiones={len(stats['region'])}, comunas={len(stats['comuna'])})")


def main():
    print("1. Cargando capa de preferencias (master validado)...")
    master = pl.read_parquet(MASTER)

    print("2. Construyendo cohortes 2025 y 2026...")
    d25 = build_cohorte(master, 2025)
    d26 = build_cohorte(master, 2026)
    data = pl.concat([d25, d26], how="vertical_relaxed")

    os.makedirs(P("data/processed"), exist_ok=True)
    data.write_parquet(P("data/processed/dataset_modelo_acceso.parquet"))
    print(f"   ✅ dataset_modelo_acceso.parquet  ({data.height:,} filas, {len(data.columns)} cols)")

    print("3. Exportando stats de carrera 2025 (corte + cupos, para el dashboard)...")
    cortes_2025 = cortes_y_cupos(master, 2025)
    stats = {str(c): {"corte": float(co), "cupos": int(cu)} for c, co, cu in
             zip(cortes_2025["COD_CARRERA_PREF"], cortes_2025["CORTE_ANTERIOR"], cortes_2025["CUPOS_ANTERIOR"])}
    with open(P("data/processed/carrera_stats.json"), "w") as f:
        json.dump(stats, f)
    print(f"   ✅ carrera_stats.json  ({cortes_2025.height:,} carreras: corte + cupos)")

    print("4. Exportando catálogo de carreras + ponderaciones (para el dashboard)...")
    # Ponderaciones oficiales → el dashboard calcula el puntaje ponderado del alumno (PTJE_PREF).
    # Se lee con pandas+latin-1 (el CSV de oferta NO es UTF-8) para conservar tildes/Ñ en los nombres.
    import pandas as pd
    of = pd.read_csv(OFERTA, sep=";", encoding="latin-1", decimal=".",
                     usecols=["CODIGO_CARRERA", "NOMBRE_CARRERA", "NOMBRE_UNIVERSIDAD",
                              "REGION_CASA_MATRIZ", "VACANTES_1SEM", "VACANTES_2SEM",
                              "CAR_VACANTES_PACE", "CDP_VACANTES_ESPECIALES", "VACANTES_GENERO",
                              "PONDERADO_MINIMO", "%_NOTAS", "%_Ranking", "%_LENG", "%_MATE1", "%_MATE2",
                              "%_HYCS", "%_CIEN", "EXIGE_MATE2"])
    of["CODIGO_CARRERA"] = pd.to_numeric(of["CODIGO_CARRERA"], errors="coerce").astype("Int64")
    of = of.dropna(subset=["CODIGO_CARRERA"])
    pl.from_pandas(of).write_parquet(P("data/processed/catalogo_carreras.parquet"))
    print(f"   ✅ catalogo_carreras.parquet  ({len(of):,} carreras, con ponderaciones)")

    print("5. Generando etiquetas legibles (región/comuna/dependencia/rama)...")
    build_labels()

    print("6. Generando curvas de equivalencia (NEM↔notas, ranking↔%superior)...")
    build_conversiones()

    print("7. Generando tasas territoriales (región/comuna)...")
    build_territorio_stats()

    print("\n✅ ETL completado.")


if __name__ == "__main__":
    main()
