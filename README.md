# Predicción de Acceso a Educación Superior — DAML 2026 · Grupo 5

Herramienta que estima la **probabilidad de que un postulante quede seleccionado en su primera
preferencia** universitaria (modalidad regular), a partir de su perfil académico y la carrera elegida.
Usa datos individuales del DEMRE (procesos de admisión 2025 y 2026).

> 📄 Metodología, decisiones y honestidad para la defensa: **[`docs/METODOLOGIA.md`](docs/METODOLOGIA.md)**

---

## Qué hace

Dos modelos calibrados (Gradient Boosting), para los dos momentos de decisión del estudiante:

| Modelo | Cuándo | Entradas | AUC (validación temporal 2025→2026) |
|---|---|---|---|
| **PRE-PAES** | Antes de rendir | NEM, ranking, notas, comuna, dependencia, dificultad de la carrera | **0.914** |
| **POST-PAES** | Con el puntaje ya conocido | + puntajes PAES + margen al corte histórico | **0.975** |

Las probabilidades están **calibradas**: "60%" significa ~60% real. El valor de la herramienta es estimar
la **incertidumbre cuando el alumno está cerca del corte** (que se mueve ~±24 pts/año) y dar una
estimación temprana **antes de la PAES**.

---

## Cómo ejecutar

```bash
pip install -r requirements.txt

# 1. ETL: construye la tabla de modelado (cohortes 2025+2026) + catálogo + etiquetas
python3 scripts/00_build_dataset.py

# 2. Entrena los modelos con validación TEMPORAL (entrena 2025 → testea 2026)
python3 scripts/01_build_models.py

# 3. Lanza el dashboard
streamlit run src/app.py        # o:  python3 -m streamlit run src/app.py
```

---

## Estructura del proyecto

```
scripts/
  00_build_dataset.py     ETL limpio → dataset_modelo_acceso.parquet, carrera_stats.json, catalogo, labels
  01_build_models.py      Entrenamiento + validación temporal + calibración → models/*.joblib
  02_build_score_model.py Predicción de puntaje PAES por prueba (regresión por cuantiles)
  03_analisis_determinantes.py  Análisis formal: brechas por origen + OLS + figuras (informe)
  04_matricula_stats.py   Matrícula efectiva por carrera (cruce admisión↔matrícula por ID_aux)
src/
  inference.py            Lógica de predicción (ponderado, encoding, predict) — sin Streamlit
  app.py                  Dashboard Streamlit (predictor de acceso PRE/POST-PAES)
models/                   modelo_acceso_{pre,post}.joblib + *_meta.json
data/
  raw/                    Archivos DEMRE (ArchivoC/D 2025 y 2026, oferta, libros de códigos)
  processed/              Datasets y artefactos generados por el ETL
docs/METODOLOGIA.md       Decisiones de diseño y validación
docs/ANALISIS_DETERMINANTES.md  Hallazgos: cómo el origen condiciona el puntaje y el acceso
reports/figures/          Figuras (EDA + curvas de calibración)
notebooks/                EDA exploratorio
legacy/                   Código/modelos previos archivados (referencia)
```

---

## Pipeline de datos (resumen)

1. **Target:** `ACCESO_1PREF = (ESTADO_PREF == 24)` en 1ª preferencia, modalidad **REGULAR**
   (filtrar a regular deja el target binario limpio: seleccionado vs. lista de espera).
2. **Features con integridad temporal:** la dificultad de cada carrera (`CORTE_ANTERIOR`, `CUPOS_ANTERIOR`)
   viene del **año previo** (2024→cohorte 2025; 2025→cohorte 2026), nunca del año que se predice.
3. **Validación temporal:** se entrena con 2025 y se evalúa con 2026 (out-of-time).
4. **Calibración isotónica** + métricas honestas (AUC, PR-AUC, Brier, curva de calibración).

---

## Trabajo futuro

- Ampliar a años anteriores (descargar ArchivoC/D 2024 y antes) para más historia.
- Segundo modelo: probabilidad de **titulación oportuna** (datos longitudinales SIES).
- Capa de exploración territorial (mapas por comuna/región).
