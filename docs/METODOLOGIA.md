# Metodología — Modelo de Acceso a 1ª Preferencia (DAML 2026 · Grupo 5)

> Documento vivo. Registra **decisiones** y **justificaciones** del pipeline de modelado.
> Objetivo del modelo: predecir si un postulante **queda seleccionado en su 1ª preferencia** (sí/no).

---

## 1. Definición del problema

- **Unidad de análisis:** una postulación en **primera preferencia** (`ORDEN_PREF == 1`).
- **Target:** `ACCESO_1PREF = 1` si `ESTADO_PREF == 24` (*seleccionado/a para esta carrera*), `0` en otro caso.
- **Población:** procesos **2025 y 2026**, modalidad **REGULAR** únicamente (cohorte 2025 para entrenar,
  cohorte 2026 para evaluar — ver §4).

### ¿Por qué solo REGULAR?
En 1ª preferencia, al incluir todas las modalidades aparece el código `26` (*seleccionado en una
preferencia anterior*) que es lógicamente inconsistente para la preferencia #1 y proviene del cruce
con modalidades paralelas (BEA, PACE, GÉNERO). **Filtrar a REGULAR deja el target binario limpio:**

| ESTADO_PREF | Significado | Conteo (2026, REGULAR, 1ª pref) | Target |
|---|---|---|---|
| 24 | Seleccionado/a | 72.780 | **1** |
| 25 | Lista de espera | 103.795 | **0** |

Balance: **~41% positivo** → NO está desbalanceado de forma severa.

### Limitación documentada
La clase 0 incluye **lista de espera (25)**, y parte de esa lista se convierte en matrícula efectiva
cuando se liberan vacantes. El modelo predice **selección en la primera asignación**, no la matrícula
final tras movimientos de lista de espera. Es una decisión consciente y se declara en la defensa.

---

## 2. Features y su justificación

Se construyen **dos modelos** para dos momentos de decisión del estudiante:

### Modelo PRE-PAES (antes de rendir la prueba)
Solo información disponible **antes** de conocer el puntaje PAES:
- `PTJE_NEM`, `PTJE_RANKING`, `PROMEDIO_NOTAS`, `PORC_SUP_NOTAS` (rendimiento de E. Media)
- `GRUPO_DEPENDENCIA`, `CODIGO_REGION`, `CODIGO_COMUNA`, `RAMA_EDUCACIONAL` (contexto socioterritorial)
- `CORTE_ANTERIOR` (dificultad de la carrera = corte del año previo)
- `CUPOS_ANTERIOR` (capacidad = n° seleccionados el año previo), `ES_CARRERA_NUEVA`

### Modelo POST-PAES (con puntaje ya conocido)
Lo anterior **más**:
- Puntajes PAES: `CLEC`, `MATE1`, `MATE2`, `HCSOC`, `CIEN` (con flags `TIENE_*`), rescatados por
  coalesce REG_ACTUAL → INV_ACTUAL → REG_ANTERIOR (recupera puntajes de invierno / año anterior)
- `PTJE_PREF` (puntaje ponderado en la preferencia)
- `MARGEN_CORTE = PTJE_PREF − CORTE_ANTERIOR` ← **predictor dominante**

---

## 3. Decisiones metodológicas clave (lecciones de la auditoría)

| # | Decisión | Justificación |
|---|---|---|
| **D1** | **Corte = mínimo puntaje de seleccionados del AÑO PREVIO** (2025 para predecir 2026), modalidad REGULAR. | Información disponible *ex-ante*. Integridad temporal: no usa resultados del año que se predice. |
| **D2** | Carreras sin corte previo → `CORTE_ANTERIOR`/`MARGEN_CORTE` quedan null (NaN nativo de HistGB) + flag `ES_CARRERA_NUEVA`. | Evita el `dropna` que sesgaba la muestra (eliminaba carreras nuevas/pequeñas). |
| **D3** | **NO usar `class_weight='balanced'`.** | Distorsiona las probabilidades (las infla). Con 41% positivo no hace falta. |
| **D4** | **Calibración isotónica** (`CalibratedClassifierCV`). | El dashboard muestra probabilidades; deben ser fieles ("60%" debe significar 60% real). |
| **D5** | Evaluar con **AUC-ROC + PR-AUC + Brier score + curva de calibración**, no solo AUC. | AUC ignora la calibración. Brier mide la calidad de la probabilidad. |
| **D6** | `HistGradientBoosting` con **manejo nativo de NaN y categóricas** (sin `StandardScaler`). | Los árboles no necesitan escalado; imputación nativa es más correcta. |
| **D7** | Sin colinealidad redundante: se conserva `MARGEN_CORTE` (no las 3 derivadas juntas). | Limpieza e interpretabilidad de importancias. |

---

## 4. Validación

- **Esquema: validación TEMPORAL (out-of-time).** Se entrena con la cohorte **2025** y se evalúa con la
  cohorte **2026** (que el modelo nunca vio). Es el estándar correcto para una herramienta que predice
  el futuro, y evita el optimismo de un split aleatorio intra-año.
- **Datos individuales de ambos años:** se procesan `ArchivoC_Adm2025` y `ArchivoC_Adm2026REG`
  (perfil académico por estudiante: NEM, ranking, PAES, comuna). Ambos comparten esquema.
- **Integridad temporal:** garantizada por D1 (los cortes de cada cohorte vienen del año previo:
  2024→cohorte 2025, 2025→cohorte 2026). Ningún feature usa resultados del año que se predice.
- **Modelo final (dashboard):** se entrena con **2025+2026** (más señal), pero las métricas reportadas
  son siempre las **temporales** (la cifra honesta).

### Resultados (temporal 2025 → 2026)

| Modelo | AUC-ROC | PR-AUC | Brier (calibrado) |
|---|---|---|---|
| PRE-PAES | 0.914 | 0.890 | 0.117 |
| POST-PAES | 0.975 | 0.967 | 0.064 |

La caída respecto a un split aleatorio es pequeña (POST: 0.990→0.975), lo que evidencia **generalización
real** a una cohorte futura. Los cortes son estables año a año (corr 2024-25 = 0.90; 2025-26 = 0.91);
su movimiento mediano (±24 pts) es la incertidumbre que la probabilidad calibrada captura.

---

## 5. Honestidad sobre el rendimiento (para la defensa)

> El AUC alto (~0.95 en POST-PAES) **no es magia del ML**: el predictor dominante es `MARGEN_CORTE`
> (usando solo esa variable cruda, AUC ≈ 0.86). El modelo aporta sobre eso capturando no-linealidades
> e interacciones (vacantes, NEM, movimiento de cortes). El valor real de la herramienta es:
> (1) **calibrar** la probabilidad cuando el alumno está cerca de la línea, y
> (2) dar una estimación **PRE-PAES** cuando aún no hay puntaje.

---

## 6. Procedencia de los datos y pipeline

**Entradas (crudas / validadas):**
- `data/raw/.../ArchivoC_Adm2025.csv` + `data/raw/ArchivoC_Adm2026REG.csv` — perfil académico individual
  por cohorte (NEM, ranking, PAES, notas, comuna). Mismo esquema en ambos años.
- `data/processed/master_admision_2018_2026.parquet` — capa de preferencias/estados (validada).
- `data/raw/OfertaAcadémica_Admisión2026.csv` — catálogo de carreras (nombres, universidad, vacantes).

**Pipeline (reproducible):**
1. `scripts/00_build_dataset.py` → `data/processed/dataset_modelo_acceso.parquet` (cohortes 2025+2026),
   `carrera_stats.json` (corte + cupos por carrera) y `catalogo_carreras.parquet`.
2. `scripts/01_build_models.py` → `models/modelo_acceso_{pre,post}.joblib` + `*_meta.json` +
   `reports/figures/calibracion_{pre,post}.png`.
3. `scripts/02_build_score_model.py` → `models/modelo_puntaje_q{10,50,90}.joblib` (nivel PAES por cuantiles).

> El código superado (3 scripts de entrenamiento inconsistentes, modelos y dashboards previos) quedó
> archivado en `legacy/` para referencia.

---

## 7. Determinantes del éxito y MEDIACIÓN (hallazgo central)

La pregunta del trabajo no es "¿quedo dado mi puntaje?" (eso es casi mecánico: ponderado vs corte).
La pregunta es **cuánto el origen condiciona el éxito**. La respuesta está en la estructura causal:

```
ORIGEN (comuna, dependencia, región, rama)  →  PUNTAJE (NEM, PAES)  →  ¿QUEDA? (vs corte)
```

El **puntaje es un mediador**: si condicionas en él (modelo POST-PAES), el origen deja de importar
(cambiar comuna con puntajes fijos mueve la probabilidad ~0). Si NO condicionas en él (modelo PRE-PAES),
el origen importa mucho, porque **predice el puntaje que probablemente obtendrás**.

**Evidencia (cohorte 2026, mismas notas, distinto origen):**
- NEM promedio: **colegio particular pagado 808** vs **municipal 756** (~60 pts de brecha estructural).
- Nivel PAES probable (modelo de cuantiles, mismo perfil de notas): mediana **~715 (pagado)** vs **~623 (municipal)**.
- En el dashboard: con las mismas notas, el tipo de colegio mueve la probabilidad PRE-PAES ~20 pts
  porcentuales; con el puntaje ya puesto (POST-PAES), la mueve ~0.

**Modelo de puntaje (componente determinantes):** regresión por cuantiles (P10/P50/P90) de
`(CLEC+MATE1)/2` desde origen + notas. Validación temporal: **cobertura P10–P90 = 79%** (objetivo 80%),
MAE de la mediana 70 pts, ancho de banda ~222 pts. La banda ancha es honesta: el origen **desplaza** la
distribución del puntaje, no la determina (la varianza individual es enorme).

**Conclusión para la defensa:** *predecir admisión desde el puntaje es trivial; el aporte del trabajo es
mostrar que el puntaje mismo está condicionado por el origen — la desigualdad opera aguas arriba del corte.*

> Limitación: **género no está disponible** a nivel individuo en los archivos DEMRE (ArchivoC/D); solo
> existe la modalidad de admisión `TIPO_PREF='GENERO'`, que no es el sexo del postulante.
