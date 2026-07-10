# Guion de defensa — DAML 2026 · Grupo 5 · Presentación final (07/07)

**Proyecto:** ¿Quedaré en mi primera preferencia? — Dashboard predictivo de acceso universitario (PAES)
**Demo:** https://pacortes2021-damlpaes-srcapp-yurxbl.streamlit.app

---

## 1 · Estructura sugerida de la presentación (10–12 min)

1. **El problema (1 min).** Cada año ~300 mil estudiantes rinden la PAES y postulan a ciegas: no saben
   su probabilidad real de quedar en su 1ª preferencia ni cuánto pesa su origen. Nuestro dashboard
   responde ambas cosas, antes y después de rendir.
2. **Los datos (2 min).** Postulaciones DEMRE completas 2018–2026 (6.5M filas), rendición individual
   (ArchivoC 2024–26, ~930k), oferta académica 2026, matrícula individual 2025–26, SIES (matrícula
   institucional 2018–24 y titulados 2024), Directorio MINEDUC (RBD→colegio). **Pipeline 100%
   reproducible desde crudos** — cada número de la app se puede regenerar con `scripts/`.
3. **Los modelos (3 min).**
   - **Acceso** (clasificación): HistGradientBoosting **calibrado isotónicamente**, dos variantes —
     PRE-PAES (notas + contexto, AUC ~0.91) y POST-PAES (con puntajes, AUC ~0.97).
   - **Puntaje** (regresión por cuantiles): P10/P50/P90 por prueba → banda de incertidumbre honesta
     (cobertura empírica ~80%, el objetivo).
   - **Validación temporal estricta:** entrena 2025 → testea 2026. Nunca se evalúa dentro del mismo año.
4. **Demo en vivo (4 min).** Flujo: perfil en la barra lateral → veredicto con gauge + "¿cuánto me
   falta?" → boxplot de seleccionados → La carrera (ficha SIES, demanda, co-postulación, titulación)
   → ¿Dónde quedo? → Mapa (brecha territorial).
5. **Hallazgo social (1 min).** Con las mismas notas, cambiar el tipo de colegio mueve la probabilidad
   estimada varios puntos: el contexto predice el puntaje → **determinante estructural**, no mérito puro.
   El mapa de puntaje por comuna lo muestra territorialmente.
6. **Limitaciones y cierre (1 min).** Ver §3.

---

## 2 · Decisiones metodológicas que hay que saber defender

### ¿Por qué el target es "seleccionado" y no "matriculado"?
El modelo predice la **selección** (hacer match con la carrera: `ESTADO_PREF=24` en 1ª pref regular).
Matricularse es una **decisión posterior del estudiante** (becas, cambio de opinión), no una capacidad
del sistema de admisión. Mezclar ambas contaminaría el target con un proceso de decisión distinto.
La **matrícula efectiva** se reporta como métrica descriptiva ("% de seleccionados que se matriculó"),
cruzando admisión↔matrícula por `ID_aux`.

### ¿Por qué la población es ESTADO_PREF ∈ {24, 25}?
24 = seleccionado, 25 = lista de espera: son los que **compitieron de verdad** (postulación válidamente
rankeada con ponderado). Los demás estados (no cumple requisitos, sin puntaje) no compitieron; incluirlos
inflaría artificialmente la clase negativa (~131k filas con PTJE_PREF nulo). El filtro es **explícito**
en `scripts/00_build_dataset.py`.

### ¿Por qué el corte de referencia es 2026 y no 2025?
La feature del modelo es "corte del último proceso cerrado". Cuando entrenamos (cohorte 2025→2026), para
la cohorte 2026 eso era el corte 2025. Para un usuario de HOY, que postulará al proceso 2027, el último
proceso cerrado es **2026**. Es el mismo concepto semántico con el valor vigente — no se reentrenó porque
la relación aprendida ("estar X puntos sobre/bajo el corte previo") no cambia. Además el corte 2026
coincide con el piso del boxplot de seleccionados 2026 (consistencia visual).

### ¿Por qué agregaron la feature de colegio (RBD) y no más años de datos?
Lo **experimentamos** (`scripts/exp_score_2024_rbd.py`), no lo supusimos:
- Duplicar el entrenamiento sumando 2024 (248k→492k filas) movió el MAE < 1 punto (incluso lo empeoró
  en M1). La imprecisión del puntaje individual es **intrínseca**, no falta de datos.
- El **historial PAES del colegio** (media por RBD, Directorio MINEDUC para nombres) bajó el MAE 3–6
  puntos y la banda 9–20 en las 5 pruebas. Señal nueva > más filas de la misma señal.
- Las medias por colegio usan ArchivoC 2024+25+26 con **shrinkage empírico-bayesiano** hacia la media
  comunal (K=20): un colegio con n=10 pesa 1/3 su propia señal → estabiliza extremos por azar.
- En la app el colegio es **opcional** y la lista es una cascada coherente (comuna→dependencia→rama→
  colegio: solo se muestra lo que existe); sin colegio se usa la media comunal (respaldo → global).

### ¿Por qué no mezclan datos anteriores a 2024 en nada que use puntajes?
El techo del ponderado salta de ~995 (2018–2023) a ~1095+ (2024–2026): **cambio de escala efectivo en
2024**. Mezclar escalas distorsionaría cortes y distribuciones. Los CONTEOS (demanda por año) sí usan
2018–2026 porque no dependen de la escala.

### ¿La banda de puntaje no es demasiado ancha (~220–280 pts)?
Es **honestidad, no debilidad**: dos estudiantes con notas idénticas de la misma comuna difieren
rutinariamente 150+ puntos. Validamos que la banda P10–P90 capture ~80% de los puntajes reales de 2026
(cobertura objetivo) — está **calibrada**, no inflada.

### ¿Cómo evitan el sesgo de "predicción = destino"?
- Probabilidades **calibradas** (isotónica), no scores crudos.
- Siempre se muestra la **incertidumbre** (banda P10–P90, escenarios bajo/alto).
- El recomendador es **área-aware**: nunca sugiere "no te alcanza para ingeniería pero sí para una
  carrera sin relación" — solo la misma carrera en otras universidades o carreras del área.
- El export dice explícitamente "estimación, no garantía".

---

## 3 · Limitaciones (decirlas ANTES de que las pregunten)

- **Género no disponible** a nivel individual en DEMRE → brecha de género solo descriptiva (SIES).
- **Arancel no existe** en DEMRE/SIES disponibles → se informa en la app, no se inventa.
- El desglose de **origen escolar** (TES) es sobre la matrícula total de la carrera, no la cohorte de
  1er año (el SIES no lo publica separado) — está rotulado así en la app.
- Cruce DEMRE↔SIES por nombre (códigos incompatibles): 86% de calce; el resto muestra "s/d".
- El modelo PRE hereda las desigualdades históricas de los datos: **predice el sistema tal cual es**,
  no como debería ser (por eso el énfasis en el hallazgo estructural).

## 4 · Números clave para tener a mano

| Qué | Valor |
|---|---|
| AUC POST-PAES / PRE-PAES (temporal 2025→2026) | ~0.975 / ~0.914 |
| Cobertura banda P10–P90 puntaje | ~76–84% (objetivo 80%) |
| Mejora MAE por feature de colegio | −3 a −6 pts por prueba |
| Postulaciones en el master | 6.5M (2018–2026) |
| Carreras en el catálogo 2026 | 2.150 |
| Colegios con historial PAES | 3.373 (3.371 con nombre MINEDUC) |
| Filtro de población | ESTADO_PREF ∈ {24,25}, 1ª pref, REGULAR |

## 5 · Posibles preguntas incómodas

- **"¿Por qué gradient boosting y no una red neuronal / regresión logística?"** → Tabular con
  categóricas de alta cardinalidad y no linealidades (margen al corte): GBM es estado del arte en
  tabular; la logística la usamos de baseline mental — el AUC del GBM la supera y la calibración
  isotónica da probabilidades interpretables. Una red no aporta con este n y estas features.
- **"¿No hay fuga de información (leakage)?"** → Validación out-of-time estricta; el corte usado como
  feature es del año PREVIO a cada cohorte; las medias de colegio en el experimento se calcularon solo
  con el año previo (sin fuga). Producción usa el pool completo porque ya no se evalúa contra ese pool.
- **"¿Qué pasa con carreras nuevas sin historia?"** → Flag `ES_CARRERA_NUEVA`, se marcan 🆕 en el
  ranking, van al final y su probabilidad se declara orientativa.
- **"¿El dashboard no desincentiva postulaciones ambiciosas?"** → Mostramos escenarios (P10–P90) y "cuánto
  te falta", no un veredicto binario; el recomendador ordena por "lo mejor que alcanzo" precisamente
  para empujar hacia arriba dentro de lo realista.
