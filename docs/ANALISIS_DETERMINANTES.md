# Análisis de determinantes del éxito — DAML 2026 · Grupo 5

> Pregunta de investigación: **¿cuánto el origen socioeconómico condiciona el éxito** en el acceso
> a la educación superior? Hallazgo central: el origen opera **aguas arriba**, a través del puntaje
> (**origen → puntaje → acceso**).

## 1. Brecha de puntaje PAES por tipo de colegio
Puntaje PAES promedio (CLEC+M1)/2, con intervalo de confianza 95% (n total = 350,888):

| Tipo de colegio | Puntaje medio | IC 95% | n |
|---|---|---|---|
| Municipal | 626 | [625, 626] | 79,771 |
| Servicio Local de Educación | 613 | [612, 615] | 21,493 |
| Particular subvencionado | 648 | [648, 649] | 194,481 |
| Particular pagado | 763 | [762, 764] | 55,143 |

**Brecha particular pagado − municipal: 138 puntos** (los IC no se solapan → diferencia estadísticamente significativa).
Ver `reports/figures/det_brecha_dependencia_puntaje.png` y `det_dist_dependencia.png`.

## 2. ¿La brecha persiste controlando por las notas? (regresión OLS)
Variable dependiente: puntaje PAES. Referencia: colegio **Municipal**.

| Modelo | R² | Efecto "Particular pagado" (pts vs. municipal) |
|---|---|---|
| A — solo origen (dependencia + región + rama) | 0.204 | +117 [116, 118] |
| B — origen **+ notas** (NEM, ranking) | 0.376 | +96 [94, 97] |

**Interpretación:** aun comparando estudiantes con **las mismas notas** (mismo NEM y ranking), los de colegio
particular pagado sacan **+96 puntos** más en la PAES que los de colegio municipal. Es decir, una
parte importante de la brecha **no se explica por el mérito académico** (las notas), sino por el contexto
(acceso a preparación, recursos). Ver `reports/figures/det_persistencia_coef.png`.

## 3. ¿Cuánto del puntaje explica el origen? (validación temporal 2025→2026)
| Predictores | R² (test 2026) |
|---|---|
| Solo origen (comuna, dependencia, región, rama) | 0.21 |
| Solo notas | 0.32 |
| Origen + notas | 0.44 |

El origen por sí solo explica ~21% de la varianza del puntaje PAES — una señal estructural fuerte.

## 4. La brecha en ACCESO es menor que en PUNTAJE
Tasa de acceso a 1ª preferencia por tipo de colegio (IC 95%):

| Tipo de colegio | Tasa de acceso | IC 95% |
|---|---|---|
| Municipal | 38.4% | [38.1%, 38.8%] |
| Servicio Local de Educación | 43.4% | [42.7%, 44.1%] |
| Particular subvencionado | 39.9% | [39.7%, 40.1%] |
| Particular pagado | 42.2% | [41.8%, 42.7%] |

**Matiz clave:** la brecha de *acceso* (~4 pts %) es **menor**
que la de *puntaje* (138 pts). ¿Por qué? Porque cada grupo **apunta distinto**: con su ventaja de puntaje,
los de colegio pagado postulan a carreras más selectivas. La desigualdad no se ve tanto en "¿quedó en su 1ª opción?"
sino en **a qué puede aspirar realmente**. Ver `reports/figures/det_acceso_dependencia.png`.

## Conclusión
Predecir admisión desde el puntaje es casi mecánico. El aporte del trabajo es mostrar que **el puntaje mismo
está condicionado por el origen**: la desigualdad opera aguas arriba del corte. El modelo PRE-PAES y el simulador
contrafactual del dashboard hacen visible y cuantificable este mecanismo.
