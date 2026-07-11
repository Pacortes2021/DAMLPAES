# Producto Mínimo Viable (MVP) - DAMLPAES

Este documento define el alcance, características y metodología del **Producto Mínimo Viable (MVP)** para el proyecto del **Grupo 5 (DAML 2026)**. El objetivo es proporcionar una herramienta interactiva y transparente que estime la probabilidad de acceso a la primera preferencia de educación superior y evalúe las condiciones socioeconómicas estructurales que determinan el éxito académico.

---

## 🎯 1. Propósito y Visión del Producto
Facilitar la toma de decisiones informada de estudiantes antes y después de rendir la PAES, haciendo visible cómo influyen las brechas socioterritoriales en el puntaje obtenido y mostrando de manera transparente el éxito formativo (retención, titulación oportuna) e ingresos futuros de cada programa académico.

---

## 🛠️ 2. Características del MVP
El MVP se divide en dos modos de análisis según la etapa del estudiante:

### A. Simulación PRE-PAES (Antes de la Prueba)
*   **Entrada de datos:** NEM (o promedio de notas de Enseñanza Media), Ranking (o porcentaje superior del curso), Región, Comuna, Dependencia del colegio (público, subvencionado, pagado), Rama (HC/TP) y código/nombre del colegio específico (opcional).
*   **Estimación de Puntajes:** Predice la distribución probable de puntajes PAES (cuantiles P10, P50, P90) para cada prueba necesaria.
*   **Probabilidad de Admisión:** Estima la probabilidad de ingresar a la carrera deseada considerando el historial y la dificultad predictiva de la postulación.

### B. Análisis POST-PAES (Puntajes Conocidos)
*   **Entrada de datos:** Puntajes PAES reales obtenidos por el postulante.
*   **Admisión Real:** Calcula la probabilidad calibrada final de admisión en base al margen sobre el corte histórico más reciente de la carrera.

### C. Análisis de Éxito Formativo y Futuro Profesional
*   **Matrícula e Historial:** Muestra estadísticas del embudo de admisión (número de vacantes, postulaciones efectivas en primera preferencia y tasa de matrícula).
*   **Titulación:** Desglosa datos de egreso y titulación (edad promedio/mediana, distribución de género).
*   **Ingresos y Empleabilidad:** Integra estadísticas oficiales de empleabilidad al primer año e ingresos promedio al cuarto año de egreso según el área de la carrera.

---

## 📊 3. Fuentes de Datos
1.  **DEMRE (Procesos de Admisión 2025 y 2026):** Archivos individuales de postulación (ArchivoD) y perfiles académicos (ArchivoC) para entrenamiento y validación de los modelos.
2.  **SIES (Titulados y Matrícula 2024/2026):** Estadísticas de matrícula institucional, desglose de establecimiento de origen, edades, y tasas de titulación.
3.  **Mifuturo.cl (Subsecretaría de Educación Superior):** Datos agregados de empleabilidad y rango de ingresos por área del conocimiento.

---

## ⚠️ 4. Limitaciones Declaradas
*   El modelo predice la **selección en la primera asignación** regular. No modela el comportamiento dinámico de los corrimientos de listas de espera posteriores o admisiones especiales.
*   Los datos de empleabilidad e ingresos futuros corresponden a agregados estadísticos a nivel de carrera genérica/área y no predicciones individualizadas.
