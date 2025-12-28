# 🏦 Predicción de Riesgo de Crédito (Loan Default Prediction)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FnyLytdXnTyRDrRixAGK30p5nUUHRkDX?usp=sharing)



## 📌 Descripción del Proyecto

Este proyecto aborda el problema de la **Asimetría de Información** en el mercado crediticio. El objetivo es desarrollar un modelo de clasificación binaria que permita a una institución financiera predecir la probabilidad de impago (*Default*) de un solicitante.

El enfoque central no es solo maximizar la precisión matemática, sino optimizar la rentabilidad del negocio minimizando los **Falsos Negativos** (aprobar créditos a clientes riesgosos), dado que la pérdida de capital es el costo más crítico para el banco.

## 🎯 Objetivos del Negocio

* **Minimización del Riesgo:** Priorizar la métrica de **Recall** (Sensibilidad) para detectar la mayor cantidad posible de morosos.
* **Gestión del Desbalance:** Manejar un dataset con una clase minoritaria de riesgo mediante técnicas de muestreo.
* **Interpretabilidad Económica:** Identificar qué variables (Ingreso, Estabilidad Familiar, Patrimonio) son los verdaderos *drivers* del riesgo.

## 🛠️ Tecnologías y Herramientas

* **Lenguaje:** Python
* **Análisis de Datos:** Pandas, NumPy, Matplotlib, Seaborn.
* **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest), XGBoost.
* **Manejo de Desbalance:** Imbalanced-learn (RandomUnderSampler, SMOTE/RandomOverSampler).

## 📊 Metodología

El ciclo de vida del proyecto incluyó las siguientes etapas:

1.  **Limpieza e Ingesta:** Detección de datos sintéticos y distribuciones uniformes.
2.  **Feature Engineering:** Creación de variables de negocio y transformación de datos.
3.  **Estrategia de Sampling (Experimento A/B):**
    * *Estrategia 1:* **Undersampling** (Reducción de clase mayoritaria).
    * *Estrategia 2:* **Oversampling** (Clonación de clase minoritaria).
4.  **Modelado:** Entrenamiento de Regresión Logística (Baseline) y XGBoost (Challenger) con optimización de hiperparámetros (`GridSearchCV`).
5.  **Evaluación Económica:** Análisis de matriz de confusión y curvas ROC-AUC.

## 📈 Resultados Clave

### 1. Comparación de Estrategias: ¿Under vs Over?
Se descubrió que la estrategia de **Undersampling** fue superior. El Oversampling generó un **Overfitting severo**, mostrando una brecha de 18 puntos porcentuales entre el rendimiento en entrenamiento (98%) y en prueba (80%).

| Modelo | Estrategia | Recall (Test) | Diagnóstico |
| :--- | :--- | :--- | :--- |
| XGBoost | **Undersampling** | **84%** | ✅ Modelo Robusto y Generalizable |
| XGBoost | Oversampling | 80% | ⚠️ Overfitting Detectado |
| Regresión Logística | Undersampling | 57% | ❌ Baja capacidad de detección |

### 2. Drivers de Riesgo (Feature Importance)
Utilizando la métrica de ganancia de información (*Gain*), el modelo reveló que el riesgo es multidimensional:

* **Estabilidad Social (Top 1):** El estado civil `Single` es el mayor predictor de riesgo. Los hogares constituidos muestran mayor resiliencia financiera.
* **Patrimonio (Top 2):** La tenencia de vehículo (`Car_Ownership`) actúa como colateral implícito y señal de solvencia.
* **Ingresos (Top 3):** Aunque importante, el `Income` por sí solo es menos predictivo que la estabilidad estructural del solicitante.

## 🚀 Conclusiones y Recomendación

Se recomienda implementar el modelo **XGBoost (Undersampled)**. Con un **Recall del 84%**, este modelo ofrece el mejor equilibrio para proteger el capital del banco. Se sugiere utilizar el score del modelo no como una decisión de rechazo automático, sino como un filtro para derivar solicitudes de alto riesgo a una revisión manual exhaustiva o para exigir mayores garantías.
