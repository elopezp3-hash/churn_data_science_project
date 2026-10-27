# 📄 Informe Técnico y Justificación Metodológica
**Proyecto:** Sistema Predictivo y Prescriptivo de Churn B2B — DataCloud Solutions  
**Metodología:** CRISP-DM  
**Autoras:** Emilys López Pérez & Kerelys Zambrano Villegas  
**Repositorio GitHub:** [https://github.com/elopezp3-hash/churn_data_science_project](https://github.com/elopezp3-hash/churn_data_science_project)  

---

## 1. Comprensión del Negocio (Business Understanding)

### 1.1 Formulación del Problema
En el modelo B2B de *DataCloud Solutions*, la pérdida de un cliente (*churn*) implica una reducción directa del ingreso recurrente mensual (MRR). La adquisición de nuevos clientes requiere una inversión en marketing y ventas significativamente superior a la retención de un cliente existente.

### 1.2 Matriz de Costos y Función de Pérdida
Desde la perspectiva de teoría de decisiones, clasificar a un cliente requiere balancear dos tipos de errores:
* **Falso Positivo ($FP$):** Predecir que un cliente abandonará cuando en realidad se quedaría. Esto genera un costo de intervención innecesario ($C_{FP}$).
* **Falso Negativo ($FN$):** Predecir que un cliente es leal cuando en realidad abandonará. Esto genera la pérdida del valor de vida del cliente ($C_{FN}$).

Dado que $C_{FN} \gg C_{FP}$, la métrica de optimización principal seleccionada es el **Recall (Sensibilidad)**:

$$\text{Recall} = \frac{TP}{TP + FN}$$

Un alto *Recall* garantiza la detección de la gran mayoría de clientes en riesgo de abandono.

---

## 2. Comprensión y Preparación de Datos (Data Understanding & Preparation)

### 2.1 Feature Engineering (Ingeniería de Características)
Se diseñaron variables derivadas para capturar patrones de comportamiento no lineales:

1. **Costo Por Minuto ($C_{min}$):** Relación de eficiencia del valor del contrato respecto al consumo real.
   $$C_{min} = \frac{\text{Monto\_Mensual}}{\text{Uso\_Diario\_Min} \times 30}$$

2. **Atención Requerida ($A_{req}$):** Variable binaria para clientes con alta fricción operativa en soporte técnico.
   $$A_{req} = \mathbb{I}(\text{Soporte\_Tickets} > 5)$$

### 2.2 Tratamiento de Variables Categoricas y Escalamiento
* **Columnas Numéricas:** Imputación por mediana + Escalamiento robusto ($RobustScaler$) para mitigar el efecto de valores atípicos (*outliers*).
* **Columnas Categóricas:** Imputación por la moda + Codificación $OneHotEncoder(handle\_unknown='ignore')$ para preservar la independencia de categorías.

---

## 3. Modelado y Justificación Matemática (Modeling)

### 3.1 Clasificación Supervisada: Random Forest
Se seleccionó un ensamble de árboles de decisión (*Random Forest*) por su capacidad de capturar interacciones no lineales entre las variables y su resistencia al sobreajuste (*overfitting*).

La predicción final para una observación $x$ en un ensamble de $B$ árboles se expresa como la promediación de las probabilidades individuales:

$$\hat{P}(Y = 1 | x) = \frac{1}{B} \sum_{b=1}^{B} P_b(Y = 1 | x)$$

La clasificación binaria adopta un umbral probabilístico predeterminado de $\tau = 0.50$:

$$\hat{y} = \begin{cases} 1 & \text{si } \hat{P}(Y = 1 | x) \ge \tau \\ 0 & \text{en otro caso} \end{cases}$$

### 3.2 Clustering No Supervisado: K-Means
Para el perfilamiento de usuarios y la asignación de acciones prescriptivas, se empleó la técnica de aglomeración $K\text{-Means}$ sobre las variables estandarizadas. El algoritmo busca minimizar la suma de cuadrados intra-cluster (Inercia, $J$):

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

donde $\mu_k$ representa el centroide del cluster $C_k$.

---

## 4. Evaluación y Explicabilidad (Evaluation & Governance)

### 4.1 Explicabilidad Global e Individual mediante Valores SHAP
Basado en la teoría de juegos cooperativos, el valor SHAP ($\phi_i$) asigna a cada variable $i$ un peso proporcional a su contribución marginal al resultado del modelo:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \Big( f(S \cup \{i\}) - f(S) \Big)$$

Esto permite auditar las predicciones ante el usuario final sin tratar al modelo de Random Forest como una "caja negra".

### 4.2 Auditoría Exploratoria de Equidad (Fairness Audit)
Con el objetivo de garantizar la no discriminación algorítmica por factores geográficos o demográficos protegidos (ej. `Genero`, `Region`), se monitorean de forma continua las métricas de disparo positivo y desempeño grupal:

$$\text{Disparidad de Recall} = \frac{\text{Recall}_{\text{grupo}}}{\text{Recall}_{\text{global}}}$$

Se exige que ningún grupo demográfico presente un *Recall* significativamente inferior al promedio del sistema.

---

## 5. Despliegue y Arquitectura de Software (Deployment)

La solución fue empaquetada e integrada en un pipeline automatizado en Python (`run_pipeline.py`) que genera los siguientes artefactos serializados:
* `modelo_churn.joblib`: Modelo clasificador entrenado.
* `modelo_clustering.joblib`: Pipeline de segmentación $K\text{-Means}$.
* `auditoria_equidad.csv` y `shap_importancia_global.csv`: Reportes de gobernanza.

La interfaz de usuario fue construida utilizando **Streamlit** en un entorno oscurecido (*Dark Mode*), ofreciendo inferencia en tiempo real para la toma de decisiones por analistas de retención humana.