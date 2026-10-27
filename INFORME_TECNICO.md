# 📄 Informe Técnico y Justificación Metodológica End-to-End
**Proyecto:** Sistema Predictivo y Prescriptivo de Churn B2B — DataCloud Solutions  
**Metodología:** CRISP-DM (Fases 1 a 5)  
**Autoras:** Emilys López Pérez & Kerelys Zambrano Villegas  
**Repositorio GitHub:** [https://github.com/elopezp3-hash/churn_data_science_project](https://github.com/elopezp3-hash/churn_data_science_project)  

---

## 1. Fase 1: Fundamentos de Ciencia de Datos (Business & Data Understanding)

### 1.1 Formulación del Problema bajo CRISP-DM
En el modelo de negocio B2B de *DataCloud Solutions*, la pérdida no detectada de clientes (*churn*) impacta de forma directa el ingreso recurrente mensual (MRR). La adquisición de un nuevo cliente corporativo requiere costos de marketing y ventas sustancialmente superiores a los de retención. El objetivo central es construir un modelo predictivo binario de clasificación ($Y = 1$ para riesgo de Churn) integrado a un módulo prescriptivo no supervisado para recomendar acciones tácticas antes de la cancelación.

### 1.2 Matriz de Costos, Función de Pérdida Financiera y Métrica Principal
Desde la perspectiva de la teoría de decisiones financieras, evaluamos la asimetría de costos entre los errores de clasificación:

* **Falso Positivo ($FP$):** Predecir abandono cuando el cliente es leal. Conlleva el costo de una campaña o descuento innecesario ($C_{FP}$).
* **Falso Negativo ($FN$):** Predecir lealtad cuando el cliente cancelará. Conlleva la pérdida total del valor de vida del cliente ($LTV$, $C_{FN}$).

La Función de Pérdida Financiera Global ($L_{Total}$) se expresa como:

$$L_{Total} = C_{FN} \cdot FN + C_{FP} \cdot FP$$

Dado que $C_{FN} \gg C_{FP}$ ($LTV \gg \text{Costo de Campaña}$), la métrica técnica priorizada es el **Recall (Sensibilidad)** sobre el *Accuracy* y el *Precision*. El *Accuracy* se descarta debido a que el desbalanceo de clases genera un sesgo donde un clasificador ingenuo puede simular un alto rendimiento sin detectar a la minoría en riesgo:

$$\text{Recall} = \frac{TP}{TP + FN}$$

### 1.3 Análisis Exploratorio de Datos (EDA) y Verificación de Hipótesis
Se evaluaron distribuciones y matrices de correlación para testear la premisa operativa principal:

$$\text{Hipótesis Inicial: } \text{Uso\_Diario\_Min} \uparrow \implies P(\text{Churn} = 1) \downarrow$$

**Resultado del EDA:** Se confirmó una correlación negativa significativa entre el tiempo de uso diario y la tasa de abandono. Los clientes con bajo engagement mostraron la mayor probabilidad de cancelación, respaldando la construcción de variables sintéticas de costo relativo por nivel de uso.

---

## 2. Fase 2: Preparación, Calidad e Ingeniería de Características

### 2.1 Limpieza y Calidad de Datos
1. **Imputación de `Soporte_Tickets`:** Se aplicó imputación por la **Mediana** (o KNN Imputer). Al ser una variable discreta asimétrica positiva, la media distorsionaría el centro de la distribución; la mediana preserva el valor representativo sin sesgarse por valores extremos.
2. **Tratamiento de Outliers en `Edad`:** Se aplicó la técnica de *Capping* basada en el Rango Intercuartílico ($IQR = Q_3 - Q_1$), estableciendo límites válidos en $[Q_1 - 1.5IQR, \; Q_3 + 1.5IQR]$ para corregir errores de digitación sin eliminar observaciones.
3. **Corrección de Formatos en `Antiguedad_Meses`:** Conversión de tipos de datos a entero sintácticamente válido (`int64`), corrigiendo registros vacíos o con inconsistencias menores a 0.

### 2.2 Ingeniería de Características (Feature Engineering)
* **Costo Por Minuto ($C_{min}$):**
  Mide la eficiencia del costo relativo del servicio para el cliente en función de su consumo real acumulado en un ciclo mensual (30 días):
  $$C_{min} = \frac{\text{Monto\_Mensual}}{\text{Uso\_Diario\_Min} \times 30}$$
* **Atención Requerida ($A_{req}$):**
  Indicador binario que captura la fricción operativa crítica en soporte técnico:
  $$A_{req} = \begin{cases} 1 & \text{si } \text{Soporte\_Tickets} > 5 \\ 0 & \text{en otro caso} \end{cases}$$
* **Transformaciones:** Codificación *One-Hot Encoding* para variables categóricas (`Region`, `Genero`) y escalamiento estandarizado (`StandardScaler`) para las continuas:
  $$z = \frac{x - \mu}{\sigma}$$

### 2.3 Selección de Características
Se analizaron colinealidades mediante matriz de correlación ($|r| > 0.85$) y se aplicó selección por impureza de Gini mediante un ensamble *Random Forest*, conservando únicamente las características con una importancia relativa $> 2\%$ para reducir el espacio de entradas.

---

## 3. Fase 3: Modelado Supervisado y Optimización de Umbral

### 3.1 División de Datos y Estratificación
El dataset se dividió en **80% Entrenamiento** y **20% Prueba** mediante muestreo estratificado (`StratifiedKFold` / `stratify=y`) para garantizar que la proporción de la clase minoritaria de Churn se preserve de forma exacta en ambas particiones.

### 3.2 Entrenamiento, Comparación y Balanceo
Se compararon dos familias de algoritmos:
1. **Modelo Base:** Regresión Logística (interpretabilidad lineal).
2. **Modelo Ensamble:** Random Forest Classifier (captura de interacciones no lineales).

Para mitigar el desbalanceo de clases se aplicaron técnicas de balanceo sintético (**SMOTE**) en el pipeline y ponderación explícita mediante pesos de clase (`class_weight='balanced'`).

### 3.3 Ajuste Operativo del Umbral de Decisión ($\tau$)
En lugar de depender del umbral probabilístico predeterminado de $\tau = 0.50$, la clasificación binaria ajusta dinámicamente el punto operativo a **$\tau = 0.35$** a partir de la curva Precision-Recall:

$$\hat{y} = \begin{cases} 1 & \text{si } \hat{P}(Y = 1 | x) \ge 0.35 \\ 0 & \text{en otro caso} \end{cases}$$

Este ajuste maximiza el **Recall**, logrando capturar un mayor número de Falsos Negativos a un costo marginal mínimo de Falsos Positivos.

---

## 4. Fase 4: Modelado No Supervisado y Estrategia Prescriptiva

### 4.1 Agrupamiento ($K\text{-Means}$) y Selección de $K$ Óptimo
Se aplicó el algoritmo $K\text{-Means}$ sobre las variables cuantitativas escaladas. La selección del número óptimo de segmentos ($K = 3$) se fundamentó matemáticamente en la minimización de la Inercia intra-cluster ($J$) mediante el **Método del Codo** y la maximización del **Coeficiente de Silueta**:

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

### 4.2 Caracterización y Acciones Prescriptivas por Cluster

| Cluster | Tipología de Cliente | Características Principales | Acción Prescriptiva Recomendada |
| :--- | :--- | :--- | :--- |
| **Cluster 0** | *Usuarios Saludables* | Alto consumo diario, bajo volumen de soporte. | Plan de fidelización y renovación orgánica. |
| **Cluster 1** | *Riesgo por Fricción de Soporte* | Soporte acumulado ($>5$ tickets), $A_{req} = 1$. | Asignación prioritaria de un Customer Success Manager (CSM). |
| **Cluster 2** | *Riesgo por Bajo Retorno ($C_{min}$ Alto)* | Bajo uso diario respecto a la tarifa contratada. | Programa de onboarding, capacitación y re-engagement de uso. |

---

## 5. Fase 5: Explicabilidad (XAI), Ética, Privacidad y Despliegue

### 5.1 Explicabilidad mediante Valores SHAP
Basado en teoría de juegos cooperativos, se calcula el aporte marginal de cada variable ($\phi_i$) para explicar las predicciones globales e individuales:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \Big( f(S \cup \{i\}) - f(S) \Big)$$

* **Factores Globales Dominantes:** 1) Costo Por Minuto ($C_{min}$), 2) Tickets de Soporte, 3) Uso Diario en Minutos.

### 5.2 Evaluación Ética, Governance y GDPR
* **Auditoría de Equidad (Fairness Audit):** Se evalúa la paridad en el rendimiento del modelo a través de variables protegidas (`Region` y `Genero`), exigiendo una Disparidad de Recall $\ge 0.80$:
  $$\text{Disparidad de Recall} = \frac{\text{Recall}_{\text{grupo}}}{\text{Recall}_{\text{global}}}$$
* **Privacidad y Cumplimiento Normativo:** Cumplimiento con marcos regulatorios GDPR y leyes de Protección de Datos Personales. La aplicación no almacena ni expone PII (*Personally Identifiable Information*); la inferencia se realiza exclusivamente con parámetros numéricos anónimos.

### 5.3 Arquitectura de Software y Producto Final
El modelo fue empaquetado mediante `run_pipeline.py` generando artefactos serializados (`.joblib`), e integrado en una interfaz web en **Streamlit** en modo oscuro (*Dark Mode*) que ejecuta diagnósticos en tiempo real, asignación de perfil prescriptivo y soporte para la toma de decisiones humanas.