# INFORME TÉCNICO: SISTEMA PREDICTIVO Y PRESCRIPTIVO DE ABANDONO DE CLIENTES (CHURN B2B) BAJO METODOLOGÍA CRISP-DM

**Empresa:** DataCloud Solutions  
**Autoras:** Emilys López Pérez & Kerelys Zambrano Villegas  
**Repositorio GitHub:** [github.com/elopezp3-hash/churn_data_science_project](https://github.com/elopezp3-hash/churn_data_science_project)  
**Fecha:** Octubre de 2026  
**Versión:** 2.4 (Release Candidate)

---

## RESUMEN EJECUTIVO

El presente informe documenta el desarrollo, validación y despliegue del sistema predictivo y prescriptivo de abandono de clientes (Churn B2B) para **DataCloud Solutions**, implementado bajo la metodología estándar **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*). El objetivo primordial del proyecto es mitigar la pérdida de ingresos recurrentes (ARR/MRR) identificando a los clientes corporativos en riesgo de cancelación antes de que hagan efectiva su salida.

El pipeline integra un modelo de clasificación supervisada basado en **Random Forest Classifier** optimizado para maximizar el *Recall* ($\tau = 0.35$), una segmentación no supervisada mediante **K-Means ($K=3$)** para tipificar el comportamiento de los usuarios, y un módulo de métricas sintéticas interactivas ($C_{min}$ y $A_{req}$) desplegado mediante un Dashboard interactivo en **Streamlit**.

---

## FASE 1: ENTENDIMIENTO DEL NEGOCIO (BUSINESS UNDERSTANDING)

### 1.1. Contexto Organizacional y Problema Central
DataCloud Solutions opera bajo un modelo de negocio *Software as a Service* (SaaS) y servicios de infraestructura Cloud B2B. En este modelo, el costo de adquisición de un nuevo cliente ($CAC$) supera significativamente el costo de retención de un cliente activo ($CRC$). Específicamente, se estima que:

$$CAC \approx 5 \times CRC$$

La pérdida imprevista de cuentas corporativas genera una contracción directa del margen operativo y reduce el valor del cliente en el tiempo (*Customer Lifetime Value* - $LTV$).

### 1.2. Función de Costo de Clasificación Asimétrica
En el contexto B2B, la matriz de costos de clasificación no es simétrica:

* **Falso Negativo ($FN$):** Un cliente en riesgo real de churn es clasificado como leal. La empresa no interviene y pierde el contrato ($LTV$).
* **Falso Positivo ($FP$):** Un cliente leal es clasificado erróneamente como en riesgo. La empresa invierte en una campaña de fidelización o descuento no requerido ($C_{fidelizacion}$).

Dado que $C_{FN} \gg C_{FP}$, la función de pérdida del negocio $L(Y, \hat{Y})$ se expresa formalmente como:

$$L(Y, \hat{Y}) = C_{FN} \cdot I(Y=1, \hat{Y}=0) + C_{FP} \cdot I(Y=0, \hat{Y}=1)$$

Donde $I(\cdot)$ es la función indicadora, $Y=1$ representa el evento de Churn, y $\hat{Y}=1$ representa la predicción positiva. Por ende, la optimización técnica del modelo debe enfocarse explícitamente en la **maximización del Recall (Sensibilidad)**:

$$\text{Recall} = \frac{TP}{TP + FN}$$

---

## FASE 2: ENTENDIMIENTO Y PREPARACIÓN DE LOS DATOS (DATA PREPARATION)

### 2.1. Ingestion y Estructura de Datos
El dataset contiene información histórica de interacciones, métricas de consumo y datos transaccionales de clientes corporativos. Las variables de entrada originales comprenden:

* **Numéricas:** `Edad`, `Antiguedad_Meses`, `Soporte_Tickets`, `Uso_Diario_Min`, `Monto_Mensual`.
* **Categorización y Demografía:** `Genero`, `Region`.
* **Variable Objetivo:** `Churn` ($\in \{0, 1\}$).

### 2.2. Ingeniería de Características (Feature Engineering)
Para capturar la fricción operativa y la ineficiencia económica percibida por los clientes, se formalizaron dos variables sintéticas clave:

#### A. Costo por Minuto Percibido ($C_{min}$)
Representa el costo efectivo que el cliente paga por cada minuto de uso real del servicio durante un período mensual (30 días):

$$C_{min} = \frac{\text{Monto\_Mensual}}{\text{Uso\_Diario\_Min} \times 30}$$

Si $\text{Uso\_Diario\_Min} = 0$, se asigna $NaN$ y se imputa formalmente mediante la mediana de la población para evitar divisiones por cero. Un valor elevado de $C_{min}$ indica un bajo retorno percibido de la inversión por parte del cliente corporativo.

#### B. Indicador Binario de Atención Requerida ($A_{req}$)
Modela el umbral de tolerancia del cliente frente a incidentes técnicos. Se define mediante la función escalón:

$$A_{req} = \begin{cases} 1, & \text{si } \text{Soporte\_Tickets} > 5 \\ 0, & \text{si } \text{Soporte\_Tickets} \le 5 \end{cases}$$

La activación de $A_{req} = 1$ señala una saturación de la mesa de ayuda y exige una intervención humana inmediata priorizada por un ejecutivo de *Customer Success*.

---

## FASE 3: MODELADO Y EVALUACIÓN SUPERVISADA (MODELING & EVALUATION)

### 3.1. Selección de Arquitectura de Aprendizaje
Se seleccionó la arquitectura **Random Forest Classifier** debido a su capacidad intrínseca para capturar relaciones no lineales y complejas interacciones entre variables sin asumir supuestos rígidos de normalidad.

Un Random Forest representa un ensamble de $B$ árboles de decisión $\{T_1(x), T_2(x), \dots, T_B(x)\}$. La predicción probabilística agregada para la clase de riesgo $Y=1$ viene dada por:

$$P(Y=1 | X=x) = \frac{1}{B} \sum_{b=1}^{B} P_b(Y=1 | X=x)$$

Para balancear el desbalance de clases nativo de los datos de churn, se aplicó la ponderación de pesos inversamente proporcional a la frecuencia de clase:

$$w_j = \frac{N}{K \cdot n_j}$$

Donde $N$ es el total de muestras, $K=2$ es el número de clases, y $n_j$ es el número de observaciones en la clase $j$.

### 3.2. Calibración Operativa del Umbral ($\tau = 0.35$)
Por defecto, los clasificadores aplican un umbral estándar de decisión $\tau_0 = 0.50$. Sin embargo, en virtud de la asimetría de costos ($C_{FN} \gg C_{FP}$), se desplazó operativamente el umbral de decisión a:

$$\tau^* = 0.35$$

La regla de clasificación final aplicada en inferencia es:

$$\hat{Y} = \begin{cases} 1 (\text{Alto Riesgo}), & \text{si } P(Y=1|X) \ge 0.35 \\ 0 (\text{Bajo Riesgo}), & \text{si } P(Y=1|X) < 0.35 \end{cases}$$

### 3.3. Métricas de Evaluación Obtenidas
Evaluación en el conjunto de prueba ($N_{test} = 200$, 20% retenido con estratificación):

* **Recall ($\tau = 0.35$):** **85.0%** — Garantiza la detección anticipada de la vasta mayoría de clientes en riesgo.
* **Precision ($\tau = 0.35$):** **68.0%** — Mantiene una tasa aceptable de falsas alarmas.
* **F1-Score:** **0.756** — Demuestra un desempeño armónico sólido.
* **ROC-AUC:** **0.892** — Alta capacidad de discriminación probabilística global.

---

## FASE 4: SEGMENTACIÓN NO SUPERVISADA Y PERFILAMIENTO (CLUSTERING)

### 4.1. Algoritmo K-Means ($K=3$)
Con el propósito de caracterizar el perfil comportamental de los clientes para la prescripción de estrategias diferenciadas, se aplicó el algoritmo **K-Means** sobre el espacio de características numéricas estandarizadas $Z$:

$$Z_j = \frac{X_j - \mu_j}{\sigma_j}$$

El objetivo de optimización de K-Means consiste en minimizar la suma de cuadrados dentro de los clusters (WCSS):

$$\arg\min_{\mathbf{S}} \sum_{i=1}^{K} \sum_{\mathbf{x} \in S_i} \left\| \mathbf{x} - \boldsymbol{\mu}_i \right\|^2$$

Donde $K=3$ fue determinado mediante el método del codo (*Elbow Method*) y el coeficiente de silueta (*Silhouette Score*).

### 4.2. Perfiles de Clusters Identificados

1. **Cluster 0 — Clientes Saludables (Bajo Riesgo):**  
   * Elevada antigüedad, alto uso diario promedio, bajo volumen de tickets.  
   * *Estrategia:* Mantenimiento y programas de Upselling / Cross-selling.
2. **Cluster 1 — Clientes en Fricción Crítica ($A_{req}=1$):**  
   * Registran más de 5 tickets por trimestre. Alta tasa de incidentes sin resolver.  
   * *Estrategia:* Asignación inmediata de un Customer Success Manager (CSM) dedicado para contención técnica.
3. **Cluster 2 — Clientes Ineficientes / Inactivos ($C_{min}$ Alto):**  
   * Alto cobro mensual combinado con bajísimo consumo de la plataforma.  
   * *Estrategia:* Despliegue de capacitaciones, programas de re-onboarding y optimización de uso.

---

## FASE 5: EXPLICABILIDAD (XAI) Y ÉTICA EN IA

### 5.1. Importancia por Permutación e Valores SHAP
Para evitar la opacidad del modelo (*Black Box*), se implementaron técnicas de Explicabilidad Agnostica (XAI). La importancia global de las características revela que los principales factores de riesgo son:

1. **`Soporte_Tickets` / $A_{req}$:** Factor dominante en la probabilidad de churn.
2. **`Uso_Diario_Min`:** Relación inversamente proporcional con la probabilidad de abandono.
3. **`Costo_Por_Minuto` ($C_{min}$):** Factor económico clave para cuentas de alto valor.

### 5.2. Auditoría de Equidad y Sesgo Demográfico
Se realizó una auditoría exploratoria para asegurar que el modelo no genere discriminación sistemática por variables sensibles demográficas (`Genero`, `Region`). Se evaluó la paridad demográfica en la tasa de falsos positivos ($FPR$) y falsos negativos ($FN$), verificando que las fluctuaciones en las métricas no superan el margen del $5\%$, cumpliendo con los estándares éticos de equidad en IA.

---

## FASE 6: ARQUITECTURA DE DESPLIEGUE Y OPERACIONALIZACIÓN

El sistema se desplegó en producción utilizando **Streamlit** en un entorno Python gestionado.