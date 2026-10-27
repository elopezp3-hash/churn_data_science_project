# Proyecto Ciencia de Datos II: Predicción y segmentación de Churn

Este repositorio implementa las cinco fases solicitadas en `TCC CIENCIA DE DATOS II.pdf`:

1. Fundamentos y análisis exploratorio (EDA), con enfoque CRISP-DM.
2. Calidad de datos e ingeniería de características.
3. Modelado supervisado (Regresión Logística y Random Forest), evaluación y ajuste.
4. Segmentación no supervisada con K-Means, método del codo y silueta.
5. Interpretabilidad SHAP, comprobaciones básicas de equidad, privacidad y aplicación Streamlit.

## Importante: datos reales

El PDF describe 10.000 registros históricos, pero el PDF compartido **no contiene el dataset**. Por tanto, este código no afirma métricas, correlaciones, clusters ni variables explicativas reales antes de ejecutar los análisis sobre el CSV real. Los resultados que se generen con el conjunto sintético son únicamente una demostración técnica.

## 1. Preparar el entorno

Se recomienda Python 3.10 o 3.11.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Preparar el dataset

Coloca el archivo CSV real en `data/clientes.csv`. Debe incluir las columnas:

- `ID_Cliente`
- `Edad`
- `Antiguedad_Meses`
- `Soporte_Tickets`
- `Uso_Diario_Min`
- `Monto_Mensual`
- `Genero`
- `Region`
- `Churn` (1 = canceló; 0 = activo)

El código acepta nombres de columnas sin distinguir mayúsculas/minúsculas y elimina espacios externos, pero los nombres deben corresponder a los anteriores. `Churn` debe estar codificada como 0/1 o como valores que puedan convertirse a 0/1.

## 3. Ejecutar

Desde la carpeta del proyecto:

```bash
python run_pipeline.py --data data/clientes.csv
```

Los informes, gráficos y modelos se guardan en `outputs/`. El flujo ejecuta limpieza, EDA, entrenamiento, clustering, explicabilidad y auditoría básica de equidad. Si SHAP no está disponible o falla por incompatibilidad, el flujo deja un aviso y continúa con el resto de resultados.

Para probar la estructura sin datos reales, crea un CSV sintético claramente etiquetado como demostración:

```bash
python generate_demo_data.py
python run_pipeline.py --data data/demo_clientes_sinteticos.csv
```

**No uses las métricas del dataset sintético como evidencia sobre clientes reales.**

## 4. Ejecutar la aplicación

Después de entrenar el pipeline:

```bash
streamlit run app.py
```

La aplicación necesita los artefactos creados por `run_pipeline.py`. Permite ingresar características de un cliente, obtener la probabilidad estimada de churn y ver el cluster asignado. La explicación SHAP se muestra si el artefacto está disponible.

## 5. Artefactos esperados

En `outputs/`:

- `eda_resumen.json`, `eda_churn.png`, `correlacion.png`
- `datos_limpios.csv`
- `metricas_modelos.json`, `matriz_confusion.png`, `curva_roc.png`
- `modelo_churn.joblib`, `preprocesador.joblib`, `modelo_clustering.joblib`, `scaler_clustering.joblib`
- `elbow_silhouette.png`, `perfiles_clusters.csv`
- `auditoria_equidad.csv`
- archivos de explicación SHAP cuando la instalación y el modelo lo permitan.

## 6. Decisiones metodológicas y límites

- Se usa `Recall` como métrica principal porque el caso busca detectar clientes que podrían cancelar: un falso negativo deja pasar a un cliente en riesgo. Se reportan también precisión, F1 y ROC-AUC.
- Se hace división estratificada 80/20. Dentro de entrenamiento se usa validación cruzada estratificada para la búsqueda de hiperparámetros.
- La imputación de `Soporte_Tickets` utiliza la mediana por robustez frente a valores extremos; los faltantes no aleatorios pueden sesgar la estimación, por lo que se crea además un indicador de faltante.
- Valores de edad fuera de 18–100 se convierten a faltantes y se imputan por la mediana de entrenamiento.
- Se calcula `Costo_Por_Minuto = Monto_Mensual / (Uso_Diario_Min * 30)`. Cuando el denominador es cero/no válido, el resultado se trata como faltante.
- `Atencion_Requerida = 1` si `Soporte_Tickets > 5`, de lo contrario 0.
- Se evita usar `ID_Cliente` como predictor. El preprocesamiento se ajusta solo con entrenamiento para evitar fuga de información.
- El escalado y la codificación se implementan en un `ColumnTransformer`. El muestreo SMOTE se aplica dentro de cada fold de entrenamiento mediante `imblearn` para evitar contaminar la validación.
- K-Means se entrena sin `Churn` ni `ID_Cliente`. Las métricas de clustering son descriptivas, no prueban que los segmentos sean “verdaderos”.
- La auditoría de equidad compara métricas por grupos de `Genero` y `Region`; grupos pequeños o faltantes limitan la interpretación. Es una comprobación exploratoria, no una certificación legal.
- El código no reemplaza revisión humana ni asesoría legal. La información debe tratarse conforme a la normativa aplicable de protección de datos.

## 7. CRISP-DM

1. Comprensión del negocio: reducir la fuga de clientes B2B.
2. Comprensión de los datos: EDA, calidad, faltantes, distribución y correlaciones.
3. Preparación: limpieza, variables derivadas y preprocesamiento.
4. Modelado: clasificación y clustering.
5. Evaluación: Recall, Precision, F1, ROC-AUC, matriz de confusión, silueta y auditoría exploratoria.
6. Despliegue: aplicación Streamlit para una demo local.

## 8. Archivos principales

- `run_pipeline.py`: orquestador de las cinco fases.
- `src/pipeline.py`: limpieza, EDA, modelos, clustering, SHAP y equidad.
- `app.py`: interfaz interactiva.
- `generate_demo_data.py`: generador opcional de datos sintéticos para comprobar que el código funciona.
