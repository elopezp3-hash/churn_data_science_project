import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

print("🚀 Iniciando Pipeline de Entrenamiento CRISP-DM...")

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# 1. Cargar o Generar Datos Demo
if Path("data/clientes.csv").exists():
    df = pd.read_csv("data/clientes.csv")
elif Path("data/churn_data.csv").exists():
    df = pd.read_csv("data/churn_data.csv")
else:
    # Generación sintética compatible
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Edad': np.random.randint(18, 70, n),
        'Antiguedad_Meses': np.random.randint(1, 60, n),
        'Soporte_Tickets': np.random.poisson(2, n),
        'Uso_Diario_Min': np.random.uniform(5, 120, n),
        'Monto_Mensual': np.random.uniform(20, 500, n),
        'Genero': np.random.choice(["Femenino", "Masculino", "Otro", "No informado"], n),
        'Region': np.random.choice(["Caribe", "Andina", "Pacífica", "Orinoquía", "Amazonía"], n),
        'Churn': np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    })

# 2. Feature Engineering
df['Soporte_Tickets'] = df['Soporte_Tickets'].fillna(df['Soporte_Tickets'].median())
df['Costo_Por_Minuto'] = df['Monto_Mensual'] / (df['Uso_Diario_Min'] * 30)
df['Atencion_Requerida'] = (df['Soporte_Tickets'] > 5).astype(int)

feature_cols = [
    "Edad", "Antiguedad_Meses", "Soporte_Tickets", "Uso_Diario_Min",
    "Monto_Mensual", "Costo_Por_Minuto", "Atencion_Requerida", "Genero", "Region"
]

X = df[feature_cols]
y = df['Churn']

# 3. Preprocesamiento (Pipeline)
num_cols = ["Edad", "Antiguedad_Meses", "Soporte_Tickets", "Uso_Diario_Min", "Monto_Mensual", "Costo_Por_Minuto", "Atencion_Requerida"]
cat_cols = ["Genero", "Region"]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

# 4. Pipeline Final de Clasificación
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

# 5. División y Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model_pipeline.fit(X_train, y_train)

# 6. Pipeline de Clustering
cluster_features = num_cols
cluster_preprocessor = StandardScaler()
X_cluster_scaled = cluster_preprocessor.fit_transform(X_train[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster_scaled)

cluster_bundle = {
    "preprocessor": cluster_preprocessor,
    "model": kmeans,
    "columns": cluster_features
}

# 7. Serialización de Artefactos
joblib.dump(model_pipeline, OUT / "modelo_churn.joblib")
joblib.dump({"feature_cols": feature_cols}, OUT / "metadata_modelo.joblib")
joblib.dump(cluster_bundle, OUT / "modelo_clustering.joblib")

print("✅ Pipeline ejecutado con éxito. Artefactos actualizados en /outputs.")