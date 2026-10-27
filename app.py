from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# 1. Configuración de página e Identidad Corporativa
# ---------------------------------------------------------
OUT = Path("outputs")

st.set_page_config(
    page_title="DataCloud Solutions | Riesgo de Churn",
    page_icon="☁️",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Inyección de CSS Personalizado (Modo Oscuro Completo)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Fondo general oscuro */
    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    
    /* Contenedor del Formulario Oscuro */
    [data-testid="stForm"] {
        background-color: #1E293B !important;
        border-radius: 12px;
        padding: 2rem;
        border-top: 4px solid #3B82F6;
        border-left: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    /* Estilo de los nombres de los campos (labels) */
    [data-testid="stForm"] label, .stMarkdown label {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }

    /* Fondo de los campos de entrada */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #334155 !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    div[data-baseweb="input"] input {
        color: white !important;
    }

    /* Botón principal de predicción */
    div.stButton > button:first-child {
        background-color: #3B82F6 !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        width: 100%;
        margin-top: 10px;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #2563EB !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }

    /* Estilo de métricas en modo oscuro */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #60A5FA !important;
    }
    
    /* Estilos del Título y Subtítulo en Azul Brillante */
    .main-title {
        color: #60A5FA !important;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #93C5FD !important;
        font-weight: 600;
        font-size: 1.3rem;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Encabezado de la Aplicación con Logo
# ---------------------------------------------------------
col_logo, col_titulo = st.columns([1, 7])

with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149678.png", width=85)

with col_titulo:
    st.markdown('<h1 class="main-title">Soluciones DataCloud</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Sistema predictivo y prescriptivo de abandono de clientes</p>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------
# 4. Carga de Modelos y Artefactos
# ---------------------------------------------------------
model_path = OUT / "modelo_churn.joblib"
meta_path = OUT / "metadata_modelo.joblib"
cluster_path = OUT / "modelo_clustering.joblib"

if not model_path.exists() or not meta_path.exists():
    st.error("No se encuentran los modelos. Ejecuta primero: python run_pipeline.py --data data/clientes.csv")
    st.stop()

model = joblib.load(model_path)
meta = joblib.load(meta_path)
cluster_bundle = joblib.load(cluster_path) if cluster_path.exists() else None

# ---------------------------------------------------------
# 5. Formulario de Entrada de Datos del Cliente
# ---------------------------------------------------------
with st.form("cliente"):
    st.markdown("### 📋 Datos del Cliente Corporativo")
    c1, c2, c3 = st.columns(3)
    with c1:
        edad = st.number_input("Edad", min_value=18, max_value=100, value=40)
        antig = st.number_input("Antigüedad (meses)", min_value=0, max_value=600, value=12)
        tickets = st.number_input("Tickets de soporte (último trimestre)", min_value=0, max_value=100, value=2)
    with c2:
        uso = st.number_input("Uso diario promedio (minutos)", min_value=0.0, max_value=1440.0, value=60.0)
        monto = st.number_input("Monto mensual (USD)", min_value=0.0, max_value=100000.0, value=99.0)
        genero = st.selectbox("Género", ["Femenino", "Masculino", "Otro", "No informado"])
    with c3:
        region = st.selectbox("Región", ["Caribe", "Andina", "Pacífica", "Orinoquía", "Amazonía", "Otra", "No informada"])
        
    submitted = st.form_submit_button("Predecir riesgo")

# ---------------------------------------------------------
# 6. Predicción y Despliegue de Resultados con Colores
# ---------------------------------------------------------
if submitted:
    costo = monto / (uso * 30) if uso > 0 else np.nan
    row = pd.DataFrame([{
        "Edad": float(edad), "Antiguedad_Meses": float(antig),
        "Soporte_Tickets": float(tickets), "Uso_Diario_Min": float(uso),
        "Monto_Mensual": float(monto), "Costo_Por_Minuto": costo,
        "Atencion_Requerida": int(tickets > 5), "Genero": genero, "Region": region
    }])
    
    row = row[meta["feature_cols"]]
    proba = float(model.predict_proba(row)[0, 1])
    pred = int(proba >= 0.5)

    st.markdown("### 📊 Resultados del Análisis de Riesgo")
    
    if pred:
        st.error(f"🚨 **ALTO RIESGO DE ABANDONO DETECTADO** (Probabilidad: {proba:.1%})")
    else:
        st.success(f"✅ **CLIENTE CON BAJO RIESGO DE ABANDONO** (Probabilidad: {proba:.1%})")

    a, b = st.columns(2)
    with a:
        st.metric("Probabilidad estimada de churn", f"{proba:.1%}")
        st.write("**Clasificación con umbral 0.50:**", "🔴 Riesgo alto" if pred else "🟢 Riesgo bajo")
        st.caption("El umbral 0.50 es ilustrativo; debe ajustarse con costos de negocio y validación.")
        
    with b:
        if cluster_bundle:
            try:
                cluster_input = row[cluster_bundle["columns"]]
                Xt = cluster_bundle["preprocessor"].transform(cluster_input)
                cluster = int(cluster_bundle["model"].predict(Xt)[0])
                st.metric("Perfil / cluster", f"Cluster {cluster}")
            except Exception as e:
                st.warning(f"No se pudo asignar cluster con estas entradas: {e}")
        st.write("**Atención requerida:**", "⚠️ Sí (Tickets > 5)" if tickets > 5 else "No")

    st.subheader("💡 Acción sugerida para revisión humana")
    if pred:
        st.warning("Contactar al cliente para entender sus necesidades, revisar tickets pendientes y ofrecer soporte o capacitación relevante. No automatizar decisiones perjudiciales.")
    else:
        st.info("Mantener seguimiento habitual y revisar cambios de uso o soporte en el tiempo.")

# ---------------------------------------------------------
# 7. Sección de Resultados y Auditoría del Pipeline con Estilos
# ---------------------------------------------------------
st.divider()
st.subheader("📂 Resultados del pipeline")

for filename, label in [
    ("metricas_modelos.json", "Métricas de evaluación"),
    ("perfiles_clusters.csv", "Perfiles de clusters"),
    ("auditoria_equidad.csv", "Auditoría exploratoria de equidad"),
    ("importancia_variables.csv", "Importancia por permutación"),
    ("shap_importancia_global.csv", "Importancia SHAP")
]:
    p = OUT / filename
    if p.exists():
        st.markdown(f"**{label}**")
        if p.suffix == ".json":
            st.json(p.read_text(encoding="utf-8"))
        else:
            df = pd.read_csv(p)
            
            if filename == "perfiles_clusters.csv":
                styled_df = df.style.background_gradient(
                    cmap="Blues", subset=["Churn"] if "Churn" in df.columns else None
                ).format(precision=3)
                st.dataframe(styled_df, use_container_width=True)
                
            elif filename == "auditoria_equidad.csv":
                cols_color = [c for c in ["recall", "precision", "tasa_churn_real"] if c in df.columns]
                styled_df = df.style.background_gradient(
                    cmap="YlGnBu", subset=cols_color
                ).format(precision=3)
                st.dataframe(styled_df, use_container_width=True)
                
            else:
                num_cols = df.select_dtypes(include=[np.number]).columns
                styled_df = df.style.background_gradient(
                    cmap="PuBu", subset=num_cols
                ).format(precision=4)
                st.dataframe(styled_df, use_container_width=True)