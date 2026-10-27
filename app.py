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

# Inicializar el historial de clientes en st.session_state
if "historial_clientes" not in st.session_state:
    st.session_state.historial_clientes = pd.DataFrame(columns=[
        "ID_Consulta", "Edad", "Antiguedad_Meses", "Soporte_Tickets",
        "Uso_Diario_Min", "Monto_Mensual", "Costo_Por_Minuto",
        "Atencion_Requerida", "Probabilidad_Churn", "Nivel_Riesgo", "Cluster"
    ])

# ---------------------------------------------------------
# 2. Inyección de CSS Personalizado (Botón Resaltado)
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

    /* Estilo de las etiquetas */
    [data-testid="stForm"] label, .stMarkdown label {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }

    /* Campos de entrada */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #334155 !important;
        border-radius: 8px !important;
        color: white !important;
    }
    div[data-baseweb="input"] input {
        color: white !important;
    }

    /* 🎨 BOTÓN RESALTADO CON GRADIENTE AZUL BRILLANTE Y DESTELLO */
    div[data-testid="stForm"] div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 18px !important;
        letter-spacing: 0.5px !important;
        border-radius: 10px !important;
        border: 2px solid #60A5FA !important;
        padding: 0.85rem 2rem !important;
        width: 100% !important;
        margin-top: 15px !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.6) !important;
        transition: all 0.3s ease-in-out !important;
    }
    
    /* Efecto Hover */
    div[data-testid="stForm"] div.stButton > button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        box-shadow: 0 0 25px rgba(96, 165, 250, 0.9) !important;
        transform: translateY(-2px) !important;
        border-color: #93C5FD !important;
    }

    /* Estilo de métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #60A5FA !important;
    }
    
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
# 3. Encabezado de la Aplicación
# ---------------------------------------------------------
col_logo, col_titulo = st.columns([1, 7])

with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149678.png", width=85)

with col_titulo:
    st.markdown('<h1 class="main-title">Soluciones DataCloud</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Sistema predictivo y prescriptivo de abandono de clientes (Churn B2B)</p>', unsafe_allow_html=True)

st.markdown("""
**Metodología:** CRISP-DM | **Autoras:** Emilys López Pérez & Kerelys Zambrano Villegas  
**Repositorio GitHub:** [github.com/elopezp3-hash/churn_data_science_project](https://github.com/elopezp3-hash/churn_data_science_project)
""")

st.markdown("---")

# ---------------------------------------------------------
# 4. Carga de Modelos y Artefactos
# ---------------------------------------------------------
model_path = OUT / "modelo_churn.joblib"
meta_path = OUT / "metadata_modelo.joblib"
cluster_path = OUT / "modelo_clustering.joblib"

if not model_path.exists() or not meta_path.exists():
    st.error("No se encuentran los modelos. Ejecuta primero: python run_pipeline.py")
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
        antig = st.number_input("Antigüedad (meses)", min_value=0, max_value=600, value=8)
        tickets = st.number_input("Tickets de soporte (último trimestre)", min_value=0, max_value=100, value=7)
    with c2:
        uso = st.number_input("Uso diario promedio (minutos)", min_value=0.0, max_value=1440.0, value=15.0)
        monto = st.number_input("Monto mensual (USD)", min_value=0.0, max_value=100000.0, value=299.0)
        genero = st.selectbox("Género", ["Femenino", "Masculino", "Otro", "No informado"])
    with c3:
        region = st.selectbox("Región", ["Caribe", "Andina", "Pacífica", "Orinoquía", "Amazonía", "Otra", "No informada"])
        
    submitted = st.form_submit_button("⚡ Predecir riesgo y Guardar")

# ---------------------------------------------------------
# 6. Predicción y Despliegue de Resultados
# ---------------------------------------------------------
if submitted:
    costo = monto / (uso * 30) if uso > 0 else np.nan
    atencion_req = int(tickets > 5)
    
    row = pd.DataFrame([{
        "Edad": float(edad),
        "Antiguedad_Meses": float(antig),
        "Soporte_Tickets": float(tickets),
        "Uso_Diario_Min": float(uso),
        "Monto_Mensual": float(monto),
        "Costo_Por_Minuto": costo,
        "Atencion_Requerida": atencion_req,
        "Genero": genero,
        "Region": region
    }])
    
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif "feature_cols" in meta:
        expected_cols = meta["feature_cols"]
    else:
        expected_cols = ['Monto_Mensual', 'Uso_Diario_Min', 'Soporte_Tickets', 'Antiguedad_Meses', 'Costo_Por_Minuto', 'Atencion_Requerida']

    row_model = row[expected_cols]
    proba = float(model.predict_proba(row_model)[0, 1])
    
    UMBRAL_RECALL = 0.35
    pred = int(proba >= UMBRAL_RECALL)

    # Identificar cluster
    cluster_label = "Cluster 1 (Fricción)"
    if cluster_bundle:
        try:
            cluster_cols = cluster_bundle.get("columns", ['Monto_Mensual', 'Uso_Diario_Min', 'Soporte_Tickets', 'Antiguedad_Meses', 'Costo_Por_Minuto', 'Atencion_Requerida'])
            cluster_input = row[cluster_cols]
            if "preprocessor" in cluster_bundle and cluster_bundle["preprocessor"] is not None:
                Xt = cluster_bundle["preprocessor"].transform(cluster_input)
            else:
                Xt = cluster_input
            c_num = int(cluster_bundle["model"].predict(Xt)[0])
            cluster_label = f"Cluster {c_num}"
        except Exception:
            cluster_label = "Cluster 1 (Fricción)"

    # Guardar en el Historial de la Sesión
    num_consulta = len(st.session_state.historial_clientes) + 1
    nueva_consulta = pd.DataFrame([{
        "ID_Consulta": f"CLI-{num_consulta:03d}",
        "Edad": int(edad),
        "Antiguedad_Meses": int(antig),
        "Soporte_Tickets": int(tickets),
        "Uso_Diario_Min": float(uso),
        "Monto_Mensual": float(monto),
        "Costo_Por_Minuto": round(costo, 4),
        "Atencion_Requerida": "Sí (Tickets>5)" if tickets > 5 else "No",
        "Probabilidad_Churn": f"{proba:.1%}",
        "Nivel_Riesgo": "🔴 Alto Riesgo" if pred else "🟢 Bajo Riesgo",
        "Cluster": cluster_label
    }])

    st.session_state.historial_clientes = pd.concat([st.session_state.historial_clientes, nueva_consulta], ignore_index=True)

    st.markdown("### 📊 Resultados del Análisis de Riesgo")
    
    if pred:
        st.error(f"🚨 **ALTO RIESGO DE ABANDONO DETECTADO** (Probabilidad: {proba:.1%})")
    else:
        st.success(f"✅ **CLIENTE CON BAJO RIESGO DE ABANDONO** (Probabilidad: {proba:.1%})")

    a, b = st.columns(2)
    with a:
        st.metric("Probabilidad estimada de churn", f"{proba:.1%}")
        st.write(f"**Clasificación con umbral operativo {UMBRAL_RECALL}:**", "🔴 Riesgo alto" if pred else "🟢 Riesgo bajo")
        st.caption(f"Se prioriza el Recall (umbral {UMBRAL_RECALL}) debido al elevado costo de los Falsos Negativos ($C_{{FN}} \\gg C_{{FP}}$).")
        
    with b:
        st.metric("Perfil / cluster", cluster_label)
        st.write("**Atención requerida ($A_{req}$):**", "⚠️ Sí (Tickets > 5)" if tickets > 5 else "No")

    st.subheader("💡 Acción prescriptiva sugerida para revisión humana")
    if pred:
        if tickets > 5:
            st.warning("⚠️ **Estrategia Prescriptiva:** Cliente con fricción crítica en soporte ($A_{req}=1$). Asignar de forma inmediata un Customer Success Manager (CSM) dedicado.")
        elif costo > 0.05:
            st.warning("⚠️ **Estrategia Prescriptiva:** Riesgo por alto costo percibido ($C_{min}$). Desplegar programa de entrenamiento, onboarding y optimización de uso.")
        else:
            st.warning("⚠️ **Estrategia Prescriptiva:** Contactar al cliente para evaluar su nivel de satisfacción y ofrecer incentivos de fidelización.")
    else:
        st.info("ℹ️ **Estrategia Prescriptiva:** Cliente en estado saludable. Mantener seguimiento habitual y explorar oportunidades de venta cruzada (upsell).")

# ---------------------------------------------------------
# 7. Tabla Interactiva de Historial con Resaltado
# ---------------------------------------------------------
if not st.session_state.historial_clientes.empty:
    st.divider()
    st.subheader("📋 Historial de Clientes Consultados (Guardado)")

    df_historial = st.session_state.historial_clientes.copy()
    
    def resaltar_ultimo(s):
        return ['background-color: #854D0E; color: #FFFFFF; font-weight: bold;' if s.name == len(df_historial) - 1 else '' for _ in s]

    df_estilizado = df_historial.style.apply(resaltar_ultimo, axis=1)
    st.dataframe(df_estilizado, use_container_width=True)

    col_d1, col_d2, col_d3 = st.columns([2, 2, 4])
    with col_d1:
        csv_data = df_historial.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Historial (CSV)",
            data=csv_data,
            file_name="historial_predicciones_churn.csv",
            mime="text/csv"
        )
    with col_d2:
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.historial_clientes = pd.DataFrame(columns=[
                "ID_Consulta", "Edad", "Antiguedad_Meses", "Soporte_Tickets",
                "Uso_Diario_Min", "Monto_Mensual", "Costo_Por_Minuto",
                "Atencion_Requerida", "Probabilidad_Churn", "Nivel_Riesgo", "Cluster"
            ])
            st.rerun()

# ---------------------------------------------------------
# 8. Visualización de Gráficas de Evaluación (ROC y Matriz de Confusión)
# ---------------------------------------------------------
st.divider()
st.subheader("📈 Evaluación Visual del Modelo (Curva ROC y Matriz de Confusión)")

col_img1, col_img2 = st.columns(2)

img_cm = OUT / "matriz_confusion.png"
img_roc = OUT / "curva_roc.png"

with col_img1:
    st.markdown("**Matriz de Confusión**")
    if img_cm.exists():
        st.image(str(img_cm), use_column_width=True, caption="Evaluación de aciertos/errores en clasificación")
    else:
        st.info("La imagen 'matriz_confusion.png' no se encuentra en /outputs.")

with col_img2:
    st.markdown("**Curva ROC**")
    if img_roc.exists():
        st.image(str(img_roc), use_column_width=True, caption="Capacidad de discriminación global (ROC-AUC)")
    else:
        st.info("La imagen 'curva_roc.png' no se encuentra en /outputs.")

# ---------------------------------------------------------
# 9. Sección de Resultados y Auditoría del Pipeline
# ---------------------------------------------------------
st.divider()
st.subheader("📂 Tablas de Resultados del Pipeline")

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
            num_cols = df.select_dtypes(include=[np.number]).columns
            styled_df = df.style.background_gradient(cmap="PuBu", subset=num_cols).format(precision=3)
            st.dataframe(styled_df, use_container_width=True)