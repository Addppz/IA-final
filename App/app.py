import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Predicción de Marketing Bancario",
    page_icon="🏦",
    layout="wide"
)

# Título de la aplicación
st.title("🏦 Sistema de Predicción de Marketing Bancario")
st.markdown("Prediga si un cliente suscribirá un depósito a plazo basado en sus características")

# Cargar modelo
try:
    model = joblib.load('modelo_svm_final.pkl')
    st.success("Modelo SVM cargado correctamente")
except:
    st.error("Error al cargar el modelo. Asegúrese que 'modelo_svm_final.pkl' existe")

# Sidebar con información
st.sidebar.header("ℹ️ Información del Modelo")
st.sidebar.markdown("""
**Modelo:** SVM (Máquina de Vectores de Soporte)  
**Kernel:** Radial (RBF)  
**Precisión:** 85.01%  
**Recall (positivos):** 76.51%
""")
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración")
threshold = st.sidebar.slider("Umbral de decisión", 0.0, 1.0, 0.5, 0.01)

# Sección de entrada de datos
st.header("📊 Ingrese los datos del cliente")

# Crear formulario con columnas
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Edad', min_value=18, max_value=100, value=40)
    job = st.selectbox('Trabajo', [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
        'management', 'retired', 'self-employed', 'services', 
        'student', 'technician', 'unemployed'
    ])
    marital = st.selectbox('Estado Civil', ['married', 'single', 'divorced'])
    education = st.selectbox('Educación', [
        'primary', 'secondary', 'tertiary', 'unknown'
    ])

with col2:
    default = st.selectbox('¿Tiene crédito en mora?', ['no', 'yes'])
    balance = st.number_input('Balance anual (€)', value=0)
    housing = st.selectbox('¿Tiene crédito hipotecario?', ['no', 'yes'])
    loan = st.selectbox('¿Tiene préstamo personal?', ['no', 'yes'])

with col3:
    day = st.number_input('Día del mes', min_value=1, max_value=31, value=15)
    month = st.selectbox('Mes', [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ])
    duration = st.number_input('Duración del último contacto (segundos)', value=300)
    campaign = st.number_input('Número de contactos en esta campaña', min_value=1, value=2)
    pdays = st.number_input('Días desde último contacto (999 = no contactado)', min_value=0, max_value=999, value=999)
    previous = st.number_input('Número de contactos antes de esta campaña', min_value=0, value=0)

# Crear dataframe con los inputs
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'day': [day],
    'month': [month],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous]
})

# Mostrar datos ingresados
st.subheader("Datos ingresados:")
st.dataframe(input_data)

# Predecir
if st.button('🔮 Predecir suscripción'):
    try:
        # Hacer predicción
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        decision_score = model.decision_function(input_data)[0]
        
        # Aplicar threshold personalizado
        final_prediction = 1 if proba >= threshold else 0
        
        # Mostrar resultados
        st.subheader("📝 Resultados de la Predicción")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("Predicción", 
                     "SÍ suscribirá" if final_prediction == 1 else "NO suscribirá",
                     delta="ALTA probabilidad" if proba > 0.7 else "BAJA probabilidad" if proba < 0.3 else "MEDIA probabilidad")
        
        with result_col2:
            st.metric("Probabilidad de suscripción", f"{proba:.2%}")
            st.progress(proba)
        
        with result_col3:
            st.metric("Score de decisión", f"{decision_score:.4f}")
        
        # Interpretación
        if final_prediction == 1:
            st.success("✅ El cliente tiene alta probabilidad de suscribir un depósito a plazo")
        else:
            st.info("ℹ️ El cliente probablemente no suscribirá un depósito a plazo")
            
        # Detalles técnicos
        with st.expander("🔍 Detalles técnicos de la predicción"):
            st.write(f"**Threshold aplicado:** {threshold:.2f}")
            st.write(f"Probabilidad clase 0 (No): {(1-proba):.2%}")
            st.write(f"Probabilidad clase 1 (Sí): {proba:.2%}")
            st.write(f"Score de decisión SVM: {decision_score:.4f}")
            
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")

# Sección de explicación del modelo
st.header("📚 Explicación del Modelo")
st.markdown("""
Este modelo predice si un cliente suscribirá un depósito a plazo basado en datos demográficos y de interacciones previas.

### Características clave:
- **Algoritmo:** Máquina de Vectores de Soporte (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **Parámetro C:** 1.0
- **Balance de clases:** Activado

### Rendimiento del modelo:
| Métrica | Valor |
|---------|-------|
| Precisión global | 85.01% |
| Recall (detección de positivos) | 76.51% |
| F1-Score | 54.03% |
| AUC-ROC | 88.38% |
""")

# Pie de página
st.markdown("---")
st.caption("Sistema de predicción desarrollado por Marketing Analytics | © 2024")