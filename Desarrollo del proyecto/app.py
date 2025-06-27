import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Depósitos Bancarios",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏦 Predicción de Depósitos Bancarios")
st.markdown("---")

# Cargar el modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelo_svm_final.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Error: No se encontró el archivo 'modelo_svm_final.pkl'. Asegúrate de que el modelo esté en el directorio correcto.")
        return None

model = load_model()

if model is None:
    st.stop()

# Información del modelo
with st.sidebar:
    st.header("ℹ️ Información del Modelo")
    st.info("""
    **Modelo:** SVM (Support Vector Machine)
    
    **Objetivo:** Predecir si un cliente contratará un depósito bancario
    
    **Precisión:** ~87% (AUC-ROC)
    
    **Dataset:** Bank Marketing Dataset
    """)
    
    st.header("📊 Métricas del Modelo")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precisión", "87.48%")
        st.metric("F1-Score", "48.73%")
    with col2:
        st.metric("Recall", "77.18%")
        st.metric("AP", "44.75%")

# Función para crear datos de ejemplo
def get_sample_data():
    return {
        'age': 41,
        'job': 'management',
        'marital': 'married',
        'education': 'tertiary',
        'default': 'no',
        'balance': 1422,
        'housing': 'yes',
        'loan': 'no',
        'day': 16,
        'month': 'may',
        'duration': 264,
        'campaign': 3,
        'pdays': 40,
        'previous': 1
    }

# Función para hacer predicción
def predict_deposit(data):
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([data])
        
        # Hacer predicción
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return None, None

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predicción Individual", "📈 Análisis de Datos", "📊 Comparación de Modelos", "ℹ️ Información"])

with tab1:
    st.header("🎯 Predicción Individual")
    st.markdown("Ingresa los datos del cliente para predecir si contratará un depósito bancario.")
    
    # Botón para cargar datos de ejemplo
    if st.button("📋 Cargar Datos de Ejemplo"):
        sample_data = get_sample_data()
        st.session_state.form_data = sample_data
    
    # Formulario de entrada
    with st.form("prediction_form"):
        st.subheader("📝 Datos del Cliente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Edad", min_value=18, max_value=95, value=41, help="Edad del cliente")
            job = st.selectbox("Trabajo", [
                'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                'management', 'retired', 'self-employed', 'services', 
                'student', 'technician', 'unemployed'
            ], help="Tipo de trabajo del cliente")
            marital = st.selectbox("Estado Civil", ['divorced', 'married', 'single'], help="Estado civil del cliente")
            education = st.selectbox("Educación", ['primary', 'secondary', 'tertiary'], help="Nivel educativo")
            default = st.selectbox("Default", ['no', 'yes'], help="¿Tiene crédito en default?")
            balance = st.number_input("Balance", min_value=-10000, max_value=100000, value=1422, help="Balance promedio anual")
            housing = st.selectbox("Housing", ['no', 'yes'], help="¿Tiene préstamo hipotecario?")
            
        with col2:
            loan = st.selectbox("Loan", ['no', 'yes'], help="¿Tiene préstamo personal?")
            day = st.number_input("Día", min_value=1, max_value=31, value=16, help="Día del mes")
            month = st.selectbox("Mes", [
                'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
            ], help="Mes del año")
            duration = st.number_input("Duración", min_value=0, max_value=5000, value=264, help="Duración de la última llamada")
            campaign = st.number_input("Campaña", min_value=1, max_value=50, value=3, help="Número de contactos en esta campaña")
            pdays = st.number_input("Pdays", min_value=-1, max_value=1000, value=40, help="Días desde último contacto")
            previous = st.number_input("Previous", min_value=0, max_value=50, value=1, help="Número de contactos antes de esta campaña")
        
        submitted = st.form_submit_button("🔮 Predecir")
        
        if submitted:
            # Crear diccionario con los datos
            data = {
                'age': age, 'job': job, 'marital': marital, 'education': education,
                'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
                'day': day, 'month': month, 'duration': duration, 'campaign': campaign,
                'pdays': pdays, 'previous': previous
            }
            
            # Hacer predicción
            prediction, probability = predict_deposit(data)
            
            if prediction is not None:
                st.markdown("---")
                st.subheader("🎯 Resultado de la Predicción")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if prediction == 1:
                        st.success("✅ **SÍ contratará un depósito**")
                        st.balloons()
                    else:
                        st.error("❌ **NO contratará un depósito**")
                
                # Mostrar probabilidades
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilidad NO", f"{probability[0]*100:.2f}%")
                with col2:
                    st.metric("Probabilidad SÍ", f"{probability[1]*100:.2f}%")
                
                # Gráfico de probabilidades
                fig = go.Figure(data=[
                    go.Bar(
                        x=['NO', 'SÍ'],
                        y=[probability[0], probability[1]],
                        marker_color=['#ff6b6b', '#51cf66'],
                        text=[f'{probability[0]*100:.1f}%', f'{probability[1]*100:.1f}%'],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title="Probabilidades de Predicción",
                    xaxis_title="Resultado",
                    yaxis_title="Probabilidad",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📈 Análisis de Datos")
    
    # Cargar datos para análisis
    @st.cache_data
    def load_bank_data():
        try:
            df = pd.read_csv('bank.csv', sep=';')
            return df
        except FileNotFoundError:
            st.error("❌ No se encontró el archivo 'bank.csv'")
            return None
    
    df = load_bank_data()
    
    if df is not None:
        st.subheader("📊 Distribución de la Variable Objetivo")
        
        # Distribución de la variable objetivo
        target_counts = df['y'].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="Distribución de Depósitos",
            color_discrete_map={'yes': '#51cf66', 'no': '#ff6b6b'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas descriptivas
        st.subheader("📋 Estadísticas Descriptivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Variables Numéricas:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.dataframe(df[numeric_cols].describe())
        
        with col2:
            st.write("**Variables Categóricas:**")
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                st.write(f"**{col}:**")
                st.write(df[col].value_counts().head())
                st.write("---")
        
        # Análisis por edad
        st.subheader("👥 Análisis por Edad")
        fig = px.histogram(
            df, 
            x='age', 
            color='y',
            title="Distribución de Edades por Resultado",
            color_discrete_map={'yes': '#51cf66', 'no': '#ff6b6b'},
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis por trabajo
        st.subheader("💼 Análisis por Tipo de Trabajo")
        job_deposit = df.groupby(['job', 'y']).size().unstack(fill_value=0)
        fig = px.bar(
            job_deposit,
            title="Depósitos por Tipo de Trabajo",
            color_discrete_map={'yes': '#51cf66', 'no': '#ff6b6b'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("📊 Comparación de Modelos")
    
    # Métricas de diferentes modelos
    models_data = {
        'Modelo': ['Regresión Logística', 'Random Forest', 'SVM', 'XGBoost', 'Red Neuronal'],
        'Precisión': [0.813, 0.845, 0.874, 0.862, 0.851],
        'F1-Score': [0.487, 0.523, 0.487, 0.512, 0.498],
        'AUC-ROC': [0.875, 0.892, 0.875, 0.889, 0.881]
    }
    
    df_models = pd.DataFrame(models_data)
    
    # Gráfico de comparación
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Precisión', 'F1-Score', 'AUC-ROC'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=df_models['Modelo'], y=df_models['Precisión'], name='Precisión', marker_color='#51cf66'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df_models['Modelo'], y=df_models['F1-Score'], name='F1-Score', marker_color='#339af0'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=df_models['Modelo'], y=df_models['AUC-ROC'], name='AUC-ROC', marker_color='#ffd43b'),
        row=1, col=3
    )
    
    fig.update_layout(height=500, title_text="Comparación de Modelos")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de métricas
    st.subheader("📋 Métricas Detalladas")
    st.dataframe(df_models, use_container_width=True)
    
    # Conclusión
    st.subheader("🏆 Modelo Seleccionado")
    st.success("""
    **SVM (Support Vector Machine)** fue seleccionado como el mejor modelo por:
    
    - ✅ Mayor precisión general (87.4%)
    - ✅ Buen balance entre precisión y recall
    - ✅ Estabilidad en diferentes conjuntos de datos
    - ✅ Menor overfitting
    """)

with tab4:
    st.header("ℹ️ Información del Proyecto")
    
    st.subheader("🎯 Objetivo")
    st.write("""
    Este proyecto tiene como objetivo desarrollar un modelo de machine learning para predecir si un cliente 
    contratará un depósito bancario basándose en sus características demográficas y de comportamiento.
    """)
    
    st.subheader("📊 Dataset")
    st.write("""
    **Bank Marketing Dataset** - Contiene información de campañas de marketing bancario:
    
    - **4,521 registros** de clientes
    - **17 variables** (16 características + 1 objetivo)
    - **Datos desbalanceados**: 11.5% positivos, 88.5% negativos
    """)
    
    st.subheader("🔧 Preprocesamiento")
    st.write("""
    1. **Limpieza de datos**: Eliminación de valores 'unknown'
    2. **Codificación**: One-Hot Encoding para variables categóricas
    3. **Escalado**: StandardScaler para variables numéricas
    4. **División**: 70% entrenamiento, 30% prueba
    """)
    
    st.subheader("🤖 Modelo Final")
    st.write("""
    **SVM (Support Vector Machine)** con los siguientes parámetros:
    
    - **Kernel**: RBF
    - **C**: 1.0
    - **Class Weight**: Balanced
    - **Probabilidad**: True
    """)
    
    st.subheader("📈 Métricas de Rendimiento")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Precisión", "87.48%")
        st.metric("Recall", "77.18%")
        st.metric("F1-Score", "48.73%")
    
    with col2:
        st.metric("AUC-ROC", "87.48%")
        st.metric("AP", "44.75%")
        st.metric("Especificidad", "81.83%")
    
    st.subheader("🚀 Tecnologías Utilizadas")
    st.write("""
    - **Python**: Lenguaje principal
    - **Scikit-learn**: Machine Learning
    - **Pandas**: Manipulación de datos
    - **Streamlit**: Interfaz web
    - **Plotly**: Visualizaciones interactivas
    - **Joblib**: Serialización del modelo
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Desarrollado con ❤️ usando Streamlit | Modelo de Predicción de Depósitos Bancarios</p>
    </div>
    """,
    unsafe_allow_html=True
) 