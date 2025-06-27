# 🏦 Predicción de Depósitos Bancarios

Una aplicación web interactiva desarrollada con Streamlit para predecir si un cliente contratará un depósito bancario utilizando un modelo de Machine Learning SVM (Support Vector Machine).

## 🎯 Características

- **Predicción Individual**: Interfaz intuitiva para ingresar datos de clientes y obtener predicciones
- **Análisis de Datos**: Visualizaciones interactivas del dataset y estadísticas descriptivas
- **Comparación de Modelos**: Análisis comparativo de diferentes algoritmos de ML
- **Información del Proyecto**: Documentación completa del modelo y metodología

## 📊 Modelo

- **Algoritmo**: SVM (Support Vector Machine)
- **Precisión**: 87.48% (AUC-ROC)
- **Dataset**: Bank Marketing Dataset (4,521 registros)
- **Variables**: 16 características demográficas y de comportamiento

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o descargar el proyecto**
   ```bash
   git clone <url-del-repositorio>
   cd <nombre-del-directorio>
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   
   # En Windows
   venv\Scripts\activate
   
   # En macOS/Linux
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar archivos necesarios**
   
   Asegúrate de tener los siguientes archivos en el directorio:
   - `modelo_svm_final.pkl` (modelo entrenado)
   - `bank.csv` (dataset original, opcional para análisis)

## 🏃‍♂️ Ejecución

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador web en la dirección `http://localhost:8501`

### Acceso desde otros dispositivos

Si quieres acceder desde otros dispositivos en la misma red:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 📱 Uso de la Aplicación

### 1. Predicción Individual

- Navega a la pestaña "🎯 Predicción Individual"
- Completa el formulario con los datos del cliente
- Haz clic en "🔮 Predecir" para obtener el resultado
- Visualiza las probabilidades y el resultado de la predicción

### 2. Análisis de Datos

- Ve a la pestaña "📈 Análisis de Datos"
- Explora las visualizaciones del dataset
- Revisa estadísticas descriptivas
- Analiza patrones por diferentes variables

### 3. Comparación de Modelos

- Accede a "📊 Comparación de Modelos"
- Compara métricas de diferentes algoritmos
- Entiende por qué se seleccionó SVM

### 4. Información del Proyecto

- Consulta "ℹ️ Información" para detalles técnicos
- Revisa la metodología y tecnologías utilizadas

## 📋 Variables del Modelo

### Variables Numéricas
- **age**: Edad del cliente (18-95)
- **balance**: Balance promedio anual
- **day**: Día del mes (1-31)
- **duration**: Duración de la última llamada
- **campaign**: Número de contactos en esta campaña
- **pdays**: Días desde el último contacto
- **previous**: Número de contactos antes de esta campaña

### Variables Categóricas
- **job**: Tipo de trabajo
- **marital**: Estado civil
- **education**: Nivel educativo
- **default**: ¿Tiene crédito en default?
- **housing**: ¿Tiene préstamo hipotecario?
- **loan**: ¿Tiene préstamo personal?
- **month**: Mes del año

## 🔧 Estructura del Proyecto

```
proyecto/
├── app.py                 # Aplicación principal de Streamlit
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Este archivo
├── modelo_svm_final.pkl  # Modelo entrenado (requerido)
├── bank.csv              # Dataset original (opcional)
└── Documentacion del proceso.ipynb  # Notebook con análisis
```

## 📈 Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| Precisión | 87.48% |
| Recall | 77.18% |
| F1-Score | 48.73% |
| AUC-ROC | 87.48% |
| AP | 44.75% |

## 🛠️ Tecnologías Utilizadas

- **Python**: Lenguaje principal
- **Streamlit**: Framework para aplicaciones web
- **Scikit-learn**: Machine Learning
- **Pandas**: Manipulación de datos
- **Plotly**: Visualizaciones interactivas
- **Joblib**: Serialización del modelo

## 🔍 Solución de Problemas

### Error: "No se encontró el archivo 'modelo_svm_final.pkl'"
- Verifica que el archivo del modelo esté en el directorio correcto
- Asegúrate de que el nombre del archivo sea exactamente `modelo_svm_final.pkl`

### Error: "No se encontró el archivo 'bank.csv'"
- Este archivo es opcional para el análisis de datos
- La aplicación funcionará sin él, pero no podrás ver las visualizaciones del dataset

### Problemas de dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📝 Notas Técnicas

- El modelo utiliza **class_weight='balanced'** para manejar el desbalance de clases
- Se aplica **StandardScaler** para normalizar variables numéricas
- Se usa **One-Hot Encoding** para variables categóricas
- El dataset original tiene 11.5% de casos positivos (desbalanceado)

## 🤝 Contribuciones

Si quieres contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

Para preguntas o soporte, por favor abre un issue en el repositorio.

---

**Desarrollado con ❤️ usando Streamlit** 