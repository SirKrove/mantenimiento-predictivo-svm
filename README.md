# ⚙️ Portafolio de Procesamiento de Señales y Machine Learning para Mantenimiento Predictivo

## 📌 Descripción General
Este repositorio contiene un conjunto integral de proyectos aplicados al **mantenimiento predictivo industrial** y monitoreo de condición (CBM). Los modelos y scripts aquí desarrollados abordan diversas fenomenologías de fallas mecánicas típicas en la minería y energía (desalineamiento en bombas, degradación de rodamientos y contaminación de aceites lubricantes).

Los proyectos fueron desarrollados mediante Python en el marco del curso de especialidad *ME7200* de la Facultad de Ciencias Físicas y Matemáticas (FCFM) de la Universidad de Chile.

## 🛠️ Tecnologías y Librerías Utilizadas
* **Procesamiento de Señales:** `SciPy` (Transformada de Fourier/FFT, Método de Welch, Ventanas de Hann).
* **Machine Learning Clásico:** `Scikit-Learn` (PCA, K-Medias, Agrupamiento Jerárquico/Ward, Isolation Forest, OCSVM, Random Forest).
* **Deep Learning y Ensambles:** `TensorFlow/Keras` (Autoencoders, Redes Neuronales Recurrentes LSTM), `XGBoost`.
* **Análisis de Datos:** `Pandas`, `NumPy`, `Matplotlib`.

---

## 📂 Proyectos Incluidos

### 1. [Análisis Espectral de Vibraciones en Bombas Centrífugas](./01_Analisis_Espectral_Vibraciones)
* **Objetivo:** Identificar causa raíz de fallas mediante análisis en el dominio de la frecuencia.
* **Metodología:** Aplicación del método de Welch y ventaneado de Hann para reducir ruido estocástico (spectral leakage). 
* **Resultados:** Diagnóstico exitoso de anomalías cinemáticas (desalineamiento paralelo evidenciado por el armónico 2X, y excentricidad del impulsor mediante bandas laterales).

### 2. [Análisis Tribológico de Aceites mediante Clustering](./02_Clustering_Aceites_Lubricantes)
* **Objetivo:** Segmentar estados de desgaste de descansos mecánicos sin etiquetas previas (Aprendizaje No Supervisado).
* **Metodología:** Reducción de dimensionalidad mediante Análisis de Componentes Principales (PCA) y partición topológica usando K-Medias y Agrupamiento Jerárquico (Enlace de Ward).
* **Resultados:** Identificación temprana de degradación físico-química y metales de desgaste para justificar mantenciones preventivas.

### 3. [Detección de Anomalías (Novedades) en Rodamientos](./03_Deteccion_Anomalias_Autoencoders)
* **Objetivo:** Detectar el instante exacto del inicio de fallas (spalling) en rodamientos sometidos a régimen de desgaste acelerado.
* **Metodología:** Contraste entre algoritmos estadísticos (Isolation Forest, Mahalanobis) y reconstrucción topológica mediante arquitecturas **Autoencoder** alimentadas por vectores espectrales (FFT).

### 4. [Pronóstico de Vida Útil Remanente (RUL)](./04_Pronostico_Vida_Util_RUL)
* **Objetivo:** Modelar la trayectoria de degradación y predecir el RUL de rodamientos de bolas.
* **Metodología:** Contraste entre modelos de regresión estáticos (XGBoost usando Curtosis y RMS) frente a modelos secuenciales (**LSTM**).
* **Resultados:** Validación de la superioridad de la arquitectura recurrente LSTM para interiorizar dependencias a largo plazo y predecir gradientes de falla con alta precisión ($R^2$ positivo y bajo MAE).

---

## 👨‍💻 Autor
**Rodrigo Opazo Bucarey**  
Estudiante de Ingeniería Civil Eléctrica | Universidad de Chile (FCFM)  
* [LinkedIn]([Pega aquí tu link])
* [Email](mailto:rodrigo.opazo.b.750@gmail.com)
