# ⚙️ Mantenimiento Predictivo de Rodamientos con Machine Learning

## 📌 Descripción del Proyecto
Este repositorio contiene el desarrollo de un modelo de Machine Learning aplicado al **mantenimiento predictivo industrial**. El objetivo principal es detectar y clasificar fallas tempranas en rodamientos mecánicos utilizando análisis de señales de vibración. Esta aplicación es un pilar crítico en la industria minera y energética para evitar detenciones operativas no planificadas y optimizar la vida útil de los activos físicos.

Proyecto desarrollado como parte del curso ME7200 en la Facultad de Ciencias Físicas y Matemáticas (FCFM) de la Universidad de Chile.

## 🛠️ Tecnologías y Metodología
El pipeline de datos y modelamiento está construido íntegramente en **Python** y se divide en tres etapas clave:

1. **Procesamiento de Señales:** Aplicación de la Transformada Rápida de Fourier (FFT) mediante `SciPy` para transitar las señales de vibración del dominio del tiempo al dominio de la frecuencia, permitiendo aislar los armónicos característicos de falla.
2. **Extracción de Características:** Implementación de **Autoencoders** para reducir la dimensionalidad de los datos y capturar las representaciones latentes más relevantes de la condición del equipo.
3. **Clasificación de Fallas:** Entrenamiento y validación de un modelo de Máquinas de Vectores de Soporte (**SVM**) a través de `Scikit-Learn` para categorizar el estado de salud del componente (ej. Operación normal, falla en pista interna, falla en elemento rodante).

## 📊 Resultados Destacados
* Se logró una precisión (*accuracy*) del **[Insertar % aquí, ej. 94%]** en la clasificación de los distintos modos de falla durante la validación cruzada.
* El pipeline demostró robustez al aislar el ruido de fondo y detectar firmas de vibración anómalas, estableciendo una prueba de concepto viable para el monitoreo de condición (CBM) en motores y chancadoras.

## 🚀 Instrucciones de Ejecución
1. Clonar este repositorio: 
   ```bash
   git clone [https://github.com/SirKrove/](https://github.com/SirKrove/)[nombre-de-tu-repositorio].git
