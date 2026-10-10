import numpy as np
import scipy.io as sio
from scipy.signal import decimate
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import os

# ==========================================
# PARTE 1: GRÁFICO DE REMUESTREO (FIGURA 1)
# ==========================================

# Cargar un archivo de ejemplo (Rodamiento Sano)
mat_data = sio.loadmat('H-B-1.mat')
# Extraer Canal 1 (Acelerómetro) y aplanar el arreglo
signal_original = mat_data['Channel_1'].flatten()

# Frecuencias
fs_original = 200000
fs_nueva = 10000
factor_decimacion = fs_original // fs_nueva

# Remuestrear usando decimate (aplica filtro antialiasing automático)
signal_remuestreada = decimate(signal_original, factor_decimacion)

# Crear vectores de tiempo para graficar un tramo corto (0.05 segundos)
t_original = np.linspace(0, 0.05, int(fs_original * 0.05), endpoint=False)
t_remuestreada = np.linspace(0, 0.05, int(fs_nueva * 0.05), endpoint=False)

# Graficar
plt.figure(figsize=(10, 5))
plt.plot(t_original, signal_original[:len(t_original)], label='Original (200 kHz)', alpha=0.5, color='gray')
plt.plot(t_remuestreada, signal_remuestreada[:len(t_remuestreada)], label='Remuestreada (10 kHz)', color='blue', marker='.')
plt.title('Demostración de Remuestreo: Señal de Vibración')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud de Aceleración')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_remuestreo.png', dpi=300)
plt.close()

# ==========================================
# PARTE 2: EXTRACCIÓN DE CARACTERÍSTICAS Y SCATTER PLOT (FIGURA 2)
# ==========================================

# Definir los archivos por clase para la condición 'Decreasing speed' (B)
archivos = {
    0: ['H-B-1.mat', 'H-B-2.mat', 'H-B-3.mat'],  # Healthy
    1: ['I-B-1.mat', 'I-B-2.mat', 'I-B-3.mat'],  # Inner Race Fault
    2: ['O-B-1.mat', 'O-B-2.mat', 'O-B-3.mat']   # Outer Race Fault
}

nombres_clases = {0: 'Healthy', 1: 'Inner Race Fault', 2: 'Outer Race Fault'}
colores = {0: 'green', 1: 'orange', 2: 'red'}

# Listas para guardar las características
todas_rms = []
todas_curtosis = []
todas_etiquetas = []

ventana_segundos = 0.1
muestras_por_ventana = int(fs_nueva * ventana_segundos) # 1000 muestras

for clase, lista_archivos in archivos.items():
    for archivo in lista_archivos:
        if os.path.exists(archivo):
            data = sio.loadmat(archivo)['Channel_1'].flatten()
            data_decimada = decimate(data, factor_decimacion)
            
            # Segmentar la señal
            num_ventanas = len(data_decimada) // muestras_por_ventana
            for i in range(num_ventanas):
                segmento = data_decimada[i*muestras_por_ventana : (i+1)*muestras_por_ventana]
                
                # Calcular características
                rms_val = np.sqrt(np.mean(segmento**2))
                curtosis_val = kurtosis(segmento, fisher=False)
                
                # Guardar
                todas_rms.append(rms_val)
                todas_curtosis.append(curtosis_val)
                todas_etiquetas.append(clase)

# Convertir a arrays para graficar fácilmente
todas_rms = np.array(todas_rms)
todas_curtosis = np.array(todas_curtosis)
todas_etiquetas = np.array(todas_etiquetas)

# Graficar Dispersión
plt.figure(figsize=(8, 6))
for clase in [0, 1, 2]:
    idx = (todas_etiquetas == clase)
    plt.scatter(todas_rms[idx], todas_curtosis[idx], 
                c=colores[clase], label=nombres_clases[clase], alpha=0.6, edgecolors='w', s=50)

plt.title('Espacio de Características: Curtosis vs RMS')
plt.xlabel('Valor RMS')
plt.ylabel('Curtosis')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('grafico_dispersion.png', dpi=300)
plt.close()

print("Proceso finalizado. Gráficos guardados como 'grafico_remuestreo.png' y 'grafico_dispersion.png'.")
