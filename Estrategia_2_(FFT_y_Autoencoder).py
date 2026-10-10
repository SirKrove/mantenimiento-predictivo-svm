import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. PARÁMETROS DEL SISTEMA Y RUTAS
# ==========================================
data_path = r"C:\Users\rodri\OneDrive\Desktop\Bearing2_4-20260531T115952Z-3-001\Bearing2_4"
num_files = 42

Fs = 25600       # Frecuencia de muestreo (Hz)
N = 32768        # Número de muestras por archivo
freq_res = Fs/N  # Resolución frecuencial (~0.78 Hz)

# Frecuencia de corte para el análisis espectral (5000 Hz)
# Justificación: Concentra la varianza de los armónicos mecánicos y descarta ruido de alta frecuencia
f_max = 5000 
idx_max = int(f_max / freq_res) 

# Matrices para almacenar los espectros concatenados
# Dimensión por archivo: 2 canales * idx_max
X_fft = np.zeros((num_files, idx_max * 2))

# ==========================================
# 2. CÁLCULO DE LA FFT (EXTRACCIÓN DE MAGNITUD)
# ==========================================
print("Extrayendo espectros FFT...")
for i in range(1, num_files + 1):
    file_path = os.path.join(data_path, f"{i}.csv")
    df = pd.read_csv(file_path)
    
    sig_H = df.iloc[:, 0].values
    sig_V = df.iloc[:, 1].values
    
    # Cálculo de la FFT (solo magnitud de las frecuencias positivas)
    fft_H = np.abs(np.fft.rfft(sig_H)) / N
    fft_V = np.abs(np.fft.rfft(sig_V)) / N
    
    # Truncamiento hasta f_max y concatenación horizontal [FFT_H, FFT_V]
    X_fft[i-1, :] = np.concatenate((fft_H[:idx_max], fft_V[:idx_max]))

# ==========================================
# 3. PREPARACIÓN DEL AUTOENCODER
# ==========================================
train_idx = 15  # Régimen nominal: Primeros 15 minutos
X_train_fft = X_fft[:train_idx, :]

# Escalamiento Min-Max (Recomendado para magnitudes de FFT y Autoencoders con ReLU/Sigmoid)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_fft)
X_scaled = scaler.transform(X_fft)

# Arquitectura de la Red Neuronal (Autoencoder Profundo)
input_dim = X_train_scaled.shape[1]

# Capa de Entrada
input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dense(128, activation='relu')(encoded)
latent_space = Dense(32, activation='relu')(encoded) # Cuello de botella

# Decoder
decoded = Dense(128, activation='relu')(latent_space)
decoded = Dense(512, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# ==========================================
# 4. ENTRENAMIENTO DEL MODELO
# ==========================================
print(f"Entrenando Autoencoder con {input_dim} características (épocas=50)...")
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=4,
    shuffle=True,
    verbose=0 # Cambiar a 1 si quieres ver el progreso en consola
)

# ==========================================
# 5. CÁLCULO DE ANOMALÍAS (ERROR DE RECONSTRUCCIÓN)
# ==========================================
# Reconstruir los datos
X_train_pred = autoencoder.predict(X_train_scaled)
X_pred = autoencoder.predict(X_scaled)

# Calcular Error Cuadrático Medio (MSE) por cada archivo (vector a vector)
mse_train = np.mean(np.power(X_train_scaled - X_train_pred, 2), axis=1)
mse_total = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

# Límite superior de normalidad: Media del error nominal + 3 Desviaciones Estándar
threshold_ae = np.mean(mse_train) + 3 * np.std(mse_train)

# Extracción del minuto de detección
minuto_ae = np.argmax(mse_total > threshold_ae) + 1
time_axis = np.arange(1, num_files + 1)

print("\n==========================================")
print(" RESULTADO AUTOENCODER (ESTRATEGIA 2)")
print("==========================================")
print(f"El Autoencoder detecta anomalía espectral en el Minuto: {minuto_ae}")
print("==========================================\n")

# ==========================================
# 6. VISUALIZACIÓN DEL RESULTADO (Con márgenes ajustados)
# ==========================================
plt.figure(figsize=(12, 4))
plt.plot(time_axis, mse_total, color='purple', marker='D', markersize=4, label='Error de Reconstrucción (MSE)')
plt.axhline(threshold_ae, color='black', linestyle='--', label='Umbral de Anomalía')

# Rellenar el área de entrenamiento para referencia visual
plt.axvspan(1, train_idx, color='gray', alpha=0.2, label='Zona de Entrenamiento')

plt.title('Detección de Anomalías mediante FFT y Autoencoder', fontweight='bold')
plt.xlabel('Tiempo de operación (minutos)')
plt.ylabel('Puntaje de Anomalía (MSE)')

# Se elimina plt.xlim() para que actúen los márgenes. 
# x=0.03 (3% extra en eje X), y=0.1 (10% extra en eje Y)
plt.margins(x=0.03, y=0.1)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')

plt.tight_layout(pad=2.0)
plt.savefig('Autoencoder_Anomaly.pdf', format='pdf', bbox_inches='tight', pad_inches=0.3)
plt.show()
