import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import scipy.linalg as la

# ==========================================
# 1. PARÁMETROS DEL SISTEMA Y RUTAS
# ==========================================
# Asegúrate de que esta ruta apunte a tu carpeta con los 42 archivos .csv
data_path = r"C:\Users\rodri\OneDrive\Desktop\Bearing2_4-20260531T115952Z-3-001\Bearing2_4"
num_files = 42

# Inicialización de matrices (42 filas x 8 columnas)
features_H = np.zeros((num_files, 8))
features_V = np.zeros((num_files, 8))

# ==========================================
# 2. DEFINICIÓN ANALÍTICA DE CARACTERÍSTICAS
# ==========================================
def extract_features(x):
    rms = np.sqrt(np.mean(x**2))
    std = np.std(x, ddof=1)
    max_abs = np.max(np.abs(x))
    p2p = np.max(x) - np.min(x)
    crest = max_abs / rms if rms > 0 else 0
    kurt = kurtosis(x, fisher=False) 
    skewness = skew(x)
    energy = np.sum(x**2)
    
    return [rms, std, max_abs, p2p, crest, kurt, skewness, energy]

# ==========================================
# 3. CÁLCULO ITERATIVO SOBRE MATRIZ DE DATOS
# ==========================================
for i in range(1, num_files + 1):
    file_path = os.path.join(data_path, f"{i}.csv")
    df = pd.read_csv(file_path)
    
    sig_H = df.iloc[:, 0].values
    sig_V = df.iloc[:, 1].values
    
    # Aquí es donde se define y llena features_H
    features_H[i-1, :] = extract_features(sig_H)
    features_V[i-1, :] = extract_features(sig_V)

# ==========================================
# 4. PREPARACIÓN DEL ESPACIO LATENTE (MODELOS DE ANOMALÍA)
# ==========================================
# Ahora sí, features_H existe y tiene datos. Extraemos el canal horizontal.
X = features_H.copy()
time_axis = np.arange(1, len(X) + 1)

# Definición del régimen nominal para entrenamiento (t = 15 min)
train_idx = 15
X_train = X[:train_idx, :]

# Estandarización (Z-score) calculada estrictamente sobre la condición sana
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(X) 

# ==========================================
# 5. IMPLEMENTACIÓN DE MODELOS DE DETECCIÓN
# ==========================================

# --- Método 1: Distancia de Mahalanobis ---
mu_train = np.mean(X_train_scaled, axis=0)
cov_train = np.cov(X_train_scaled, rowvar=False)
inv_cov = la.pinv(cov_train) 

mahalanobis_dist = np.zeros(len(X))
for i in range(len(X)):
    delta = X_scaled[i] - mu_train
    mahalanobis_dist[i] = np.sqrt(np.dot(np.dot(delta, inv_cov), delta.T))

threshold_mah = np.max(mahalanobis_dist[:train_idx]) + 3 * np.std(mahalanobis_dist[:train_idx])

# --- Método 2: One-Class SVM ---
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
ocsvm.fit(X_train_scaled)
score_ocsvm = ocsvm.decision_function(X_scaled)
threshold_ocsvm = 0.0 

# --- Método 3: Isolation Forest ---
iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iforest.fit(X_train_scaled)
score_iforest = iforest.decision_function(X_scaled)
threshold_iforest = 0.0

# ==========================================
# 6. CÁLCULO DE INSTANTES DE DETECCIÓN Y VISUALIZACIÓN
# ==========================================
minuto_mah = np.argmax(mahalanobis_dist > threshold_mah) + 1
minuto_svm = np.argmax(score_ocsvm < threshold_ocsvm) + 1
minuto_ifo = np.argmax(score_iforest < threshold_iforest) + 1

print("\n==========================================")
print(" RESULTADOS DE DETECCIÓN DE ANOMALÍAS")
print("==========================================")
print(f"Distancia de Mahalanobis : Minuto {minuto_mah}")
print(f"One-Class SVM            : Minuto {minuto_svm}")
print(f"Isolation Forest         : Minuto {minuto_ifo}")
print("==========================================\n")

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Subplot Mahalanobis
axes[0].plot(time_axis, mahalanobis_dist, color='red', marker='o', markersize=3, label='Distancia')
axes[0].axhline(threshold_mah, color='black', linestyle='--', label='Umbral')
axes[0].set_title('Distancia de Mahalanobis', fontweight='bold')
axes[0].set_ylabel('Distancia')

# Subplot OCSVM
axes[1].plot(time_axis, score_ocsvm, color='blue', marker='s', markersize=3, label='Función de Decisión')
axes[1].axhline(threshold_ocsvm, color='black', linestyle='--', label='Frontera (0.0)')
axes[1].set_title('One-Class SVM', fontweight='bold')
axes[1].set_ylabel('Puntaje')

# Subplot IForest
axes[2].plot(time_axis, score_iforest, color='green', marker='^', markersize=3, label='Función de Decisión')
axes[2].axhline(threshold_iforest, color='black', linestyle='--', label='Frontera (0.0)')
axes[2].set_title('Isolation Forest', fontweight='bold')
axes[2].set_xlabel('Tiempo de operación (minutos)')
axes[2].set_ylabel('Puntaje')

for ax in axes:
    ax.margins(x=0.03, y=0.1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

plt.tight_layout(pad=1.5)
plt.savefig('Anomaly_Detection_Comparison.pdf', format='pdf', bbox_inches='tight')
print("Gráfico comparativo de anomalías exportado exitosamente.")
