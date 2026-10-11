import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# ==========================================
# 1. PARÁMETROS DEL SISTEMA Y RUTAS
# ==========================================
data_path = r"C:\Users\rodri\OneDrive\Desktop\Bearing2_4-20260531T115952Z-3-001\Bearing2_4"
num_files = 42

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
# 3. CÁLCULO ITERATIVO 
# ==========================================
for i in range(1, num_files + 1):
    file_path = os.path.join(data_path, f"{i}.csv")
    df = pd.read_csv(file_path)
    
    sig_H = df.iloc[:, 0].values
    sig_V = df.iloc[:, 1].values
    
    features_H[i-1, :] = extract_features(sig_H)
    features_V[i-1, :] = extract_features(sig_V)

# ==========================================
# 4. VISUALIZACIÓN Y EXPORTACIÓN VECTORIAL
# ==========================================
feature_names = [
    'RMS', 'Standard_Deviation', 'Maximum_Absolute', 'Peak-to-Peak', 
    'Crest_Factor', 'Kurtosis', 'Skewness', 'Energy'
]

plot_titles = [
    'Valor RMS', 'Desviación Estándar', 'Máximo Absoluto', 'Valor Pico a Pico', 
    'Factor de Cresta', 'Curtosis', 'Asimetría (Skewness)', 'Energía'
]

time_axis = np.arange(1, num_files + 1)

for j in range(8):
    plt.figure(figsize=(12, 4))
    
    plt.plot(time_axis, features_H[:, j], label='Horizontal', color='blue', marker='o', markersize=4, linewidth=1.5)
    plt.plot(time_axis, features_V[:, j], label='Vertical', color='orange', marker='s', markersize=4, linewidth=1.5)
    
    plt.title(f'Evolución Temporal: {plot_titles[j]}', fontweight='bold')
    plt.xlabel('Tiempo de operación (minutos)')
    plt.ylabel('Amplitud')
    
    # Se elimina plt.xlim() para que actúen los márgenes. 
    # x=0.03 (3% extra en eje X), y=0.1 (10% extra en eje Y)
    plt.margins(x=0.03, y=0.1)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Condicional de posición geométrica para la leyenda
    if feature_names[j] == 'Skewness':
        plt.legend(loc='upper right')
    else:
        plt.legend(loc='upper left')
        
    plt.tight_layout(pad=2.0)
    
    file_name = f"{feature_names[j]}_evolution.pdf"
    plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.3)
    plt.close()
