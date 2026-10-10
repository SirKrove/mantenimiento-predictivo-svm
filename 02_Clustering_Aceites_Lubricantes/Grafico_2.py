import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Carga de datos directa desde la nueva carpeta
ruta_excel = r'C:\Users\rodri\OneDrive\Desktop\Archivos de Tarea 2\datos_tarea2.xlsx'

if not os.path.exists(ruta_excel):
    raise FileNotFoundError(f"No se encontró el archivo en la ruta especificada: {ruta_excel}")

df = pd.read_excel(ruta_excel, engine='openpyxl')
print("Archivo Excel cargado con éxito para la Etapa 2.")

# Aislar subespacio numérico X
columnas_no_numericas = ['Equipo', 'Desc Componente', 'Tag', 'Correlativo Muestra', 'Fecha']
X = df.drop(columns=columnas_no_numericas)

# Replicar el filtrado de la Etapa 1 para garantizar continuidad matemática
varianzas = X.var()
umbral_varianza = 0.05
X_var = X.loc[:, varianzas > umbral_varianza]

R = X_var.corr(method='pearson')
umbral_corr = 0.85
upper_tri = R.where(np.triu(np.ones(R.shape), k=1).astype(bool))
eliminadas_por_corr = [column for column in upper_tri.columns if any(upper_tri[column] > umbral_corr)]
X_final = X_var.drop(columns=eliminadas_por_corr)

# --- INICIO DESARROLLO ETAPA 2 ---

# 2. Estandarización Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# 3. Reducción de dimensionalidad mediante PCA a 3 componentes
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
var_explicada = pca.explained_variance_ratio_

print("\n--- RESULTADOS NUMÉRICOS DE PCA ---")
print(f"Varianza explicada por PC1: {var_explicada[0]*100:.2f}%")
print(f"Varianza explicada por PC2: {var_explicada[1]*100:.2f}%")
print(f"Varianza explicada por PC3: {var_explicada[2]*100:.2f}%")
print(f"Varianza total acumulada en 3D: {sum(var_explicada)*100:.2f}%")

# 4. Generación y exportación del gráfico tridimensional
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Proyección de las observaciones en el espacio latente
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='blue', marker='o', alpha=0.6, edgecolors='k')

ax.set_title('Proyección Espacial de Muestras mediante PCA (3 Componentes)')
ax.set_xlabel(f'PC1 ({var_explicada[0]*100:.2f}%)')
ax.set_ylabel(f'PC2 ({var_explicada[1]*100:.2f}%)')
ax.set_zlabel(f'PC3 ({var_explicada[2]*100:.2f}%)')

plt.tight_layout()
plt.savefig('pca_3d.png', dpi=300)
plt.close()
print("\nGráfico PCA 3D exportado con éxito como 'pca_3d.png'")
