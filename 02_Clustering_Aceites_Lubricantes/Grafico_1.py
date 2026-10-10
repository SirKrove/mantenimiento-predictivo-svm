import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carga de datos directa
ruta_excel = r'C:\Users\rodri\OneDrive\Desktop\Archivos de Tarea 2\datos_tarea2.xlsx'

# Lectura de la matriz estática
df = pd.read_excel(ruta_excel, engine='openpyxl')
print("Archivo Excel cargado con éxito.")

# Aislar subespacio numérico X
columnas_no_numericas = ['Equipo', 'Desc Componente', 'Tag', 'Correlativo Muestra', 'Fecha']
X = df.drop(columns=columnas_no_numericas)

# 2. Evaluación de elementos nulos
missing_values = X.isnull().sum().sum()
print(f"Total de valores NaN: {missing_values}")

# 3. Filtrado por varianza (umbral epsilon = 0.05)
varianzas = X.var()
umbral_varianza = 0.05
X_var = X.loc[:, varianzas > umbral_varianza]
eliminadas_por_var = varianzas[varianzas <= umbral_varianza].index.tolist()
print(f"Variables eliminadas por varianza (sigma^2 <= {umbral_varianza}): {eliminadas_por_var}")

# 4. Matriz de correlación R
R = X_var.corr(method='pearson')

plt.figure(figsize=(14, 10))
sns.heatmap(R, annot=False, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Matriz de Correlación de Pearson ($R$)')
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300)
plt.close()
print("Gráfico de correlación exportado: matriz_correlacion.png")

# 5. Filtrado por colinealidad (|R| > 0.85)
umbral_corr = 0.85
upper_tri = R.where(np.triu(np.ones(R.shape), k=1).astype(bool))
eliminadas_por_corr = [column for column in upper_tri.columns if any(upper_tri[column] > umbral_corr)]
X_final = X_var.drop(columns=eliminadas_por_corr)

print(f"Variables eliminadas por alta correlación: {eliminadas_por_corr}")
print(f"Subconjunto final (M' = {X_final.shape[1]}): {X_final.columns.tolist()}")
