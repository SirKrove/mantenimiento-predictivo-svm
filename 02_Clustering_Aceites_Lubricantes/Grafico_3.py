import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

# --- 1. PREPROCESAMIENTO CONSOLIDADO EN SEGUNDO PLANO ---
ruta_excel = r'C:\Users\rodri\OneDrive\Desktop\Archivos de Tarea 2\datos_tarea2.xlsx'

if not os.path.exists(ruta_excel):
    raise FileNotFoundError("Verifica que el Excel esté en 'Archivos de Tarea 2' en el Escritorio.")

df = pd.read_excel(ruta_excel, engine='openpyxl')
X = df.drop(columns=['Equipo', 'Desc Componente', 'Tag', 'Correlativo Muestra', 'Fecha'])

# Filtro de Varianza
varianzas = X.var()
X_var = X.loc[:, varianzas > 0.05]

# Filtro de Correlación
R = X_var.corr(method='pearson')
upper_tri = R.where(np.triu(np.ones(R.shape), k=1).astype(bool))
X_final = X_var.drop(columns=[c for c in upper_tri.columns if any(upper_tri[c] > 0.85)])

# Estandarización y PCA a 3 componentes
X_scaled = StandardScaler().fit_transform(X_final)
X_pca = PCA(n_components=3).fit_transform(X_scaled)

# --- 2. APLICACIÓN DE CLUSTERING: ETAPA 3 ---
K_optimo = 3 # Definido analíticamente para la visualización topológica

# K-Medias
kmeans = KMeans(n_clusters=K_optimo, init='k-means++', random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_pca)
centroides = kmeans.cluster_centers_

# Jerárquico (Enlace de Ward)
jerarquico = AgglomerativeClustering(n_clusters=K_optimo, linkage='ward')
labels_jerarquico = jerarquico.fit_predict(X_pca)

# --- 3. EXPORTACIÓN DE GRÁFICOS ---
fig = plt.figure(figsize=(14, 6))

# Gráfico 1: K-Means
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_kmeans, cmap='viridis', alpha=0.6, edgecolors='k')
ax1.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], c='red', marker='X', s=150, label='Centroides $\mu_k$')
ax1.set_title('Clustering Particional: K-Medias')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.legend()

# Gráfico 2: Jerárquico (Ward)
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_jerarquico, cmap='plasma', alpha=0.6, edgecolors='k')
ax2.set_title('Clustering Aglomerativo: Enlace de Ward')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')

plt.tight_layout()
plt.savefig('clustering_comparativa.png', dpi=300)
plt.close()

print("Script consolidado de la Etapa 3 ejecutado con éxito.")
print("Gráfico comparativo exportado como 'clustering_comparativa.png'")
