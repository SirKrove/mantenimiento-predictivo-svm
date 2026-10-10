import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# --- 1. PREPROCESAMIENTO EN SEGUNDO PLANO ---
ruta_excel = r'C:\Users\rodri\OneDrive\Desktop\Archivos de Tarea 2\datos_tarea2.xlsx'

df = pd.read_excel(ruta_excel, engine='openpyxl')
X = df.drop(columns=['Equipo', 'Desc Componente', 'Tag', 'Correlativo Muestra', 'Fecha'])

varianzas = X.var()
X_var = X.loc[:, varianzas > 0.05]

R = X_var.corr(method='pearson')
upper_tri = R.where(np.triu(np.ones(R.shape), k=1).astype(bool))
X_final = X_var.drop(columns=[c for c in upper_tri.columns if any(upper_tri[c] > 0.85)])

X_scaled = StandardScaler().fit_transform(X_final)
X_pca = PCA(n_components=3).fit_transform(X_scaled)

# --- 2. CÁLCULO DE MÉTRICAS: ETAPA 4 ---
rango_K = range(2, 9)
silueta_kmeans, ch_kmeans = [], []
silueta_ward, ch_ward = [], []

for k in rango_K:
    # K-Medias
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels_k = kmeans.fit_predict(X_pca)
    silueta_kmeans.append(silhouette_score(X_pca, labels_k))
    ch_kmeans.append(calinski_harabasz_score(X_pca, labels_k))
    
    # Jerárquico (Ward)
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels_w = ward.fit_predict(X_pca)
    silueta_ward.append(silhouette_score(X_pca, labels_w))
    ch_ward.append(calinski_harabasz_score(X_pca, labels_w))

# --- 3. EXPORTACIÓN DE GRÁFICOS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de Silueta
ax1.plot(rango_K, silueta_kmeans, marker='o', label='K-Medias', color='blue')
ax1.plot(rango_K, silueta_ward, marker='s', label='Jerárquico (Ward)', color='orange')
ax1.set_title('Coeficiente de Silueta ($S$) vs $K$')
ax1.set_xlabel('Número de Clústeres ($K$)')
ax1.set_ylabel('Puntaje de Silueta')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Gráfico Calinski-Harabasz
ax2.plot(rango_K, ch_kmeans, marker='o', label='K-Medias', color='blue')
ax2.plot(rango_K, ch_ward, marker='s', label='Jerárquico (Ward)', color='orange')
ax2.set_title('Índice Calinski-Harabasz ($CH$) vs $K$')
ax2.set_xlabel('Número de Clústeres ($K$)')
ax2.set_ylabel('Puntaje $CH$')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig('metricas_evaluacion.png', dpi=300)
plt.close()

print("Gráfico de métricas exportado como 'metricas_evaluacion.png'")
print(f"\n--- RESULTADOS PARA EL INFORME (K=3) ---")
print(f"K-Means -> Silueta: {silueta_kmeans[1]:.4f} | CH: {ch_kmeans[1]:.2f}")
print(f"Ward    -> Silueta: {silueta_ward[1]:.4f} | CH: {ch_ward[1]:.2f}")
