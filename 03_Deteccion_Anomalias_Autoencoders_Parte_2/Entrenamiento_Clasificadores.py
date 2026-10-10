import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from scipy.signal import decimate
from scipy.stats import kurtosis, skew

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# ==========================================
# CANDADO DE SEGURIDAD PARA WINDOWS/IDLE
# ==========================================
if __name__ == '__main__':
    
    # ==========================================
    # 1. EXTRACCIÓN DE CARACTERÍSTICAS (5 METRICAS)
    # ==========================================
    archivos = {
        0: ['H-B-1.mat', 'H-B-2.mat', 'H-B-3.mat'],  # Healthy
        1: ['I-B-1.mat', 'I-B-2.mat', 'I-B-3.mat'],  # Inner Race Fault
        2: ['O-B-1.mat', 'O-B-2.mat', 'O-B-3.mat']   # Outer Race Fault
    }

    fs_original = 200000
    fs_nueva = 10000
    factor_decimacion = fs_original // fs_nueva
    ventana_segundos = 0.1
    muestras_por_ventana = int(fs_nueva * ventana_segundos)

    X = []
    y = []

    print("Extrayendo características de las señales...")
    for clase, lista_archivos in archivos.items():
        for archivo in lista_archivos:
            if os.path.exists(archivo):
                data = sio.loadmat(archivo)['Channel_1'].flatten()
                data_decimada = decimate(data, factor_decimacion)
                
                num_ventanas = len(data_decimada) // muestras_por_ventana
                for i in range(num_ventanas):
                    segmento = data_decimada[i*muestras_por_ventana : (i+1)*muestras_por_ventana]
                    
                    # Calcular las 5 características solicitadas en el informe
                    rms_val = np.sqrt(np.mean(segmento**2))
                    curtosis_val = kurtosis(segmento, fisher=False)
                    skewness_val = skew(segmento)
                    crest_factor = np.max(np.abs(segmento)) / rms_val if rms_val > 0 else 0
                    p2p_val = np.max(segmento) - np.min(segmento)
                    
                    X.append([rms_val, crest_factor, curtosis_val, skewness_val, p2p_val])
                    y.append(clase)

    X = np.array(X)
    y = np.array(y)

    # ==========================================
    # 2. PREPARACIÓN DE DATOS (TRAIN/TEST Y ESCALADO)
    # ==========================================
    # División 70% entrenamiento (que GridSearch usará para validación cruzada internamente) y 30% prueba aislada
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

    # Estandarización Z-score
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 3. BÚSQUEDA DE HIPERPARÁMETROS (GRID SEARCH)
    # ==========================================
    print("\nIniciando Grid Search (esto puede tardar un par de minutos, paciencia)...")

    # Diccionarios para guardar los mejores modelos y resultados
    mejores_modelos = {}
    nombres_clases = ['Sano', 'Pista Interna', 'Pista Externa']

    # --- Modelo 1: k-NN ---
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 11],
        'metric': ['euclidean', 'manhattan'],
        'weights': ['uniform', 'distance']
    }
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='balanced_accuracy', n_jobs=1)
    grid_knn.fit(X_train_scaled, y_train)
    mejores_modelos['k-NN'] = grid_knn.best_estimator_
    print(f"Mejores parámetros k-NN: {grid_knn.best_params_}")

    # --- Modelo 2: Random Forest ---
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='balanced_accuracy', n_jobs=1)
    grid_rf.fit(X_train_scaled, y_train)
    mejores_modelos['Random Forest'] = grid_rf.best_estimator_
    print(f"Mejores parámetros Random Forest: {grid_rf.best_params_}")

    # --- Modelo 3: SVC ---
    param_grid_svc = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1]
    }
    grid_svc = GridSearchCV(SVC(random_state=42), param_grid_svc, cv=5, scoring='balanced_accuracy', n_jobs=1)
    grid_svc.fit(X_train_scaled, y_train)
    mejores_modelos['SVC'] = grid_svc.best_estimator_
    print(f"Mejores parámetros SVC: {grid_svc.best_params_}")

    # --- Modelo 4: Red Neuronal (MLP) ---
    param_grid_mlp = {
        'hidden_layer_sizes': [(64,), (128, 64), (64, 32)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    }
    # max_iter aumentado para asegurar convergencia
    grid_mlp = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42), param_grid_mlp, cv=5, scoring='balanced_accuracy', n_jobs=1)
    grid_mlp.fit(X_train_scaled, y_train)
    mejores_modelos['Red Neuronal'] = grid_mlp.best_estimator_
    print(f"Mejores parámetros Red Neuronal: {grid_mlp.best_params_}")

    # ==========================================
    # 4. EVALUACIÓN EN CONJUNTO DE PRUEBA Y GRÁFICOS
    # ==========================================
    print("\n--- MÉTRICAS DE EVALUACIÓN PARA LA TABLA DE LATEX ---")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (nombre, modelo) in enumerate(mejores_modelos.items()):
        # Predecir
        y_pred = modelo.predict(X_test_scaled)
        
        # Calcular métricas
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='macro') * 100
        rec = recall_score(y_test, y_pred, average='macro') * 100
        f1 = f1_score(y_test, y_pred, average='macro') * 100
        
        # Imprimir para rellenar la tabla
        print(f"\n{nombre}:")
        print(f"Accuracy: {acc:.1f} % | Precision: {prec:.1f} % | Recall: {rec:.1f} % | F1-Score: {f1:.1f} %")
        
        # Graficar Matriz de Confusión
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=nombres_clases, cmap='Blues', ax=axes[idx], colorbar=False)
        axes[idx].set_title(f'Matriz de Confusión: {nombre}')

    plt.tight_layout()
    plt.savefig('matrices_confusion.png', dpi=300)
    print("\nProceso finalizado. Gráfico guardado como 'matrices_confusion.png'.")
