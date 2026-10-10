import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

# ============================================================
# Función de la Prof. Viviana (Mantiene la lógica intacta)
# ============================================================
def calcular_rpm_desde_encoder(filename, fs=200000, umbral=2.0, pulsos_por_vuelta=1024, ventana_suavizado=200):
    mat = sio.loadmat(filename)
    encoder = mat["Channel_2"].squeeze()
    tiempo = np.arange(len(encoder)) / fs
    encoder_binario = encoder > umbral
    
    # Flancos ascendentes
    flancos = np.where(np.diff(encoder_binario.astype(int)) == 1)[0]
    tiempo_flancos = flancos / fs
    
    # Calcular dt y filtrar anomalías
    dt = np.diff(tiempo_flancos)
    dt_mediana = np.median(dt)
    mascara = ((dt > 0.3 * dt_mediana) & (dt < 3.0 * dt_mediana))
    dt_filtrado = dt[mascara]
    tiempo_rpm = tiempo_flancos[1:][mascara]
    
    # Calcular RPM y suavizar
    rpm_pulsos = 60 / (dt_filtrado * pulsos_por_vuelta)
    if ventana_suavizado > 1:
        rpm_suave = pd.Series(rpm_pulsos).rolling(window=ventana_suavizado, center=True, min_periods=1).mean().to_numpy()
    else:
        rpm_suave = rpm_pulsos
        
    rpm_interpolada = np.interp(tiempo, tiempo_rpm, rpm_suave)
    df = pd.DataFrame({"Tiempo (s)": tiempo, "RPM": rpm_interpolada})
    
    info = {"rpm_min": np.min(rpm_interpolada), "rpm_max": np.max(rpm_interpolada)}
    return df, info

# ============================================================
# Procesamiento de Archivos Locales (Condición B)
# ============================================================
if __name__ == '__main__':
    
    # Solo necesitamos 1 ensayo por clase para visualizar el perfil decreciente
    archivos_b = {
        "Estado Nominal (Sano)": "H-B-1.mat",
        "Falla Pista Interna": "I-B-1.mat",
        "Falla Pista Externa": "O-B-1.mat"
    }

    fs_original = 200000
    datos_rpm = {}

    print("Extrayendo señal del encoder y calculando RPM...")
    
    for clase, archivo in archivos_b.items():
        if os.path.exists(archivo):
            df, info = calcular_rpm_desde_encoder(archivo, fs=fs_original)
            datos_rpm[clase] = df
            print(f"[{clase}] Procesado exitosamente. Bajó de {info['rpm_max']:.1f} a {info['rpm_min']:.1f} RPM.")
        else:
            print(f"Error: No se encontró el archivo {archivo}.")

    # ============================================================
    # Generación de Gráfico para Overleaf
    # ============================================================
    plt.figure(figsize=(10, 5))
    
    # Colores consistentes con la estética académica
    colores = {"Estado Nominal (Sano)": "green", "Falla Pista Interna": "orange", "Falla Pista Externa": "red"}

    for clase, df in datos_rpm.items():
        plt.plot(df["Tiempo (s)"], df["RPM"], label=clase, color=colores[clase], linewidth=1.5)

    plt.title("Perfil Cinemático: Variación de la Velocidad Rotacional (Decreciente)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad de giro [RPM]")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper right")
    plt.xlim([0, 10])

    plt.tight_layout()
    plt.savefig("perfil_velocidad_decreciente.png", dpi=300)
    print("\n¡Gráfico guardado como 'perfil_velocidad_decreciente.png'!")
    plt.show()
