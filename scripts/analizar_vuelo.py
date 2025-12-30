import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar el CSV
try:
    df = pd.read_csv('unity_replay_ep0.csv')
    print("✅ Archivo cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: No encuentro 'unity_replay_ep0.csv'. Asegúrate de que esté en la carpeta.")
    exit()

# Configurar la gráfica
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
time = df['time']

# --- 1. GRÁFICA DE MOTORES (La Prueba de Fuego) ---
# Si esto parece un código de barras sólido, es Bang-Bang (MALO).
# Si son líneas curvas que suben y bajan, es Control Suave (BUENO).
axs[0].plot(time, df['m1_rpm'], label='M1', alpha=0.8, linewidth=1)
axs[0].plot(time, df['m2_rpm'], label='M2', alpha=0.8, linewidth=1)
axs[0].plot(time, df['m3_rpm'], label='M3', alpha=0.8, linewidth=1)
axs[0].plot(time, df['m4_rpm'], label='M4', alpha=0.8, linewidth=1)
axs[0].set_ylabel('RPM de Motores')
axs[0].set_title('ANÁLISIS DE MOTORES (¿Suavidad o Bang-Bang?)')
axs[0].legend(loc='upper right')
axs[0].grid(True, alpha=0.3)

# --- 2. GRÁFICA DE ACCIONES (Lo que piensa el cerebro) ---
# Aquí vemos qué está pidiendo la red neuronal (-1 a 1)
axs[1].plot(time, df['action_0'], label='Action 0', alpha=0.6)
axs[1].plot(time, df['action_1'], label='Action 1', alpha=0.6)
axs[1].plot(time, df['action_2'], label='Action 2', alpha=0.6)
axs[1].plot(time, df['action_3'], label='Action 3', alpha=0.6)
axs[1].set_ylabel('Acción Normalizada (-1 a 1)')
axs[1].set_title('COMANDOS DE LA IA')
axs[1].grid(True, alpha=0.3)

# --- 3. POSICIÓN Z (Altura) ---
# Para ver si mantiene la altura estable o rebota
axs[2].plot(time, df['pos_z'], label='Altura Real (Z)', color='green', linewidth=2)
# Dibujar línea del objetivo (asumiendo que es hover, buscamos estabilidad)
axs[2].set_ylabel('Altura (metros)')
axs[2].set_title('ESTABILIDAD DE VUELO')
axs[2].grid(True, alpha=0.3)
axs[2].set_xlabel('Tiempo (segundos)')

plt.tight_layout()
plt.show()

# --- ANÁLISIS NUMÉRICO DE "JERK" (Tirones) ---
# Calculamos cuánto cambian las RPM promedio por paso
rpm_changes = np.diff(df[['m1_rpm', 'm2_rpm', 'm3_rpm', 'm4_rpm']], axis=0)
avg_jerk = np.mean(np.abs(rpm_changes))
max_jerk = np.max(np.abs(rpm_changes))

print("\nResultados del Análisis:")
print(f"Cambio Promedio de RPM por paso: {avg_jerk:.2f} RPM")
print(f"Cambio Máximo de RPM (Pico):     {max_jerk:.2f} RPM")

if avg_jerk > 1000:
    print("\n⚠️ DIAGNÓSTICO: POSIBLE BANG-BANG.")
    print("   Los motores cambian demasiado rápido. Peligroso para hardware real.")
elif avg_jerk < 200:
    print("\n✅ DIAGNÓSTICO: CONTROL SUAVE (Sim2Real Ready).")
    print("   Los cambios son graduales. Seguro para motores reales.")
else:
    print("\n⚠️ DIAGNÓSTICO: AGRESIVO.")
    print("   Es volable, pero los motores se calentarán.")