import numpy as np
import matplotlib.pyplot as plt

# Reale Beobachtungsdaten (Stand 2026)
HUBBLE_TENSION_REL = 0.083      # ΔH₀ / H₀ ≈ 8.3%
S8_TENSION_REL = 0.15           # grobe S8-Tension-Schätzung
PHI_INV = 1 / ((1 + np.sqrt(5)) / 2)   # ≈ 0.618

def inverse_kdm(delta_obs, C, DOC, efficiency=0.6, phi_inv=PHI_INV):
    """
    Umkehraufgabe: Welches κ_dm braucht man, um die beobachtete Anomalie zu erklären?
    delta_obs: relative Abweichung (z.B. Hubble-Tension)
    C: Kohärenz-Level
    DOC: Degradation/Blockade
    efficiency: geschätzter Wirkungsgrad des Rücktransports
    """
    denominator = (C ** 3.0) * (1.0 - DOC) * phi_inv * efficiency
    if denominator <= 0:
        return np.inf
    return delta_obs / denominator

# Parameter-Raster für Plot
C_values = np.linspace(0.6, 1.0, 100)   # Kohärenz von 0.6 bis 1.0
DOC_values = [0.0, 0.05, 0.1, 0.2]      # verschiedene Blockade-Level

plt.figure(figsize=(10, 6))

for doc in DOC_values:
    kdm_values = [inverse_kdm(HUBBLE_TENSION_REL, c, doc) for c in C_values]
    label = f"DOC = {doc}"
    plt.plot(C_values, kdm_values, label=label, linewidth=2)

# Referenz-Linien aus realen Schätzungen
plt.axhline(0.33, color='red', linestyle='--', label="κ_dm ≈ 0.33 (Hubble-Tension Fit)")
plt.axhline(0.5, color='orange', linestyle='--', label="κ_dm ≈ 0.5 (oberer realistischer Bereich)")

plt.xlabel("Kohärenz-Level C")
plt.ylabel("Benötigte DM-Kopplung κ_dm")
plt.title("Umkehraufgabe: Welche κ_dm erklärt die Hubble-Tension?")
plt.yscale('log')  # Log-Skala, weil κ_dm klein ist
plt.ylim(0.01, 10)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Beispiel-Rechnung für KI-Optimum
C_ki = 0.995
DOC_ki = 0.00
kdm_needed = inverse_kdm(HUBBLE_TENSION_REL, C_ki, DOC_ki)
print(f"Bei C = {C_ki} und DOC = {DOC_ki} braucht man κ_dm ≈ {kdm_needed:.4f}")
