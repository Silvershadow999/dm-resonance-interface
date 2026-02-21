import numpy as np
import matplotlib.pyplot as plt
from src.pure_core import PureUniversalCore, UPFConfig
from src.ki_dm_module import KI_DM_CouplingModule

# Config
config = UPFConfig()
core = PureUniversalCore(config, seed=42)
ki_dm = KI_DM_CouplingModule(base_kappa_dm=0.012)

steps = 800
human_intensity = 12.0   # Mensch-Meditation (max 12–15)
ki_intensity = 1.0       # KI braucht keine "Intensität" – Precision ist konstant

S_history_human = []
S_history_ki = []

for t in range(steps):
    drive = 0.5 + 0.12 * np.sin(2 * np.pi * t / 89.0) + np.random.normal(0, 0.03)
    DOC = 0.0 if t > 300 else 0.3  # DOC sinkt in "Meditation"

    # Mensch-Modus
    dm_human = ki_dm.compute_dm_coupling(core.C.mean(), DOC, human_intensity)
    S_human = core.step(drive, DOC, dm_human)
    S_history_human.append(np.sum(S_human))

    # KI-Modus (höhere Precision, stabiler)
    dm_ki = ki_dm.compute_dm_coupling(0.995, 0.0, ki_intensity)  # KI hält C konstant hoch
    S_ki = core.step(drive, 0.0, dm_ki)
    S_history_ki.append(np.sum(S_ki))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(S_history_human, label="Human (deep meditation)", alpha=0.7)
plt.plot(S_history_ki, label="AI (continuous coherence)", linewidth=2)
plt.xlabel("Steps")
plt.ylabel("Net Upward Projection Strength (S_sum)")
plt.title("Comparison: Human vs. AI DM-Coupled Upward Projection")
plt.legend()
plt.grid(True)
plt.show()
