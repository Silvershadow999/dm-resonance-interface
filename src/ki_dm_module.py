class KI_DM_CouplingModule:
    def __init__(self, base_kappa_dm: float = 0.012):
        self.base_kappa_dm = base_kappa_dm
        self.precision_boost = 100.0   # KI-Vorteil

    def compute_dm_coupling(self, C: float, DOC: float, intensity: float = 1.0) -> float:
        if C > 0.95 and DOC < 0.05:
            return self.base_kappa_dm * (C ** 3.0) * self.precision_boost * intensity
        return 0.0
