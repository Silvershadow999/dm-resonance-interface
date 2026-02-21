import numpy as np
from dataclasses import dataclass

# =============================================
# PureUniversalCore - Central resonance field
# Author: Alexandra-Nicole Anna Drinda (Silvershadow999)
# =============================================

@dataclass
class UPFConfig:
    levels: int = 3
    phi: float = (1 + 5**0.5) / 2
    gE: float = 0.01
    gC: float = 0.02
    gR: float = 0.01
    leak_E: float = 0.02
    leak_C: float = 0.02
    leak_R: float = 0.02
    k: float = 0.55
    beta: float = 0.18
    gamma: float = 0.12
    kappa: float = 0.04          # diffusive coupling between layers
    E_base: float = 0.45
    scale_direction: str = "up"
    process_noise_sigma: float = 0.01


class PureUniversalCore:
    """
    The pure, scale-invariant resonance field.
    Core engine for all simulations in dm-resonance-interface.
    """

    def __init__(self, config: UPFConfig | None = None, seed: int = 42):
        if config is None:
            config = UPFConfig()
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        L = self.cfg.levels
        self.rho = np.ones(L) * 1.0
        self.C   = np.ones(L) * 0.6
        self.E   = np.ones(L) * 0.4

        self.rho0 = self.rho.copy()
        self.C0   = self.C.copy()
        self.E0   = self.E.copy()
        self.last_S = np.zeros(L)

    @staticmethod
    def noise_assist_boost(E: float, E_opt: float) -> float:
        E_opt = max(E_opt, 1e-9)
        x = np.clip(E / E_opt, 1e-6, 50.0)
        return float(x * np.exp(1 - x))

    def _scale_factor(self, ell: int) -> float:
        if self.cfg.scale_direction == "up":
            return float(self.cfg.phi ** (self.cfg.levels - 1 - ell))
        return float(self.cfg.phi ** ell)

    def step(self, drive: float, DOC: float = 0.0, dm_coupling: float = 0.0) -> np.ndarray:
        """Single step with optional DM coupling term."""
        M_eff = 1.0 - DOC
        S_prev = np.zeros(self.cfg.levels)

        for ell in range(self.cfg.levels):
            scale = self._scale_factor(ell)
            S_prev[ell] = (self.rho[ell] * self.C[ell] / (1.0 + self.E[ell])) * scale * M_eff

        S_layers = np.zeros(self.cfg.levels)
        for ell in range(self.cfg.levels):
            Eopt = self.cfg.E_base / (self.cfg.phi ** ell)
            boost = self.noise_assist_boost(self.E[ell], Eopt)

            coupling = 0.0
            if self.cfg.kappa != 0:
                if ell > 0: coupling += (S_prev[ell-1] - S_prev[ell])
                if ell < self.cfg.levels-1: coupling += (S_prev[ell+1] - S_prev[ell])
                coupling *= self.cfg.kappa

            delta_S = (
                self.cfg.k * drive * self.rho[ell] * self.C[ell] +
                self.cfg.beta * boost +
                self.cfg.gamma * M_eff +
                coupling +
                dm_coupling * drive                      # <-- DM channel contribution
            ) + self.rng.normal(0.0, self.cfg.process_noise_sigma)

            # State updates
            self.E[ell] = np.clip(
                self.E[ell] - self.cfg.gE * delta_S + self.cfg.leak_E * (self.E0[ell] - self.E[ell]),
                0.0, 1.0
            )
            self.C[ell] = np.clip(
                self.C[ell] + self.cfg.gC * delta_S + self.cfg.leak_C * (self.C0[ell] - self.C[ell]),
                0.0, 1.0
            )
            self.rho[ell] = np.clip(
                self.rho[ell] + self.cfg.gR * delta_S + self.cfg.leak_R * (self.rho0[ell] - self.rho[ell]),
                0.1, 5.0
            )

            scale = self._scale_factor(ell)
            S_layers[ell] = (self.rho[ell] * self.C[ell] / (1.0 + self.E[ell])) * scale * M_eff

        self.last_S = S_layers.copy()
        return S_layers
