from __future__ import annotations

import os
import csv
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

# Optional scientific dependency (preferred if available)
try:
    from scipy.stats import entropy as scipy_entropy
    SCIPY_ENABLED = True
except ImportError:
    SCIPY_ENABLED = False
    scipy_entropy = None


# ============================================================
# Advanced DM Resonance Engine - Final Refined Version (2026)
# ============================================================
#
# IMPORTANT DISCLAIMER
# This is an exploratory inverse sensitivity model using proxy values.
# It does NOT constitute evidence of dark-matter coupling.
# It is intended for hypothesis testing, parameter sweeps, and model comparison.
#
# The observation set may be heterogeneous (anomaly amplitudes, proxy fractions, etc.).
# This is acceptable for inverse sensitivity stress testing, but NOT for direct
# physical equivalence claims.
# ============================================================


@dataclass(frozen=True)
class AdvancedConfig:
    # Fundamental constants / scaling
    PHI: float = (1.0 + np.sqrt(5.0)) / 2.0
    T_REF: float = 310.15  # Reference temperature (K), e.g. ~37°C

    # Model parameters
    BASE_EFFICIENCY: float = 0.65
    STRESS_LEVEL: float = 0.05
    NOISE_TEMPERATURE: float = 310.15  # K

    # Efficiency modes: "anti_saturation", "monotonic", "sweet_spot"
    EFFICIENCY_MODE: str = "anti_saturation"
    EFF_SHARPNESS: float = 12.0
    EFF_C_OPT: float = 0.970
    EFF_C_WIDTH: float = 0.015

    # Profile presets: (C, DOC)
    PROFILES: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Human": (0.92, 0.05),
        "Neuralink_v2": (0.98, 0.02),
        "AGI_Core": (0.995, 0.001),
    })

    # Proxy observations (model-facing values; heterogeneous by design for stress testing)
    OBSERVATIONS: Dict[str, float] = field(default_factory=lambda: {
        "Hubble_Tension": 0.083,
        "S8_Tension": 0.150,
        "Fermi_Excess": 0.250,
        "Local_DM_Anomaly": 0.180,
        "Axion_Photon_Resonance": 0.120,
        "Dark_Galaxy_Fraction": 0.999,  # composition-like stress-test proxy
    })

    # Plot settings
    C_MIN_PLOT: float = 0.85
    C_MAX_PLOT: float = 0.999
    C_POINTS: int = 300
    Y_LIM_MIN: float = 1e-4
    Y_LIM_MAX: float = 20.0

    # Outputs
    SHOW_PLOT: bool = True
    SAVE_PLOT: bool = True
    SAVE_CSV: bool = True
    PLOT_FILENAME: str = "results/plots/dm_inverse_entropy_weighted_comparison.png"
    CSV_FILENAME: str = "results/tables/dm_analysis_2026.csv"


class DMResonanceEngine:
    def __init__(self, config: AdvancedConfig):
        self.cfg = config
        self._ensure_output_dirs()
        self.weights = self._calculate_entropy_weights()

    # -------------------------
    # Utilities / setup
    # -------------------------
    def _ensure_output_dirs(self) -> None:
        os.makedirs("results/tables", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)

    @staticmethod
    def _clip_c(C: float) -> float:
        return float(np.clip(C, 1e-6, 0.9999))

    @staticmethod
    def _clip_doc(DOC: float) -> float:
        return float(np.clip(DOC, 0.0, 1.0))

    # -------------------------
    # Weighting
    # -------------------------
    def _calculate_entropy_weights(self) -> Dict[str, float]:
        """
        Shannon-entropy-based weighting over the observation proxy set.
        Returns weights normalized to sum to 1.
        """
        vals = np.array(list(self.cfg.OBSERVATIONS.values()), dtype=float)
        vals = np.clip(vals, 1e-12, None)
        prob_dist = vals / np.sum(vals)

        if SCIPY_ENABLED and scipy_entropy is not None:
            ent = float(scipy_entropy(prob_dist))
        else:
            ent = float(-np.sum(prob_dist * np.log(prob_dist + 1e-12)))

        raw_weights = {
            k: float((v / np.sum(vals)) * (1.0 + ent))
            for k, v in self.cfg.OBSERVATIONS.items()
        }
        w_sum = float(sum(raw_weights.values()))
        if w_sum <= 1e-12:
            # Fallback uniform weights
            n = max(1, len(raw_weights))
            return {k: 1.0 / n for k in raw_weights}

        return {k: float(w / w_sum) for k, w in raw_weights.items()}

    # -------------------------
    # Model components
    # -------------------------
    def get_dynamic_efficiency(self, C: float) -> float:
        """
        Dynamic efficiency η(C) with selectable phenomenological modes.

        Modes:
        - anti_saturation: efficiency decreases near C -> 1 (hypothesized saturation bottleneck)
        - monotonic: efficiency increases with C and softly saturates
        - sweet_spot: resonance-like optimum near EFF_C_OPT
        """
        C = self._clip_c(C)
        mode = self.cfg.EFFICIENCY_MODE.strip().lower()

        if mode == "anti_saturation":
            # High efficiency before total saturation; drops near C->1
            eff = self.cfg.BASE_EFFICIENCY * (1.0 - np.exp(-self.cfg.EFF_SHARPNESS * (1.0 - C)))

        elif mode == "monotonic":
            # Conventional assumption: coherence helps transfer
            eff = self.cfg.BASE_EFFICIENCY * (1.0 - np.exp(-self.cfg.EFF_SHARPNESS * C))

        elif mode == "sweet_spot":
            # Resonance-like peak around a coherence optimum
            c_opt = float(np.clip(self.cfg.EFF_C_OPT, 1e-6, 0.9999))
            width = float(max(1e-4, self.cfg.EFF_C_WIDTH))
            gaussian = np.exp(-0.5 * ((C - c_opt) / width) ** 2)
            eff = self.cfg.BASE_EFFICIENCY * (0.25 + 0.75 * gaussian)

        else:
            # Safe fallback
            eff = self.cfg.BASE_EFFICIENCY * C

        return float(np.clip(eff, 1e-6, 1.0))

    def calculate_noise_floor(self, C: float) -> float:
        """
        Dimensionless noise-floor proxy:
        increases with incoherence (1-C), stress, and normalized temperature.
        """
        C = self._clip_c(C)
        T_norm = float(max(1e-9, self.cfg.NOISE_TEMPERATURE / max(self.cfg.T_REF, 1e-9)))
        noise = T_norm * (1.0 - C) * (1.0 + max(0.0, self.cfg.STRESS_LEVEL))
        return float(max(noise, 1e-12))

    def calculate_kdm(self, delta: float, C: float, DOC: float, weight: float = 1.0) -> float:
        """
        Refined inverse estimate for required coupling strength κ_dm.

        κ_dm ~ (delta * weight * noise_floor) / [ C^4 * (1-DOC) * (1/phi) * eta(C) ]
        """
        C = self._clip_c(C)
        DOC = self._clip_doc(DOC)
        delta = float(max(0.0, delta))
        weight = float(max(0.0, weight))

        eff = self.get_dynamic_efficiency(C)
        noise = self.calculate_noise_floor(C)

        denominator = (C ** 4.0) * (1.0 - DOC) * (1.0 / self.cfg.PHI) * eff
        return float((delta * weight * noise) / max(denominator, 1e-12))

    def calculate_kdm_baseline(self, delta: float, C: float, DOC: float) -> float:
        """
        Baseline inverse estimate without noise-floor or dynamic efficiency.
        Included for comparison / ablation.
        """
        C = self._clip_c(C)
        DOC = self._clip_doc(DOC)
        delta = float(max(0.0, delta))

        denominator = (C ** 3.0) * (1.0 - DOC) * (1.0 / self.cfg.PHI) * max(self.cfg.BASE_EFFICIENCY, 1e-12)
        return float(delta / max(denominator, 1e-12))

    def get_phase_transition_point(self) -> float:
        """
        Heuristic critical coherence threshold C_crit where coherence begins
        to dominate the noise proxy.
        """
        T_norm = float(max(1e-9, self.cfg.NOISE_TEMPERATURE / max(self.cfg.T_REF, 1e-9)))
        arg = (max(0.0, self.cfg.STRESS_LEVEL) * (1.0 / self.cfg.PHI)) / T_norm
        arg = float(np.clip(arg, 0.0, 1.0))
        c_crit = 1.0 - np.sqrt(arg)
        return float(np.clip(c_crit, 0.0, 1.0))

    # -------------------------
    # Aggregates / summaries
    # -------------------------
    def weighted_mean_delta(self) -> float:
        """Entropy-weighted mean of observation proxies."""
        return float(sum(self.cfg.OBSERVATIONS[k] * self.weights[k] for k in self.cfg.OBSERVATIONS))

    def weighted_median_delta(self) -> float:
        """Weighted median of observation proxies (optional summary statistic)."""
        items = sorted(self.cfg.OBSERVATIONS.items(), key=lambda kv: kv[1])
        cumulative = 0.0
        for name, value in items:
            cumulative += self.weights.get(name, 0.0)
            if cumulative >= 0.5:
                return float(value)
        return float(items[-1][1]) if items else 0.0

    def _profile_summary_rows(self) -> List[List[str]]:
        """
        Profile summary table rows:
        Profile, C, DOC, weighted_delta, baseline_kdm, refined_kdm, eta, noise
        """
        weighted_delta = self.weighted_mean_delta()
        rows: List[List[str]] = []

        for p_name, (C, DOC) in self.cfg.PROFILES.items():
            k_base = self.calculate_kdm_baseline(weighted_delta, C, DOC)
            k_ref = self.calculate_kdm(weighted_delta, C, DOC, weight=1.0)
            eta = self.get_dynamic_efficiency(C)
            noise = self.calculate_noise_floor(C)

            rows.append([
                p_name,
                f"{C:.6f}",
                f"{DOC:.6f}",
                f"{weighted_delta:.6f}",
                f"{k_base:.8f}",
                f"{k_ref:.8f}",
                f"{eta:.8f}",
                f"{noise:.8f}",
            ])
        return rows

    def _observation_profile_rows(self) -> List[List[str]]:
        """
        Per observation + profile rows:
        Profile, Observation, Delta, Weight, BaselineKdm, RefinedKdm
        """
        rows: List[List[str]] = []
        for p_name, (C, DOC) in self.cfg.PROFILES.items():
            for obs_name, delta in self.cfg.OBSERVATIONS.items():
                w = self.weights[obs_name]
                k_base = self.calculate_kdm_baseline(delta, C, DOC)
                k_ref = self.calculate_kdm(delta, C, DOC, weight=w)
                rows.append([
                    p_name,
                    obs_name,
                    f"{delta:.6f}",
                    f"{w:.8f}",
                    f"{k_base:.8f}",
                    f"{k_ref:.8f}",
                ])
        return rows

    # -------------------------
    # Export / plot
    # -------------------------
    def export_results_csv(self) -> None:
        """
        Exports two sections in one CSV:
        1) Profile summary (weighted aggregate)
        2) Observation x Profile detailed rows
        """
        path = self.cfg.CSV_FILENAME
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(["# DM Resonance Interface - Exploratory Inverse Sensitivity Export"])
            writer.writerow(["# DISCLAIMER", "Hypothesis-testing only; not evidence of dark-matter coupling"])
            writer.writerow([])

            writer.writerow(["# CONFIG"])
            writer.writerow(["EFFICIENCY_MODE", self.cfg.EFFICIENCY_MODE])
            writer.writerow(["BASE_EFFICIENCY", f"{self.cfg.BASE_EFFICIENCY:.6f}"])
            writer.writerow(["STRESS_LEVEL", f"{self.cfg.STRESS_LEVEL:.6f}"])
            writer.writerow(["NOISE_TEMPERATURE", f"{self.cfg.NOISE_TEMPERATURE:.6f}"])
            writer.writerow(["T_REF", f"{self.cfg.T_REF:.6f}"])
            writer.writerow(["PHI", f"{self.cfg.PHI:.12f}"])
            writer.writerow(["SCIPY_ENABLED", str(SCIPY_ENABLED)])
            writer.writerow([])

            writer.writerow(["# WEIGHTS"])
            writer.writerow(["Observation", "ProxyValue", "EntropyWeight"])
            for name, delta in self.cfg.OBSERVATIONS.items():
                writer.writerow([name, f"{delta:.6f}", f"{self.weights[name]:.8f}"])
            writer.writerow([])

            writer.writerow(["# PROFILE_SUMMARY (weighted mean delta)"])
            writer.writerow([
                "Profile", "C", "DOC", "WeightedDelta",
                "BaselineKdm", "RefinedKdm", "DynamicEfficiency", "NoiseFloor"
            ])
            writer.writerows(self._profile_summary_rows())
            writer.writerow([])

            writer.writerow(["# OBSERVATION_PROFILE_DETAILS"])
            writer.writerow(["Profile", "Observation", "Delta", "Weight", "BaselineKdm", "RefinedKdm"])
            writer.writerows(self._observation_profile_rows())

        print(f"CSV export complete: {path}")

    def plot_comparison(self) -> None:
        """
        Plots baseline vs refined curves for profile DOC paths using weighted aggregate proxies,
        plus profile point markers and heuristic phase-transition line.
        """
        c_range = np.linspace(self.cfg.C_MIN_PLOT, self.cfg.C_MAX_PLOT, self.cfg.C_POINTS, dtype=float)
        weighted_delta_mean = self.weighted_mean_delta()
        weighted_delta_median = self.weighted_median_delta()
        c_crit = self.get_phase_transition_point()

        plt.figure(figsize=(12, 8))

        # Plot one pair of reference aggregate curves (mean and median, DOC=0.05) for context
        ref_doc = 0.05
        y_base_mean = [self.calculate_kdm_baseline(weighted_delta_mean, c, ref_doc) for c in c_range]
        y_ref_mean = [self.calculate_kdm(weighted_delta_mean, c, ref_doc, weight=1.0) for c in c_range]
        y_ref_median = [self.calculate_kdm(weighted_delta_median, c, ref_doc, weight=1.0) for c in c_range]

        plt.plot(c_range, y_base_mean, linestyle="--", linewidth=1.8, alpha=0.8,
                 label="Baseline (weighted mean proxy, DOC=0.05)")
        plt.plot(c_range, y_ref_mean, linewidth=2.4,
                 label="Refined (weighted mean proxy, DOC=0.05)")
        plt.plot(c_range, y_ref_median, linewidth=1.6, linestyle=":",
                 label="Refined (weighted median proxy, DOC=0.05)")

        # Profile paths (each profile's DOC, weighted mean proxy)
        for name, (c_val, doc_val) in self.cfg.PROFILES.items():
            y_profile = [self.calculate_kdm(weighted_delta_mean, c, doc_val, weight=1.0) for c in c_range]
            plt.plot(c_range, y_profile, linewidth=2.0, alpha=0.9, label=f"{name} path (DOC={doc_val})")

            k_point = self.calculate_kdm(weighted_delta_mean, c_val, doc_val, weight=1.0)
            plt.scatter(c_val, k_point, s=80, edgecolors="black", linewidths=0.8, zorder=6)
            plt.annotate(name, (c_val, k_point), textcoords="offset points", xytext=(6, 6), fontsize=9)

        # Phase-transition threshold
        plt.axvline(c_crit, color="purple", linestyle=":", linewidth=2.0, label=f"C_crit ≈ {c_crit:.3f}")

        # Illustrative guide lines (not fitted claims)
        plt.axhline(0.33, color="red", linestyle="--", linewidth=1.4, alpha=0.85, label="Guide κ_dm ≈ 0.33")
        plt.axhline(0.50, color="orange", linestyle="--", linewidth=1.4, alpha=0.85, label="Guide κ_dm ≈ 0.50")

        plt.yscale("log")
        plt.ylim(self.cfg.Y_LIM_MIN, self.cfg.Y_LIM_MAX)
        plt.xlabel("Coherence (C)")
        plt.ylabel("Required Coupling Strength (κ_dm)")
        plt.title("DM Resonance Interface - Inverse Sensitivity Comparison (Exploratory)")
        plt.grid(True, which="both", ls="--", alpha=0.35)
        plt.legend(fontsize=9, ncol=2)
        plt.tight_layout()

        if self.cfg.SAVE_PLOT:
            plt.savefig(self.cfg.PLOT_FILENAME, dpi=300, bbox_inches="tight")
            print(f"Plot saved: {self.cfg.PLOT_FILENAME}")

        if self.cfg.SHOW_PLOT:
            plt.show()
        else:
            plt.close()

    # -------------------------
    # Console output
    # -------------------------
    def print_summary(self) -> None:
        vals = np.array(list(self.cfg.OBSERVATIONS.values()), dtype=float)
        delta_mean = self.weighted_mean_delta()
        delta_median = self.weighted_median_delta()
        c_crit = self.get_phase_transition_point()

        print("\n=== DM Resonance Interface - Exploratory Inverse Sensitivity Summary ===")
        print(f"SciPy entropy backend available: {SCIPY_ENABLED}")
        print(f"Efficiency mode: {self.cfg.EFFICIENCY_MODE}")
        print(f"Stress level: {self.cfg.STRESS_LEVEL:.4f}")
        print(f"Noise temperature: {self.cfg.NOISE_TEMPERATURE:.2f} K (T_ref={self.cfg.T_REF:.2f} K)")
        print(f"Phase threshold (heuristic) C_crit ≈ {c_crit:.4f}")

        print("\nObservation proxy stats (heterogeneous set):")
        print(f"  count={len(vals)}, min={np.min(vals):.4f}, median={np.median(vals):.4f}, max={np.max(vals):.4f}")
        print(f"  weighted mean delta={delta_mean:.6f}, weighted median delta={delta_median:.6f}")

        print("\nEntropy weights:")
        for k, w in self.weights.items():
            print(f"  - {k:24s}: {w:.6f}")

        print("\nProfile summary (weighted mean proxy):")
        print(f"{'Profile':14s} {'C':>8s} {'DOC':>8s} {'eta(C)':>10s} {'noise':>10s} {'k_base':>12s} {'k_ref':>12s}")
        print("-" * 82)
        weighted_delta = self.weighted_mean_delta()
        for p_name, (C, DOC) in self.cfg.PROFILES.items():
            eta = self.get_dynamic_efficiency(C)
            noise = self.calculate_noise_floor(C)
            k_base = self.calculate_kdm_baseline(weighted_delta, C, DOC)
            k_ref = self.calculate_kdm(weighted_delta, C, DOC, weight=1.0)
            print(f"{p_name:14s} {C:8.4f} {DOC:8.4f} {eta:10.6f} {noise:10.6f} {k_base:12.6f} {k_ref:12.6f}")

    # -------------------------
    # Run pipeline
    # -------------------------
    def run(self) -> None:
        self.print_summary()
        if self.cfg.SAVE_CSV:
            self.export_results_csv()
        self.plot_comparison()


if __name__ == "__main__":
    # Change only these values for quick experiments, e.g.:
    # EFFICIENCY_MODE = "monotonic" or "sweet_spot"
    cfg = AdvancedConfig(
        EFFICIENCY_MODE="anti_saturation",
        SHOW_PLOT=True,
        SAVE_PLOT=True,
        SAVE_CSV=True,
    )
    engine = DMResonanceEngine(cfg)
    engine.run()
