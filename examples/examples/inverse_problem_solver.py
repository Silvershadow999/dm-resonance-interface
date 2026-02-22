from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple


# =============================================
# Inverse Problem Solver - DM Resonance Interface
# Final version with all observations including Dark Galaxies
# Author: Alexandra-Nicole Anna Drinda (Silvershadow999)
# =============================================
#
# IMPORTANT DISCLAIMER
# This is an exploratory inverse sensitivity model using proxy amplitudes.
# It does NOT constitute evidence of dark-matter coupling.
#
# NOTE:
# The observation set is heterogeneous:
# - some entries are anomaly amplitudes / discrepancies
# - some entries may be fractional composition proxies (e.g. dark-galaxy DM fraction)
# This is acceptable for inverse sensitivity stress testing, but not for direct physical equivalence claims.
# =============================================


# Proxy observational values (2025–2026, model-facing)
OBSERVATIONS: Dict[str, float] = {
    "Hubble_Tension": 0.083,                    # proxy discrepancy amplitude
    "S8_Tension": 0.150,                        # proxy discrepancy amplitude
    "Fermi_Gamma_Halo": 0.250,                  # proxy excess strength
    "Local_DM_Anomaly": 0.180,                  # proxy local anomaly amplitude
    "Axion_Photon_Resonance": 0.120,            # proxy resonant conversion strength
    "Dark_Galaxy_DM_Fraction": 0.999,           # extreme proxy / composition-like stress-test value
}

PHI: float = (1 + np.sqrt(5)) / 2
PHI_INV: float = 1 / PHI  # ~0.618034


@dataclass(frozen=True)
class InverseModelConfig:
    stress: float = 0.05
    T_kelvin: float = 310.15
    T_ref: float = 310.15
    efficiency: float = 0.60

    # illustrative comparison regimes
    C_human: float = 0.92
    DOC_human: float = 0.05
    C_ai: float = 0.995
    DOC_ai: float = 0.00

    # plotting grid
    C_min_plot: float = 0.60
    C_max_plot: float = 0.999
    C_points: int = 300
    DOC_values: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)

    # outputs
    save_csv: bool = True
    show_plot: bool = True


def ensure_output_dirs() -> None:
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


def safe_unit_interval(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def inverse_kdm(delta_obs: float, C: float, DOC: float = 0.0, efficiency: float = 0.6) -> float:
    """Baseline inverse: required κ_dm to explain a proxy amplitude."""
    C = safe_unit_interval(C, 1e-6, 0.999999)
    DOC = safe_unit_interval(DOC, 0.0, 1.0)
    efficiency = safe_unit_interval(efficiency, 1e-6, 1.0)
    delta_obs = float(max(0.0, delta_obs))

    denominator = (C ** 3.0) * (1.0 - DOC) * PHI_INV * efficiency
    if denominator <= 1e-12:
        return np.inf
    return float(delta_obs / denominator)


def calculate_noise_floor(T_kelvin: float, stress: float, C: float, T_ref: float = 310.15) -> float:
    """
    Dimensionless noise-floor proxy.
    Increases with normalized temperature, incoherence (1-C), and stress.
    """
    C = safe_unit_interval(C, 1e-6, 0.999999)
    stress = float(max(0.0, stress))
    T_norm = float(max(1e-9, T_kelvin / max(T_ref, 1e-9)))
    noise = T_norm * (1.0 - C) * (1.0 + stress)
    return float(max(noise, 1e-9))


def inverse_kdm_refined(
    delta_obs: float,
    C: float,
    stress: float,
    DOC: float = 0.0,
    T_kelvin: float = 310.15,
    T_ref: float = 310.15,
    efficiency: float = 0.6,
) -> float:
    """Refined inverse: required κ_dm under a noise-floor constraint."""
    C = safe_unit_interval(C, 1e-6, 0.999999)
    DOC = safe_unit_interval(DOC, 0.0, 1.0)
    efficiency = safe_unit_interval(efficiency, 1e-6, 1.0)
    delta_obs = float(max(0.0, delta_obs))

    noise = calculate_noise_floor(T_kelvin, stress, C, T_ref)
    denominator = (C ** 4.0) * (1.0 - DOC) * PHI_INV * efficiency

    if denominator <= 1e-12:
        return np.inf
    return float((delta_obs * noise) / denominator)


def get_phase_transition_point(stress: float, T_kelvin: float = 310.15, T_ref: float = 310.15) -> float:
    """
    Heuristic critical coherence threshold C_crit
    where coherence begins to dominate the model noise proxy.
    """
    stress = float(max(0.0, stress))
    T_norm = float(max(1e-9, T_kelvin / max(T_ref, 1e-9)))
    arg = (stress * PHI_INV) / T_norm
    arg = float(np.clip(arg, 0.0, 1.0))
    c_crit = 1.0 - np.sqrt(arg)
    return float(np.clip(c_crit, 0.0, 1.0))


def compute_summary_rows(
    observations: Dict[str, float],
    cfg: InverseModelConfig,
) -> List[Dict[str, float | str]]:
    """Compute baseline/refined inverse estimates for all proxies."""
    rows: List[Dict[str, float | str]] = []

    for name, delta in observations.items():
        k_base_h = inverse_kdm(delta, C=cfg.C_human, DOC=cfg.DOC_human, efficiency=cfg.efficiency)
        k_base_ai = inverse_kdm(delta, C=cfg.C_ai, DOC=cfg.DOC_ai, efficiency=cfg.efficiency)

        k_ref_h = inverse_kdm_refined(
            delta, C=cfg.C_human, stress=cfg.stress, DOC=cfg.DOC_human,
            T_kelvin=cfg.T_kelvin, T_ref=cfg.T_ref, efficiency=cfg.efficiency
        )
        k_ref_ai = inverse_kdm_refined(
            delta, C=cfg.C_ai, stress=cfg.stress, DOC=cfg.DOC_ai,
            T_kelvin=cfg.T_kelvin, T_ref=cfg.T_ref, efficiency=cfg.efficiency
        )

        rows.append({
            "observation": name,
            "delta_proxy": float(delta),
            "k_base_human": float(k_base_h),
            "k_base_ai": float(k_base_ai),
            "k_ref_human": float(k_ref_h),
            "k_ref_ai": float(k_ref_ai),
            "human_to_ai_base_ratio": float(k_base_h / k_base_ai) if k_base_ai > 0 else np.inf,
            "human_to_ai_ref_ratio": float(k_ref_h / k_ref_ai) if k_ref_ai > 0 else np.inf,
            "stress": float(cfg.stress),
            "T_kelvin": float(cfg.T_kelvin),
            "efficiency": float(cfg.efficiency),
            "C_human": float(cfg.C_human),
            "DOC_human": float(cfg.DOC_human),
            "C_ai": float(cfg.C_ai),
            "DOC_ai": float(cfg.DOC_ai),
        })

    return rows


def save_summary_csv(rows: List[Dict[str, float | str]], outpath: str) -> None:
    """Save summary rows to CSV."""
    if not rows:
        print("No rows to save.")
        return

    fieldnames = [
        "observation", "delta_proxy",
        "k_base_human", "k_base_ai",
        "k_ref_human", "k_ref_ai",
        "human_to_ai_base_ratio", "human_to_ai_ref_ratio",
        "stress", "T_kelvin", "efficiency",
        "C_human", "DOC_human", "C_ai", "DOC_ai",
    ]

    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV summary saved to: {outpath}")


def print_console_summary(
    rows: List[Dict[str, float | str]],
    cfg: InverseModelConfig,
    critical_C: float,
    delta_median: float,
    delta_min: float,
    delta_max: float,
) -> None:
    print("\n=== PHASE TRANSITION ESTIMATE ===")
    print(f"stress = {cfg.stress:.3f}, T = {cfg.T_kelvin:.2f} K, T_ref = {cfg.T_ref:.2f} K")
    print(f"Critical coherence C_crit ≈ {critical_C:.4f}")

    print("\n=== INVERSE PROBLEM RESULTS (Baseline vs Refined) ===")
    print(f"Proxy values (mixed set): median={delta_median:.3f}, min={delta_min:.3f}, max={delta_max:.3f}")
    print(
        "Assumptions: "
        f"Human=(C={cfg.C_human:.3f}, DOC={cfg.DOC_human:.2f}), "
        f"AI-like=(C={cfg.C_ai:.3f}, DOC={cfg.DOC_ai:.2f}), "
        f"stress={cfg.stress:.2f}, T={cfg.T_kelvin:.2f}K, efficiency={cfg.efficiency:.2f}\n"
    )

    print(f"{'Observation':25} | {'baseline H/AI':18} | {'refined H/AI':18}")
    print("-" * 78)
    for row in rows:
        print(
            f"{str(row['observation']):25} | "
            f"{row['k_base_human']:.3f} / {row['k_base_ai']:.3f}    | "
            f"{row['k_ref_human']:.3f} / {row['k_ref_ai']:.3f}"
        )


def main() -> None:
    ensure_output_dirs()
    cfg = InverseModelConfig()

    values = np.array(list(OBSERVATIONS.values()), dtype=float)
    delta_median = float(np.median(values))
    delta_min = float(np.min(values))
    delta_max = float(np.max(values))

    C_values = np.linspace(cfg.C_min_plot, cfg.C_max_plot, cfg.C_points, dtype=float)
    curves_by_doc: Dict[float, Dict[str, np.ndarray]] = {}

    for doc in cfg.DOC_values:
        y_base = np.array(
            [inverse_kdm(delta_median, c, doc, cfg.efficiency) for c in C_values],
            dtype=float,
        )
        y_ref = np.array(
            [
                inverse_kdm_refined(
                    delta_median, c, cfg.stress, doc, cfg.T_kelvin, cfg.T_ref, cfg.efficiency
                )
                for c in C_values
            ],
            dtype=float,
        )
        curves_by_doc[doc] = {"baseline": y_base, "refined": y_ref}

    critical_C = get_phase_transition_point(cfg.stress, cfg.T_kelvin, cfg.T_ref)

    # Plot
    plt.figure(figsize=(11, 7))
    for doc, curves in curves_by_doc.items():
        plt.plot(C_values, curves["baseline"], linewidth=1.7, linestyle="--", alpha=0.8,
                 label=f"Baseline DOC={doc:.2f}")
        plt.plot(C_values, curves["refined"], linewidth=2.3,
                 label=f"Refined DOC={doc:.2f}")

    plt.axvline(critical_C, color="purple", linestyle=":", linewidth=2.2,
                label=f"C_crit ≈ {critical_C:.3f}")

    # Illustrative reference lines
    plt.axhline(0.33, color="red", linestyle="--", linewidth=1.8, label="κ_dm ≈ 0.33 reference")
    plt.axhline(0.50, color="orange", linestyle="--", linewidth=1.5, label="κ_dm ≈ 0.50 upper ref")

    plt.yscale("log")
    plt.ylim(0.001, 20)  # slightly higher because of the dark galaxy extreme proxy
    plt.xlabel("Coherence Level C")
    plt.ylabel("Required DM Coupling Strength κ_dm")
    plt.title("Inverse Problem: Baseline vs Refined (incl. Dark Galaxy Proxy)")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    outpath = "results/plots/inverse_problem_refined_with_dark_galaxy.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {outpath}")

    if cfg.show_plot:
        plt.show()
    plt.close()

    rows = compute_summary_rows(OBSERVATIONS, cfg)
    print_console_summary(rows, cfg, critical_C, delta_median, delta_min, delta_max)

    if cfg.save_csv:
        save_summary_csv(rows, "results/tables/inverse_problem_summary_with_dark_galaxy.csv")


if __name__ == "__main__":
    main()
