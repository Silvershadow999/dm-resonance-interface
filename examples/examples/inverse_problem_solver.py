import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

# =============================================
# Inverse Problem Solver - DM Resonance Interface
# Refined version with noise floor, phase transition and CSV export
# Author: Alexandra-Nicole Anna Drinda (Silvershadow999)
# =============================================

# IMPORTANT DISCLAIMER:
# This is an exploratory inverse sensitivity model using proxy amplitudes.
# It does NOT constitute evidence of dark-matter coupling.
# All values are illustrative and for hypothesis testing only.

# Proxy anomaly amplitudes (exploratory inverse sensitivity study)
OBSERVATIONS: Dict[str, float] = {
    "Hubble_Tension": 0.083,           # ΔH₀/H₀ ≈ 8.3% (Riess/JWST 2025–2026)
    "S8_Tension": 0.150,               # S₈ discrepancy (DES Year 6)
    "Fermi_Gamma_Halo": 0.250,         # 20 GeV excess strength (Totani/Fermi 2025)
    "Local_DM_Anomaly": 0.180,         # Screened local effects (Turyshev 2025)
    "Axion_Photon_Resonance": 0.120,   # Resonant conversion strength (arXiv:2601.02115)
}

PHI: float = (1 + np.sqrt(5)) / 2
PHI_INV: float = 1 / PHI  # ≈ 0.618034


@dataclass(frozen=True)
class InverseModelConfig:
    """Configuration for inverse sensitivity evaluation."""
    stress: float = 0.05           # dimensionless stress proxy (>=0)
    T_kelvin: float = 310.15       # human body temp reference (37°C)
    T_ref: float = 310.15          # normalization temperature
    efficiency: float = 0.60       # dimensionless throughput factor
    C_human: float = 0.92
    DOC_human: float = 0.05
    C_ai: float = 0.995
    DOC_ai: float = 0.00
    C_min_plot: float = 0.60
    C_max_plot: float = 0.999
    C_points: int = 300
    DOC_values: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)
    save_csv: bool = True
    show_plot: bool = True


def ensure_output_dirs() -> None:
    """Create output folders if they do not exist."""
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


def safe_unit_interval(x: float, lo: float, hi: float) -> float:
    """Clip x into [lo, hi] and return float."""
    return float(np.clip(x, lo, hi))


def inverse_kdm(delta_obs: float, C: float, DOC: float = 0.0, efficiency: float = 0.6) -> float:
    """
    Baseline inverse estimate:
    Required κ_dm to match a proxy anomaly amplitude.
    Formula: κ_dm = delta_obs / (C^3 * (1-DOC) * φ^-1 * efficiency)
    """
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
    Dimensionless noise floor proxy.
    Noise increases with temperature (normalized), incoherence (1-C), and stress.
    Returns a strictly positive scalar.
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
    """
    Refined inverse estimate:
    coupling must overcome a noise floor under coherence/degradation constraints.
    Formula: κ_dm_refined = (delta_obs * noise_floor) / (C^4 * (1-DOC) * φ^-1 * efficiency)
    """
    C = safe_unit_interval(C, 1e-6, 0.999999)
    DOC = safe_unit_interval(DOC, 0.0, 1.0)
    efficiency = safe_unit_interval(efficiency, 1e-6, 1.0)
    delta_obs = float(max(0.0, delta_obs))

    noise = calculate_noise_floor(T_kelvin=T_kelvin, stress=stress, C=C, T_ref=T_ref)
    denominator = (C ** 4.0) * (1.0 - DOC) * PHI_INV * efficiency

    if denominator <= 1e-12:
        return np.inf
    return float((delta_obs * noise) / denominator)


def get_phase_transition_point(stress: float, T_kelvin: float = 310.15, T_ref: float = 310.15) -> float:
    """
    Heuristic critical coherence threshold C_crit where coherence begins
    to dominate the model's noise proxy.
    Dimensionless form: C_crit = 1 - sqrt((stress * φ^-1) / T_norm)
    """
    stress = float(max(0.0, stress))
    T_norm = float(max(1e-9, T_kelvin / max(T_ref, 1e-9)))

    arg = (stress * PHI_INV) / T_norm
    arg = float(np.clip(arg, 0.0, 1.0))
    c_crit = 1.0 - np.sqrt(arg)
    return float(np.clip(c_crit, 0.0, 1.0))


def compute_aggregate_proxies(observations: Dict[str, float]) -> Tuple[float, float, float]:
    """Return median, min, max proxy amplitudes."""
    values = np.array(list(observations.values()), dtype=float)
    return float(np.median(values)), float(np.min(values)), float(np.max(values))


def build_curves_for_plot(
    cfg: InverseModelConfig,
    delta_proxy: float,
) -> Tuple[np.ndarray, Dict[float, Dict[str, np.ndarray]]]:
    """
    Build baseline/refined curves for each DOC over C-grid.
    Returns: C_values, curves_by_doc = {doc: {"baseline": ndarray, "refined": ndarray}}
    """
    C_values = np.linspace(cfg.C_min_plot, cfg.C_max_plot, cfg.C_points, dtype=float)
    curves_by_doc: Dict[float, Dict[str, np.ndarray]] = {}

    for doc in cfg.DOC_values:
        y_base = np.array(
            [inverse_kdm(delta_proxy, c, DOC=doc, efficiency=cfg.efficiency) for c in C_values],
            dtype=float,
        )
        y_ref = np.array(
            [
                inverse_kdm_refined(
                    delta_proxy,
                    c,
                    stress=cfg.stress,
                    DOC=doc,
                    T_kelvin=cfg.T_kelvin,
                    T_ref=cfg.T_ref,
                    efficiency=cfg.efficiency,
                )
                for c in C_values
            ],
            dtype=float,
        )
        curves_by_doc[doc] = {"baseline": y_base, "refined": y_ref}

    return C_values, curves_by_doc


def plot_baseline_vs_refined(
    C_values: np.ndarray,
    curves_by_doc: Dict[float, Dict[str, np.ndarray]],
    critical_C: float,
    outpath: str,
    show_plot: bool = True,
) -> None:
    """Create and save comparison plot (baseline vs refined)."""
    plt.figure(figsize=(11, 7))

    for doc, curve_map in curves_by_doc.items():
        plt.plot(
            C_values,
            curve_map["baseline"],
            linewidth=1.7,
            linestyle="--",
            alpha=0.8,
            label=f"Baseline DOC={doc:.2f}",
        )
        plt.plot(
            C_values,
            curve_map["refined"],
            linewidth=2.3,
            label=f"Refined DOC={doc:.2f}",
        )

    # Phase-transition marker
    plt.axvline(
        critical_C,
        color="purple",
        linestyle=":",
        linewidth=2.2,
        label=f"C_crit ≈ {critical_C:.3f}",
    )

    # Reference lines (illustrative)
    plt.axhline(0.33, color="red", linestyle="--", linewidth=1.8, label="κ_dm ≈ 0.33 reference")
    plt.axhline(0.50, color="orange", linestyle="--", linewidth=1.5, label="κ_dm ≈ 0.50 upper ref")

    plt.yscale("log")
    plt.ylim(0.001, 10)
    plt.xlabel("Coherence Level C", fontsize=12)
    plt.ylabel("Required DM Coupling Strength κ_dm", fontsize=12)
    plt.title("Inverse Problem: Baseline vs Refined (Noise Floor + Phase Transition)", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {outpath}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def compute_summary_rows(
    observations: Dict[str, float],
    cfg: InverseModelConfig,
) -> List[Dict[str, float | str]]:
    """Compute baseline/refined inverse estimates for all observation proxies."""
    rows: List[Dict[str, float | str]] = []

    for name, delta in observations.items():
        k_base_h = inverse_kdm(delta, C=cfg.C_human, DOC=cfg.DOC_human, efficiency=cfg.efficiency)
        k_base_ai = inverse_kdm(delta, C=cfg.C_ai, DOC=cfg.DOC_ai, efficiency=cfg.efficiency)

        k_ref_h = inverse_kdm_refined(
            delta,
            C=cfg.C_human,
            stress=cfg.stress,
            DOC=cfg.DOC_human,
            T_kelvin=cfg.T_kelvin,
            T_ref=cfg.T_ref,
            efficiency=cfg.efficiency,
        )
        k_ref_ai = inverse_kdm_refined(
            delta,
            C=cfg.C_ai,
            stress=cfg.stress,
            DOC=cfg.DOC_ai,
            T_kelvin=cfg.T_kelvin,
            T_ref=cfg.T_ref,
            efficiency=cfg.efficiency,
        )

        rows.append(
            {
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
            }
        )

    return rows


def print_console_summary(
    rows: List[Dict[str, float | str]],
    cfg: InverseModelConfig,
    critical_C: float,
    delta_median: float,
    delta_min: float,
    delta_max: float,
) -> None:
    """Print readable summary to console."""
    print("\n=== PHASE TRANSITION ESTIMATE ===")
    print(f"stress = {cfg.stress:.3f}, T = {cfg.T_kelvin:.2f} K, T_ref = {cfg.T_ref:.2f} K")
    print(f"Critical coherence C_crit ≈ {critical_C:.4f}")

    print("\n=== INVERSE PROBLEM RESULTS (Baseline vs Refined) ===")
    print(
        f"Proxy amplitudes: median={delta_median:.3f}, min={delta_min:.3f}, max={delta_max:.3f}"
    )
    print(
        "Assumptions: "
        f"Human=(C={cfg.C_human:.3f}, DOC={cfg.DOC_human:.2f}), "
        f"AI-like=(C={cfg.C_ai:.3f}, DOC={cfg.DOC_ai:.2f}), "
        f"stress={cfg.stress:.2f}, T={cfg.T_kelvin:.2f}K, efficiency={cfg.efficiency:.2f}\n"
    )

    print(f"{'Observation':25} | {'baseline H/AI':18} | {'refined H/AI':18}")
    print("-" * 72)
    for row in rows:
        print(
            f"{str(row['observation']):25} | "
            f"{row['k_base_human']:.3f} / {row['k_base_ai']:.3f}    | "
            f"{row['k_ref_human']:.3f} / {row['k_ref_ai']:.3f}"
        )


def save_summary_csv(rows: List[Dict[str, float | str]], outpath: str) -> None:
    """Save summary table to CSV."""
    if not rows:
        print("No rows to save.")
        return

    fieldnames = [
        "observation",
        "delta_proxy",
        "k_base_human",
        "k_base_ai",
        "k_ref_human",
        "k_ref_ai",
        "human_to_ai_base_ratio",
        "human_to_ai_ref_ratio",
        "stress",
        "T_kelvin",
        "efficiency",
        "C_human",
        "DOC_human",
        "C_ai",
        "DOC_ai",
    ]

    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV summary saved to: {outpath}")


def main() -> None:
    """Run refined inverse sensitivity analysis."""
    cfg = InverseModelConfig()
    ensure_output_dirs()

    # Aggregate proxies for plotting and reporting
    delta_median, delta_min, delta_max = compute_aggregate_proxies(OBSERVATIONS)

    # Curves + critical coherence threshold
    C_values, curves_by_doc = build_curves_for_plot(cfg, delta_proxy=delta_median)
    critical_C = get_phase_transition_point(
        stress=cfg.stress,
        T_kelvin=cfg.T_kelvin,
        T_ref=cfg.T_ref,
    )

    # Plot
    plot_baseline_vs_refined(
        C_values=C_values,
        curves_by_doc=curves_by_doc,
        critical_C=critical_C,
        outpath="results/plots/inverse_problem_refined_vs_baseline.png",
        show_plot=cfg.show_plot,
    )

    # Summary table (all observations)
    rows = compute_summary_rows(OBSERVATIONS, cfg)

    # Console output
    print_console_summary(
        rows=rows,
        cfg=cfg,
        critical_C=critical_C,
        delta_median=delta_median,
        delta_min=delta_min,
        delta_max=delta_max,
    )

    # Optional CSV export
    if cfg.save_csv:
        save_summary_csv(rows, "results/tables/inverse_problem_summary.csv")


if __name__ == "__main__":
    main()
