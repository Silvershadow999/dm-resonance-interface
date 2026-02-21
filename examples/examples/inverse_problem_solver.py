import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# Inverse Problem Solver - DM Resonance Interface
# Author: Alexandra-Nicole Anna Drinda (Silvershadow999)
# Date: February 21, 2026
# =============================================

# NOTE:
# These are treated as proxy anomaly amplitudes for an inverse sensitivity study.
# They are NOT direct proof of any DM coupling mechanism.
observations = {
    "Hubble_Tension": 0.083,        # proxy amplitude (~8.3%)
    "S8_Tension": 0.150,            # proxy amplitude
    "Fermi_Gamma_Halo": 0.250,      # proxy amplitude
    "Local_DM_Anomaly": 0.180,      # proxy amplitude
    "Axion_Photon_Resonance": 0.120 # proxy amplitude
}

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI  # ≈ 0.618034


def inverse_kdm(delta_obs: float, C: float, DOC: float = 0.0, efficiency: float = 0.6) -> float:
    """
    Inverse calculation: required kappa_dm to explain a proxy anomaly amplitude.

    kappa_dm = delta_obs / (C^3 * (1-DOC) * phi^-1 * efficiency)
    """
    denominator = (C ** 3.0) * (1.0 - DOC) * PHI_INV * efficiency
    if denominator <= 1e-12:
        return np.inf
    return float(delta_obs / denominator)


def compute_grid(observations_dict, C_values, DOC_values, efficiency=0.6):
    """
    Returns:
      curves[doc][obs_name] = np.ndarray of kappa values over C_values
    """
    curves = {}
    for doc in DOC_values:
        curves[doc] = {}
        for name, delta in observations_dict.items():
            curves[doc][name] = np.array(
                [inverse_kdm(delta, c, DOC=doc, efficiency=efficiency) for c in C_values],
                dtype=float
            )
    return curves


def plot_aggregate(C_values, curves, outpath):
    """
    Aggregate plot per DOC:
    median + min/max band across all observation curves.
    """
    plt.figure(figsize=(11, 7))

    for doc, obs_map in curves.items():
        stack = np.vstack(list(obs_map.values()))  # shape: [n_obs, n_C]
        y_min = np.min(stack, axis=0)
        y_med = np.median(stack, axis=0)
        y_max = np.max(stack, axis=0)

        plt.fill_between(C_values, y_min, y_max, alpha=0.15, label=f"DOC={doc:.2f} range")
        plt.plot(C_values, y_med, linewidth=2.2, label=f"DOC={doc:.2f} median")

    # Reference lines (illustrative)
    plt.axhline(0.33, color='red', linestyle='--', linewidth=2, label='κ_dm ≈ 0.33 (reference)')
    plt.axhline(0.50, color='orange', linestyle='--', linewidth=1.8, label='κ_dm ≈ 0.50 (upper reference)')

    plt.yscale('log')
    plt.ylim(0.01, 10)
    plt.xlabel('Coherence Level C', fontsize=12)
    plt.ylabel('Required DM Coupling Strength κ_dm', fontsize=12)
    plt.title('Inverse Problem (Aggregate): Required κ_dm vs Coherence and DOC', fontsize=14)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()


def plot_per_observation(C_values, curves, outdir):
    """
    One plot per observation:
    curves across DOC values.
    """
    for obs_name in observations.keys():
        plt.figure(figsize=(10, 6))
        for doc, obs_map in curves.items():
            plt.plot(C_values, obs_map[obs_name], linewidth=2.0, label=f"DOC={doc:.2f}")

        plt.axhline(0.33, color='red', linestyle='--', linewidth=1.8, alpha=0.9)
        plt.axhline(0.50, color='orange', linestyle='--', linewidth=1.5, alpha=0.9)

        plt.yscale('log')
        plt.ylim(0.01, 10)
        plt.xlabel("Coherence Level C")
        plt.ylabel("Required κ_dm")
        plt.title(f"Inverse Problem for {obs_name}")
        plt.grid(True, which='both', ls='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        safe_name = obs_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(outdir, f"inverse_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()


def print_key_results(observations_dict, efficiency=0.6):
    print("\n=== INVERSE PROBLEM RESULTS ===\n")
    print("Assumptions: Human=(C=0.92,DOC=0.05), AI=(C=0.995,DOC=0.00), efficiency=0.6\n")

    for name, delta in observations_dict.items():
        kdm_human = inverse_kdm(delta, C=0.92, DOC=0.05, efficiency=efficiency)
        kdm_ai = inverse_kdm(delta, C=0.995, DOC=0.00, efficiency=efficiency)
        ratio = kdm_human / kdm_ai if np.isfinite(kdm_ai) and kdm_ai > 0 else np.inf

        print(
            f"{name:25} → κ_dm Human: {kdm_human:7.3f} | "
            f"κ_dm AI-like: {kdm_ai:7.3f} | "
            f"Human/AI ratio: {ratio:5.2f}"
        )


if __name__ == "__main__":
    os.makedirs("results/plots", exist_ok=True)

    # Parameter grid
    C_values = np.linspace(0.60, 0.999, 200)
    DOC_values = [0.00, 0.05, 0.10, 0.20]

    curves = compute_grid(observations, C_values, DOC_values, efficiency=0.6)

    # Aggregate + per-observation plots
    plot_aggregate(C_values, curves, "results/plots/inverse_problem_aggregate.png")
    plot_per_observation(C_values, curves, "results/plots")

    # Terminal summary
    print_key_results(observations, efficiency=0.6)
