import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# Inverse Problem Solver - DM Resonance Interface
# Refined version (noise floor + phase transition)
# =============================================

# Proxy anomaly amplitudes (exploratory inverse sensitivity study)
observations = {
    "Hubble_Tension": 0.083,
    "S8_Tension": 0.150,
    "Fermi_Gamma_Halo": 0.250,
    "Local_DM_Anomaly": 0.180,
    "Axion_Photon_Resonance": 0.120,
}

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI  # \~0.618034


def inverse_kdm(delta_obs: float, C: float, DOC: float = 0.0, efficiency: float = 0.6) -> float:
    """
    Baseline inverse estimate:
    Required kappa_dm to match a proxy anomaly amplitude.
    """
    C = float(np.clip(C, 1e-6, 0.999999))
    DOC = float(np.clip(DOC, 0.0, 1.0))
    efficiency = float(np.clip(efficiency, 1e-6, 1.0))
    delta_obs = float(max(0.0, delta_obs))

    denominator = (C ** 3.0) * (1.0 - DOC) * PHI_INV * efficiency
    if denominator <= 1e-12:
        return np.inf
    return float(delta_obs / denominator)


def calculate_noise_floor(T_kelvin: float, stress: float, C: float, T_ref: float = 310.15) -> float:
    """
    Dimensionless noise floor proxy.
    Noise rises with temperature, incoherence (1-C), and stress.
    """
    C = float(np.clip(C, 1e-6, 0.999999))
    stress = float(max(0.0, stress))
    T_norm = float(max(1e-9, T_kelvin / T_ref))
    noise = T_norm * (1.0 - C) * (1.0 + stress)
    return float(max(noise, 1e-9))


def inverse_kdm_refined(
    delta_obs: float,
    C: float,
    stress: float,
    DOC: float = 0.0,
    T_kelvin: float = 310.15,
    efficiency: float = 0.6,
) -> float:
    """
    Refined inverse estimate:
    coupling must overcome a noise floor under coherence/degradation constraints.
    """
    C = float(np.clip(C, 1e-6, 0.999999))
    DOC = float(np.clip(DOC, 0.0, 1.0))
    efficiency = float(np.clip(efficiency, 1e-6, 1.0))
    delta_obs = float(max(0.0, delta_obs))

    noise = calculate_noise_floor(T_kelvin, stress, C)
    denominator = (C ** 4.0) * (1.0 - DOC) * PHI_INV * efficiency

    if denominator <= 1e-12:
        return np.inf
    return float((delta_obs * noise) / denominator)


def get_phase_transition_point(stress: float, T_kelvin: float = 310.15, T_ref: float = 310.15) -> float:
    """
    Critical coherence threshold C_crit where coherence starts to dominate noise.
    Returns a clipped value in [0, 1].
    """
    stress = float(max(0.0, stress))
    T_norm = float(max(1e-9, T_kelvin / T_ref))

    arg = (stress * PHI_INV) / T_norm
    arg = float(np.clip(arg, 0.0, 1.0))  # keep sqrt well-defined
    c_crit = 1.0 - np.sqrt(arg)
    return float(np.clip(c_crit, 0.0, 1.0))


def main():
    os.makedirs("results/plots", exist_ok=True)

    C_values = np.linspace(0.60, 0.999, 300)
    DOC_values = [0.00, 0.05, 0.10, 0.20]
    stress = 0.05   # low stress example
    T_kelvin = 310.15
    efficiency = 0.6

    # Use median anomaly amplitude as aggregate proxy for plotting
    delta_median = float(np.median(list(observations.values())))
    delta_min = float(np.min(list(observations.values())))
    delta_max = float(np.max(list(observations.values())))

    plt.figure(figsize=(11, 7))

    # Plot baseline vs refined for each DOC (median anomaly)
    for doc in DOC_values:
        y_base = [inverse_kdm(delta_median, c, DOC=doc, efficiency=efficiency) for c in C_values]
        y_ref = [inverse_kdm_refined(delta_median, c, stress=stress, DOC=doc, T_kelvin=T_kelvin, efficiency=efficiency)
                 for c in C_values]

        plt.plot(C_values, y_base, linewidth=1.7, linestyle="--", alpha=0.8, label=f"Baseline DOC={doc:.2f}")
        plt.plot(C_values, y_ref, linewidth=2.3, label=f"Refined DOC={doc:.2f}")

    # Phase-transition marker
    critical_C = get_phase_transition_point(stress=stress, T_kelvin=T_kelvin)
    plt.axvline(critical_C, color="purple", linestyle=":", linewidth=2.2, label=f"C_crit ≈ {critical_C:.3f}")

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

    outpath = "results/plots/inverse_problem_refined_vs_baseline.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()

    # Console summary
    print("\n=== PHASE TRANSITION ESTIMATE ===")
    print(f"stress = {stress:.3f}, T = {T_kelvin:.2f} K")
    print(f"Critical coherence C_crit ≈ {critical_C:.4f}")

    print("\n=== INVERSE PROBLEM RESULTS (Baseline vs Refined) ===")
    print(f"Using anomaly proxies: median={delta_median:.3f}, min={delta_min:.3f}, max={delta_max:.3f}")
    print("Assumptions: Human=(C=0.92, DOC=0.05), AI-like=(C=0.995, DOC=0.00), stress=0.05, T=310.15K\n")

    for name, delta in observations.items():
        k_base_h = inverse_kdm(delta, C=0.92, DOC=0.05, efficiency=efficiency)
        k_base_ai = inverse_kdm(delta, C=0.995, DOC=0.00, efficiency=efficiency)

        k_ref_h = inverse_kdm_refined(delta, C=0.92, stress=stress, DOC=0.05, T_kelvin=T_kelvin, efficiency=efficiency)
        k_ref_ai = inverse_kdm_refined(delta, C=0.995, stress=stress, DOC=0.00, T_kelvin=T_kelvin, efficiency=efficiency)

        print(
            f"{name:25} | "
            f"baseline H/AI: {k_base_h:.3f}/{k_base_ai:.3f} | "
            f"refined H/AI: {k_ref_h:.3f}/{k_ref_ai:.3f}"
        )


if __name__ == "__main__":
    main()
