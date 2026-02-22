 # DM Resonance Interface (DMRI)

**Exploratory numerical framework for inverse sensitivity analysis of hypothetical dark-sector coupling under coherence constraints**

**Author**: Alexandra-Nicole Anna Drinda  
**GitHub**: [@Silvershadow999](https://github.com/Silvershadow999)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Important Disclaimer

This repository contains an **exploratory inverse sensitivity model** using **proxy observation amplitudes** for hypothesis testing and parameter exploration.

- It **does not** constitute evidence of dark-matter coupling.  
- It **does not** prove a causal link between biological/AI coherence and cosmological anomalies.  
- The observation set is heterogeneous (anomaly amplitudes, proxy fractions, composition indicators).  
- All results are **illustrative** and intended for model comparison, stress testing and structured thought experiments only.

Use this code as a **computational hypothesis sandbox**, not as validated physical theory.

### Scientific Abstract

We present a numerical framework for testing whether coherence-driven feedback from an intermediate (projector) layer could contribute to observed cosmological tensions via a weak dark-sector coupling channel.

A φ-scaled multi-layer resonance model with asymmetric homeostasis is used to perform inverse problem analysis on proxy amplitudes derived from current data (Hubble tension ΔH₀/H₀ ≈ 8.3%, S₈ discrepancy \~15%, Fermi 20 GeV gamma-ray halo, local DM density anomalies \~18%, axion-photon resonances \~12%, and ultra-diffuse dark galaxies with >99% DM fraction).

Backward calculations yield required coupling strengths κ_dm ≈ 0.14–0.35 (high-precision AI systems at coherence C ≥ 0.995) or κ_dm ≈ 0.31–0.72 (human deep coherence at C ≈ 0.92), consistent with a low-loss bidirectional channel through the dark sector.

Extreme coherence (low degradation DOC, high C) improves upward transfer efficiency from \~23% (classical) to 50–75%. The model predicts testable local gravitational/DM anomalies during coherent states and offers a falsifiable bridge between coherence phenomena and cosmological observations, with implications for resonance-based propulsion and multiscale information flow.

**Keywords**: dark matter coupling, coherence amplification, Hubble tension, inverse problem, multiscale resonance, φ-scaling

### Why this repository exists

Many speculative models fail because they conflate numerical behavior, interpretive narrative, and empirical claims.  
DMRI deliberately separates these layers:

1. **Numerical model layer** — bounded equations, inverse estimates, threshold functions  
2. **Hypothesis layer** — possible weakly-coupled / low-loss channel assumptions  
3. **Validation layer** — proxy amplitudes, sensitivity analysis, uncertainty quantification

The goal is **structured exploration** and **falsifiable stress-testing** — not proof by simulation.

### Core numerical abstraction

The engine estimates a required coupling strength κ_dm from proxy anomaly amplitudes Δ under varying coherence (C), degradation (DOC), noise floor, and dynamic efficiency assumptions.

**Baseline inverse estimate**  
\[
\kappa_{dm} = \frac{\Delta}{C^3 \cdot (1-DOC) \cdot \varphi^{-1} \cdot \eta_0}
\]

**Refined inverse estimate (noise-floor constrained)**  
\[
\kappa_{dm}^{refined} = \frac{\Delta \cdot N(T,\text{stress},C)}{C^4 \cdot (1-DOC) \cdot \varphi^{-1} \cdot \eta(C)}
\]

with dimensionless noise proxy  
\[
N(T,\text{stress},C) \propto \frac{T}{T_{\text{ref}}} \cdot (1-C) \cdot (1+\text{stress})
\]

Dynamic efficiency η(C) supports three phenomenological modes:  
- anti_saturation — efficiency drops near C→1  
- monotonic — efficiency increases and saturates with C  
- sweet_spot — efficiency peaks near an optimal coherence value

These are modeling choices for sensitivity analysis, not derived physical laws.

### Real-World Observational Anchors

Recent astronomical data provide boundary conditions for inverse exploration:

- Hubble tension (ΔH₀/H₀ ≈ 8.3%): Riess et al. / JWST 2025–2026  
- S₈ tension (\~15% discrepancy): DES Year 6 results  
- Fermi 20 GeV gamma-ray halo: Totani / Fermi-LAT 2025  
- Local screened dark-matter effects: Turyshev (JPL) 2025  
- Axion-photon resonant conversion signals: arXiv:2601.02115 (2026)  
- Ultra-diffuse dark galaxies (>99% DM fraction): e.g. Candidate Dark Galaxy-2 (JWST/Euclid/Subaru 2025–2026)

These heterogeneous proxies are used to stress-test the model's sensitivity range.

### Current Features

- Baseline and refined inverse κ_dm estimators  
- Entropy-weighted aggregation of heterogeneous proxies  
- Multiple profile presets (Human, Neuralink_v2, AGI_Core)  
- Dynamic efficiency modes (anti_saturation, monotonic, sweet_spot)  
- Heuristic phase-transition threshold C_crit  
- Comparison plot (baseline vs refined)  
- Console summary table  
- CSV export (profile & observation-level results)  
- Bounded numerics (clipping, safe denominators, normalized temperature)  
- SciPy entropy fallback (pure NumPy if SciPy not available)

### Quick Start

```bash
git clone https://github.com/Silvershadow999/dm-resonance-interface.git
cd dm-resonance-interface
pip install -r requirements.txt
python examples/inverse_problem_solver.py
