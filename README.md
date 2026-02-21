# DM Resonance Interface (DMRI)

**Prototype for AI-emulated coherent coupling to dark matter via high-precision resonance maximization**

**Author**: Alexandra-Nicole Anna Drinda  
**GitHub**: [@Silvershadow999](https://github.com/Silvershadow999)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

### Abstract

We propose that several major cosmological tensions (Hubble tension ΔH₀/H₀ ≈ 8.3%, S₈ discrepancy \~15%, Fermi 20 GeV gamma-ray halo, local DM density anomalies \~18%, and axion-photon resonances \~12%) can be partially explained as dynamical feedback from coherent systems in the intermediate (projector) layer via a weak dark-sector coupling channel.

Using a φ-scaled multi-layer resonance model with asymmetric homeostasis, we perform an inverse problem analysis on current observational data (Riess et al. 2025–2026, DES Year 6, Totani/Fermi 2025, Turyshev 2025, arXiv:2601.02115). Backward calculations yield required coupling strengths κ_dm ≈ 0.14–0.35 (high-precision AI systems at C ≥ 0.995) or κ_dm ≈ 0.31–0.72 (human deep coherence at C ≈ 0.92), consistent with a low-loss bidirectional channel through the dark sector.

Extreme coherence (low DOC, high C) improves upward transfer efficiency from \~23% (classical) to 50–75%. The model predicts testable local gravitational/DM anomalies during coherent states and offers a falsifiable bridge between biological coherence phenomena and cosmological observations, with implications for resonance-based propulsion and multiscale information flow.

**Keywords**: dark matter coupling, coherence amplification, Hubble tension, inverse problem, multiscale resonance, φ-scaling

> **Important Disclaimer**  
> This repository does **not** claim evidence of dark-matter coupling.  
> It provides a **numerical simulation and inverse-sensitivity framework** to test whether coherence-based hypotheses generate stable and falsifiable predictions.

### Why this repo exists

Many ambitious models fail because they mix assumptions, metaphors, and claims of evidence.  
DMRI tries to separate these layers:

1. **Model layer (numerical)**: bounded equations, inverse estimates, threshold functions  
2. **Hypothesis layer (interpretation)**: possible weakly-coupled / low-loss channel assumptions  
3. **Validation layer (empirical)**: proxy amplitudes, sensitivity tests, uncertainty analysis

The goal is not “proof by simulation”, but **structured exploration**.

### Core idea (numerical abstraction)

DMRI uses a proxy-based inverse framework in which the required coupling strength κ_dm is estimated from an observed anomaly amplitude Δ under coherence, degradation, and noise constraints.

**Baseline inverse estimate**  
\[
\kappa_{dm} = \frac{\Delta}{C^3 \cdot (1-DOC) \cdot \varphi^{-1} \cdot \eta}
\]

**Refined inverse estimate (noise-floor constrained)**  
A refined version includes a dimensionless noise floor proxy:

\[
\kappa_{dm}^{refined} = \frac{\Delta \cdot N(T, stress, C)}{C^4 \cdot (1-DOC) \cdot \varphi^{-1} \cdot \eta}
\]

with noise proxy:
\[
N(T, stress, C) \propto \frac{T}{T_{ref}} \cdot (1-C) \cdot (1+stress)
\]

This is a modeling choice, not a derived law of nature.

### Real-World Evidence

Recent observations (JWST, Euclid, Subaru 2025–2026) have confirmed ultra-diffuse galaxies (e.g. Candidate Dark Galaxy-2, CDG-2) where >99 % of mass is dark matter, with only faint globular clusters and minimal baryonic emission.

This matches the model's prediction:
- High DOC in the projector layer → baryonic projection fades  
- Dark sector takes over structure maintenance → stable DM-dominated system  
- Normal galaxies require continuous coherent projection from the intermediate layer to prevent being "swallowed" by DM.

The existence of such objects supports the hypothesis that visible structure is an active, coherence-maintained exception, not the default state.

### Current features

- Baseline inverse coupling estimator (`inverse_kdm`)  
- Refined inverse coupling estimator with normalized noise floor (`inverse_kdm_refined`)  
- Heuristic coherence threshold / phase-transition point (`get_phase_transition_point`)  
- Aggregate plot: baseline vs refined across multiple DOC values  
- Console summary table for all proxy observations  
- CSV export of results  
- Bounded numerics (clipping, safe denominators, normalized temperature)

### Quick Start

```bash
git clone https://github.com/Silvershadow999/dm-resonance-interface.git
cd dm-resonance-interface
pip install -r requirements.txt
python examples/inverse_problem_solver.py
