# Supplementary Mathematical Derivations

This directory contains the complete, self-contained derivations
supporting the lemmas and the main theorem of the paper

> *Sufficient Conditions for Sonar-Based Contour-Tracking Measurement
> Systems: Field-of-View Geometry and a Scanning-versus-Forward
> Comparison.*

The paper presents the results with proof *sketches* for length reasons;
these notes expand each sketch to the level of detail of a journal
supplementary appendix. Notation follows the paper exactly. Math renders
on GitHub via MathJax (inline `$...$`, display `$$...$$`).

The closed loop is modelled as a hybrid automaton with four modes —
tracking ($T$), noise-driven loss ($L_N$), geometry-driven loss ($L_G$),
and re-acquisition ($R$). The derivations characterise each mode and then
compose them into a single finite-horizon probabilistic guarantee.

## Files

1. **`01-tracking-conditions.md`** — Cascade Lyapunov / ISS analysis
   ($V=V_o+V_u+V_r$) yielding the static condition (C1) and the bandwidth
   condition (C2), including the origin of each admissibility-budget term
   and the half-sample / transport-delay contributions to (C2).
2. **`02-curvature-uncertainty.md`** — Closed-form curvature-estimator
   variance $\sigma_{\hat\kappa}^{2}=720\,\sigma_\eta^{2}/(L^{5}\bar s^{4})$
   from the least-squares normal equations, with CRLB achievability.
3. **`03-loss-dwell.md`** — The two loss-mode dwell bounds $T^{\star}_N$
   and $T^{\star}_G$, including the geometric-rigidity property of
   $T^{\star}_G$.
4. **`04-reacquisition.md`** — Koopman random-search re-acquisition
   probability and the zeroing-CBF safety guarantee.
5. **`05-joint-trackability.md`** — Average-dwell-time + union-bound
   composition into Theorem 1 and the admissible set $\Phi^{\mathrm{adm}}$.

## Correspondence table

| File | Paper section | Lemma / Theorem | Equation(s) derived |
|------|---------------|-----------------|---------------------|
| `01-tracking-conditions.md` | Section III | Lemma 1 (Tracking-mode ISS) | (4) = (C1); (5) = (C2); Table I budget terms $\alpha_{\mathrm{noise}},\alpha_{\mathrm{cur}},\alpha_{\mathrm{sample}},\alpha_{\mathrm{delay}},\alpha_{\mathrm{rate}}$ |
| `02-curvature-uncertainty.md` | Section IV | (estimator uncertainty) | (6) $\sigma_{\hat\kappa}^{2}=720\,\sigma_\eta^{2}/(L^{5}\bar s^{4})$ |
| `03-loss-dwell.md` | Section IV | Lemma 2 (Loss-mode dwell) | (7) $T^{\star}_N$; (8) $T^{\star}_G$ |
| `04-reacquisition.md` | Section IV | Lemma 3 (Re-acquisition) | (9) $P[\text{reacquire}]\ge 1-e^{-\lambda}$, $\lambda=R_{\mathrm{FOV}}w_{\mathrm{FOV}}\underline u\,T_R/A_{\mathrm{unc}}$ |
| `05-joint-trackability.md` | Section V | Theorem 1 (Joint trackability) | (10) joint bound, $\delta(\mathcal T)\le(\mathcal T/T_D)(\delta_L+e^{-\lambda}+\delta_G)$; (11) $\Phi^{\mathrm{adm}}$ |

## Symbol glossary

| Symbol | Meaning |
|--------|---------|
| $d^{\star}$ | desired standoff distance |
| $d_{\min}$ | collision lower bound |
| $\underline u$ | worst-case advance speed |
| $\bar V_c$ | current-magnitude bound |
| $\kappa_{\max}$ | curvature bound of the wall class $\mathbb W$ |
| $\kappa'_{\max}$ | curvature-rate bound (m$^{-2}$) |
| $w_{\mathrm{FOV}}$ | full angular field of view (rad); $\beta_{\mathrm{FOV}}=w_{\mathrm{FOV}}/2$ |
| $R_{\mathrm{FOV}}$ | sensor range (m) |
| $\sigma_\eta$ | per-sample range-noise standard deviation (m) |
| $\sigma_{\hat\kappa}$ | curvature-estimate standard deviation (m$^{-2}$ for variance) |
| $\tau_d$ | end-to-end processing (transport) delay (s) |
| $f_s$ | sampling rate (Hz) |
| $L$ | sliding-window sample count |
| $\bar s$ | along-track sample spacing (m) |
| $\Delta$ | LOS lookahead distance |
| $T^{\star}_N,T^{\star}_G$ | noise- / geometry-driven loss dwell bounds |
| $A_{\mathrm{unc}}$ | re-acquisition target-uncertainty area (m$^2$) |
| $\delta_L,e^{-\lambda},\delta_G$ | per-mode failure budgets |
| $\Phi^{\mathrm{adm}}$ | admissible configuration set |
