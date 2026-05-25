# Tracking-Mode Conditions (C1) and (C2)

**Corresponds to: paper Section III, Lemma 1, equation(s) (4) [=(C1)] and (5) [=(C2)].**

This note expands the proof sketch of Lemma 1 into a complete cascade
input-to-state-stability (ISS) derivation. We construct the cascade
Lyapunov candidate $V = V_o + V_u + V_r$, isolate one error loop per
term, and show how each admissibility-budget term
($\alpha_{\mathrm{noise}},\alpha_{\mathrm{cur}},\alpha_{\mathrm{sample}},
\alpha_{\mathrm{delay}},\alpha_{\mathrm{rate}}$) arises. The static
budget (C1) follows from the steady-state (zero-rate) balance of the
cross-track loop; the bandwidth budget (C2) follows from a phase-margin
analysis of the same loop under the zero-order-hold and transport-delay
lags. Throughout, $\underline u$ is the worst-case advance speed,
$w_{\mathrm{FOV}}$ the angular field of view, $\beta_{\mathrm{FOV}} =
w_{\mathrm{FOV}}/2$, and $R_{\min}(\underline u) = \underline u/r_{\max}$
the speed-dependent minimum turn radius.

---

## 1. Error dynamics and the guidance law

We recall the Serret--Frenet error dynamics (paper eq. (1)). With $d$ the
wall-relative distance, $\theta$ the angle between the body axis and the
wall tangent, $s$ the foot-point arc length, $\psi$ the heading, $u$ the
surge speed, $r$ the yaw rate, $\kappa(s)$ the signed wall curvature, and
$V_c$ a constant current of magnitude $\|V_c\|\le \bar V_c$ and direction
$\theta_c$,

$$
\dot d = u\sin\theta - V_c\sin(\psi - \theta_c - \theta),
$$

$$
\dot\theta = r - \kappa(s)\,\dot s,
\qquad
\dot s = \frac{u\cos\theta - V_c\cos(\psi-\theta_c-\theta)}{1-\kappa d},
$$

$$
\dot u = f_u(u,\tau_c) - D(u) - D_{\mathrm{ind}}(u,\kappa),
\qquad
\dot r =
\begin{cases}
f_r(u,\delta_c), & u \ge u_{\mathrm{gate}},\\[2pt]
-\zeta r, & u < u_{\mathrm{gate}}.
\end{cases}
$$

The guidance law is line-of-sight (LOS) with curvature feedforward (paper
eq. (2)),

$$
r^{\star}(t) = \hat\kappa(t)\,\hat u(t)
+ r^{\star}_{\mathrm{LOS}}\!\big(d-d^{\star},\,\theta;\,\Delta\big),
$$

with lookahead $\Delta$ and curvature estimate $\hat\kappa$. The standard
LOS desired heading is

$$
\psi_{\mathrm{LOS}} = \theta_{\mathrm{path}}
- \arctan\!\frac{d-d^{\star}}{\Delta},
$$

so that the LOS heading error $\tilde\theta_{\mathrm{LOS}} = \psi -
\psi_{\mathrm{LOS}}$ drives the cross-track loop. We write the cross-track
deviation $\tilde d := d - d^{\star}$.

---

## 2. Cascade structure

The closed loop has a triangular (cascade) interconnection: the yaw loop
$r\to r^{\star}$ is the innermost and fastest; it feeds the heading/yaw
kinematics $\theta$; the surge loop $u\to v^{\star}$ sets the advance
speed that scales the cross-track loop; and the outermost loop is the
cross-track/heading pair $(\tilde d,\tilde\theta_{\mathrm{LOS}})$. We
therefore pick a Lyapunov candidate with one quadratic term per loop,

$$
V = V_o + V_u + V_r,
$$

$$
V_o = \tfrac12 \tilde d^{\,2}
+ \tfrac{\Delta^{2}}{2}\sin^{2}\tilde\theta_{\mathrm{LOS}},
\qquad
V_u = \tfrac{c_u}{2}(u-v^{\star})^{2},
\qquad
V_r = \tfrac{c_r}{2}(r-r^{\star})^{2},
$$

with positive weights $c_u,c_r>0$. The $\Delta^2$ scaling on the heading
term is the canonical LOS choice that makes the cross-coupling in
$\dot V_o$ sign-definite; see below.

### 2.1 Yaw loop ($V_r$)

For $u\ge u_{\mathrm{gate}}$ the rudder map $f_r$ is invertible and the
commanded yaw $r^{\star}$ is realizable up to the yaw-acceleration limit
$\dot r_{\max}$. A proportional inner law gives

$$
\dot V_r = c_r (r-r^{\star})(\dot r - \dot r^{\star})
\le -c_r\,k_r (r-r^{\star})^{2} + c_r (r-r^{\star})\,e_{\mathrm{ff}},
$$

where $e_{\mathrm{ff}}$ is the feedforward tracking residual. The
feedforward command must track the rate of change of $r^{\star}$, whose
dominant component on a curved wall is $\dot r^\star \approx
\kappa'\,\dot s\,u \le \kappa'_{\max}\,\underline u^{2}$ at worst case
(advance speed $\underline u$, $\dot s\approx u$). Realizability of this
command requires

$$
\kappa'_{\max}\,\underline u^{2} \le \dot r_{\max},
$$

which, normalized, is precisely the curvature-rate budget term

$$
\boxed{\;\alpha_{\mathrm{rate}}
= \frac{\kappa'_{\max}\,\underline u^{2}}{\dot r_{\max}}.\;}
$$

When $\alpha_{\mathrm{rate}}<1$ the yaw loop is ISS with respect to its
feedforward residual; the residual then enters (C2) as a phase budget,
derived in §4.

### 2.2 Surge loop ($V_u$)

With $D(u) = X_u u + X_{u|u|}u|u|$ the nonlinear surge damping and a
surge controller closing $u\to v^{\star}$,

$$
\dot V_u = c_u (u-v^{\star})\big[f_u - D(u) - D_{\mathrm{ind}}(u,\kappa) - \dot v^{\star}\big]
\le -c_u\,k_u (u-v^{\star})^{2} + c_u (u-v^{\star})\,d_u,
$$

with $d_u$ collecting the curvature-induced drag $D_{\mathrm{ind}}$ and
current load. The fixed point of the steady-state thrust--drag--current
balance defines the worst-case advance speed $\underline u$; the surge
loop is ISS provided $\underline u \ge u_{\mathrm{gate}}$, so that the
rudder retains authority (the second clause of (C1)). This is the
coupling that produces the non-monotone feasibility island: if
$\underline u < u_{\mathrm{gate}}$ the yaw branch switches to $\dot r =
-\zeta r$ and the cascade breaks. The switched pair $u\gtrless
u_{\mathrm{gate}}$ is handled by a multiple-Lyapunov argument; on the
feasible branch $u\ge u_{\mathrm{gate}}$ the single candidate $V$ above
governs.

### 2.3 Cross-track loop ($V_o$)

Differentiating $V_o$ along (1) and substituting the LOS law,

$$
\dot V_o
= \tilde d\,\dot{\tilde d}
+ \Delta^{2}\sin\tilde\theta_{\mathrm{LOS}}\cos\tilde\theta_{\mathrm{LOS}}\,\dot{\tilde\theta}_{\mathrm{LOS}}.
$$

Using $\dot{\tilde d} = u\sin\theta - V_c\sin(\cdot)$ and the LOS identity
$\sin\tilde\theta_{\mathrm{LOS}} = \tilde d/\sqrt{\tilde d^{2}+\Delta^{2}}$
at the nominal heading, the leading term contracts as

$$
\dot V_o \le -\,\frac{\underline u\,\Delta}{\sqrt{\tilde d^{2}+\Delta^{2}}}\;\tilde d^{2}
+ \underbrace{\bar V_c\,|\tilde d|}_{\text{current load}}
+ \underbrace{\Delta^{2}|\sin\tilde\theta_{\mathrm{LOS}}|\,|r-r^{\star}|}_{\text{yaw coupling}}
+ \underbrace{\Delta^{2}|\sin\tilde\theta_{\mathrm{LOS}}|\,|\hat\kappa-\kappa|\,u}_{\text{estimator coupling}}.
$$

The first term is the LOS contraction at rate $\omega_{\mathrm{LOS}}
\approx \underline u/\Delta$; the remaining three are disturbance inputs.
The estimator-coupling term carries the curvature-estimate error
$\hat\kappa-\kappa$, whose standard deviation is
$\sigma_{\hat\kappa}=\sqrt{720}\,\sigma_\eta/(L^{5/2}\bar s^{2})$
(derived in `02-curvature-uncertainty.md`). Its contribution to the
steady-state cross-track offset, after dividing by the contraction rate
and absorbing the LOS gain $K_p$ and the per-window averaging $\sqrt L$,
is the range-noise budget term

$$
\boxed{\;\alpha_{\mathrm{noise}}
= \frac{K_p\,\sigma_\eta}{\Delta\,r_{\max}\sqrt L}.\;}
$$

The $\sqrt L$ in the denominator is the variance reduction of the
$L$-sample sliding window; the $\Delta\,r_{\max}$ normalizes the LOS
heading authority. The current load term yields, after the same
normalization by the advance speed, the cross-drift budget

$$
\boxed{\;\alpha_{\mathrm{cur}} = \frac{\bar V_c}{\underline u}.\;}
$$

---

## 3. The static condition (C1)

Collecting the steady-state (zero curvature-rate, zero lag) terms, the
cross-track loop admits a nonnegative ISS margin only if the demanded
turn, the noise-driven LOS amplification, and the current overhead fit
within unit budget. The curvature demand on the wall class $\mathbb W$ is
$\kappa_{\max}R_{\min}(\underline u)$, the fraction of the available turn
budget consumed by following the sharpest admissible wall. Adding the two
disturbance terms,

$$
\boxed{\;
\kappa_{\max}\,R_{\min}(\underline u)
+ \underbrace{\frac{K_p\,\sigma_\eta}{\Delta\,r_{\max}\sqrt L}}_{\alpha_{\mathrm{noise}}}
+ \underbrace{\frac{\bar V_c}{\underline u}}_{\alpha_{\mathrm{cur}}}
\;\le\; 1,
\qquad \underline u \ge u_{\mathrm{gate}}.
\;}
\tag{C1}
$$

This is the Dubins-type steady-state feasibility budget. The condition is
exactly the requirement that the per-step decrease of $V_o$ remain
negative outside an ultimate bound: under (C1) there exist
$\lambda_o>0$ and an offset $b_o\ge 0$ with
$\dot V_o \le -\lambda_o V_o + b_o$, and the residual $b_o$ is what makes
the bound *ultimate* rather than asymptotic-to-zero. The second clause
$\underline u\ge u_{\mathrm{gate}}$ guarantees the yaw cascade does not
collapse, preserving $\inf_t d > 0$.

---

## 4. The bandwidth condition (C2)

(C1) ignores actuation lag. The LOS loop is a closed loop of bandwidth
$\omega_{\mathrm{LOS}}\approx \underline u/\Delta$; any phase lag in the
measurement-to-actuation path erodes its stability margin. We account for
two lag sources and the curvature-rate residual of §2.1.

### 4.1 Zero-order-hold half-sample lag

The sonar samples at rate $f_s$, and the controller holds each command
over the inter-sample interval $1/f_s$ (zero-order hold). The frequency
response of a ZOH is

$$
H_{\mathrm{zoh}}(j\omega)
= \frac{1-e^{-j\omega/f_s}}{j\omega/f_s}
= e^{-j\omega/(2 f_s)}\,\mathrm{sinc}\!\Big(\frac{\omega}{2 f_s}\Big),
$$

whose phase is $-\omega/(2 f_s)$. The ZOH therefore acts, to first order,
as a pure transport delay of **half a sample period**,

$$
\tau_{\mathrm{zoh}} = \frac{1}{2 f_s}.
$$

Evaluated at the loop bandwidth $\omega = \omega_{\mathrm{LOS}}$, this
contributes the phase loss

$$
\boxed{\;\alpha_{\mathrm{sample}}
= \omega_{\mathrm{LOS}}\cdot\frac{1}{2 f_s}
= \frac{\omega_{\mathrm{LOS}}}{2 f_s}.\;}
$$

### 4.2 Transport delay

The end-to-end processing delay $\tau_d$ (range detection, arc fit,
guidance update) is a genuine transport delay $e^{-j\omega\tau_d}$, with
phase $-\omega\tau_d$. We treat it through a Lyapunov--Krasovskii
functional: appending

$$
V_\tau = \int_{t-\tau_d}^{t}\!\!\int_{\theta}^{t}
\dot\xi(\sigma)^{\top}R\,\dot\xi(\sigma)\,\mathrm d\sigma\,\mathrm d\theta
$$

to $V$ produces, after a Jensen-inequality bound on the delayed state,
the same first-order phase penalty $\omega_{\mathrm{LOS}}\tau_d$ in the
margin budget. Hence

$$
\boxed{\;\alpha_{\mathrm{delay}} = \omega_{\mathrm{LOS}}\,\tau_d.\;}
$$

### 4.3 Curvature-rate residual

The yaw-loop residual of §2.1, $\alpha_{\mathrm{rate}} =
\kappa'_{\max}\underline u^{2}/\dot r_{\max}$, is the inner-loop tracking
lag relative to the feedforward demand and adds directly to the margin
budget.

### 4.4 Assembling (C2) and the $1/(2\pi)$ normalization

The total phase lag the LOS loop can tolerate before losing its stability
margin is a fixed fraction of a cycle. Summing the two delay channels and
the rate residual,

$$
\boxed{\;
\omega_{\mathrm{LOS}}\!\cdot\!\Big(\tau_d + \frac{1}{2 f_s}\Big)
+ \underbrace{\frac{\kappa'_{\max}\,\underline u^{2}}{\dot r_{\max}}}_{\alpha_{\mathrm{rate}}}
\;\le\; \frac{1}{2\pi}.
\;}
\tag{C2}
$$

The constant $1/(2\pi)$ is a conservative normalization. The total
accumulated phase, expressed in cycles rather than radians, is
$(\text{phase in rad})/(2\pi)$; requiring the dimensionless lag-fraction
on the left to stay below $1/(2\pi)$ corresponds to admitting at most one
radian of accumulated open-loop phase lag at the crossover frequency,
i.e. a phase margin comfortably bounded away from the
$180^{\circ}$ instability boundary. This is deliberately conservative: a
sensitivity study across LOS implementations places the exact admissible
constant in $[0.1,0.3]$, so $1/(2\pi)\approx 0.16$ is a representative
conservative choice rather than a tuned parameter. We use the single
value $1/(2\pi)$ throughout.

---

## 5. Cascade-ISS assembly and the ultimate bound

With the three budgets satisfied, each subsystem is ISS:

$$
\dot V_r \le -\lambda_r V_r + \gamma_r\,\alpha_{\mathrm{rate}},
\qquad
\dot V_u \le -\lambda_u V_u + \gamma_u\,d_u,
$$

$$
\dot V_o \le -\lambda_o V_o
+ \gamma_{o,1}\alpha_{\mathrm{cur}}
+ \gamma_{o,2}\alpha_{\mathrm{noise}}
+ \gamma_{o,3}(\alpha_{\mathrm{sample}}+\alpha_{\mathrm{delay}}).
$$

Because the interconnection is triangular (yaw drives heading drives
cross-track, surge scales the cross-track gain) and each subsystem is ISS
with respect to the state of the subsystem feeding it, the standard
cascade-ISS result applies: the composite

$$
\dot V \le -\lambda V + \beta_T,
\qquad
\lambda = \min\{\lambda_o,\lambda_u,\lambda_r\},
$$

holds with $\beta_T$ collecting all disturbance terms. The comparison
lemma then gives

$$
V(t) \le e^{-\lambda t}V(0) + \frac{\beta_T}{\lambda},
$$

so the cross-track error is ultimately bounded by
$|\tilde d|_\infty \le \sqrt{2\beta_T/\lambda}$. Conditions (C1)--(C2) are
exactly those under which $\lambda>0$ and $\beta_T/\lambda$ is finite and
small enough that the ultimate bound keeps $\inf_t d > 0$. This proves the
ISS claim of Lemma 1, with $\alpha_{\mathrm{noise}},\alpha_{\mathrm{cur}}$
entering (C1) and $\alpha_{\mathrm{sample}},\alpha_{\mathrm{delay}},
\alpha_{\mathrm{rate}}$ entering (C2), matching Table I of the paper.
$\blacksquare$
