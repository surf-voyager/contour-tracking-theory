# Joint Trackability

**Corresponds to: paper Section V, Theorem 1, equation(s) (10) [joint bound] and (11) [$\Phi^{\mathrm{adm}}$].**

This note composes the three per-mode results — tracking-mode ISS
(Lemma 1), loss-mode dwell (Lemma 2), and re-acquisition (Lemma 3) — into
the finite-horizon probabilistic guarantee of Theorem 1. The composition
uses an average-dwell-time (ADT) argument for sample-path stability and a
Boole/union bound for the failure probability, yielding

$$
\delta(\mathcal T) \ \le\ \frac{\mathcal T}{T_D}\,
\big(\delta_L + e^{-\lambda} + \delta_G\big).
$$

We define the three failure budgets and the admissible set
$\Phi^{\mathrm{adm}}$, and state explicitly that the result is
sufficient, not necessary.

Notation: modes $q\in\{T,L_N,L_G,R\}$ (tracking, noise-driven loss,
geometry-driven loss, re-acquisition); horizon $\mathcal T$; average
dwell time $T_D$; $\epsilon$ steady-state error tolerance; $d^{\star}$
desired standoff; $d_{\min}$ collision bound; $T^{\star} =
\max\{T^{\star}_N,T^{\star}_G\}$ the joint dwell budget.

---

## 1. The three per-mode guarantees

From the preceding notes we have, in isolation:

* **Tracking $T$** (Lemma 1, `01-tracking-conditions.md`). Under (C1)
  and (C2) the error dynamics are ISS, with a cascade Lyapunov function
  $V_T$ satisfying $\dot V_T \le -\lambda V_T + \beta_T$, so the distance
  error is ultimately bounded by $\epsilon$ and $\inf_t d > 0$.

* **Loss $L_N, L_G$** (Lemma 2, `03-loss-dwell.md`). Each loss episode is
  safe for a dwell up to $T^{\star}_N$ (noise channel) or $T^{\star}_G$
  (geometry channel); beyond that, safety can fail with the respective
  budget probability.

* **Re-acquisition $R$** (Lemma 3, `04-reacquisition.md`). The wall is
  recaptured within $T_R$ with probability $\ge 1-e^{-\lambda}$, while a
  zeroing CBF holds $d\ge d_{\min}$ throughout.

Each is a *conditional* statement about one mode. The task is to show
that switching among them preserves the guarantees, provided the
switching is not too fast.

---

## 2. Sample-path stability by average dwell time

### 2.1 Dwell-weighted Lyapunov function

Let $\tau$ be the time elapsed in the current mode and $W_s(z)$ the
active per-mode Lyapunov function ($W_s\in\{V_T,V_{L_N},V_{L_G},V_R\}$).
Define the dwell-weighted composite

$$
V(x) = \exp(\mu\tau)\,W_s(z),
$$

with $\mu>0$ chosen below. Within a mode, $\dot W_s\le -\lambda_s W_s +
\beta_s$, so $V$ behaves like the per-mode function modulated by the
clock $\tau$. At a switch $q^{-}\to q^{+}$ the Lyapunov value may jump by
a factor $\mu_{\mathrm{jmp}} := \sup_{q^-,q^+} W_{q^+}/W_{q^-} \ge 1$
(re-initialisation cost across the guard).

### 2.2 The average-dwell-time condition

Over a horizon $[0,\mathcal T]$ with $N_\sigma(\mathcal T)$ switches, an
average-dwell-time switching signal satisfies

$$
N_\sigma(\mathcal T) \ \le\ N_0 + \frac{\mathcal T}{T_D},
$$

for chatter bound $N_0$ and average dwell time $T_D$. Accumulating the
within-mode contraction and the per-switch jumps,

$$
W(\mathcal T)
\ \le\ \mu_{\mathrm{jmp}}^{\,N_\sigma(\mathcal T)}\,
e^{-\lambda \mathcal T}\,W(0)
\ =\ \exp\!\Big[\big(\tfrac{\ln\mu_{\mathrm{jmp}}}{T_D} - \lambda\big)\mathcal T\Big]\,
\mu_{\mathrm{jmp}}^{N_0}\,W(0).
$$

The exponent is negative — and the composite is uniformly stable along
sample paths — iff

$$
T_D \ >\ \frac{\ln\mu_{\mathrm{jmp}}}{\lambda}.
\tag{ADT}
$$

This is condition (iii) of Theorem 1: switching slower than (ADT)
guarantees that the per-mode contractions outrun the per-switch jumps. A
multiple-Lyapunov cross-check (each $W_s$ non-increasing at successive
entries to the same mode) gives the same conclusion geometrically.
*Conditional on* a switching signal that obeys (ADT), the closed loop is
sample-path stable: the distance error stays within $\epsilon$ and
$\inf_t d > 0$.

---

## 3. Failure probability by Boole/union bound

Sample-path stability above is conditional on the random realisations
(range noise $\eta$, re-acquisition search outcome $\omega_R$, wall
draw from $\mathbb W$) producing a well-behaved switching signal. We now
bound the probability that they do not.

### 3.1 Per-episode failure budgets

Three independent randomness sources can each cause a single mode episode
to violate its guarantee:

* **$\delta_L$** — the noise-gate false-loss probability of Lemma 2: the
  probability that the curvature error exceeds its $\delta_L$-tail
  quantile, so the noise-driven dwell $T^{\star}_N$ underestimates the
  true safe time. (This is the same $\delta_L$ that sets
  $\beta_N=1/\sqrt{2\ln(1/\delta_L)}$ in eq. (7).)

* **$e^{-\lambda}$** — the re-acquisition miss probability of Lemma 3:
  the probability that the spiral search fails to recapture the wall
  within $T_R$, with $\lambda = R_{\mathrm{FOV}}\,w_{\mathrm{FOV}}\,
  \underline u\,T_R/A_{\mathrm{unc}}$.

* **$\delta_G$** — the geometry-loss overshoot probability: the
  probability over the wall class $\mathbb W$ that a geometry-driven loss
  episode requires dwell beyond $T^{\star}_G$ before the wall re-enters
  the field of view.

### 3.2 Union bound over episodes

Each episode begins at a mode switch; over the horizon there are at most
$N_\sigma(\mathcal T)\le N_0 + \mathcal T/T_D$ switches, hence at most
$O(\mathcal T/T_D)$ episodes. By Boole's inequality, the probability that
*any* episode fails is at most the sum over episodes of the per-episode
failure probability. Since a single episode can fail through any of the
three independent sources, its per-episode failure probability is at most
$\delta_L + e^{-\lambda} + \delta_G$. Multiplying by the episode count
(dropping the $O(1)$ chatter term $N_0$ into the leading $\mathcal T/T_D$
factor),

$$
P[\text{any episode fails over }[0,\mathcal T]]
\ \le\ \frac{\mathcal T}{T_D}\,\big(\delta_L + e^{-\lambda} + \delta_G\big)
\ =:\ \delta(\mathcal T).
$$

---

## 4. Theorem 1

Combining §2 (conditional sample-path stability under (ADT)) with §3
(probability $\ge 1-\delta(\mathcal T)$ that no episode fails) gives the
finite-horizon guarantee. Suppose the configuration $\Phi$ admits
controller parameters $(\Delta,K_p,K_{\mathrm{ff}},\rho_0,a,
w_{\mathrm{FOV}})$ such that simultaneously:

**(i)** Lemma 1 holds (i.e. (C1) and (C2));
**(ii)** $\mathbb E[T_R]\le T^{\star}=\max\{T^{\star}_N,T^{\star}_G\}$;
**(iii)** the guard switching signal is an average-dwell signal with
$T_D > \ln\mu_{\mathrm{jmp}}/\lambda$.

Then for any finite horizon $\mathcal T$,

$$
\boxed{\;
\begin{aligned}
P_{\eta,\omega_R}\Big[\,\forall\,\mathcal W\in\mathbb W:\
&\sup_{t\le\mathcal T}|d-d^{\star}|\le\epsilon \\
&\wedge\ \inf_{t\le\mathcal T} d > 0
\ \wedge\ T_{\mathrm{lost}}\le T^{\star}\,\Big]
\ \ge\ 1-\delta(\mathcal T),
\end{aligned}
\;}
\tag{10}
$$

with

$$
\delta(\mathcal T) \ \le\ \frac{\mathcal T}{T_D}\,
\big(\delta_L + e^{-\lambda} + \delta_G\big).
$$

The bound degrades linearly in the horizon $\mathcal T$ (more time, more
chances to fail) and improves with longer average dwell $T_D$ (fewer
risky switches) and with smaller per-mode budgets.

---

## 5. The admissible configuration set $\Phi^{\mathrm{adm}}$

Theorem 1 induces an implicit object on configuration space: the set of
instrument-and-controller configurations for which *some* tuning meets
all three conditions with acceptable failure probability,

$$
\boxed{\;
\Phi^{\mathrm{adm}}
= \big\{\,\Phi : \text{(i)--(iii) satisfiable}\ \wedge\
\delta^{\sup}(\Phi)\le\bar\delta\,\big\}.
\;}
\tag{11}
$$

Here $\delta^{\sup}(\Phi)$ is the worst-case failure probability over the
admissible tunings and $\bar\delta$ the task tolerance. The boundary
$\partial\Phi^{\mathrm{adm}}$ is the queryable phase-diagram boundary,
sliced along the engineering axes $(\kappa_{\max},v^{\star})$,
$(f_s,\tau_d)$, $(w_{\mathrm{FOV}},\sigma_\eta)$, and
$(\kappa_{\max},\bar V_c)$. Each slice maps a region of instrument
specifications to admissible task performance.

---

## 6. Sufficiency, not necessity

Theorem 1 is a **sufficient-conditions** result. It asserts the
implication

$$
\text{(i)} \wedge \text{(ii)} \wedge \text{(iii)}
\ \Longrightarrow\ \text{(10)},
$$

and makes **no** claim that (i)--(iii) are necessary. Several steps are
deliberately conservative: the $1/(2\pi)$ normalisation in (C2)
(`01-tracking-conditions.md` §4.4), the worst-case curvature-rate
accumulation in $T^{\star}_G$ (`03-loss-dwell.md` §3.2), the worst-case
advance speed $\underline u$, and the Boole union bound in §3.2 (which
ignores favourable correlations between episodes). A configuration
outside $\Phi^{\mathrm{adm}}$ may therefore still be empirically
trackable; the theorem certifies a *guaranteed* region, not the *maximal*
one. Quantifying the gap between the certified and the empirically
trackable region is an estimation question handled in the validation and
discussion sections of the paper, not a defect of the sufficiency claim.
$\blacksquare$
