# Re-acquisition

**Corresponds to: paper Section IV, Lemma 3, equation (12) [=eq. (9) re-acquisition bound; $P[\text{reacquire within }T_R]\ge 1-e^{-\lambda}$ with $\lambda = R_{\mathrm{FOV}}\,w_{\mathrm{FOV}}\,\underline u\,T_R/A_{\mathrm{unc}}$].**

This note derives the re-acquisition success probability as a Koopman
random-search detection law specialised to a no-gap Archimedean spiral,
and establishes forward invariance of the safe set
$\{d\ge d_{\min}\}$ via a zeroing control barrier function (CBF). We make
explicit the dimensional bookkeeping that renders the exponent $\lambda$
dimensionless.

Notation: $R_{\mathrm{FOV}}$ sensor range (m), $w_{\mathrm{FOV}}$ full
angular field of view (rad), $\underline u$ advance speed (m/s), $T_R$
re-acquisition time budget (s), $A_{\mathrm{unc}}$ target-uncertainty
area (m$^2$), $d_{\min}$ collision lower bound (m).

---

## 1. The re-acquisition policy

On entering re-acquisition mode $R$, the vehicle first visits the
maximum-likelihood predicted contact point $\hat p_R$, obtained by
tangent-and-curvature extrapolation of the last tracked state. If the
wall is not captured there, the prediction error places the true contact
$p_R$ somewhere in a region of area $A_{\mathrm{unc}}$ around $\hat p_R$
(the propagated state-estimate covariance footprint). The vehicle then
executes an exhaustive search of that region with an Archimedean spiral

$$
r(\phi) = \rho_0 + a\,\phi,
$$

with $\rho_0$ the starting radius and $a$ the radial growth per radian.

### 1.1 The no-gap pitch condition

For the spiral to sweep the uncertainty region with **no unsearched
gaps**, consecutive turns must overlap within the sensor's angular swath.
The radial advance per turn is the spiral pitch $2\pi a$; the sensor
covers an arc subtending the full angular width $w_{\mathrm{FOV}}$ at
range, so successive passes leave no gap provided

$$
2\pi a \ \le\ w_{\mathrm{FOV}}.
$$

Under this no-gap condition the search is *complete*: every point of the
region is brought within sensor view as the path length grows.

---

## 2. Koopman random-search detection law

### 2.1 The exponential law

Koopman's classical result for search over a region of area $A$ with a
sensor of effective sweep width $W$ following a path of length
$\mathcal L$, when the target location is uniform over the region and the
path does not retrace, gives the cumulative detection probability

$$
P[\text{detect}] = 1 - \exp\!\Big(-\frac{W\,\mathcal L}{A}\Big).
$$

The exponent is the *coverage ratio*: $W\mathcal L$ is the swept area
(swath width times path length), and dividing by the region area $A$
gives the expected number of times the region is covered. The
exponential form is the no-retrace (Poisson-coverage) idealisation, which
for the no-gap spiral is exact rather than approximate because the spiral
sweeps the area monotonically.

### 2.2 Specialisation to the spiral

We map Koopman's quantities onto the re-acquisition geometry:

* **Sweep width $W$.** The sonar sees the wall over angular width
  $w_{\mathrm{FOV}}$ at range $R_{\mathrm{FOV}}$, so the cross-range
  swath presented to the wall has *metric* width

  $$
  W \ \mapsto\ R_{\mathrm{FOV}}\,w_{\mathrm{FOV}}\quad[\mathrm m].
  $$

  (Range in metres times angle in radians is an arc length in metres;
  this is the key step that makes the swath dimensionally a width.)

* **Path length $\mathcal L$.** The vehicle advances at speed
  $\underline u$ for the budget $T_R$, so

  $$
  \mathcal L \ \mapsto\ \underline u\,T_R\quad[\mathrm m].
  $$

* **Region area $A$.** The uncertainty footprint,

  $$
  A \ \mapsto\ A_{\mathrm{unc}}\quad[\mathrm m^{2}].
  $$

### 2.3 The dimensionless exponent

Substituting,

$$
\lambda \ =\ \frac{W\,\mathcal L}{A}
\ =\ \frac{(R_{\mathrm{FOV}}\,w_{\mathrm{FOV}})\,(\underline u\,T_R)}{A_{\mathrm{unc}}}.
$$

The units are
$[\mathrm m]\cdot[\mathrm m]/[\mathrm m^{2}] = 1$, so $\lambda$ is
dimensionless, as required of the argument of an exponential. Hence the
re-acquisition success probability obeys

$$
\boxed{\;
P[\text{re-acquire within }T_R] \ \ge\ 1 - \exp(-\lambda),
\qquad
\lambda = \frac{R_{\mathrm{FOV}}\,w_{\mathrm{FOV}}\,\underline u\,T_R}{A_{\mathrm{unc}}}.
\;}
\tag{9}
$$

The inequality (rather than equality) reflects that the predicted-point
visit of §1 may itself capture the wall before the spiral begins, and that
the no-gap spiral never *under*-covers; (9) is therefore a guaranteed
lower bound. The miss probability is $e^{-\lambda}$, which is the term
that enters the joint failure budget $\delta(\mathcal T)$ of Theorem 1.

### 2.4 Expected re-acquisition time

The total time decomposes into the prediction visit and the spiral
sweep,

$$
\mathbb E[T_R] \ \le\ \tau_{\mathrm{predict}}
+ \tau_{\mathrm{spiral}}\big(\|p_R-\hat p_R\|;\,w_{\mathrm{FOV}}\big),
$$

where $\tau_{\mathrm{predict}}$ is the time to reach $\hat p_R$ and
$\tau_{\mathrm{spiral}}$ grows with the prediction error
$\|p_R-\hat p_R\|$ and shrinks with wider $w_{\mathrm{FOV}}$ (a wider cone
needs fewer, coarser-pitched turns to cover the same area). Wider field
of view thus both raises $\lambda$ in (9) and lowers
$\mathbb E[T_R]$.

---

## 3. Forward invariance of the safe set via a zeroing CBF

The spiral search must not drive the vehicle into the wall. We must
guarantee that the safe set

$$
\mathcal C = \{\,x : h(x)\ge 0\,\},
\qquad h(x) = d(x) - d_{\min},
$$

is forward invariant throughout re-acquisition, i.e. $d(t)\ge d_{\min}$
for all $t$.

### 3.1 Zeroing CBF condition

Let $h$ be a zeroing control barrier function with a linear class-
$\mathcal K$ extension $\alpha(h)=\gamma h$, $\gamma>0$. The CBF
forward-invariance condition requires the control input to keep

$$
\dot h(x,u_{\mathrm{ctrl}}) \ \ge\ -\,\alpha\big(h(x)\big)
\ =\ -\,\gamma\,h(x).
$$

Integrating this differential inequality (comparison lemma) from any
initial condition with $h(x(0))\ge 0$,

$$
h(x(t)) \ \ge\ h(x(0))\,e^{-\gamma t} \ \ge\ 0
\qquad\forall t\ge 0,
$$

so $h$ never crosses zero: $\mathcal C$ is forward invariant and
$d(t)\ge d_{\min}$ holds for all time. The exponential envelope shows the
barrier acts softly — it permits approach toward $d_{\min}$ but forbids
crossing, allowing the spiral to search aggressively while remaining
provably safe.

### 3.2 Compatibility with the spiral and the tightened start radius

The spiral command of §1 is filtered through a CBF-based quadratic
program that minimally modifies the nominal spiral input to satisfy
$\dot h\ge -\gamma h$. To ensure feasibility from the first instant, the
starting radius $\rho_0$ is *tightened* so the spiral begins strictly
inside $\{d\ge d_{\min}\}$,

$$
\rho_0 \ \ge\ \rho_0^{\mathrm{safe}}
\quad\Longrightarrow\quad
h(x(0)) > 0.
$$

With $h(x(0))>0$ and the CBF constraint active, §3.1 gives forward
invariance for the entire episode. Hence re-acquisition achieves the
detection bound (9) *while* maintaining $\inf_t d > d_{\min}$, so the
safety guarantee underlying Theorem 1 holds across the re-acquisition
mode. Note that the estimator uncertainty $\sigma_{\hat\kappa}$ enters
here — through the prediction error that sizes $A_{\mathrm{unc}}$ and
through the conservative tightening of $\rho_0$ — and *only* here; it does
not enter the geometric dwell $T^{\star}_G$ (cf.
`03-loss-dwell.md` §3.4). $\blacksquare$
