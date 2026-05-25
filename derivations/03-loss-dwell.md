# Loss-Mode Dwell

**Corresponds to: paper Section IV, Lemma 2, equation(s) (7) [=$T^{\star}_N$] and (8) [=$T^{\star}_G$].**

This note derives the two loss-mode dwell bounds. The noise-driven dwell
$T^{\star}_N$ (eq. (7)) follows from inverting the quadratic chord/arc
extrapolation error against the available distance margin
$d^{\star}-d_{\min}$ under a Gaussian tail bound. The geometry-driven
dwell $T^{\star}_G$ (eq. (8)) follows from consuming the angular
field-of-view margin under worst-case curvature-rate accumulation. We
emphasise the *geometric-rigidity* property: $T^{\star}_G$ depends only on
$\{w_{\mathrm{FOV}},\kappa'_{\max},\underline u\}$ and on no estimator
quantity.

Notation: $d^{\star}$ desired standoff, $d_{\min}$ collision lower bound,
$\underline u$ worst-case advance speed, $\sigma_{\hat\kappa}$ curvature-
estimate standard deviation, $w_{\mathrm{FOV}}$ full angular field of
view, $\beta_{\mathrm{FOV}} = w_{\mathrm{FOV}}/2$ the half-angle,
$\kappa'_{\max}$ the curvature-rate bound, $\delta_L\in(0,1)$ the
loss-failure budget.

---

## 1. Setup: coasting in a loss mode

When the wall is no longer reliably observed the vehicle leaves tracking
mode and coasts along its best geometric extrapolation. There is no fresh
measurement, so the open-loop deviation grows deterministically with the
elapsed dwell time. The dwell bound is the largest coasting time after
which we can still guarantee, at confidence $1-\delta_L$, that the safety
constraint $\inf_t d > d_{\min}$ is not violated. We treat the two loss
channels separately because the per-sample range noise is independent of
the wall geometry; their dwell bounds therefore admit separate closed
forms and combine by the maximum, $T^{\star} = \max\{T^{\star}_N,
T^{\star}_G\}$.

---

## 2. Noise-driven dwell $T^{\star}_N$

### 2.1 The quadratic extrapolation error

In the noise-driven channel $L_N$ the vehicle coasts along the
curvature-extrapolated tangent. Let $t^{-}$ be the instant of loss. The
true wall continues to curve while the extrapolation, built from the
last curvature estimate $\hat\kappa$, follows a fixed arc. The
chord-versus-arc deviation between true and extrapolated paths is, to
leading order, the second-order Taylor remainder of the displacement.
With advance speed $\underline u$, the arc length coasted in time
$t-t^{-}$ is $\underline u\,(t-t^{-})$, and the lateral deviation of a
constant-curvature extrapolation from the true path accumulates as

$$
e(t) = \tfrac12\,\big|\kappa - \hat\kappa\big|\,\big[\underline u\,(t-t^{-})\big]^{2}
\ \le\ \tfrac12\,\big|\kappa - \hat\kappa\big|\,\underline u^{2}\,(t-t^{-})^{2}.
$$

The deviation is quadratic in elapsed time, the signature of an
open-loop second-order extrapolation: position error integrates the
curvature error twice over the coasted arc length.

### 2.2 Gaussian tail bound on the curvature error

The curvature error $\kappa-\hat\kappa$ is the estimator error of the
sliding-window arc fit, which under i.i.d. Gaussian range noise is itself
Gaussian with mean zero and standard deviation $\sigma_{\hat\kappa}$ (see
`02-curvature-uncertainty.md`). We require the dwell to be safe with
confidence $1-\delta_L$, i.e. we bound $|\kappa-\hat\kappa|$ by its
$\delta_L$-tail quantile. For a zero-mean Gaussian,

$$
P\big[\,|\kappa-\hat\kappa| > z\,\sigma_{\hat\kappa}\,\big]
\ \le\ \exp\!\Big(-\tfrac12 z^{2}\Big),
$$

the standard sub-Gaussian tail bound. Setting the right-hand side equal
to $\delta_L$ gives the multiplier

$$
z = \sqrt{2\ln(1/\delta_L)},
$$

so that with probability at least $1-\delta_L$,

$$
|\kappa-\hat\kappa| \ \le\ z\,\sigma_{\hat\kappa}
= \sqrt{2\ln(1/\delta_L)}\;\sigma_{\hat\kappa}.
$$

It is convenient to absorb the multiplier into the parameter

$$
\beta_N := \frac{1}{\sqrt{2\ln(1/\delta_L)}} = \frac1z,
$$

so that $|\kappa-\hat\kappa| \le \sigma_{\hat\kappa}/\beta_N$ with the
required confidence. The worst-case (high-confidence) extrapolation error
is then

$$
e(t) \ \le\ \frac12\,\frac{\sigma_{\hat\kappa}}{\beta_N}\,\underline u^{2}\,(t-t^{-})^{2}.
$$

### 2.3 Inverting against the distance margin

The vehicle starts a loss episode at the desired standoff $d^{\star}$ and
must not breach the collision bound $d_{\min}$. The available lateral
margin is $d^{\star}-d_{\min}$. The dwell is admissible as long as the
high-confidence deviation does not consume the margin:

$$
\frac12\,\frac{\sigma_{\hat\kappa}}{\beta_N}\,\underline u^{2}\,(t-t^{-})^{2}
\ \le\ d^{\star}-d_{\min}.
$$

Solving for the largest admissible $t-t^{-}$,

$$
(t-t^{-})^{2} \ \le\
\frac{2(d^{\star}-d_{\min})\,\beta_N}{\sigma_{\hat\kappa}\,\underline u^{2}},
$$

and taking the positive root gives the noise-driven dwell bound

$$
\boxed{\;
T^{\star}_N
= \sqrt{\frac{2(d^{\star}-d_{\min})\,\beta_N}{\sigma_{\hat\kappa}\,\underline u^{2}}}\,,
\qquad
\beta_N = \frac{1}{\sqrt{2\ln(1/\delta_L)}}.
\;}
\tag{7}
$$

The structure is transparent: more margin $d^{\star}-d_{\min}$ buys
longer dwell ($\propto\sqrt{\text{margin}}$); higher curvature
uncertainty $\sigma_{\hat\kappa}$ or higher speed $\underline u$ shortens
it; and a stricter failure budget $\delta_L$ (smaller) shrinks $\beta_N$
and hence the dwell. Because $\sigma_{\hat\kappa}\propto L^{-5/2}$, the
dwell improves only as $L^{5/4}$ with window length — the noise channel
*is* coupled to the estimator, in contrast to the geometry channel below.

---

## 3. Geometry-driven dwell $T^{\star}_G$

### 3.1 The angular margin and its dynamics

In the geometry-driven channel $L_G$ the loss is triggered by the
instantaneous event $vis_t : 1\to 0$: the wall tangent leaves the
field-of-view cone. Geometric occlusion cannot be resolved by
re-observation; the relevant quantity is how long the wall stays inside
the cone given how fast the cone must rotate to keep up with the wall.
Let $\phi(t)$ be the bearing of the nearest wall contact within the cone,
and define the angular margin

$$
\Theta(t) = \beta_{\mathrm{FOV}} - |\phi(t)|,
\qquad \beta_{\mathrm{FOV}} = \tfrac12 w_{\mathrm{FOV}},
$$

i.e. how far the contact bearing is from the edge of the cone. Its rate
is the difference between how fast the vehicle yaws and how fast the
wall-tangent direction advances,

$$
\dot\Theta = \dot\psi - \dot\gamma_p,
$$

with $\gamma_p$ the path-tangent angle. In tracking the controller keeps
$\dot\psi\approx\dot\gamma_p$, so $\Theta$ is held; in loss the vehicle
coasts and the difference is driven by the *unmodelled* change in wall
geometry.

### 3.2 Worst-case curvature-rate accumulation

The path-tangent angle obeys $\dot\gamma_p = \kappa\,\dot s \approx
\kappa\,\underline u$. Over a loss episode the vehicle's yaw cannot
anticipate changes in $\kappa$, so the *rate mismatch* is governed by how
fast the wall curvature itself changes, bounded by $\kappa'_{\max}$
(units $\mathrm m^{-2}$). The angular acceleration of the margin is

$$
\ddot\Theta \ =\ -\,\ddot\gamma_p
\ =\ -\,\frac{\mathrm d}{\mathrm dt}\big(\kappa\,\underline u\big)
\ =\ -\,\kappa'\,\dot s\,\underline u
\ \ \Rightarrow\ \
|\ddot\Theta| \ \le\ \kappa'_{\max}\,\underline u^{2},
$$

using $\dot s\approx \underline u$ and $|\kappa'|\le\kappa'_{\max}$. The
worst case is a constant adverse angular acceleration
$\kappa'_{\max}\underline u^{2}$ acting from the onset of the episode,
when $\dot\Theta(0)=0$ (the controller had been holding the margin). The
margin then decays as

$$
\Theta(t) \ =\ \Theta(0) - \tfrac12\,\kappa'_{\max}\,\underline u^{2}\,t^{2}.
$$

### 3.3 Consuming the full FOV margin

Take the most demanding admissible onset, $\Theta(0)=\beta_{\mathrm{FOV}}
= w_{\mathrm{FOV}}/2$: the contact starts at the centre of the cone and
the entire angular half-width is available before the wall exits. Setting
$\Theta(t)=0$,

$$
\tfrac12\,\kappa'_{\max}\,\underline u^{2}\,t^{2}
= \frac{w_{\mathrm{FOV}}}{2}
\quad\Longrightarrow\quad
t^{2} = \frac{w_{\mathrm{FOV}}}{\kappa'_{\max}\,\underline u^{2}}.
$$

Taking the positive root gives the geometry-driven dwell

$$
\boxed{\;
T^{\star}_G
= \frac{1}{\underline u}\,\sqrt{\frac{w_{\mathrm{FOV}}}{\kappa'_{\max}}}\,.
\;}
\tag{8}
$$

(The factors of $2$ cancel: the half-width $w_{\mathrm{FOV}}/2$ matched
against the half from $\tfrac12 t^2$.)

### 3.4 Geometric rigidity / channel orthogonality

The arguments of $T^{\star}_G$ are exactly
$\{w_{\mathrm{FOV}},\kappa'_{\max},\underline u\}$. It contains **no**
$\sigma_{\hat\kappa}$, no $\sigma_\eta$, no $L$, and no $\bar s$ — none of
the estimator quantities that appear in $T^{\star}_N$. This is the
geometric-rigidity property: a geometry-driven loss is an instantaneous
*geometric* event (the tangent leaves the cone), determined entirely by
how fast the wall bends ($\kappa'_{\max}$), how wide the instrument sees
($w_{\mathrm{FOV}}$), and how fast the vehicle advances ($\underline u$).
No amount of estimation accuracy can postpone it, because the wall is
simply not in the field of view. Consequently the two loss channels are
*orthogonal*: $L_N$ lives entirely in the estimator/noise subspace, $L_G$
entirely in the geometric subspace. The estimator uncertainty
$\sigma_{\hat\kappa}$ enters only the safety layer of the re-acquisition
mode (see `04-reacquisition.md`), never the geometric dwell.

A practical consequence: $T^{\star}_G$ separates a mechanically scanning
sonar ($w_{\mathrm{FOV}}=2\pi$) from a forward-looking unit
($w_{\mathrm{FOV}}\in[60^{\circ},120^{\circ}]$) by field of view alone,
through the explicit $\sqrt{w_{\mathrm{FOV}}}$ dependence, independent of
any choice of estimator window or range-noise level.

---

## 4. Joint dwell budget

Since a loss episode is one channel or the other and each bound is the
maximum safe dwell for its channel, the admissible loss-mode dwell is the
maximum of the two,

$$
T^{\star} = \max\{T^{\star}_N,\,T^{\star}_G\}.
$$

In an estimator-rich, wide-FOV regime $T^{\star}_N$ dominates (geometry
rarely the binding constraint); in a narrow-FOV, sharply-curving regime
$T^{\star}_G$ dominates and is, by §3.4, irreducible by sensing accuracy.
$\blacksquare$
