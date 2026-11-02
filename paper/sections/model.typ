#import "../fig_law.typ": law-diagram
#import "../fig_circuit.typ": circuit

= The Model <sec:law>

The model is presented as an equivalent circuit of the wound core in continuous time (@sec:circuit). Its discretization
at the sampling step is the difference equation that is fitted to the data (@sec:dt); its elements are interpreted in
@sec:interp and gain-scheduled in @sec:gs. The model was found in its discrete form, by fitting candidate structures to
the data; the circuit is the follow-up interpretation, not the design starting point.

== Equivalent Circuit <sec:circuit>

#figure(
  circuit,
  placement: top, scope: "parent",
  caption: [Equivalent circuit of the model. The reversible curve is a flux-controlled
  nonlinear inductor. Each viscous play is an inductor in series with a viscous Coulomb slider: a diode bridge that
  carries the DC current $I_(c,i)$ (zero voltage while $|I_i| < I_(c,i)$), in parallel with $R_i$. Each relaxation is an
  $R$–$L$ branch. Every operator is driven through a current-controlled voltage source $L_(gamma,i) thin d I \/ d t$
  (the mean-field feedback). Its linear part $w_i v_i$ is the parallel inductor $-L_i$; without feedback these merge with
  $cal(L)_s$. The measurement path reproduces the low-pass of the acquisition: the measured flux follows from $V_B$, the
  voltage across $C_m$; the current sense records $H$ through the lag @eq:Hmc as $V_H$. Neither loads the core. Dashed: the elements whose
  values the scheduling MLP sets at every step in the gain-scheduled circuit (@sec:gs).],
) <fig:circuit>

Let $u$ be the corrected flux density and $H$ the field strength of the material. With $M = 6$ viscous plays and $J = 2$
relaxations, the model reads
$ u &= B + tau_1 dot(B) + tau_2 dot.double(B), $ <eq:uc>
$ v_i &= u + gamma_i H, $ <eq:vc>
$ dot(p)_i &= kappa_alpha thin delta_(r_i) (v_i - p_i), quad i <= M, $ <eq:pc>
$ dot(z)_j &= kappa_j (v_(M+j) - z_j), quad j <= J, $ <eq:zc>
$ H &= h_s (u) + sum_(i=1)^M w_i p_i + sum_(j=1)^J c_j z_j + H_"off", $ <eq:Hc>
$ tau_h dot(H)^"m" &= H - H^"m", $ <eq:Hmc>
with the dead-zone function $delta_r (x) = op("sign")(x) max(|x| - r, 0)$ and the reversible curve
$ h_s (u) = a_1 u + a_3 u^3 + H_"kn" [e^((u - B_s)\/b) - e^((-u - B_s)\/b)], $ <eq:hs>
$H_"kn" = 10$ A/m. A play state $p_i$ follows its drive $v_i$ only once the drive leaves the band of width $r_i$ around
it, and then with the rate $kappa_alpha$; a relaxation state $z_j$ follows its drive as a first-order lag. $H^"m"$ is the
field strength as the instrument records it.

For a core with $N$ turns, cross-section $A_e$ and path length $l_e$ the flux linkage is $psi = N A_e u$ and the winding
current $I = l_e H \/ N$. Each additive term of $H$ is then a current branch in parallel across the winding
(@fig:circuit): the reversible curve is the nonlinear inductor $I = (l_e \/ N) thin h_s (psi \/ (N A_e))$, a term
$w_i p_i$ is the play inductor $L_i = N^2 A_e \/ (l_e |w_i|)$ with the slider current $I_(c,i) = l_e |w_i| r_i \/ N$
and the slider resistor $R_i = kappa_alpha L_i$, a relaxation is the branch $L_j = N^2 A_e \/ (l_e |c_j|)$,
$R_j = kappa_j L_j$, and the feedback source is $L_(gamma,i) thin d I \/ d t$ with $L_(gamma,i) = N^2 A_e gamma_i \/ l_e$.
The measurement path has $R_m C_m = tau_1$, $L_m C_m = tau_2$ and $R_h C_h = tau_h$. All values are linear or constant
in $theta.alt$, as the model's coefficients.

The play branches are the electrical form of the Jenkins element of mechanics, a spring in series with a Coulomb
friction slider; a parallel bank of them is the Iwan model, the mechanical form of the Prandtl–Ishlinskii operator.
While its current stays below $I_(c,i)$ the slider does not move and the branch is a plain inductor. Above it, the
excess current flows through $R_i$, which sets the viscous drag of the play. This matches the picture of domain walls
that are pinned until the field exceeds a threshold and then move against eddy-current damping.

The model is trained with the constraints $w_i <= 0$ and $c_j <= 0$ (@sec:dt), so that every play and relaxation
branch consists of positive elements; an unconstrained fit already gives $w_i < 0$ for all materials, but mostly
$c_j > 0$, i.e., negative $R$–$L$ branches. The bound $beta_j <= 1\/2$ keeps the relaxation time constants above 1.4
samples, below which the discrete model no longer approximates its continuous reading. The parallel inductors $-L_i$,
$-L_j$ combine with the reversible inductor to a linear inductance that is positive for all materials at 25 and 70 °C
except 3E6 at 70 °C; apart from that case, the circuit without the feedback sources is passive. The feedback sources
are active, as is the effective field of Jiles–Atherton theory. The measurement path does not load the core.


== Discretization and Step Size <sec:dt>

#figure(
  law-diagram,
  placement: top, scope: "parent",
  caption: [Structure of the model. The play and relaxation states are driven by the corrected flux density
  plus a feedback of the field strength; $H$ is the sum of a reversible curve and weighted states, and the
  instrument records it through a first-order lag. During warm-up an observer (not shown) corrects the states
  with the measured $H$.],
) <fig:law>

The circuit is fitted at the sampling step $Delta t = 62.5$ ns of the data. Over one step, the drive of every
operator is held constant (zero-order hold) and the feedback uses the field of the previous sample, which breaks the
algebraic loop between @eq:vc and @eq:Hc. For a constant drive, @eq:pc, @eq:zc and @eq:Hmc have exact solutions over
one step, so that the rates become $alpha = 1 - e^(-kappa_alpha Delta t)$, $beta_j = 1 - e^(-kappa_j Delta t)$ and the lag
pole $eta = e^(-Delta t \/ tau_h)$; the derivatives in @eq:uc become central differences in units of samples. Let $n$ be
the sample index and $tilde(theta.alt) = (theta.alt - 50 °"C") / (25 °"C")$ the normalized temperature. Every coefficient
written as $phi(theta.alt)$ below is linear in $tilde(theta.alt)$, $phi(theta.alt) = phi_0 + phi_1 tilde(theta.alt)$. The
discrete model (@fig:law) reads

$ u_n &= B_n + tau_1(theta.alt) dot(B)_n + tau_2(theta.alt) dot.double(B)_n, $ <eq:u>
$ v_(i,n) &= u_n + gamma_i (theta.alt) H_(n-1), $ <eq:v>
$ p_(i,n) &= p_(i,n-1) + alpha_i (theta.alt) thin delta_(r_i (theta.alt)) (v_(i,n) - p_(i,n-1)), $ <eq:p>
$ H_n &= h_s (u_n; theta.alt) + sum_i w_i (theta.alt) thin p_(i,n) + H_"off", $ <eq:H>
$ H^"m"_n &= eta thin H^"m"_(n-1) + (1 - eta) H_n, $ <eq:Hm>

for $i = 1, ..., M+J$, where the relaxations are written as operators of zero width, $p_(M+j) = z_j$, $r_(M+j) = 0$,
$w_(M+j) = c_j$, with their own rates $alpha_(M+j) = beta_j$; the plays share the rate $alpha$. $H^"m"$ is the quantity
that is predicted and, during the warm-up, measured. The feedback @eq:v uses the field $H$ of the material, not the
recorded $H^"m"$; during the warm-up, $H$ is the de-lagged measurement. Only the offset $H_"off"$ is not a parameter; it
is part of the initial state.

The operators are not parameterized individually. The play widths are log-equidistant between $r_1$ and $r_M$
with a common temperature factor,
$ ln r_i (theta.alt) = ln r_1 + (i-1)/(M-1) (ln r_M - ln r_1) + rho tilde(theta.alt). $
The play weights follow a density that is linear in the standardized logarithm $chi_i$ of the width
(a linear function of the play index from $-1.34$ to $1.34$),
$w_i (theta.alt) = d_0 (theta.alt) + d_1 (theta.alt) chi_i$, and the feedback gains are $gamma_i = gamma(theta.alt) + g_p chi_i$ for the plays and
$gamma(theta.alt) + g_j$ for the relaxations.
Again, the circuit is constrained to consist of positive elements apart from the
feedback (@sec:circuit): $w_i <= 0$, $c_j <= 0$ and $beta_j <= 1\/2$. With the pole $eta$ of the measurement lag, this
gives 32 parameters per material (@tab:params).

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, right, left),
    stroke: none,
    table.hline(),
    table.header([*Term*], [*\#*], [*Parameters*]),
    table.hline(),
    [measurement], [5], [$tau_1, tau_2$ linear in $tilde(theta.alt)$; lag pole $eta$],
    [reversible], [7], [$a_1, a_3, B_s$ linear in $tilde(theta.alt)$; knee width $b$],
    [play widths], [3], [$r_1, r_6$ (log-equidistant); temperature slope $rho$],
    [play rate], [2], [$alpha = sigma(alpha_0 + alpha_1 tilde(theta.alt))$],
    [play density], [4], [$d_0, d_1$ linear in $tilde(theta.alt)$],
    [relaxations], [6], [$beta_1, beta_2$; $c_1, c_2$ linear in $tilde(theta.alt)$],
    [feedback], [5], [$gamma$ linear in $tilde(theta.alt)$; $g_p$; $g_1, g_2$],
    table.hline(),
    [*total*], [*32*], [],
    table.hline(),
  ),
  caption: [Parameters of the model per material. Five belong to the measurement chain, 27 to the material.
  The observer has its own parameters, which are not counted (@sec:observer).],
) <tab:params>

*Step size.* Since the material part of the model is the discretization of @eq:vc–@eq:Hc, it can be evaluated at any
step $Delta t$ by $alpha arrow.r 1 - (1 - alpha)^(Delta t \/ 62.5 "ns")$. Run open-loop on test windows of all 15
materials with 8 sub-steps per sample, the model changes by a median of 0.07 % (RMS, relative) against the native step,
most for FEC014 (median 0.6 %), which combines a fast play rate with a relaxation at its bound. Applied on a finer grid,
the measurement correction instead turns the corners of the PWM waveforms into ever narrower spikes (median deviation
3.8 % at 8-fold upsampling for 3C94): it is a property of the 16 MHz acquisition, so $u$ is computed on the 16 MHz grid
and then resampled, and the lag is omitted when the true field is wanted.

*SPICE.* In ngspice 42, with the slider written as the dead-zone resistor $v = R_i thin d_(I_c) (i)$ that the diode bridge implements, the circuit driven with the corrected flux of 60 test windows reproduces the continuous-time model with a median deviation of 0.01 % (maximum 0.38 %): the circuit is an exact realization of the model.


== Interpretation <sec:interp>

*Measurement correction @eq:u.* The term inverts, to second order, a low-pass
$B_"meas" (s) = B(s)\/(1 + tau_1 s + tau_2 s^2 + ...)$ between the true and the recorded flux density (or a relative
delay of the voltage and current channels plus the winding's parasitic dynamics), a property of the specimen and its
winding, not of the material.

*Measurement lag @eq:Hm.* The recorded field strength follows the field of the model through a first-order lag with pole $eta$.
It was not designed in but found in the fitted model. Without the lag and without the passivity
constraints, the model drives both relaxations of 3C92 and FEC014 to $beta_j = 1$ with $c_j > 0$, in every one of ten training seeds.
Such a relaxation follows its drive instantly, $z_j = u + gamma_j H_(n-1)$, so that the two together add $sum_j c_j gamma_j H_(n-1)$ to $H_n$: an autoregression of the output with coefficient 0.59 (3C92), 0.51 (FEC014) and 0.1 (N87), and close to zero for all other materials.
The autoregression is a first-order lag of $H$, which a passive model cannot form; with the lag as an explicit element, the passive model is as accurate as the unconstrained one.
The fitted poles are at or close to the bound $eta <= 0.6$ (about 0.1 µs) for N87, 3C92, 3C95, FEC014 and N27, and below 0.4 for the rest.
A lag in the current channel of the acquisition, e.g., of the current probe or shunt, is the natural reading. During the warm-up, the measured $H^"m"$ is de-lagged by the exact inverse of @eq:Hm, which amplifies noise by $(1+eta)\/(1-eta)$; hence the bound on $eta$.

*Reversible curve @eq:hs.* A linear term, a cubic term and a smooth exponential knee at $plus.minus B_s (theta.alt)$
describe the anhysteretic part; the knee is needed for the low-frequency recordings that reach saturation.

*Viscous plays.* For $alpha = 1$ each state is the classical play operator, and $sum w_i p_i$ is a
Prandtl–Ishlinskii model, i.e., a Preisach model with a one-dimensional density @krasnoselskii1989 @brokate1996; $-w_i$
is the discretized density, and for all materials wide plays carry more weight than narrow ones. With $alpha < 1$ the
operator follows its input with a first-order lag once the dead zone is exceeded, a discrete form of the dry-friction
pinning picture @bergqvist1997 with a viscous term, which makes the loops widen with $dot(B)$ as observed for ferrites
@bertotti1988 @salomon2020. The fitted time constants are 4–28 samples (0.25–1.8 µs); only on 3F4 are the plays
rate-independent.

*Relaxations and feedback.* With $c_j <= 0$, a step of the drive raises $H$ at once and lets it decay toward the
static value, as an eddy-current field would: magnetic after-effect, the time-domain counterpart of the relaxation term
of the i²GSE @muehlethaler2012. The slow relaxation has time constants of 2.5–35 µs for most materials, the fast one at
most ten samples on nine materials. The mean feedback gain of the plays is 0.2–12 mT per A/m; since $H$ reaches several
hundred A/m, $gamma H$ is comparable to the flux amplitude, as the effective field of the Jiles–Atherton model
@jiles1986 written in the $B$-driven direction.

*Specimen and material.* The fitted parameters of all materials and their transfer to other specimens are given in
the supplementary material#footnote[#link("https://github.com/wkirgsn/ferrite-bh-circuit")]. The raw MC1 data @chen2025magnet contain the ten training materials measured with the same
instruments on other cores, mostly with other numbers of turns. Refitting only the five measurement parameters of the
MC2 model on these recordings recovers two thirds of the gap between no refit (10.8 % SRE) and a full refit (7.1 %),
the 14 amplitude parameters together with them 84 %, and the remaining 13 dynamic parameters add 0.6 points. The
parameters thus fall into three tiers: the measurement chain differs most between setups, the amplitudes follow the
specimen, and the dynamics carry over almost unchanged.


== Gain Scheduling <sec:gs>

The element values of the circuit depend on temperature only. Its remaining error is largely structural: the
constraints that make it readable, one rate for all plays, a density linear in $ln r$, two relaxations and element values
that are constant in time, also limit how closely it can follow the data. Gain scheduling relaxes these limits but keeps
the structure: the element values become functions of the excitation, as in gain-scheduled control @rugh2000 and in
the thermal neural network @kirchgaessner2023tnn.

Per material, an MLP $f$ with two hidden layers of $h$ $tanh$ units maps the scheduling variables
$ bold(x)_n = (u_n \/ u_0, thin op("asinh")(dot(B)_n \/ b_1), thin op("asinh")(dot.double(B)_n \/ b_2), thin tilde(theta.alt)), $
with $u_0 = 0.2$ T, $b_1 = 1$ mT and $b_2 = 0.1$ mT (derivatives per sample), to 16 shifts
$ bold(z)_n = 1/2 [f(bold(x)_n) + f(bold(S) bold(x)_n)], $ <eq:z>
where $bold(S) = op("diag")(-1, -1, -1, 1)$ reverses the excitation, so that the model stays odd, $H(-B) = -H(B)$. In the circuit, every coefficient $phi$ is computed from an
unconstrained raw value $phi^"raw"$ through a fixed map $g_phi$ that enforces its constraint, e.g., a
logistic function for $alpha$ and $beta_j$, $-op("softplus")(-dot)$ for $w_i$ and $-op("softplus")$ for $c_j$. The
schedule adds the shift to the raw value,
$ phi_n = g_phi (phi^"raw" (theta.alt) + z_(phi,n)). $ <eq:gs>
The 16 shifts act on five output-side coefficients, $a_1$, $a_3$, $B_s$, the knee width $b$ and a gain $e^(z)$ of the
whole hysteretic part $sum_i w_i p_i$, and on eleven dynamic ones: the play rate $alpha$, the density coefficients
$d_0$ and $d_1$, the relaxation rates $beta_j$ and weights $c_j$, and two shifts each of the play widths and the
feedback gains, a common one and one that tilts them along the play index,
$ ln r_(i,n) &= ln r_i (theta.alt) + z_(r,n) + z'_(r,n) thin chi_i, \
  gamma_(i,n) &= gamma_i (theta.alt) + z_(gamma,n) + z'_(gamma,n) thin chi_i, $
with $chi_i = 0$ for the feedback of the relaxations. The last layer of
$f$ is initialized to zero, so that training starts at the classical circuit, the special case $bold(z) = 0$. With
$h = 16$ hidden units the network adds 624 parameters, 656 per material in total; with $h = 64$ it adds 5520, 5552 in total. The 32 parameters of the classical circuit are trained jointly with the network; they become the operating point of the scheduled circuit and need not equal the classical fit.

The scheduling variables depend on $B$ and $theta.alt$ only, so the shifts of a whole window are computed before the
recurrence, and the network has no state of its own: the memory of the model remains in the play and relaxation
states. In the circuit (dashed in @fig:circuit), the nonlinear inductor, the play and relaxation branches and the
feedback gains become time-varying elements. Because every value passes through the same map as in the classical
circuit, every element is positive at every step. A time-varying inductor is nevertheless not passive: changing its
value at a fixed flux changes its stored energy, which is quantified in @sec:gsres.

