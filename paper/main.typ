#import "@preview/charged-ieee:0.1.4": ieee
#import "fig_law.typ": law-diagram
#import "fig_circuit.typ": circuit

// Numbers that change when the remaining training seeds of the scheduled circuits are complete (red until then).
#let tbd(body) = text(fill: rgb("#c0392b"), body)

#set page(numbering: "1")

#show: ieee.with(
  title: [A Data-Driven, Gain-Scheduled Equivalent Circuit for the $B$–$H$ Behavior of Power Ferrites],
  abstract: [
    An equivalent circuit for the dynamic $B$–$H$ behavior of power ferrites is identified from measured transients.
    It consists of classical elements of hysteresis modeling: a nonlinear inductor for the reversible curve, six viscous play branches (inductors in series with diode-bridge sliders) that form a Prandtl–Ishlinskii model, two $R$–$L$ branches for the magnetic after-effect, a mean-field feedback of $H$ as in the Jiles–Atherton model, and a measurement path.
    All elements are positive apart from the feedback sources, and the circuit can run in SPICE.
    Discretized at the sampling step, the circuit is a difference equation with 32 parameters per material, fitted by backpropagation through time.
    On the 15 materials of the MagNet Challenge 2, the mean sequence relative error is 7.33 ± 0.03 % against 7.77 % for the challenge-winning recurrent network with 336 parameters, and the circuit is more accurate on 12 of the 15 materials.
    The accuracy can be scaled up by gain scheduling, where a small network per material sets the element values at every step from the excitation, while the hysteresis memory stays in the circuit, such that the error drops to 5.27 % with 656 parameters per material.
    Thus, an interpretable model is demonstrated that achieves state-of-the-art accuracy.
  ],
  authors: (
    (
      name: "Wilhelm Kirchgässner",
      organization: "Beckhoff Automation",
      email: "wk.research@web.de",
    ),
  ),
  index-terms: ("Hysteresis", "power ferrites", "play operator", "Prandtl–Ishlinskii model", "equivalent circuit", "gain scheduling", "magnetic relaxation", "system identification", "MagNet Challenge", "interpretable machine learning"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

// All figures and tables float (top or bottom of a column) so that the text flows without gaps.
#set figure(placement: auto)

// ------------------------------------------------------------------------------------------------
= Introduction

#figure(
  image("fig/overview.pdf", width: 100%),
  placement: top,
  caption: [The problem. (a) MagNet Challenge 2 task: in a window of 1000 samples (62.5 µs), $B$, $H$ and $theta.alt$
  are known during a warm-up, and $H$ must then be predicted from $B$ and $theta.alt$ alone. The excitation is a
  switching ripple on a slowly wandering flux density, so the material traverses nested minor loops of
  all sizes. (b) Deep saturation (3C90, 50 kHz). (c) Minor loops with fast ripple on a drifting bias
  (FEC014, 800 kHz). (d) The same flux range at 25, 50 and 70 °C (3C94, 200 kHz) and (e) at 50, 200 and
  800 kHz (3C94, 25 °C): temperature and rate impact both the slope and the width of the loops.],
) <fig:overview>

The design of power electronics requires models of magnetic core materials that
predict the field strength $H(t)$, and with it the core loss, for arbitrary excitation waveforms,
frequencies and temperatures. Classical approaches are either empirical loss formulas of the
Steinmetz family @venkatachalam2002 @muehlethaler2012 or hysteresis models such as the Preisach
@preisach1935 @mayergoyz1991, Jiles–Atherton @jiles1986, Coleman–Hodgdon @coleman1986 or Bouc–Wen
@wen1976 models, which are hard to parameterize for the rate- and temperature-dependent behavior of
MnZn and NiZn ferrites at hundreds of kilohertz.

// the magnet challenge 2 has its own paper coming soon to IEEE XPlore, update mc2repo citation with another arxiv update
The MagNet project @li2023how @serrano2023why @li2023magnetai and the MagNet Challenges
@chen2025magnet @mc2repo turned the problem into a data-driven benchmark. The first challenge (MC1)
asked for the steady-state core loss under periodic excitation where the winning entries were neural
networks with a few thousand parameters per material @kirchgaessner2025hardcore. The second challenge
(MC2), built on the transient MagNetX database @kwon2025magnetx @wang2025magnetx, asks for the
sequence-to-sequence prediction of $H$ from magnetic flux density $B$ and temperature $theta.alt$, given a short warm-up segment in which $H$ is known.
The winner of the performance track, RHINO-MAG @vater2026rhino, is a gated
recurrent unit (GRU) @cho2014 with eight hidden states and 336 parameters per material; other
strong entries use LSTMs @yang2026maglearn2 or neural operators @guo2026fno with many more parameters.

These models are accurate but opaque: the eight GRU states have little physical meaning, the model cannot
be inspected for consistency with magnetics, and it is unclear which parts of the material behavior
are learned and which are artifacts of the measurement setup. This paper asks whether the information the GRU
extracts from the data can be expressed as a _governing equation_ — rather than as a neural ODE @chen2018node — with few, interpretable parameters, built from the established vocabulary of hysteresis modeling @krasnoselskii1989
@brokate1996 @visintin1994 @bertotti1998, but trained on the data like a neural network.

No single element of the resulting model is new (@sec:related). What is new is the evidence that this minimal
combination, fitted end-to-end, suffices. The contributions are:
+ *Classical elements are accurate enough.* A viscous, moving Prandtl–Ishlinskii model with 32 interpretable
  parameters per material is more accurate than the challenge-winning recurrent network, selected from a Pareto front, on 12 of the 15 MC2 materials with a tenth of its parameters, and an ablation shows that every element contributes (@sec:results).
+ *An equivalent circuit that can be read.* The model is a circuit of positive elements apart from the feedback
  sources, verified in SPICE; the fit revealed a lag in the $H$ measurement, and the parameters of the material part are valid at any time step (@sec:law).
+ *Gain scheduling.* Small networks that set the element values at every step, while the memory stays in the circuit,
  close most of the gap to the best contestants: with 64 hidden units, second of 33 by both challenge metrics (@sec:gs,
  @sec:gsres).
+ *Extrapolation and data efficiency.* Outside its training range in frequency and temperature, the circuit degrades
  far less than the recurrent network (@sec:ood).

// ------------------------------------------------------------------------------------------------
= Related Work <sec:related>

*Hysteresis operators.* Preisach models @preisach1935 @mayergoyz1991 superpose elementary relay hysterons
with a density over their switching thresholds. Prandtl–Ishlinskii (PI) models superpose play (backlash) or
stop operators instead; they form the subclass of Preisach models whose density depends only on the width of
the elementary loop, and their mathematical theory is well developed @krasnoselskii1989 @brokate1996
@visintin1994. Play-type hysterons with a friction-like pinning interpretation have also been used for vector
hysteresis of soft magnetic materials @bergqvist1997. In mechanics the same structure is the Iwan model, a
parallel bank of Jenkins elements, i.e., springs in series with Coulomb friction sliders @iwan1966; this is the
analogy behind the equivalent circuit of @sec:circuit.

*Rate dependence and after-effect.* Bertotti's dynamic Preisach model lets each hysteron switch at a rate
proportional to the excess of the field over its threshold @bertotti1992; the viscous play used here is the
corresponding operator restricted to the PI class, with one common rate. Rate-dependent PI models with
rate-dependent thresholds and densities have been developed for piezoelectric and other smart actuators
@aljanaideh2008, and Kuhnen's modified PI model adds creep operators, first-order lags, to the plays
@kuhnen2003. The two relaxations of the present model play the same role; physically they represent the magnetic
after-effect or magnetic viscosity @street1949, the subject of Preisach's original paper @preisach1935. Loss
separation into hysteresis, classical eddy-current and excess contributions @bertotti1988 motivates the dynamic
terms.

*Mean-field feedback and circuit models.* The moving Preisach model shifts the Preisach density by a term
proportional to the magnetization @dellatorre1966, and the Jiles–Atherton model uses the effective field
$H + alpha_"JA" cal(M)$ @jiles1986; the present model, driven by $B$, feeds $H$ back into the operator inputs, the
corresponding construction for a flux-driven model. Coleman–Hodgdon @coleman1986 and Bouc–Wen @wen1976 models
describe rate-independent loops with few parameters but no minor-loop memory of Preisach type. In circuit simulation,
hysteresis models enter mostly as behavioral elements @wilson2002 @salomon2020; the present model maps onto standard
elements one to one.

*Gain scheduling and hypernetworks.* Gain scheduling makes the parameters of a linear controller or plant model
functions of a measured scheduling variable @rugh2000, and hypernetworks generate the weights of one network with
another @ha2017hypernetworks. The thermal neural network @kirchgaessner2023tnn combines both with a physical structure:
Multilayer perceptrons (MLPs) set the conductances and power loss terms of a lumped-parameter thermal network from the operating point.
The scheduled circuit in this work applies this idea to the magnetic circuit; unlike in the thermal case, the memory of the model, the hysteresis, stays in the states of the circuit. Rate-dependent PI densities @aljanaideh2008 are an analytic form of
such a schedule.

*Data-driven models.* The MagNet challenges produced accurate neural models, from convolutional networks for
steady-state loss @kirchgaessner2025hardcore to recurrent networks for transient sequences @vater2026rhino
@yang2026maglearn2. Equation discovery @brunton2016 and gray-box identification @schoukens2019 seek compact models
instead; the approach here is gray-box, with the structure taken from the hysteresis vocabulary above and all
parameters fitted jointly by gradient descent through the recurrence.

// ------------------------------------------------------------------------------------------------
= Problem Statement and Data <sec:data>

The MC2 data @mc2repo consist of measurements on toroidal (and, for one material, E-shaped)
ferrite cores of ten materials (Ferroxcube 3C90, 3C94, 3E6, 3F4; Fair-Rite 77, 78; TDK N27, N30, N49,
N87), excited with PWM-type and other transient waveforms at up to seven fundamental frequencies
between 50 and 800 kHz and at 25, 50 and 70 °C. Each record holds $B$ and $H$ sampled at 16 MHz
(62.5 ns per sample) over 100 periods. Five further materials (3C92, 3C95, FEC007, FEC014, T37) were
used for the final evaluation of the challenge.

The task is defined on windows of 1000 samples, see @fig:overview. In the first $k$ samples both $B$ and $H$ are known
(warm-up); for the remaining $1000 - k$ samples only $B$ and $theta.alt$ are given and $H$ must be predicted.
Three scenarios $k in {100, 500, 900}$ ("10/90", "50/50", "90/10") are evaluated. The main metric is
the sequence relative error
$ "SRE" = sqrt(sum_(n>=k) (hat(H)_n - H_n)^2) / sqrt(sum_(n>=k) H_n^2), $
averaged over windows and scenarios. The energy error is measured by the normalized energy relative error
$ "NERE" = abs(sum_(n>=k) (hat(H)_n - H_n) Delta B_n) / sum_n abs(H_n Delta B_n), $
the error of the energy $integral H dif B$ that enters the core over the predicted part, relative to the energy
exchanged over the whole window. The challenge ranked its contestants by the 95th percentile of both errors, for
which it normalizes the energy error by the core loss of the recording instead; @fig:ranking uses that definition.
Since the official MC2 test labels of the training materials are not
public, every recording is split 70/15/15 % into training, validation and test sequences,
stratified by temperature, and 300 fixed test windows are drawn per frequency file. The published RHINO-MAG
GRU8 models @vater2026rhino are evaluated on exactly the same windows.

// ------------------------------------------------------------------------------------------------
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

// ------------------------------------------------------------------------------------------------
= State Estimation During Warm-Up <sec:observer>

In every test window the initial state — the magnetization history encoded in the play operators — is unknown. It is estimated from the warm-up by an estimator that is kept separate from the model; its parameters are not counted, since they describe the initial state and not the material; in circuit simulation the model starts from a known (e.g., demagnetized) state. All quantities below use the de-lagged $H$.

*Prior.* A play state lies within its dead zone around the drive, $|u - p_i| <= r_i$, but where depends on the history
before the window. The prior places each play within its band as a smooth function of the level $u_0$ and the
direction of the excitation at the first sample,
$ p_i = u_0 - r_i tanh(xi_(2,i) u_0 \/ r_i + xi_i op("sign")(dot(u)_0)), $
with two learned coefficients $xi_i$ and $xi_(2,i)$ per play and material. A large $xi_i$ puts the play at the trailing edge of its band,
as on a branch along which it is dragged by the drive; $xi_(2,i) > 0$ puts it below the drive at a high level, as at
the top of a loop reached from below. The fit uses the direction mainly for the narrowest play. The relaxations start
at $u_0$.

*Mismatch distribution and observer.* The prior leaves a mismatch $e_0 = H_0 - hat(H)_0$. Absorbing it in the offset
$H_"off"$ alone is correct only until the first reversal of $B$, after which the plays move and the wrong offset
becomes visible as a jump; it is therefore distributed over all states with learned gains $K_0$,
$p_i arrow.l p_i + K_(0,i) e_0$, $H_"off" = H_"pr" + K_(0,"off") e_0$. During the warm-up, the innovation
$e_n = H_n - hat(H)_n$ corrects every state with learned constant gains, as in a Luenberger observer @luenberger1964.

*Amplitude estimate.* The amplitude of $H$ varies from window to window beyond what $B$ and $theta.alt$ explain: it is
set by the excitation history before the window. Two factors $q$ and $q_h$ for the reversible curve without the knee
and for the hysteretic part are fitted to the detrended mismatch of an open-loop pass over the warm-up by a Bayesian
ridge regression whose prior is worth $nu_0 = e^(lambda_0) (k \/ 300)^(lambda_1)$ samples of the window's mean square
field, clamped to $|q| <= 0.5$, and the main pass scales both parts accordingly. Long warm-ups thus determine the
amplitudes from the data, short ones fall back to the model's own. During training, a penalty $0.5 (q^2 + q_h^2)\/2$
makes the model carry the mean amplitude of a material.

Since the whole warm-up is given at once, the estimator need not be causal; but for the state at the end of the
warm-up, which the prediction needs, the causal observer already uses all the warm-up data, and choosing the initial
state among candidate reversal histories by the open-loop error was worse at every warm-up length.

In total, the estimator learns 33 parameters per material, which are not part of the 32 of the model: the 12
coefficients $xi_i, xi_(2,i)$ of the play prior, the offset prior $H_"pr"$, the nine gains $K_0$ that distribute the initial
mismatch, the nine observer gains and the two coefficients $lambda_0, lambda_1$ of the amplitude prior.

// ------------------------------------------------------------------------------------------------
= Implementation and Training <sec:impl>

The model is a nonlinear recursion over up to $10^3$ samples per window and cannot be parallelized
over time. A straightforward PyTorch @paszke2019 loop is limited by kernel launch overhead
(about 0.4 s per optimizer step regardless of batch size). The forward pass and its
hand-derived adjoint were therefore written as a single fused Triton @tillet2019 kernel in which each GPU thread
processes one window and keeps all $M+J$ states in registers; the backward pass stores the states at
every step and recomputes the step-local Jacobians. On an AMD RX 9070 XT this increases the
throughput by a factor of about 400, which makes gradient-based fitting to $10^9$ samples practical.

Parameters are fitted with Adam @kingma2015 and a one-cycle learning-rate schedule @smith2019
(peak $2 dot 10^(-2)$, 20 000 steps). Each step draws 768 random windows of 1000 samples per
material, weighted by the number of sequences per frequency file, and a random warm-up length
$k in {100, 500, 900}$ as in the evaluation. The loss is the mean of SRE + NERE of the predicted part, summed over
materials.
Initial values of $a_1$ and $w_i$ are scaled by the least-squares slope of $H$ over $B$ of each
material. Several model variants are trained simultaneously in one kernel launch. One training run of all 15
materials takes about one hour.

*Scheduled circuit.* The kernel takes the operator coefficients per step, and its adjoint returns their gradients
per step, through which the network is trained. Training starts from the fitted classical circuit, with the same
warm-up estimator and loss, 512 windows per material and step, 20 000 steps, and peak learning rates of $10^(-3)$ for
the network and $2 dot 10^(-3)$ for the 32 circuit parameters; one run of all 15 materials takes about one hour with
16 hidden units and two hours with 64.

// ------------------------------------------------------------------------------------------------
= Results <sec:results>

#include "tables/main_results.typ"

#figure(
  image("fig/examples.pdf", width: 100%),
  placement: top, scope: "parent",
  caption: [Representative test windows (median error of the model in the 50/50 scenario). Shaded: warm-up with
  known $H$; afterwards the model predicts $H$ from $B$ and $theta.alt$ alone. FEC007 belongs to the official MC2 test set.],
) <fig:examples>

== Accuracy <sec:accuracy>

#figure(
  image("fig/ranking.pdf", width: 100%),
  placement: top, scope: "parent",
  caption: [The classical and the gain-scheduled circuits among the contestants of the MagNet Challenge 2: parameters per material over the average 95th percentile of the SRE (left) and of the energy error (right) on the official test set of the five final-evaluation materials; contestants from the official results, the better half by each metric shown (energy error: up to 10 %). Circuits: mean and range over the training seeds (ten each); scheduled circuits with 16 or 64 hidden units.],
) <fig:ranking>

@tab:main compares the classical circuit (32 parameters, passive, with measurement lag) with the published GRU8 models of
RHINO-MAG on identical test windows, with one configuration for all 15 materials; @fig:examples shows representative
predictions. Over ten training seeds the mean SRE is 7.33 ± 0.03 % against 7.77 % for the GRU (7.31 % for the run in
@tab:main), and the model is more accurate on 12 of the 15 materials. Its advantage is largest in the 50/50 and 90/10
scenarios (6.85 % and 6.11 % against 8.14 % and 6.59 %), where the warm-up determines the amplitude estimate well;
with a 100-sample warm-up it is behind (9.04 % vs 8.59 %). The 95th percentile of the SRE is lower than the GRU's on
the same 12 materials (18.78 % vs 21.06 %). The energy error is the weaker side: the mean NERE is 0.82 % against
0.73 %, better on 8 of the 15 materials, with most of the difference on 3C92 (95th percentile 10.7 % vs 3.8 %).

With 32 parameters per material, the classical circuit is smaller than every submitted model, the smallest of which had 83 (@fig:ranking), and its average 95th-percentile SRE on the official test set is on par with that of the challenge winner RHINO-MAG (19.4 % vs 20.4 %).
Gain-scheduled with 16 hidden units (656 parameters), it would rank third by both the SRE (13.0 %) and the energy
error (1.93 %, level with Fuzhou at 1.91 %); with 64 hidden units (5552 parameters), second by both (12.1 % and 1.62 %;
ten seeds each). Only Hefei, with about 6600 parameters per material, is more accurate by both metrics. Its lead over
the MLP64-scheduled circuit in the SRE (9.2 % vs 12.1 %) is spread over all five materials, 2.2 to 4.0 percentage
points each, most on 3C95 and FEC014; in the energy error (1.57 % vs 1.62 %), the scheduled circuit is ahead on
FEC007 (2.2 % vs 3.7 %) and behind on the other four by 0.2 to 0.8 points.

The GRU remains better on three materials, for different reasons. On N87 the model is behind only with the short
warm-up (9.6 % vs 7.8 % at $k = 100$) and at 50 and 80 kHz, where the excitation reaches deep saturation and the single
exponential knee is a rough description. On 3C92 it is behind at every warm-up length; with only 15 training sequences
per frequency, the saturation knee is poorly determined at 70 °C. On FEC014 it is behind even at $k = 900$ (5.2 % vs
4.1 %), a deficit of the model structure rather than of state estimation; the lag pole of FEC014 sits at its bound.

== Gain Scheduling <sec:gsres>

#include "tables/schedule.typ"

@tab:schedule compares the scheduled circuits with the classical one. With 16 hidden units, the mean test SRE is
5.27 ± 0.04 % (ten seeds), 28 % below the classical circuit and 32 % below the GRU, and the mean NERE drops from
0.82 % to 0.45 %; with 64 hidden units, the SRE is 4.81 ± 0.02 % and the NERE 0.39 % (ten seeds). On the official
test set, the average 95th-percentile SRE falls from 19.4 % to 13.0 ± 0.1 % and 12.1 ± 0.1 %, and the energy error
from 5.1 % to 1.9 ± 0.1 % and 1.62 ± 0.02 %.


The schedule acts mainly at the reversals of $B$: on 3C94 and N87, for example, the play rate
and the widths of the narrow plays change abruptly at every reversal and relax along the following branch. On 3C94
the play weights also rise with $|B|$ by up to 30 %, i.e., the hysteresis density depends on the amplitude. Jointly
trained, the operating point moves away from the classical fit: the mean play rate of N87 is about 0.13 against 0.49
in the classical circuit. The 32 parameters of the scheduled circuit therefore describe it only together with the
network.

The energy exchanged by the time-varying elements was evaluated on 200 test windows per material in the 50/50 scenario
(open-loop part). Per step, the energy that the schedule adds to the inductors at fixed state is compared with the
energy dissipated in the sliders and relaxation resistors over the same window (the feedback sources are not counted;
they are active in the classical circuit as well). Within a window, the schedule adds and removes energy of the same
order as the dissipation, but on net it adds a median of 4 % of the dissipated energy (95th percentile 47 %), and only
in 1.4 % of the windows more than is dissipated, most often on T37 (11 %) and 3E6 (9 %). The scheduled circuit is thus
not passive in the strict sense, but it is dominated by dissipation in almost every window.

== Ablation <sec:ablation>

@tab:ablation removes one classical element at a time from the classical circuit, with the same data and seed. Every
element contributes. By far the most important is the measurement correction @eq:u: without it the error rises by
5.3 percentage points and the GRU is ahead on all 15 materials; it is a property of the acquisition, but without it
the model cannot follow the recorded $B$. Among the material elements, the mean-field feedback (+0.8, worse on 10
materials by more than 0.3 points) and the saturation knee (+0.6, up to +3.4 on N49) matter most, followed by the
relaxations (+0.4) and the cubic term (+0.3). Rate-independent plays are as good on most materials but lose strongly
on 3C90 and 77, where their fit fails. The lag of $H$ matters only for the lagged materials (3C92 +1.5, 3C95 +0.5).

#include "tables/ablation.typ"

Conversely, the model can be made smaller at a moderate cost (supplementary material): with a
temperature-independent measurement correction and a common feedback gain for all operators, 27 parameters give
7.79 ± 0.08 % (ten seeds), and with three more coefficients independent of temperature, 24 parameters reach 7.81 %,
about the accuracy of the GRU with a fourteenth of its parameters. Most of the loss falls on the materials with a
measurement lag.

== Extrapolation and Data Efficiency <sec:ood>

A model built from classical elements should need less data and degrade more gracefully outside its training range
than a recurrent network. The classical circuit and a GRU8 replica (the RHINO-MAG architecture, 336 parameters, trained
with the same pipeline) were fitted to four materials (3C94, N87, 3F4, 77) under five restrictions of the training
data and evaluated on the test windows outside the restriction, next to the full-data model (@tab:ood; the replica
is weaker than the published network and serves only this comparison). Both need little data: trained on 2 % of the
sequences, the model is as accurate as with all of them (6.5 % vs 6.4 %), and so is the GRU, at a higher level. Outside
the training range they differ. Trained up to 320 kHz, the GRU fails at 500 and 800 kHz (54.5 %, up to 106 % on 77),
while the model loses 4.9 points (10.2 % vs 5.3 %), most of it on N87. Across temperature, the GRU loses about twice
as much: interpolating to 50 °C costs the model 0.8 points against 6.4, extrapolating to 70 °C 3.7 against 8.5, the
latter mostly on 77.

#include "tables/ood.typ"

// ------------------------------------------------------------------------------------------------
= Discussion

*Why so few parameters suffice.* The GRU has to learn, from data alone, a representation of
hysteresis memory. The play operators provide this memory structurally: their states store the
reversal points of the input in exactly the way the wiping-out and congruency properties of
Preisach-type models require @mayergoyz1991. The data then only have to determine a one-dimensional
weight density over the operator widths, a small number of time constants and the reversible curve.

*Why gain scheduling works.* The classical circuit underfits: the constraints that make it readable cap its
accuracy. Gain scheduling puts more parameters into the element values while the structure, the memory and the
constraints of the circuit remain; what is learned is how the elements depend on the excitation, a dependence that
classical hysteresis models introduce analytically, e.g., through rate-dependent densities @aljanaideh2008. Because the
network sees only the excitation, the scheduled circuit remains a circuit driven by $B$ and $theta.alt$, and the
schedule can be inspected: it acts at the reversals of $B$ and makes the hysteresis density depend
on the amplitude. The price is that the 32 parameters lose their stand-alone meaning, since the circuit
is described by them and the network together, and that the time-varying inductors make the circuit active in a small
fraction of windows. Where readable parameters are needed, the classical circuit remains the model; where accuracy
matters most, the scheduled circuit gives it at a small multiple of the parameters and without a recurrent network.

*Limitations.* The model stays behind the GRU on N87, 3C92 and FEC014 (@sec:accuracy). The warm-up estimator is learned per material. With only three training temperatures, the temperature dependence is restricted to linear terms, and DC-biased minor loops have not yet been tested.

= Conclusion

For the MagNet Challenge 2 data, what a challenge-winning recurrent network has learned fits into a compact
equivalent circuit of classical elements: a nonlinear inductor, a Prandtl–Ishlinskii bank of viscous play branches,
two relaxation branches and a mean-field feedback, between a correction of the measured flux density and a lag of the
measured field strength. With 32 parameters per material it is more accurate than the 336-parameter GRU (7.33 % vs
7.77 %) and better on 12 of 15 materials. It can be read: its parameters separate into the acquisition and the
material, it revealed a lag in the measurement of the field strength, and its material part is valid at any time step
and runs in circuit simulation. Gain-scheduled by small networks that set its element values from the excitation, it
reaches 5.27 % with 656 parameters per material and 4.81 % with 5552, which would rank second of 33 in the challenge
by both metrics.

The code (PyTorch with the fused Triton kernel), the fitted parameters of all materials with a NumPy implementation
and SPICE subcircuits, the supplementary material, and the scripts that reproduce every table are available at
#link("https://github.com/wkirgsn/ferrite-bh-circuit"). Open directions are an amplitude estimator for short warm-ups, a
model-based observer without learned gains, a saturation description that extrapolates safely in temperature,
schedules shared across materials or constrained to remain passive, and measurements of the same material with
different acquisition chains.

#heading(numbering: none)[Acknowledgment]

Claude Opus 5.5 (Anthropic) assisted in every facet of this work: the literature research, the analysis of the
data, the design, implementation and training of the models and of the GPU kernel, the experiments and their
evaluation, the equivalent-circuit verification, the figures, and the writing of this manuscript.
