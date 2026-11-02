= Results <sec:results>

#include "../tables/main_results.typ"

#figure(
  image("../fig/examples.pdf", width: 100%),
  placement: top, scope: "parent",
  caption: [Representative test windows (median error of the model in the 50/50 scenario). Shaded: warm-up with
  known $H$; afterwards the model predicts $H$ from $B$ and $theta.alt$ alone. FEC007 belongs to the official MC2 test set.],
) <fig:examples>

== Accuracy <sec:accuracy>

#figure(
  image("../fig/ranking.pdf", width: 100%),
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

#include "../tables/schedule.typ"

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

#include "../tables/ablation.typ"

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
latter mostly on 77. The MLP64-scheduled circuit, trained in the same way (one seed), also needs little data (4.5 % with
2 % of the sequences vs 4.2 %), but outside the trained temperatures and frequencies its advantage over the classical
circuit vanishes: extrapolated to 70 °C, it is less accurate (11.1 % vs 9.3 %), above 320 kHz on par (10.7 % vs
10.2 %), and in both cases still far more accurate than the GRU.

#include "../tables/ood.typ"

