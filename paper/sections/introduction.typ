= Introduction

#figure(
  image("../fig/overview.pdf", width: 100%),
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
@preisach1935 @mayergoyz1986, Jiles–Atherton @jiles1986, Coleman–Hodgdon @coleman1986 or Bouc–Wen
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

