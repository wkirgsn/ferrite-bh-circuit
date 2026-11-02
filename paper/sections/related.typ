= Related Work <sec:related>

*Hysteresis operators.* Preisach models @preisach1935 @mayergoyz1986 superpose elementary relay hysterons
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

