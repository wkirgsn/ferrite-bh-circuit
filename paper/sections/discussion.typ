= Discussion

*Why so few parameters suffice.* The GRU has to learn, from data alone, a representation of
hysteresis memory. The play operators provide this memory structurally: their states store the
reversal points of the input in exactly the way the wiping-out and congruency properties of
Preisach-type models require @mayergoyz1986. The data then only have to determine a one-dimensional
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

