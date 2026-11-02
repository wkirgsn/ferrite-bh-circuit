An equivalent circuit for the dynamic $B$–$H$ behavior of power ferrites is identified from measured transients.
It consists of classical elements of hysteresis modeling: a nonlinear inductor for the reversible curve, six viscous play branches (inductors in series with diode-bridge sliders) that form a Prandtl–Ishlinskii model, two $R$–$L$ branches for the magnetic after-effect, a mean-field feedback of $H$ as in the Jiles–Atherton model, and a measurement path.
All elements are positive apart from the feedback sources, and the circuit can run in SPICE.
Discretized at the sampling step, the circuit is a difference equation with 32 parameters per material, fitted by backpropagation through time.
On the 15 materials of the MagNet Challenge 2, the mean sequence relative error is 7.33 ± 0.03 % against 7.77 % for the challenge-winning recurrent network with 336 parameters, and the circuit is more accurate on 12 of the 15 materials.
The accuracy can be scaled up by gain scheduling, where a small network per material sets the element values at every step from the excitation, while the hysteresis memory stays in the circuit, such that the error drops to 5.27 % with 656 parameters per material.
Thus, an interpretable model is demonstrated that achieves state-of-the-art accuracy.
