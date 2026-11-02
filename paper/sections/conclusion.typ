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

