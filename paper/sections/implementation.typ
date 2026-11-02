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

