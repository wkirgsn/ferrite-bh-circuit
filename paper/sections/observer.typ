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

