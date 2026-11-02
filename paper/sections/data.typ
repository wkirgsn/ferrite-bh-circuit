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

