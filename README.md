# A Data-Driven, Gain-Scheduled Equivalent Circuit for the B–H Behavior of Power Ferrites

Code, trained models and results of the paper (Wilhelm Kirchgässner, 2026; arXiv link follows).

An equivalent circuit of the $B$–$H$ behavior of power ferrites is designed from classical elements of hysteresis
modeling, discretized, and fitted to the measured transients of the
[MagNet Challenge 2](https://github.com/minjiechen/magnetchallenge-2) (MC2) data set by gradient descent through the
recurrence, with a fused GPU kernel in Triton for the time loop. The classical circuit has 32 parameters per
material. Its element values can be gain-scheduled with a small MLP per material, which sets them at every step from
the excitation, while the hysteresis memory stays in the states of the circuit.

![Average 95th-percentile SRE and energy error of the 33 challenge entries and the circuits on the five final-evaluation materials of MC2](paper/fig/ranking.png)

On the 15 materials of MC2:

| | parameters per material | mean test SRE (%) | official p95 SRE / energy error (%) |
|---|---:|---:|---:|
| RHINO-MAG GRU8 (challenge winner) | 336 | 7.77 | 20.4 / 3.4 |
| classical circuit (10 training seeds) | 32 | 7.33 ± 0.03 | 19.4 / 5.1 |
| MLP16-scheduled circuit (5 seeds) | 656 | 5.25 ± 0.04 | 13.0 / 1.9 |
| MLP64-scheduled circuit (5 seeds) | 5552 | 4.80 ± 0.03 | 12.1 / 1.6 |

The MLP64-scheduled circuit would rank second of 33 in the challenge by both metrics. All circuits additionally use
the 33 parameters per material of the warm-up state estimator.

## Model

The classical circuit, a 32-parameter model built only from classical elements of hysteresis modeling, predicts the
field strength $H$ from the flux density $B$ of power ferrites under arbitrary excitation:

- a second-order correction of the measured flux density and a first-order lag of the measured field strength
  (the acquisition chain),
- a reversible saturation curve,
- six rate-dependent (viscous) play operators forming a Prandtl–Ishlinskii model with a weight density linear in
  the logarithm of the play width,
- two linear relaxation states (magnetic after-effect),
- a mean-field feedback of $H$ into every operator, as in the moving Preisach and Jiles–Atherton models.

All coefficients are at most linear in temperature. The scheduled circuits keep this structure: per material, an
MLP with 16 or 64 hidden units maps the excitation at each step to the reversible curve, the hysteresis gain and the
operator dynamics (play widths and feedback gains), and it is trained jointly with the 32 circuit parameters.

The classical circuit can be read: measurement and material parameters separate, the material part is a discretized
ODE that can be evaluated at any time step, and it has an exact equivalent circuit of positive inductors, resistors
and diode-bridge sliders plus controlled sources, which reproduces it in SPICE.

## Repository

```
ferrite_bh/   NumPy implementation, fitted parameters (JSON) and SPICE subcircuits of the classical circuit
src/          model, training, evaluation, checks, tables and figures (one script each, docstring = usage)
configs/      run configurations (model variants trained together in one run)
models/       trained checkpoints of every run in the paper (all variants of a run per file)
results/      evaluation results that the tables and figures are built from
data/         not included: raw data (data/raw/), cached windows and the RHINO-MAG GRU8 results (see Data)
paper/        Typst source of the paper and its supplement, generated tables (paper/tables/) and figures (paper/fig/)
scripts/      reproduce.sh: every experiment, table and figure
```

Main files:

| file | content |
|---|---|
| `src/compact_m.py` | the model (`CompactM`, all variants as options) and its training |
| `src/scan.py` | fused Triton kernel of the recurrence with hand-derived adjoint; `scan_ref` is the plain PyTorch reference |
| `src/mc2lib.py` | data loading, window sampling, metrics (SRE, NERE) |
| `src/eval_m.py`, `src/official_eval.py`, `src/main_stats.py` | test evaluation, official MC2 metric (95th percentiles), statistics |
| `src/finetune_m.py` | refitting parameter groups on another specimen (MC2 → MC1 transfer) |
| `src/spice_check.py`, `src/ode_check.py`, `src/upsample_check.py`, `src/ekf_eval.py` | equivalent circuit in ngspice, step-size consistency, measurement correction at other steps, Kalman-filter estimator |
| `src/hyper_m.py`, `src/scan_tv.py` | the gain-scheduled circuit (per-material MLP on the element values) and its kernel with per-step coefficients |
| `src/hyper_analysis.py` | energy balance of the scheduled circuit (parametric pumping of the time-varying inductors) |

## Setup

[uv](https://docs.astral.sh/uv/) installs the environment from `pyproject.toml` / `uv.lock`:

```
uv sync
```

The lock file pins PyTorch for ROCm 6.4 (AMD GPU), on which all results were computed. For an NVIDIA GPU or a CPU,
change the `pytorch-rocm` index in `pyproject.toml` to the matching PyTorch index (and `pytorch-triton-rocm` to
`triton`) before `uv sync`. Without a GPU the model runs on the plain PyTorch reference of the kernel: evaluation and
all tables and figures work, training is impractically slow.

## Data

The measurement data are not redistributed here. Download them into `data/raw/` and convert them:

| source | place as |
|---|---|
| MC2 training data, 10 materials ([MagNet Challenge 2](https://github.com/minjiechen/magnetchallenge-2)) | `data/raw/mc2/Training Data/<material>.zip` |
| MC2 training data of the 5 final-evaluation materials | `data/raw/mc2/evaltrain/Material <A-E>/` |
| MC2 labeled test set of these 5 materials | `data/raw/mc2/test-set-5-eval-materials.zip` |
| MagNet raw data, for the transfer study ([MagNet Challenge 1](https://github.com/minjiechen/magnetchallenge)) | `data/raw/mc1/Raw Data.zip` |

```
cd src && uv run python prepare_data.py
```

The test and validation windows are drawn reproducibly from the data on first use and cached in `data/`.

The comparison with the GRU8 baseline needs its predictions on the same windows, `data/<material>_rhino_res.npy`. They
are not included; `src/eval_rhino.py` computes them from the published RHINO-MAG models, run in the
[RHINO-MAG](https://github.com/upb-lea/RHINO-MAG) environment.

## Reproducing the paper

```
scripts/reproduce.sh                 # everything, about 25 GPU hours on one AMD RX 9070 XT
scripts/reproduce.sh tables figures  # only the tables and figures, from the included results and checkpoints
```

Steps: `final` (classical circuit and energy-error weights, 10 seeds), `path` (smaller models, 10 seeds of the 31- and 27-parameter variants),
`ablation`, `ood`
(extrapolation and data efficiency, with a GRU8 replica), `transfer` (MC2 → MC1 specimens), `checks` (SPICE,
step size, Kalman filter), `schedule` (gain-scheduled circuits, 16 and 64 hidden units, 5 seeds each, and the energy
check), `export` (JSON,
SPICE subcircuits and their checks), `tables`, `figures`, `paper` (compiles `paper/main.pdf` and `paper/supplement.pdf` with
Typst). The GRU8 baseline results must be computed first (see Data).

## Using the model

Without PyTorch, from the repository root (NumPy only; see `AGENTS.md` for conventions and pitfalls):

```python
import ferrite_bh as fb

fb.materials()                                # the 15 materials
H = fb.simulate("N87", B, T=25)               # open loop from the demagnetized state, B in T at 16 MHz starting at 0
H = fb.predict("N87", B, H_warmup, T=25)      # MC2 setting: measured H over the first k samples sets the state
fb.coefficients("N87", T=25)                  # a1, Bs, play widths and weights, relaxations, feedback gains, ...
```

```
python -m ferrite_bh simulate N87 --T 25 --B b.csv --out h.csv
```

- `ferrite_bh/params/<material>.json`: the 32 fitted parameters by name, with units and descriptions, plus the
  warm-up estimator, validity range and test errors.
- `ferrite_bh/spice/<material>.lib`: the equivalent circuit of the material as a SPICE subcircuit,
  `X1 p n N87_core params: Tc=25 N=5 Ae=82.6e-6 le=0.08206`, with temperature as a parameter.

The NumPy implementation reproduces the PyTorch model to 1e-4 % (`results/check_numpy.txt`), the subcircuits the
NumPy model to a median of 0.1 % (`results/check_spice_lib.txt`).

With PyTorch, the checkpoints of every run in the paper:

```python
# from src/
import torch
from loader import load

model, variants, materials = load("nere")                      # checkpoint models/nere.pt
e, i = variants.index("final_e1"), materials.index("N87")       # classical circuit, material N87
# B, H: (N, W) tensors at 16 MHz, T: (N,) in °C; H is used only for the warm-up n < k
H_pred = model(B, H, T, torch.full((len(B),), i), k)[e]
```

## License

MIT (see `LICENSE`). The measurement data are subject to the terms of the MagNet Challenge.

## Citation

```bibtex
@misc{kirchgaessner2026circuit,
  author = {Kirchg{\"a}ssner, Wilhelm},
  title  = {A Data-Driven, Gain-Scheduled Equivalent Circuit for the {$B$--$H$} Behavior of Power Ferrites},
  year   = {2026},
  note   = {arXiv preprint}
}
```
