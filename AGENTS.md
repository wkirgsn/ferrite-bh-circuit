# AGENTS.md — ferrite-bh-circuit

Guide for AI agents and scripts. Humans: see README.md.

## What this repository provides

A 32-parameter model of the B–H behavior of 15 power ferrites (MagNet Challenge 2 materials), in three equivalent forms:

| form | where | needs |
|---|---|---|
| fitted parameters, plain JSON | `ferrite_bh/params/<material>.json` | nothing |
| NumPy implementation | `ferrite_bh/model.py` (package `ferrite_bh`) | Python ≥ 3.10, NumPy |
| SPICE subcircuits (material only) | `ferrite_bh/spice/<material>.lib` | ngspice (or LTspice) |
| training / evaluation code (paper) | `src/` | PyTorch, see README |

Materials: 3C90, 3C92, 3C94, 3C95, 3E6, 3F4, 77, 78, FEC007, FEC014, N27, N30, N49, N87, T37
(`python -m ferrite_bh list`).

## Conventions (read before using the model)

- Sample period 62.5 ns (16 MHz). All time constants are in samples. Resample other waveforms to 16 MHz first.
- B in T, H in A/m, temperature in °C. The model is per unit core: winding and geometry enter only through
  B = Φ/(N Ae) and H = N i / le.
- Trained temperatures: 25, 50, 70 °C. All coefficients are at most linear in t = (T − 50)/25; outside
  25–70 °C this is extrapolation.
- Measurement vs. material: the parameters tau1, tau2 (correction u = B + tau1 B′ + tau2 B″) and eta (first-order lag
  of H) describe the MC2 test bench, not the material. `simulate(..., material_only=True)` and the SPICE subcircuits
  exclude them; `predict()` includes them because it reproduces the challenge's measured H.
- Do not apply the measurement correction to a waveform at another sample rate; it is a 16 MHz filter.

## Python API (`import ferrite_bh as fb`, from the repository root)

```python
fb.materials()                         # list of names
fb.load("N87")                         # JSON record: parameters, estimator, validity, test errors
fb.coefficients("N87", T=25)           # dict: a1, a3, Bs, bw, r (6), alpha, w (6), beta (2), c (2), gamma (8), tau1, tau2, eta
fb.simulate("N87", B, T=25)            # open loop from the demagnetized state; B: (W,) or (N, W), must start at 0
fb.simulate("N87", B, T=25, states=s)  # from given states s: (8,) = 6 play states, 2 relaxation states (in T)
fb.predict("N87", B, H_warmup, T=25)   # MC2 setting: measured H over the first k samples fixes the state
```

`predict` returns the full window: the first k samples are the measured warm-up, the rest the prediction. With
k < 50 the amplitude factors of the warm-up estimator are skipped.

## Command line

```
python -m ferrite_bh list
python -m ferrite_bh coefficients N87 --T 25
python -m ferrite_bh simulate N87 --T 25 --B b.csv --out h.csv   # one column in, one column out
python -m ferrite_bh predict N87 --T 25 --B b.csv --H h_warmup.csv
python -m ferrite_bh spice N87                                   # regenerate the .lib from the JSON
```

## SPICE

```
.include ferrite_bh/spice/N87.lib
X1 p n N87_core params: Tc=25 N=5 Ae=82.6e-6 le=0.08206    * the MC2 N87 specimen, R34.0x20.5x12.5
```

V(p, n) = N Ae dB/dt; the current into p is H le / N. Start the simulation at B = 0 (demagnetized). The circuit:
reversible curve as a nonlinear inductor, six plays (inductor + viscous Coulomb slider), two relaxations (inductor +
resistor), each behind a feedback source gamma_k dH/dt. Small time steps (≤ 62.5 ns / 4) and `method=gear` are
recommended.

## Accuracy and checks

- Test errors per material: `test_errors_percent` in each JSON (mean SRE over the MC2-style test windows and per
  warm-up length k = 100, 500, 900). Mean over the 15 materials 7.3 %.
- NumPy vs. PyTorch: ≤ 1e-4 % of rms H on 270 test windows (`src/check_numpy.py`, `results/check_numpy.txt`).
- SPICE vs. NumPy (material only, 60 runs at 25 and 70 °C): median 0.10 %, max 1.6 % (FEC014, whose fast plays make the
  continuous-time circuit differ from the 16 MHz recurrence; `src/check_spice_lib.py`, `results/check_spice_lib.txt`).
- Weak spots: 3C92 (SRE 12 %), deep saturation (N87 at 50 kHz), short warm-ups (k = 100).

## Do not

- Do not edit `ferrite_bh/params/*.json` by hand; they are exported from `models/nere.pt` by `src/export_params.py`.
- Do not expect the model to cover materials other than the 15 listed; refit with `src/compact_m.py`.
- Do not treat the warm-up estimator parameters (`estimator` in the JSON) as material parameters.
