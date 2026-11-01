"""Convert MagNet (MC1) raw .mat records into 16 MHz B/H sequences comparable to the MC2 data.
Usage: python mc1_convert.py <mat file> <out .npz>

Each record: voltage and current at Ts (8 ns), 10000 samples. B = integral of the secondary voltage / (N2 Ae),
H = N1 i / Le. B is integrated over an integer number of periods with the voltage mean removed and made zero-mean,
then both are resampled to 16 MHz with a polyphase anti-alias filter. The measured volumetric loss
P = f * oint H dB is computed at the native rate over the same integer number of periods."""
import sys, numpy as np, h5py
from fractions import Fraction
from scipy.signal import resample_poly

FS = 16e6
src, dst = sys.argv[1], sys.argv[2]
d = h5py.File(src, "r")["Data"]
txt = lambda k: "".join(map(chr, d[k][:, 0]))
N1, N2 = float(d["Primary_Turns"][0, 0]), float(d["Secondary_Turns"][0, 0])
Ae, Le = float(d["Effective_Area"][0, 0]), float(d["Effective_Length"][0, 0])
Ts = d["Sampling_Time"][0].astype(np.float64); f = d["Frequency_command"][0]
T = d["Temperature_command"][0]; Hdc = d["Hdc_command"][0]; duty = d["DutyP_command"][0]
n = len(f); V, I = d["Voltage"], d["Current"]
L16 = None; Bs, Hs, P, keep = [], [], [], []
for j in range(0, n, 500):
    Vb = V[:, j:j + 500].T.astype(np.float64); Ib = I[:, j:j + 500].T.astype(np.float64)
    for q in range(len(Vb)):
        r = j + q; per = 1 / (f[r] * Ts[r]); npd = int(len(Vb[q]) // per); m = int(round(npd * per))
        if npd < 1:
            continue
        v = Vb[q, :m] - Vb[q, :m].mean(); B = np.cumsum(v) * Ts[r] / (N2 * Ae); B -= B.mean()
        H = N1 * Ib[q, :m] / Le
        dB = np.gradient(B)
        P.append(f[r] * npd ** -1 * np.sum(H * dB))  # W/m^3: f * (sum over npd periods of H dB) / npd
        fr = Fraction(FS * Ts[r]).limit_denominator(1000)       # 16 MHz * 8 ns = 16/125
        k = 2 * fr.denominator                                  # wrap-pad (records span whole periods) against edge effects
        rs = lambda x: resample_poly(np.concatenate([x[-k:], x, x[:k]]), fr.numerator, fr.denominator)[2 * fr.numerator:-2 * fr.numerator]
        b16, h16 = rs(B), rs(H)
        Bs.append(b16.astype(np.float32)); Hs.append(h16.astype(np.float32)); keep.append(r)
    print(f"{j + 500}/{n}", flush=True)
keep = np.array(keep); n16 = np.array([len(b) for b in Bs]); L = n16.max()   # n16: samples of whole periods at 16 MHz
pad = lambda X: np.stack([np.pad(x, (0, L - len(x)), constant_values=np.nan) for x in X])
np.savez(dst, B=pad(Bs), H=pad(Hs), n16=n16, T=T[keep].astype(np.float32),
         f=f[keep], Hdc=Hdc[keep], duty=duty[keep], P=np.array(P), N1=N1, N2=N2, Ae=Ae, Le=Le,
         shape=txt("Shape"), material=txt("Material"), Ts=Ts[keep])
print(dst, len(keep), "records, length", n16.min(), "-", L)
