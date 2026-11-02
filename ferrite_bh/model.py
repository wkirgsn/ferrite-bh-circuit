"""NumPy reference implementation of the 32-parameter B-H model (no PyTorch, no GPU).

Reproduces the trained PyTorch model (src/compact_m.py, variant final_e1 of models/nere.pt) to floating-point
rounding. One sample = 62.5 ns (16 MHz); B in T, H in A/m, temperature in degC; t = (T - 50) / 25.

    u    = B + tau1(t) B' + tau2(t) B''                          measurement correction (16 MHz acquisition chain)
    v_k  = u + gamma_k(t) H_(n-1)                                drive of operator k (mean-field feedback of H)
    p_i <- p_i + alpha(t) [relu(v_i - p_i - r_i) - relu(p_i - v_i - r_i)]     6 viscous plays, widths r_i(t)
    z_j <- z_j + beta_j (v_j - z_j)                              2 linear relaxations
    H    = a1(t) u + a3(t) u^3 + 10 A/m [e^((u - Bs)/b) - e^((-u - Bs)/b)] + sum w_i(t) p_i + sum c_j(t) z_j + c
    H_meas = first-order lag of H with pole eta (H-measurement chain)

The material model is everything except tau1, tau2 (measurement correction) and eta (lag); use
simulate(..., material_only=True) for the true field of the material.
predict() runs the warm-up estimator (observer, initial-state prior, amplitude factors) on the first k samples of a
measured H, as in the MagNet Challenge 2; simulate() runs open-loop from given (default: demagnetized) states.
"""
import json
from pathlib import Path
import numpy as np

DT = 62.5e-9
PARAMS = Path(__file__).parent / "params"


def materials():
    """Names of the materials with fitted parameters."""
    return sorted(p.stem for p in PARAMS.glob("*.json"))


def load(material):
    """Parameter record of a material (dict, see params/<material>.json)."""
    return json.loads((PARAMS / f"{material}.json").read_text())


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def coefficients(material, T):
    """Coefficients of the model at temperature T (degC) as the model uses them: dict of floats / arrays.
    Keys: tau1, tau2 (samples), a1 (A/m/T), a3 (A/m/T^3), Bs (T), bw (T), r (6, T), alpha, w (6, A/m/T),
    beta (2), c (2, A/m/T), gamma (8, T per A/m; plays then relaxations), eta (lag pole)."""
    P = load(material)["parameters"] if isinstance(material, str) else material["parameters"]
    v = lambda n: P[n]["value"]
    t = (T - 50.0) / 25.0
    M = 6
    lr = v("log_r_first") + (v("log_r_last") - v("log_r_first")) * np.linspace(0, 1, M) + v("rT") * t
    x = (lr - lr.mean()) / (lr.std(ddof=1) + 1e-6)                  # standardized log width (independent of T)
    w = (v("dw0_0") + v("dw0_1") * t) + (v("dw1_0") + v("dw1_1") * t) * x
    return dict(
        tau1=v("tau1_0") + v("tau1_1") * t, tau2=v("tau2_0") + v("tau2_1") * t,
        a1=v("a1_0") + v("a1_1") * t, a3=100.0 * (v("a3_0") + v("a3_1") * t),
        Bs=max(v("Bs_0") + v("Bs_1") * t, 0.15), bw=float(np.exp(v("log_bw"))),
        r=np.exp(lr), x=x, alpha=float(_sig(v("alpha_0") + v("alpha_1") * t)),
        w=-_softplus(-w),                                            # passive: w_i <= 0
        beta=0.5 * _sig(np.array([v("beta_1"), v("beta_2")])),       # passive: beta_j <= 1/2
        c=-_softplus(np.array([v("c1_0") + v("c1_1") * t, v("c2_0") + v("c2_1") * t])),   # passive: c_j <= 0
        gamma=1e-3 * ((v("gam_0") + v("gam_1") * t) + np.r_[v("gop_play") * x, v("gop_rel1"), v("gop_rel2")]),
        eta=float(0.6 * _sig(v("hl"))))


def static_curve(co, u):
    """Reversible curve a1 u + a3 u^3 + knee and its knee part (A/m)."""
    hsat = 10.0 * (np.exp(np.minimum((u - co["Bs"]) / co["bw"], 15.0)) - np.exp(np.minimum((-u - co["Bs"]) / co["bw"], 15.0)))
    return co["a1"] * u + co["a3"] * u ** 3 + hsat, hsat


def _scan(u, st, H, p0, c0, co, k, L=None, Lc=None, m=None):
    """Recurrence over the window, N windows in parallel: u, st, H (N, W); p0 (N, 8) states; c0 (N,) offset.
    The first k samples are corrected by the observer gains L (8,), Lc and replaced by the measured H."""
    N, W = u.shape
    a = np.r_[np.full(6, co["alpha"]), co["beta"]]; r = np.r_[co["r"], 0.0, 0.0]
    wc = np.r_[co["w"], co["c"]]; g = co["gamma"]
    m = np.ones((N, W)) if m is None else m
    p, c = p0.copy(), c0.copy(); hp = H[:, 0].copy(); out = np.empty((N, W))
    for j in range(W):
        x = u[:, j:j + 1] + g * hp[:, None] - p
        p = p + a * (np.maximum(x - r, 0.0) - np.maximum(-x - r, 0.0))
        h = st[:, j] + m[:, j] * (p * wc).sum(1) + c
        if j < k:
            e = H[:, j] - h
            p = p + L * e[:, None]; c = c + Lc * e
            h = H[:, j].copy()
        out[:, j] = h; hp = h
    return out


def _lag(x, eta, y0):
    """y_n = eta y_(n-1) + (1 - eta) x_n with y_(-1) = y0."""
    y = np.empty_like(x); yp = y0.copy()
    for j in range(x.shape[1]):
        yp = eta * yp + (1 - eta) * x[:, j]; y[:, j] = yp
    return y


def drive(B, co):
    """Measurement-corrected flux density u = B + tau1 B' + tau2 B'' (derivatives per sample)."""
    d1 = np.gradient(B, axis=-1); d2 = np.gradient(d1, axis=-1)
    return B + co["tau1"] * d1 + co["tau2"] * d2


def predict(material, B, H_warmup, T, k=None):
    """H (A/m) for windows of B (T) at 16 MHz, given the measured H over the first k samples (warm-up).
    B: (W,) or (N, W); H_warmup: (k,) / (N, k) or full-length arrays (only the first k samples are used); T: degC,
    scalar or (N,). Returns H with the shape of B: the warm-up samples as measured (lagged), then the prediction.
    This is the MagNet Challenge 2 setting: the output is the field as measured by the MC2 test bench."""
    rec = load(material) if isinstance(material, str) else material
    one = np.ndim(B) == 1
    B = np.atleast_2d(np.asarray(B, float)); N, W = B.shape
    Hw = np.atleast_2d(np.asarray(H_warmup, float))
    k = Hw.shape[1] if k is None else k
    H = np.zeros((N, W)); H[:, :k] = Hw[:, :k]
    T = np.broadcast_to(np.asarray(T, float), (N,))
    out = np.empty((N, W))
    for Tv in np.unique(T):                                           # coefficients per temperature
        i = np.where(T == Tv)[0]
        out[i] = _predict(rec, B[i], H[i], float(Tv), k)
    return out[0] if one else out


def _predict(rec, B, H, T, k):
    co = coefficients(rec, T); est = rec["estimator"]; N, W = B.shape
    eta = co["eta"]
    HE = (H - eta * np.concatenate([H[:, :1], H[:, :-1]], 1)) / (1 - eta)    # de-lagged warm-up
    u = drive(B, co)
    static, hsat = static_curve(co, u)
    r = co["r"]; g, g2 = np.array(est["g"]), np.array(est["g2"])
    u0 = u[:, :1]
    p = u0 - r * np.tanh(g2 * u0 / r + np.sign(u[:, 5:6] - u0) * g)         # prior initial play states
    z = np.repeat(u0, 2, 1)
    wc = np.r_[co["w"], co["c"]]
    e0 = HE[:, 0] - static[:, 0] - (np.concatenate([p, z], 1) * wc).sum(1)
    K0 = np.array(est["K0"])
    p = p + 1e-3 * K0[:6] * e0[:, None]; z = z + 1e-3 * K0[6:8] * e0[:, None]
    c = est["cpr"] + K0[8] * e0
    Lv = np.array(est["L"]); L, Lc = 1e-3 * Lv[:8], Lv[8]
    s0 = np.concatenate([p, z], 1)
    m = np.ones((N, W))
    if k >= 50:                                                        # amplitude factors q, qh of the window
        Ho = _scan(u[:, :k], static[:, :k], HE[:, :k], s0, c, co, 1, L, Lc)
        tt = np.linspace(-1, 1, k - 1)
        def dtr(x):
            y = x - x.mean(-1, keepdims=True)
            return y - (y * tt).sum(-1, keepdims=True) / (tt * tt).sum() * tt
        yd = dtr(HE[:, 1:k] - Ho[:, 1:])
        n0 = np.exp(est["lq"] + est["lqk"] * np.log(k / 300)); Sh = (dtr(HE[:, 1:k]) ** 2).mean(-1)
        h1 = dtr(static[:, 1:k] - hsat[:, 1:k]); h2 = dtr(Ho[:, 1:] - static[:, 1:k] - c[:, None])
        s11 = (h1 * h1).sum(-1) + n0 * Sh + 1e-6; s22 = (h2 * h2).sum(-1) + n0 * Sh + 1e-6; s12 = (h1 * h2).sum(-1)
        b1, b2 = (yd * h1).sum(-1), (yd * h2).sum(-1); det = s11 * s22 - s12 ** 2
        q = np.clip((s22 * b1 - s12 * b2) / det, -0.5, 0.5); qh = np.clip((s11 * b2 - s12 * b1) / det, -0.5, 0.5)
        static = static + q[:, None] * (static - hsat); m = m * (1 + qh[:, None])
    out = _scan(u, static, HE, s0, c, co, k, L, Lc, m)
    return _lag(out, eta, H[:, 0])


def simulate(material, B, T, states=None, material_only=True):
    """Open-loop H (A/m) for B (T) at 16 MHz from given states (default: demagnetized, all states zero; start B at 0,
    e.g. with a ramp). B: (W,) or a batch (N, W); T: degC, scalar or (N,). material_only=True: the material alone
    (u = B, no lag), the true field of the material and what the SPICE subcircuit computes; False: with the measurement
    correction and lag of the MC2 test bench. states: optional (8,) / (N, 8) initial play (6) and relaxation (2) states.
    Returns H with the shape of B."""
    rec = load(material) if isinstance(material, str) else material
    B2 = np.atleast_2d(np.asarray(B, float)); N, W = B2.shape
    T = np.broadcast_to(np.asarray(T, float), (N,))
    s0 = np.zeros((N, 8)) if states is None else np.broadcast_to(np.asarray(states, float), (N, 8)).copy()
    out = np.empty((N, W))
    for Tv in np.unique(T):                                           # coefficients per temperature
        i = np.where(T == Tv)[0]
        co = coefficients(rec, float(Tv))
        u = B2[i] if material_only else drive(B2[i], co)
        static, _ = static_curve(co, u)
        o = _scan(u, static, np.zeros((len(i), W)), s0[i], np.zeros(len(i)), co, 0)
        out[i] = o if material_only else _lag(o, co["eta"], o[:, 0])
    return out[0] if np.ndim(B) == 1 else out
