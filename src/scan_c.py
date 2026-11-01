"""Fused GPU scan (Triton) for the hyper-network option C (hyper_m.py): the coefficients of every step come from a
per-row MLP that also sees the previous output H_(n-1), so the MLP runs inside the recurrence. Per step j:
    a1_s = tanh(pre_sj + w1h_s phi(h_(j-1))),  phi = asinh(h / 10),  s = +, - (mirrored branch, w1h_- = -w1h_+)
    a2_s = tanh(a1_s W2 + b2),  z = mean_s(a2_s W3) + b3                                     (16 shifts, see hyper_m.IXC)
    coefficients from z and the backbone values (static curve, plays, relaxations, gamma), then as in scan.py:
    x = u_j + gamma h_(j-1) - p;  p += alpha (relu(x - r) - relu(-x - r));  h = s'_j + m' sum wc p + c
    s' = (1 + q)(a1 u + a3 u^3) + hsat,  m' = m (1 + qh);  j < k: observer correction, h = H_j
pre (the MLP input layer of the exogenous inputs) is computed outside. The backward is hand-derived through the whole
step including the feedback of h into the MLP; it returns dL/dz per step, from which the MLP weight gradients follow by
one vectorized torch pass (z depends on the weights only through the step-local MLP once h_(j-1) is given).
The backward also returns the gradients of the backbone values (per-row static and operator values, observer gains), of
the drive u and of the measured H (through its lag), so the backbone can be trained with the MLP (--joint).
"""
import torch, triton, triton.language as tl

NO = 16          # outputs of the MLP (hyper_m.IXC order)


@triton.jit
def _tanh(x):
    return 1.0 - 2.0 / (tl.exp(2.0 * x) + 1.0)


@triton.jit
def _sig(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _softplus(x):
    return tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))


@triton.jit
def _col(z, oc, o: tl.constexpr):
    return tl.sum(tl.where(oc[None, :] == o, z, 0.0), 1)


@triton.jit
def _mlp(pp, pm, w1p, w1m, w2, b2, w3, b3, hp):
    ah = tl.abs(hp) / 10.0
    phi = tl.where(hp >= 0, 1.0, -1.0) * tl.log(ah + tl.sqrt(ah * ah + 1.0))
    a1p = _tanh(pp + w1p * phi[:, None]); a1m = _tanh(pm + w1m * phi[:, None])
    a2p = _tanh(tl.sum(a1p[:, :, None] * w2, 1) + b2); a2m = _tanh(tl.sum(a1m[:, :, None] * w2, 1) + b2)
    z = 0.5 * (tl.sum(a2p[:, :, None] * w3, 1) + tl.sum(a2m[:, :, None] * w3, 1)) + b3
    dphi = 0.1 / tl.sqrt(ah * ah + 1.0)
    return a1p, a1m, a2p, a2m, z, dphi


@triton.jit
def _coef(z, oc, uj, a1b, a3b, bsb, lbw, al0, dw0, dw1, lrK, xK, xgK, g0K, rawK, cz0K, ks, M: tl.constexpr):
    A1 = a1b * tl.exp(_col(z, oc, 0)); A3 = 100.0 * (a3b + _col(z, oc, 1))
    bsr = bsb + 0.05 * _col(z, oc, 2); Bs = tl.maximum(bsr, 0.15); bsm = (bsr > 0.15).to(tl.float32)
    bw = tl.exp(lbw + _col(z, oc, 3))
    v1 = (uj - Bs) / bw; v2 = (-uj - Bs) / bw
    E1 = tl.exp(tl.minimum(v1, 15.0)); E2 = tl.exp(tl.minimum(v2, 15.0))
    k1 = (v1 < 15.0).to(tl.float32); k2 = (v2 < 15.0).to(tl.float32)
    hs = 10.0 * (E1 - E2)
    poly = A1 * uj + A3 * uj * uj * uj
    mx = tl.exp(_col(z, oc, 4))
    al = _sig(al0 + _col(z, oc, 5))
    isp = ks[None, :] < M; is0 = ks[None, :] == M
    zbe = tl.where(is0, _col(z, oc, 10)[:, None], _col(z, oc, 11)[:, None])
    zc = tl.where(is0, _col(z, oc, 12)[:, None], _col(z, oc, 13)[:, None])
    bet = 0.5 * _sig(rawK + zbe)
    aK = tl.where(isp, al[:, None], bet)
    rK = tl.where(isp, tl.exp(lrK + _col(z, oc, 6)[:, None] + _col(z, oc, 7)[:, None] * xK), 0.0)
    vK = (dw0 + _col(z, oc, 8))[:, None] + (dw1 + _col(z, oc, 9))[:, None] * xK
    yK = cz0K + zc
    wK = tl.where(isp, -_softplus(-vK), -_softplus(yK))
    gK = g0K + 1e-3 * (_col(z, oc, 14)[:, None] + _col(z, oc, 15)[:, None] * xgK)
    return poly, hs, mx, aK, rK, wK, gK, A1, A3, Bs, bsm, bw, E1, E2, k1, k2, al, vK, yK


@triton.jit
def _fwd(pre, u, Hh, p0, c0, q, qh, LL, Lc, BS, LN, W1H, W2, B2, W3, B3, out, sto, hso, Ps, Cs, R, W, k,
         K: tl.constexpr, M: tl.constexpr, HD: tl.constexpr, NOUT: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR); rm = rows < R
    ks = tl.arange(0, K); i2 = rows[:, None] * K + ks[None, :]; m2 = rm[:, None]
    hc = tl.arange(0, HD); oc = tl.arange(0, NOUT); ih = rows[:, None] * HD + hc[None, :]
    w1p = tl.load(W1H + ih, mask=m2, other=0.0); w1m = tl.load(W1H + R * HD + ih, mask=m2, other=0.0)
    w2 = tl.load(W2 + rows[:, None, None] * HD * HD + hc[None, :, None] * HD + hc[None, None, :], mask=rm[:, None, None], other=0.0)
    b2 = tl.load(B2 + ih, mask=m2, other=0.0)
    w3 = tl.load(W3 + rows[:, None, None] * HD * NOUT + hc[None, :, None] * NOUT + oc[None, None, :], mask=rm[:, None, None], other=0.0)
    b3 = tl.load(B3 + rows[:, None] * NOUT + oc[None, :], mask=m2, other=0.0)
    a1b = tl.load(BS + rows * 8 + 0, mask=rm, other=1.0); a3b = tl.load(BS + rows * 8 + 1, mask=rm, other=0.0)
    bsb = tl.load(BS + rows * 8 + 2, mask=rm, other=0.4); lbw = tl.load(BS + rows * 8 + 3, mask=rm, other=0.0)
    al0 = tl.load(BS + rows * 8 + 4, mask=rm, other=0.0); dw0 = tl.load(BS + rows * 8 + 5, mask=rm, other=0.0)
    dw1 = tl.load(BS + rows * 8 + 6, mask=rm, other=0.0)
    lrK = tl.load(LN + i2, mask=m2, other=0.0); xK = tl.load(LN + R * K + i2, mask=m2, other=0.0)
    xgK = tl.load(LN + 2 * R * K + i2, mask=m2, other=0.0); g0K = tl.load(LN + 3 * R * K + i2, mask=m2, other=0.0)
    rawK = tl.load(LN + 4 * R * K + i2, mask=m2, other=0.0); cz0K = tl.load(LN + 5 * R * K + i2, mask=m2, other=0.0)
    p = tl.load(p0 + i2, mask=m2, other=0.0); L = tl.load(LL + i2, mask=m2, other=0.0)
    c = tl.load(c0 + rows, mask=rm, other=0.0); lc = tl.load(Lc + rows, mask=rm, other=0.0)
    qq = tl.load(q + rows, mask=rm, other=0.0); qhh = tl.load(qh + rows, mask=rm, other=0.0)
    hp = tl.load(Hh + rows, mask=rm, other=0.0)
    for j in range(W):
        tl.store(Ps + j * R * K + i2, p, mask=m2); tl.store(Cs + j * R + rows, c, mask=rm)
        pp = tl.load(pre + (2 * j) * R * HD + ih, mask=m2, other=0.0); pm = tl.load(pre + (2 * j + 1) * R * HD + ih, mask=m2, other=0.0)
        uj = tl.load(u + j * R + rows, mask=rm, other=0.0); hj = tl.load(Hh + j * R + rows, mask=rm, other=0.0)
        a1p, a1m, a2p, a2m, z, dphi = _mlp(pp, pm, w1p, w1m, w2, b2, w3, b3, hp)
        poly, hs, mx, aK, rK, wK, gK, A1, A3, Bs, bsm, bw, E1, E2, k1, k2, al, vK, yK = _coef(
            z, oc, uj, a1b, a3b, bsb, lbw, al0, dw0, dw1, lrK, xK, xgK, g0K, rawK, cz0K, ks, M)
        st = (1 + qq) * poly + hs; mm = mx * (1 + qhh)
        x = uj[:, None] + gK * hp[:, None] - p
        p = p + aK * (tl.maximum(x - rK, 0.0) - tl.maximum(-x - rK, 0.0))
        h = st + mm * tl.sum(wK * p, 1) + c
        g = (j < k).to(tl.float32)
        e = (hj - h) * g
        p = p + L * e[:, None]; c = c + lc * e
        hp = g * hj + (1 - g) * h
        tl.store(out + j * R + rows, hp, mask=rm); tl.store(sto + j * R + rows, st, mask=rm); tl.store(hso + j * R + rows, hs, mask=rm)


@triton.jit
def _bwd(go, gso, gho, pre, u, Hh, q, qh, LL, Lc, BS, LN, W1H, W2, B2, W3, B3, out, Ps, Cs, GZ, gp0, gc0, gq, gqh,
         GU, GH, GH0, GLL, GLc, GBS, GLN, R, W, k,
         K: tl.constexpr, M: tl.constexpr, HD: tl.constexpr, NOUT: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR); rm = rows < R
    ks = tl.arange(0, K); i2 = rows[:, None] * K + ks[None, :]; m2 = rm[:, None]
    hc = tl.arange(0, HD); oc = tl.arange(0, NOUT); ih = rows[:, None] * HD + hc[None, :]
    w1p = tl.load(W1H + ih, mask=m2, other=0.0); w1m = tl.load(W1H + R * HD + ih, mask=m2, other=0.0)
    w2 = tl.load(W2 + rows[:, None, None] * HD * HD + hc[None, :, None] * HD + hc[None, None, :], mask=rm[:, None, None], other=0.0)
    b2 = tl.load(B2 + ih, mask=m2, other=0.0)
    w3 = tl.load(W3 + rows[:, None, None] * HD * NOUT + hc[None, :, None] * NOUT + oc[None, None, :], mask=rm[:, None, None], other=0.0)
    b3 = tl.load(B3 + rows[:, None] * NOUT + oc[None, :], mask=m2, other=0.0)
    a1b = tl.load(BS + rows * 8 + 0, mask=rm, other=1.0); a3b = tl.load(BS + rows * 8 + 1, mask=rm, other=0.0)
    bsb = tl.load(BS + rows * 8 + 2, mask=rm, other=0.4); lbw = tl.load(BS + rows * 8 + 3, mask=rm, other=0.0)
    al0 = tl.load(BS + rows * 8 + 4, mask=rm, other=0.0); dw0 = tl.load(BS + rows * 8 + 5, mask=rm, other=0.0)
    dw1 = tl.load(BS + rows * 8 + 6, mask=rm, other=0.0)
    lrK = tl.load(LN + i2, mask=m2, other=0.0); xK = tl.load(LN + R * K + i2, mask=m2, other=0.0)
    xgK = tl.load(LN + 2 * R * K + i2, mask=m2, other=0.0); g0K = tl.load(LN + 3 * R * K + i2, mask=m2, other=0.0)
    rawK = tl.load(LN + 4 * R * K + i2, mask=m2, other=0.0); cz0K = tl.load(LN + 5 * R * K + i2, mask=m2, other=0.0)
    L = tl.load(LL + i2, mask=m2, other=0.0); lc = tl.load(Lc + rows, mask=rm, other=0.0)
    qq = tl.load(q + rows, mask=rm, other=0.0); qhh = tl.load(qh + rows, mask=rm, other=0.0)
    isp = (ks[None, :] < M).to(tl.float32); is0 = (ks[None, :] == M).to(tl.float32); is1 = (ks[None, :] == M + 1).to(tl.float32)
    carry = tl.zeros((BR,), tl.float32)
    gp = tl.zeros((BR, K), tl.float32); gc = tl.zeros((BR,), tl.float32)
    gqa = tl.zeros((BR,), tl.float32); gqha = tl.zeros((BR,), tl.float32)
    gL = tl.zeros((BR, K), tl.float32); glc = tl.zeros((BR,), tl.float32)
    ga1b = tl.zeros((BR,), tl.float32); ga3b = tl.zeros((BR,), tl.float32); gbsb = tl.zeros((BR,), tl.float32)
    glbw = tl.zeros((BR,), tl.float32); gal0 = tl.zeros((BR,), tl.float32)
    gdw0 = tl.zeros((BR,), tl.float32); gdw1 = tl.zeros((BR,), tl.float32)
    glr = tl.zeros((BR, K), tl.float32); gxk = tl.zeros((BR, K), tl.float32); gxg = tl.zeros((BR, K), tl.float32)
    gg0 = tl.zeros((BR, K), tl.float32); graw = tl.zeros((BR, K), tl.float32); gcz = tl.zeros((BR, K), tl.float32)
    for jj in range(W):
        j = W - 1 - jj
        p = tl.load(Ps + j * R * K + i2, mask=m2, other=0.0); c = tl.load(Cs + j * R + rows, mask=rm, other=0.0)
        pp = tl.load(pre + (2 * j) * R * HD + ih, mask=m2, other=0.0); pm = tl.load(pre + (2 * j + 1) * R * HD + ih, mask=m2, other=0.0)
        uj = tl.load(u + j * R + rows, mask=rm, other=0.0); hj = tl.load(Hh + j * R + rows, mask=rm, other=0.0)
        goj = tl.load(go + j * R + rows, mask=rm, other=0.0)
        gsj = tl.load(gso + j * R + rows, mask=rm, other=0.0); ghj = tl.load(gho + j * R + rows, mask=rm, other=0.0)
        hp = tl.load(out + (j - 1) * R + rows, mask=rm & (j > 0), other=0.0) + tl.load(Hh + rows, mask=rm & (j == 0), other=0.0)
        a1p, a1m, a2p, a2m, z, dphi = _mlp(pp, pm, w1p, w1m, w2, b2, w3, b3, hp)
        poly, hs, mx, aK, rK, wK, gK, A1, A3, Bs, bsm, bw, E1, E2, k1, k2, al, vK, yK = _coef(
            z, oc, uj, a1b, a3b, bsb, lbw, al0, dw0, dw1, lrK, xK, xgK, g0K, rawK, cz0K, ks, M)
        mm = mx * (1 + qhh)
        x = uj[:, None] + gK * hp[:, None] - p
        act = (tl.abs(x) > rK).to(tl.float32)
        dz = tl.maximum(x - rK, 0.0) - tl.maximum(-x - rK, 0.0)
        p1 = p + aK * dz
        wp = tl.sum(wK * p1, 1)
        g = (j < k).to(tl.float32)
        hcur = (1 + qq) * poly + hs + mm * wp + c
        e = (hj - hcur) * g
        ge = tl.sum(L * gp, 1) + lc * gc
        gL += gp * e[:, None]; glc += gc * e
        tl.store(GH + j * R + rows, g * (ge + goj + carry), mask=rm)            # dL/dH_j (observer and output)
        gh = g * (-ge) + (1 - g) * (goj + carry)
        gst = gh + gsj                                                          # dL/ds'
        gmm = gh * wp
        gp1 = gp + (gh * mm)[:, None] * wK
        gw = (gh * mm)[:, None] * p1
        gc = gc + gh
        d = aK * act
        gx = gp1 * d
        ga = gp1 * dz
        sx = tl.where(x > 0, 1.0, -1.0)
        gr = gp1 * (-aK * sx * act)
        gg = gx * hp[:, None]
        ghp = tl.sum(gx * gK, 1)
        gp = gp1 * (1 - d)
        # coefficients -> z
        gqa += gst * poly; gqha += gmm * mx
        gpoly = gst * (1 + qq); ghs = gst + ghj
        dBs = 10.0 * (-E1 * k1 + E2 * k2) / bw
        dbw = 10.0 * (-E1 * k1 * (uj - Bs) + E2 * k2 * (-uj - Bs)) / (bw * bw)
        z0 = gpoly * uj * A1
        z1 = 100.0 * gpoly * uj * uj * uj
        z2 = ghs * dBs * 0.05 * bsm
        z3 = ghs * dbw * bw
        z4 = gmm * (1 + qhh) * mx
        z5 = tl.sum(ga * isp, 1) * al * (1 - al)
        dbet = aK * (1 - 2 * aK)
        z10 = tl.sum(ga * is0 * dbet, 1); z11 = tl.sum(ga * is1 * dbet, 1)
        z6 = tl.sum(gr * isp * rK, 1); z7 = tl.sum(gr * isp * rK * xK, 1)
        sv = _sig(-vK)
        z8 = tl.sum(gw * isp * sv, 1); z9 = tl.sum(gw * isp * sv * xK, 1)
        sy = _sig(yK)
        z12 = -tl.sum(gw * is0 * sy, 1); z13 = -tl.sum(gw * is1 * sy, 1)
        z14 = 1e-3 * tl.sum(gg, 1); z15 = 1e-3 * tl.sum(gg * xgK, 1)
        # backbone values (per row) and the drive u
        ga1b += gpoly * uj * tl.exp(_col(z, oc, 0)); ga3b += z1; gbsb += ghs * dBs * bsm; glbw += z3; gal0 += z5
        gdw0 += z8; gdw1 += z9
        glr += gr * isp * rK
        gxk += isp * (gr * rK * _col(z, oc, 7)[:, None] + gw * sv * (dw1 + _col(z, oc, 9))[:, None])
        gxg += gg * 1e-3 * _col(z, oc, 15)[:, None]; gg0 += gg
        graw += ga * (1 - isp) * dbet; gcz += -gw * (1 - isp) * sy
        dhu = 10.0 * (E1 * k1 + E2 * k2) / bw
        tl.store(GU + j * R + rows, tl.sum(gx, 1) + gpoly * (A1 + 3.0 * A3 * uj * uj) + ghs * dhu, mask=rm)
        o = oc[None, :]
        gz = (tl.where(o == 0, z0[:, None], 0.0) + tl.where(o == 1, z1[:, None], 0.0) + tl.where(o == 2, z2[:, None], 0.0)
              + tl.where(o == 3, z3[:, None], 0.0) + tl.where(o == 4, z4[:, None], 0.0) + tl.where(o == 5, z5[:, None], 0.0)
              + tl.where(o == 6, z6[:, None], 0.0) + tl.where(o == 7, z7[:, None], 0.0) + tl.where(o == 8, z8[:, None], 0.0)
              + tl.where(o == 9, z9[:, None], 0.0) + tl.where(o == 10, z10[:, None], 0.0) + tl.where(o == 11, z11[:, None], 0.0)
              + tl.where(o == 12, z12[:, None], 0.0) + tl.where(o == 13, z13[:, None], 0.0) + tl.where(o == 14, z14[:, None], 0.0)
              + tl.where(o == 15, z15[:, None], 0.0))
        tl.store(GZ + j * R * NOUT + rows[:, None] * NOUT + oc[None, :], gz, mask=m2)
        # z -> previous output through the MLP
        ga2 = 0.5 * tl.sum(w3 * gz[:, None, :], 2)
        gq2p = ga2 * (1 - a2p * a2p); gq2m = ga2 * (1 - a2m * a2m)
        gq1p = tl.sum(w2 * gq2p[:, None, :], 2) * (1 - a1p * a1p); gq1m = tl.sum(w2 * gq2m[:, None, :], 2) * (1 - a1m * a1m)
        ghp += (tl.sum(gq1p * w1p, 1) + tl.sum(gq1m * w1m, 1)) * dphi
        carry = ghp
    tl.store(gp0 + i2, gp, mask=m2); tl.store(gc0 + rows, gc, mask=rm)
    tl.store(gq + rows, gqa, mask=rm); tl.store(gqh + rows, gqha, mask=rm)
    tl.store(GH0 + rows, carry, mask=rm)                                        # h_(-1) = H_0
    tl.store(GLL + i2, gL, mask=m2); tl.store(GLc + rows, glc, mask=rm)
    tl.store(GBS + rows * 8 + 0, ga1b, mask=rm); tl.store(GBS + rows * 8 + 1, ga3b, mask=rm)
    tl.store(GBS + rows * 8 + 2, gbsb, mask=rm); tl.store(GBS + rows * 8 + 3, glbw, mask=rm)
    tl.store(GBS + rows * 8 + 4, gal0, mask=rm); tl.store(GBS + rows * 8 + 5, gdw0, mask=rm)
    tl.store(GBS + rows * 8 + 6, gdw1, mask=rm); tl.store(GBS + rows * 8 + 7, gdw1 * 0, mask=rm)
    tl.store(GLN + i2, glr, mask=m2); tl.store(GLN + R * K + i2, gxk, mask=m2); tl.store(GLN + 2 * R * K + i2, gxg, mask=m2)
    tl.store(GLN + 3 * R * K + i2, gg0, mask=m2); tl.store(GLN + 4 * R * K + i2, graw, mask=m2)
    tl.store(GLN + 5 * R * K + i2, gcz, mask=m2)


def mlp_z(pre, w1h, w2, b2, w3, b3, hp):
    """The MLP of all steps at once given the previous outputs hp (W,R): z (W,R,NO). pre (W,2,R,h), w1h (2,R,h)."""
    a = torch.tanh(pre + w1h[None] * torch.asinh(hp / 10.0)[:, None, :, None])
    a = torch.tanh(torch.einsum("wsrh,rhg->wsrg", a, w2) + b2[None, None])
    return torch.einsum("wsrh,rho->wsro", a, w3).mean(1) + b3[None]


class _ScanC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre, w1h, w2, b2, w3, b3, p0, c0, q, qh, uT, HT, LL, Lc, BS, LN, k, M):
        W, R = uT.shape; K = p0.shape[1]; HD = w2.shape[-1]; BR = max(2, 128 // HD)
        out, sto, hso = torch.empty_like(uT), torch.empty_like(uT), torch.empty_like(uT)
        Ps = torch.empty(W, R, K, device=uT.device); Cs = torch.empty(W, R, device=uT.device)
        _fwd[(triton.cdiv(R, BR),)](pre, uT, HT, p0, c0, q, qh, LL, Lc, BS, LN, w1h, w2, b2, w3, b3, out, sto, hso, Ps, Cs,
                                    R, W, k, K=K, M=M, HD=HD, NOUT=NO, BR=BR)
        ctx.save_for_backward(pre, w1h, w2, b2, w3, b3, q, qh, uT, HT, LL, Lc, BS, LN, out, Ps, Cs); ctx.k, ctx.M = k, M
        return out, sto, hso

    @staticmethod
    def backward(ctx, go, gso, gho):
        pre, w1h, w2, b2, w3, b3, q, qh, uT, HT, LL, Lc, BS, LN, out, Ps, Cs = ctx.saved_tensors
        W, R = uT.shape; K = LL.shape[1]; HD = w2.shape[-1]; BR = max(2, 128 // HD)
        z0 = lambda x: torch.zeros_like(uT) if x is None else x.contiguous()
        GZ = torch.empty(W, R, NO, device=uT.device)
        gp0, gc0, gq, gqh = torch.empty(R, K, device=uT.device), torch.empty_like(Lc), torch.empty_like(Lc), torch.empty_like(Lc)
        GU, GH, GH0 = torch.empty_like(uT), torch.empty_like(uT), torch.empty_like(Lc)
        GLL, GLc, GBS, GLN = torch.empty_like(LL), torch.empty_like(Lc), torch.empty_like(BS), torch.empty_like(LN)
        _bwd[(triton.cdiv(R, BR),)](z0(go), z0(gso), z0(gho), pre, uT, HT, q, qh, LL, Lc, BS, LN, w1h, w2, b2, w3, b3, out, Ps, Cs,
                                    GZ, gp0, gc0, gq, gqh, GU, GH, GH0, GLL, GLc, GBS, GLN, R, W, ctx.k, K=K, M=ctx.M, HD=HD, NOUT=NO, BR=BR)
        GH[0] += GH0
        hp = torch.cat([HT[:1], out[:-1]])
        with torch.enable_grad():
            xs = [x.detach().requires_grad_() for x in (pre, w1h, w2, b2, w3, b3)]
            gr = torch.autograd.grad(mlp_z(*xs, hp), xs, GZ)
        return (*gr, gp0, gc0, gq, gqh, GU, GH, GLL, GLc, GBS, GLN, None, None)


def scan_c(pre, w1h, w2, b2, w3, b3, p0, c0, q, qh, u, H, L, Lc, base, lanes, k, M):
    """pre (2,R,W,h); w1h (2,R,h) (mirror already negated); w2 (R,h,h); b2 (R,h); w3 (R,h,NO); b3 (R,NO); p0, L (R,K);
    c0, q, qh, Lc (R); u, H (R,W); base (R,8) = a1b, a3b, Bsb, log bw, logit alpha, dw0, dw1, 0;
    lanes (6,R,K) = log r, x, x for gamma (0 on relaxations), gamma0, raw beta (relaxations), raw c (relaxations).
    Returns h, s', hsat (R,W)."""
    f = lambda x: x.contiguous()
    o = _ScanC.apply(f(pre.permute(2, 0, 1, 3)), f(w1h), f(w2), f(b2), f(w3), f(b3), f(p0), f(c0), f(q), f(qh),
                     f(u.T), f(H.T), f(L), f(Lc), f(base), f(lanes), int(k), M)
    return tuple(x.T for x in o)
