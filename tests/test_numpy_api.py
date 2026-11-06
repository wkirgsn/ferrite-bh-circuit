"""ferrite_bh (NumPy): shapes and batching, physical properties of the classical circuit for every material."""
import numpy as np
import pytest
import ferrite_bh as fb
from conftest import excitation

MATS = fb.materials()


def test_materials():
    """The 15 MC2 materials have fitted parameters."""
    assert len(MATS) == 15 and {"N87", "3C90", "T37"} <= set(MATS)


@pytest.mark.parametrize("mat", MATS)
def test_passive_coefficients(mat):
    """The paper's constraints: non-positive play weights and relaxation gains, relaxation rates <= 1/2, lag pole
    <= 0.6, positive play widths, Bs >= 0.15 T, and sign-consistent operator coefficients at all trained temperatures."""
    for T in (25.0, 50.0, 70.0):
        co = fb.coefficients(mat, T)
        assert np.all(co["w"] <= 0) and np.all(co["c"] <= 0)
        assert np.all((co["beta"] > 0) & (co["beta"] <= 0.5)) and 0 < co["alpha"] < 1
        assert 0 <= co["eta"] <= 0.6 and np.all(co["r"] > 0) and co["Bs"] >= 0.15 and co["bw"] > 0


@pytest.mark.parametrize("material_only", [True, False])
def test_batch_equals_single(material_only):
    """A batch of windows gives exactly the results of the windows one by one."""
    B, T = excitation(N=5)
    Hb = fb.simulate("N87", B, T, material_only=material_only)
    assert Hb.shape == B.shape
    for j in range(len(B)):
        assert np.array_equal(Hb[j], fb.simulate("N87", B[j], T[j], material_only=material_only))


def test_predict_batch_and_shapes():
    """predict: shapes, warm-up returned as measured, batch equals single windows, full-length H with k."""
    B, T = excitation(N=5); H = fb.simulate("3C90", B, T, material_only=False)
    B, H = B[:, -1000:], H[:, -1000:]
    P = fb.predict("3C90", B, H[:, :500], T)
    assert P.shape == B.shape
    assert np.allclose(P[:, :500], H[:, :500])                        # warm-up samples are returned as measured
    for j in range(len(B)):
        assert np.allclose(P[j], fb.predict("3C90", B[j], H[j, :500], T[j]), rtol=0, atol=1e-9)
    assert np.allclose(fb.predict("3C90", B, H, T, k=500), P)         # full-length H with explicit k


@pytest.mark.parametrize("mat", MATS)
def test_odd_symmetry(mat):
    """From the demagnetized state the circuit is odd: H(-B) = -H(B)."""
    B, T = excitation(N=2, seed=1)
    assert np.allclose(fb.simulate(mat, -B, T), -fb.simulate(mat, B, T), rtol=0, atol=1e-9)


@pytest.mark.parametrize("mat", MATS)
@pytest.mark.parametrize("T", [25.0, 70.0])
def test_dissipation(mat, T):
    """Passivity: in periodic steady state the loop encloses a positive area, the energy per cycle oint H dB > 0."""
    per = 160                                                         # 100 kHz at 16 MHz: 160 samples per period
    n = np.arange(40 * per)
    B = 0.15 * np.minimum(n / (10 * per), 1) * np.sin(2 * np.pi * n / per)
    H = fb.simulate(mat, B, T)
    Bl, Hl = B[-per - 1:], H[-per - 1:]
    assert np.sum(0.5 * (Hl[1:] + Hl[:-1]) * np.diff(Bl)) > 0


def test_steady_state_reached():
    """The last two periods of a periodic excitation coincide (no drift of the states)."""
    per = 160; n = np.arange(40 * per)
    H = fb.simulate("N87", 0.2 * np.minimum(n / (10 * per), 1) * np.sin(2 * np.pi * n / per), 25.0)
    assert np.abs(H[-per:] - H[-2 * per:-per]).max() < 1e-3 * np.abs(H[-per:]).max()


def test_states_and_zero_input():
    """Zero input gives zero field; explicit zero states equal the demagnetized default."""
    assert np.array_equal(fb.simulate("N87", np.zeros(100), 25.0), np.zeros(100))
    B, T = excitation(N=1)
    s = np.zeros(8)
    assert np.array_equal(fb.simulate("N87", B[0], 25.0, states=s), fb.simulate("N87", B[0], 25.0))


def test_range_warning():
    """predict and simulate warn outside the training range (temperature, |B|, slope) and not on measured data."""
    with pytest.warns(fb.OutOfTrainingRangeWarning, match="T from"):
        fb.simulate("3C90", np.zeros(16), 100.0)
    with pytest.warns(fb.OutOfTrainingRangeWarning, match=r"\|B\| up to"):
        fb.predict("3C90", np.full(64, 1.0), np.zeros(16), 25.0)
    t = np.arange(320) / 16e6
    b = 0.4 * (1.0 - 4.0 * np.abs((t * 100e3 + 0.25) % 1.0 - 0.5))      # 0.4 T triangle at 100 kHz
    with pytest.warns(fb.OutOfTrainingRangeWarning, match="dB/dt"):
        fb.simulate("3C90", b, 25.0)
    b = 0.1 * np.sin(2 * np.pi * 100e3 * t)                               # 0.1 T sine at 100 kHz: within range
    assert fb.check_range(fb.load("3C90"), b, 25.0) == []
