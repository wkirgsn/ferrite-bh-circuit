"""Challenge metrics (src/mc2lib.metrics): sequence relative error and normalized energy error."""
import numpy as np
from mc2lib import metrics


def test_sre_nere():
    """Zero error for a perfect prediction, 10 % SRE for a 10 % gain error, the warm-up not scored."""
    W, k = 1000, 500
    t = np.linspace(0, 4 * np.pi, W)[None]
    B = 0.1 * np.sin(t); H = 50 * np.sin(t + 0.3)
    sre, nere = metrics(H, H, B, k)
    assert np.allclose(sre, 0) and np.allclose(nere, 0)
    sre, _ = metrics(1.1 * H, H, B, k)
    assert np.allclose(sre, 0.1)
    Hp = H.copy(); Hp[:, :k] += 100                                   # the warm-up part is not scored
    assert np.allclose(metrics(Hp, H, B, k)[0], 0)
