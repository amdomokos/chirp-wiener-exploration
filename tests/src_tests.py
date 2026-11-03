
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
import chirps, filters, utils
 
FS, F0, F1, DUR = 20e6, 1e6, 5e6, 20e-6
 
 
def _chirp():
    _, x = chirps.linear_chirp(F0, F1, DUR, FS)
    return np.real(x)
 
 
def test_chirp_sweeps_correct_band():
    """Frequency ramps upward and spans ~(f1-f0) -- catches the 1/2 chirp-rate bug."""
    from scipy.signal import hilbert
    _, x = chirps.linear_chirp(F0, F1, DUR, FS, taper=0.0)
    f = np.diff(np.unwrap(np.angle(hilbert(np.real(x))))) * FS / (2 * np.pi)
    n = len(f)
    f_lo = np.median(f[n // 4 - n // 20:n // 4 + n // 20])
    f_hi = np.median(f[3 * n // 4 - n // 20:3 * n // 4 + n // 20])
    assert f_hi > f_lo                                   # up-chirp
    assert np.isclose(f_hi - f_lo, 0.5 * (F1 - F0), rtol=0.2)
 
 
def test_bandwidth_matches_sweep():
    """-3 dB bandwidth is roughly the swept range."""
    x = _chirp()
    bw = chirps.minus_3db_bandwidth(x, FS)
    assert 0.6 * (F1 - F0) < bw < 1.4 * (F1 - F0)
 
 
def test_matched_filter_compresses_to_correct_delay():
    """A scatterer at a known delay compresses to a sharp peak there."""
    x = _chirp()
    echo = utils.make_echo(x, [500], [1.0])
    y = filters.matched_filter(echo, x)
    assert abs(np.argmax(np.abs(y)) - 500) < 20          # right location
    assert utils.mainlobe_width(y) < len(x) / 5          # actually compressed
 
 
def test_wiener_interpolates_matched_and_inverse():
    """Large gamma -> matched-like; small gamma -> sharper (toward inverse)."""
    x = _chirp()
    echo = utils.make_echo(x, [500], [1.0])
    w_mf = utils.mainlobe_width(filters.matched_filter(echo, x))
    w_big = utils.mainlobe_width(filters.wiener_filter(echo, x, gamma=1e9))
    w_small = utils.mainlobe_width(filters.wiener_filter(echo, x, gamma=1e-3))
    assert w_small <= w_mf                               # small gamma sharpens
    assert abs(w_big - w_mf) <= 2                        # large gamma ~ matched
 
 
def test_naive_inverse_is_noisier_than_matched():
    """The naive inverse amplifies noise more than the matched filter."""
    x = _chirp()
    echo = utils.add_noise(utils.make_echo(x, [500], [1.0]), 5, np.random.default_rng(0))
    def floor(y):
        y = np.abs(y) / (np.max(np.abs(y)) + 1e-12)
        p = np.argmax(y); m = np.ones(len(y), bool); m[max(0, p-50):p+50] = False
        return np.sqrt(np.mean(y[m]**2))
    assert floor(filters.naive_inverse_filter(echo, x)) > floor(filters.matched_filter(echo, x))
 
 
def test_utils_echo_and_noise():
    """make_echo places/superposes correctly; add_noise hits the target SNR."""
    ref = np.array([1.0, 1.0, 1.0])
    echo = utils.make_echo(ref, [0, 1], [1.0, 1.0], total_len=10)
    assert echo[1] == 2.0 and echo[0] == 1.0             # overlap adds (linearity)
 
    sig = np.sin(2 * np.pi * 0.01 * np.arange(100000))
    noisy = utils.add_noise(sig, 10.0, np.random.default_rng(42))
    realized = 10 * np.log10(np.mean(sig**2) / np.mean((noisy - sig)**2))
    assert abs(realized - 10.0) < 0.5
 
 
def test_sidelobe_measurement():
    """A 1/10-height sidelobe reads ~ -20 dB below the main lobe."""
    idx = np.arange(1000)
    y = np.exp(-((idx-500)**2)/128) + 0.1*np.exp(-((idx-700)**2)/128)
    assert utils.peak_sidelobe_db(y) < -15.0
 