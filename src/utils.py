import numpy as np


def make_echo(reference, delays_samples, amplitudes=None, total_len=None):
    delays_samples = list(delays_samples)
    if amplitudes is None:
        amplitudes = [1.0] * len(delays_samples)
    if total_len is None:
        total_len = max(delays_samples) + len(reference) + 10
    echo = np.zeros(total_len)
    for d, a in zip(delays_samples, amplitudes):
        echo[d:d + len(reference)] += a * reference
    return echo

def add_noise(signal, snr_db, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    p_signal = np.mean(signal**2)
    p_noise = p_signal / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=signal.shape)
    return signal + noise

def envelope(y):
    from scipy.signal import hilbert
    env = np.abs(hilbert(y))
    return env / (np.max(env) + 1e-12)

def envelope_db(y):
    env = envelope(y)
    return 20 * np.log10(env + 1e-12)

def mainlobe_width(y, drop_db=-6.0):
    edb = envelope_db(y)
    peak = np.argmax(edb)
    above = edb > drop_db
    r = peak
    while r < len(edb) - 1 and above[r + 1]:
        r += 1
    l = peak
    while l > 0 and above[l - 1]:
        l -= 1
    return r - l

def peak_sidelobe_db(y, drop_db=-6.0):
    edb = envelope_db(y)
    peak = np.argmax(edb)
    r = peak
    while r < len(edb) - 1 and edb[r + 1] <= edb[r]:
        r += 1
    l = peak
    while l > 0 and edb[l - 1] <= edb[l]:
        l -= 1
    masked = edb.copy()
    masked[l:r + 1] = -np.inf
    if np.all(np.isneginf(masked)):
        return -np.inf
    return np.max(masked)