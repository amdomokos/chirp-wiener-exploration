import numpy as np
from scipy.signal import windows

def linear_chirp(f0, f1, duration, fs, taper = 0.08):
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    k = 0.5 *(f1 - f0) / duration
    untapered_chirp = np.exp(2j*np.pi*(f0*t+k*(t**2)))
    tapered_chirp = windows.tukey(n, alpha=taper) * untapered_chirp
    return t, tapered_chirp

def spectrum(signal, fs):
    n = len(signal)
    f_signal = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mag = np.abs(f_signal)
    mag_db = 20 * np.log10(mag/np.max(mag) + 1e-12)
    return freqs, mag_db

def minus_3db_bandwidth(signal, fs):
    freqs, mag_db = spectrum(signal, fs)
    above = np.where(mag_db >= -3.0)[0]
    if len(above) < 2:
        return 0.0
    return freqs[above[-1]] - freqs[above[0]]