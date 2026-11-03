import numpy as np

"""recreates the three pulse-compression filters introduced in Oelze (2007)"""


def _fft_pair(echo, reference):
    n = len(echo) + len(reference)
    nfft = 1 << (n-1).bit_length()
    E = np.fft.fft(echo, nfft)
    V = np.fft.fft(reference, nfft)
    return E, V, nfft

def matched_filter(echo, reference):
    E, V, nfft = _fft_pair(echo, reference)
    Y = E * np.conj(V) # autocorrelation
    y = np.fft.ifft(Y)
    return np.real(y)

def naive_inverse_filter(echo, reference):
    E, V, nfft = _fft_pair(echo, reference)
    Y = E / (V + 1e-12) # inverse filtering
    y = np.fft.ifft(Y)
    return np.real(y)

def wiener_filter(echo, reference, gamma=1.0, esnr=None):
    
    E, V, nfft = _fft_pair(echo, reference)
    if esnr is None:
        inv_esnr = 1.0
    else:
        inv_esnr = 1.0 / np.asarray(esnr)

    Y = (E * np.conj(V)) / (np.abs(V)**2 + gamma * inv_esnr) # wiener
    y = np.fft.ifft(Y)
    return np.real(y)


    
