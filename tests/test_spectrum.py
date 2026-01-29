import numpy as np
import xarray as xr
from reconn_wave_power.spectrum import compute_psd_time, compute_psd_space

def test_compute_psd_time_sine():
    # Sine wave at f0 = 5 Hz
    f0 = 5.0
    dt = 0.001
    t = np.arange(0, 2.0, dt)
    signal_data = np.sin(2 * np.pi * f0 * t)
    da = xr.DataArray(signal_data, dims=("time",), coords={"time": t})
    f, Pxx = compute_psd_time(da, dt=dt, nperseg=1024)
    peak_idx = np.argmax(Pxx)
    peak_freq = f[peak_idx]
    assert abs(peak_freq - f0) < 0.5  # allow tolerance

def test_compute_psd_space_sine():
    # Spatial sine with wavelength lambda = 2.0 -> k_cycles = 1/lambda
    lam = 2.0
    dx = 0.1
    x = np.arange(0, 20.0, dx)
    k_cycles_true = 1.0 / lam
    data = np.sin(2 * np.pi * k_cycles_true * x)
    da = xr.DataArray(data, dims=("x",), coords={"x": x})
    k_rad, Pk = compute_psd_space(da, dx=dx, axis="x", window=None)
    # choose peak in k_cycles -> convert first positive k_rad to cycles
    k_cycles = k_rad / (2.0 * np.pi)
    # find positive frequencies
    pos_mask = k_cycles > 0
    if not np.any(pos_mask):
        raise AssertionError("No positive k found")
    # average power across returned shape (Pk may be 1D or have extra dims)
    Pk_mean = Pk
    if Pk_mean.ndim > 1:
        Pk_mean = Pk_mean.reshape((Pk_mean.shape[0], -1)).mean(axis=1)
    peak_idx = np.argmax(Pk_mean[pos_mask])
    peak_k = k_cycles[pos_mask][peak_idx]
    assert abs(peak_k - k_cycles_true) / k_cycles_true < 0.1