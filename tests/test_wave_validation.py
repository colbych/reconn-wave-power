"""
Tests using synthetic Alfven wave to validate FFT methods.

Alfven dispersion relation: omega = k * v_A
Wave signal: B_z(x, t) = B0 * sin(k0*x - omega0*t)
"""
import numpy as np
import xarray as xr
import pytest

from reconn_wave_power.spectrum import compute_psd_time, compute_psd_space, compute_komega_2d


# Wave parameters
V_A = 1.0  # Normalized Alfven speed
WAVELENGTH = 10.0  # Grid units
K0 = 2 * np.pi / WAVELENGTH  # Wavenumber (rad/unit)
OMEGA0 = K0 * V_A  # Angular frequency (rad/unit time)
F0 = OMEGA0 / (2 * np.pi)  # Frequency in cycles
B0 = 1.0  # Amplitude


def generate_alfven_wave(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Generate B_z(x, t) = B0 * sin(k0*x - omega0*t) on a 2D grid.

    Returns array with shape (len(x), len(t)).
    """
    X, T = np.meshgrid(x, t, indexing='ij')
    return B0 * np.sin(K0 * X - OMEGA0 * T)


def test_alfven_wave_time_spectrum():
    """Verify time PSD recovers omega0 from time series at fixed x."""
    # Grid setup
    dt = 0.1
    t = np.arange(0, 100.0, dt)  # Long enough for good frequency resolution
    x = np.arange(0, 50.0, 1.0)

    # Generate wave
    Bz = generate_alfven_wave(x, t)

    # Extract time series at fixed x (middle of domain)
    x_idx = len(x) // 2
    time_series = Bz[x_idx, :]

    # Create DataArray
    da = xr.DataArray(time_series, dims=("time",), coords={"time": t})

    # Compute time PSD
    f, Pxx = compute_psd_time(da, dt=dt, nperseg=256)

    # Find peak frequency
    peak_idx = np.argmax(Pxx)
    peak_freq = f[peak_idx]

    # Check peak matches expected f0 within tolerance
    # Frequency resolution is roughly 1/(N*dt), allow some tolerance
    freq_resolution = 1.0 / (len(t) * dt)
    tolerance = 3 * freq_resolution

    assert abs(peak_freq - F0) < tolerance, \
        f"Peak frequency {peak_freq:.4f} does not match expected {F0:.4f} (tolerance {tolerance:.4f})"


def test_alfven_wave_space_spectrum():
    """Verify spatial PSD recovers k0 from spatial profile at fixed t."""
    # Grid setup
    dx = 0.5
    x = np.arange(0, 100.0, dx)  # Long enough for good k resolution
    t = np.arange(0, 50.0, 1.0)

    # Generate wave
    Bz = generate_alfven_wave(x, t)

    # Extract spatial profile at fixed t (middle of time range)
    t_idx = len(t) // 2
    spatial_profile = Bz[:, t_idx]

    # Create DataArray
    da = xr.DataArray(spatial_profile, dims=("x",), coords={"x": x})

    # Compute spatial PSD (returns k in rad/unit)
    k_rad, Pk = compute_psd_space(da, dx=dx, axis="x", window=None)

    # Find peak in positive k
    pos_mask = k_rad > 0
    Pk_pos = Pk[pos_mask]
    k_pos = k_rad[pos_mask]

    peak_idx = np.argmax(Pk_pos)
    peak_k = k_pos[peak_idx]

    # Check peak matches expected k0 within tolerance
    # k resolution is roughly 2*pi/(N*dx)
    k_resolution = 2 * np.pi / (len(x) * dx)
    tolerance = 3 * k_resolution

    assert abs(peak_k - K0) < tolerance, \
        f"Peak wavenumber {peak_k:.4f} does not match expected {K0:.4f} (tolerance {tolerance:.4f})"


def test_alfven_wave_komega():
    """Verify 2D k-omega spectrum shows power along omega = k * v_A."""
    # Grid setup - need good resolution in both dimensions
    dx = 0.5
    dt = 0.1
    x = np.arange(0, 100.0, dx)
    t = np.arange(0, 100.0, dt)

    # Generate wave
    Bz = generate_alfven_wave(x, t)

    # Compute k-omega spectrum
    # axes=(0, 1) means axis 0 is space, axis 1 is time
    k_rad, omega, P = compute_komega_2d(Bz, dx=dx, dt=dt, axes=(0, 1))

    # The wave sin(k0*x - omega0*t) produces peaks at (+k0, -omega0) and (-k0, +omega0)
    # due to FFT sign conventions. Find the global peak and check |k| and |omega|.
    peak_idx = np.unravel_index(np.argmax(P), P.shape)

    peak_k = k_rad[peak_idx[0]]
    peak_omega = omega[peak_idx[1]]

    # Check both |k| and |omega| match expected values
    k_resolution = 2 * np.pi / (len(x) * dx)
    omega_resolution = 2 * np.pi / (len(t) * dt)

    k_tolerance = 3 * k_resolution
    omega_tolerance = 3 * omega_resolution

    assert abs(abs(peak_k) - K0) < k_tolerance, \
        f"Peak |k| {abs(peak_k):.4f} does not match expected {K0:.4f} (tolerance {k_tolerance:.4f})"

    assert abs(abs(peak_omega) - OMEGA0) < omega_tolerance, \
        f"Peak |omega| {abs(peak_omega):.4f} does not match expected {OMEGA0:.4f} (tolerance {omega_tolerance:.4f})"

    # Verify dispersion relation: |omega| = |k| * v_A
    recovered_va = abs(peak_omega) / abs(peak_k)
    va_tolerance = 0.1 * V_A  # 10% tolerance

    assert abs(recovered_va - V_A) < va_tolerance, \
        f"Recovered v_A {recovered_va:.4f} does not match expected {V_A:.4f}"
