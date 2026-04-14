"""
FFT and spectral routines for time and space PSDs and simple k-omega 2D spectra.

APIs:
- compute_psd_time(da, dt, method='welch', **kwargs) -> frequencies, psd
- compute_psd_space(da, dx, axis='x', **kwargs) -> wavenumbers, psd_k
- compute_komega_2d(data, dx, dt, axes=('x','time')) -> k, omega, Pkomega
"""
from typing import Tuple, Optional, Sequence
import numpy as np
import xarray as xr
from scipy import signal


def compute_psd_time(
    da: xr.DataArray,
    dt: float,
    method: str = "welch",
    nperseg: Optional[int] = None,
    window: str = "hann",
    detrend: bool = True,
    axis: str = "time",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D PSD over the time axis for an xarray.DataArray.

    Parameters:
      da: DataArray with the time axis present
      dt: temporal spacing
      method: "welch" for Welch's method (segment-averaged), or "fft" for a
              single windowed FFT (better frequency resolution, noisier).
      nperseg: samples per segment (Welch only; ignored for "fft")
      window: window function name (applied in both methods)
      detrend: if True, subtract the mean along the axis before computing
      axis: name of the time dimension

    Returns:
      f: frequency bins [Hz] (one-sided, length n//2+1)
      Pxx: PSD values with shape matching da but with `axis` replaced by
           frequency axis length
    """
    if axis not in da.dims:
        raise ValueError(f"Axis '{axis}' not in DataArray dims: {da.dims}")

    arr = da
    if detrend:
        arr = arr - arr.mean(dim=axis)

    data = arr.values
    axis_num = arr.get_axis_num(axis)
    fs = 1.0 / dt

    if method == "welch":
        f, Pxx = signal.welch(data, fs=fs, window=window, nperseg=nperseg, axis=axis_num)
    elif method == "fft":
        n = data.shape[axis_num]
        # Apply window
        w = signal.get_window(window, n)
        shape = [1] * data.ndim
        shape[axis_num] = n
        data = data * w.reshape(tuple(shape))
        # One-sided rfft
        fft_vals = np.fft.rfft(data, axis=axis_num)
        f = np.fft.rfftfreq(n, d=dt)
        # PSD: |FFT|^2 normalized by fs*N (consistent with Welch scaling)
        Pxx = (np.abs(fft_vals) ** 2) / (fs * n)
        # Double the one-sided bins (except DC and Nyquist) to conserve power
        slices = [slice(None)] * data.ndim
        slices[axis_num] = slice(1, -1)
        Pxx[tuple(slices)] *= 2.0
    else:
        raise ValueError(f"method must be 'welch' or 'fft', got '{method}'")

    return f, Pxx


def compute_psd_space(
    da: xr.DataArray,
    dx: float,
    axis: str = "x",
    window: Optional[str] = "hann",
    return_k_cycles: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial PSD (using FFT) across a spatial axis.

    Parameters:
      da: DataArray with the spatial axis present
      dx: spacing in physical units (meters)
      axis: name of spatial axis e.g., 'x'
      window: window name to apply before FFT (None for no window)
      return_k_cycles: if True return cycles/m instead of angular wavenumber.

    Returns:
      k: wavenumber array (rad/m by default unless return_k_cycles=True)
      Pk: power spectral density vs k (one-sided for real input)
    """
    if axis not in da.dims:
        raise ValueError(f"Axis '{axis}' not in DataArray dims: {da.dims}")

    data = da.values
    axis_num = da.get_axis_num(axis)
    n = data.shape[axis_num]

    # Apply window
    if window:
        w = signal.get_window(window, n)
        # Broadcast window to data shape
        shape = [1] * data.ndim
        shape[axis_num] = n
        w_shaped = w.reshape(tuple(shape))
        data_win = data * w_shaped
    else:
        data_win = data

    # Compute FFT along axis
    # Use rfft for real-valued signals -> returns n//2+1 points
    fft = np.fft.rfft(data_win, axis=axis_num)
    # Frequency bins in cycles per length: freq_index / L where L = n*dx
    L = n * dx
    k_cycles = np.fft.rfftfreq(n, d=dx)  # cycles per unit length (1/m)
    k_rad = 2.0 * np.pi * k_cycles  # rad/m

    # Power spectral density: |FFT|^2 normalized by length
    # Normalize so Parseval theorem approximately holds: (1/L) * |FFT|^2
    Pk = (np.abs(fft) ** 2) / L

    if return_k_cycles:
        return k_cycles, Pk
    return k_rad, Pk


def compute_komega_2d(
    data: np.ndarray,
    dx: float,
    dt: float,
    axes: Sequence[int] = (0, 1),
    window_time: Optional[str] = "hann",
    window_space: Optional[str] = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a simple 2D k-omega spectrum from a 2D array where axes specify [space_axis, time_axis]
    or vice versa. This is a generic helper operating on a NumPy array.

    Parameters:
      data: ndarray with at least 2 dims
      dx: spatial spacing
      dt: temporal spacing
      axes: tuple (space_axis_index, time_axis_index)
    Returns:
      k (rad/m), omega (rad/s), P (2D ndarray with shape (len(k), len(omega)))
    """
    sa, ta = axes
    # move axes to (space, time)
    arr = np.moveaxis(data, (sa, ta), (0, 1))
    ns, nt = arr.shape[0], arr.shape[1]

    # apply windows
    if window_space:
        ws = signal.get_window(window_space, ns).reshape((ns, 1))
    else:
        ws = np.ones((ns, 1))
    if window_time:
        wt = signal.get_window(window_time, nt).reshape((1, nt))
    else:
        wt = np.ones((1, nt))

    arr_win = arr * ws * wt

    # 2D FFT: space (k) and time (f)
    F2 = np.fft.fft2(arr_win)
    # shift zero-frequency component to center for visualization if desired
    F2_shift = np.fft.fftshift(F2, axes=(0, 1))

    # k and omega axes
    k_cycles = np.fft.fftfreq(ns, d=dx)
    k_rad = 2.0 * np.pi * k_cycles
    k_rad = np.fft.fftshift(k_rad)

    f = np.fft.fftfreq(nt, d=dt)
    omega = 2.0 * np.pi * f
    omega = np.fft.fftshift(omega)

    P = np.abs(F2_shift) ** 2 / (ns * nt)

    return k_rad, omega, P


def compute_fft_map(
    da: xr.DataArray,
    dt: float,
    window: str = "hann",
    detrend: bool = True,
) -> xr.DataArray:
    """Compute a per-pixel complex FFT over a 2-D spatial subregion.

    For every (x, y) point in *da*, the time-series is windowed and
    FFT'd, returning the complex one-sided spectrum.  Storing the complex
    amplitudes (rather than squaring to get a PSD) preserves phase, which
    is required to compute cross-spectral quantities such as the Poynting
    flux S = Re(E × B*).

    Normalisation
    -------------
    The stored coefficients are scaled by ``1 / sqrt(fs * n)`` so that
    ``|fft_map|**2`` equals the one-sided PSD *before* the factor-of-2
    correction applied in :func:`compute_psd_time`.  The cross-spectral
    Poynting flux is then::

        Sx = Re(fft_Ey * conj(fft_Bz) - fft_Ez * conj(fft_By))

    and has units consistent with the auto-PSDs in the companion
    ``psd_`` files (up to the one-sided factor of 2 on interior bins).

    Parameters
    ----------
    da : xr.DataArray
        Must have dimensions ``("time", "x", "y")``.
    dt : float
        Temporal spacing between samples (``dt_output = dt * ndump``).
    window : str
        Window function name (e.g. ``"hann"``).
    detrend : bool
        If True, subtract the mean along the time axis before computing.

    Returns
    -------
    xr.DataArray
        Complex128, dimensions ``("x", "y", "frequency")``.
        Coordinates ``x``, ``y`` are inherited from *da*;
        ``frequency`` is the one-sided rfft frequency array in Hz.
    """
    for dim in ("time", "x", "y"):
        if dim not in da.dims:
            raise ValueError(
                f"DataArray must have dim '{dim}', got dims {tuple(da.dims)}"
            )

    arr = da
    if detrend:
        arr = arr - arr.mean(dim="time")

    data = arr.values                          # shape: (n_time, nx, ny)
    axis_num = arr.get_axis_num("time")
    n = data.shape[axis_num]
    fs = 1.0 / dt

    # Apply window along time axis
    w = signal.get_window(window, n)
    shape = [1] * data.ndim
    shape[axis_num] = n
    data = data * w.reshape(tuple(shape))

    # One-sided complex FFT; normalise so |fft|^2 ~ psd (pre one-sided factor)
    fft_vals = np.fft.rfft(data, axis=axis_num)
    fft_vals = fft_vals / np.sqrt(fs * n)

    f = np.fft.rfftfreq(n, d=dt)

    # Move frequency axis from position 0 → last: (n_freq, nx, ny) → (nx, ny, n_freq)
    fft_map = np.moveaxis(fft_vals, axis_num, -1)

    return xr.DataArray(
        fft_map,
        dims=("x", "y", "frequency"),
        coords={
            "x": da.coords["x"],
            "y": da.coords["y"],
            "frequency": f,
        },
    )


def compute_psd_map(
    da: xr.DataArray,
    dt: float,
    method: str = "welch",
    nperseg: Optional[int] = None,
    window: str = "hann",
    detrend: bool = True,
) -> xr.DataArray:
    """Compute a per-pixel frequency PSD over a 2-D spatial subregion.

    For every (x, y) point in *da*, the time-series is extracted and a
    frequency PSD is computed, returning a 2-D spatial map of spectra.

    Parameters
    ----------
    da : xr.DataArray
        Must have dimensions ``("time", "x", "y")``.
    dt : float
        Temporal spacing between samples.
    method : str
        ``"welch"`` or ``"fft"``; forwarded to :func:`compute_psd_time`.
    nperseg : int, optional
        Welch segment length; ignored for ``"fft"``.
    window : str
        Window function name (e.g. ``"hann"``).
    detrend : bool
        If True, subtract the mean along the time axis before computing.

    Returns
    -------
    xr.DataArray
        Dimensions ``("x", "y", "frequency")``.  Each pixel contains the PSD
        of the corresponding time series.  Coordinates ``x``, ``y`` are
        inherited from *da*; ``frequency`` is in Hz.
    """
    for dim in ("time", "x", "y"):
        if dim not in da.dims:
            raise ValueError(
                f"DataArray must have dim '{dim}', got dims {tuple(da.dims)}"
            )

    f, Pxx = compute_psd_time(
        da,
        dt=dt,
        method=method,
        nperseg=nperseg,
        window=window,
        detrend=detrend,
        axis="time",
    )
    # compute_psd_time returns Pxx with shape (n_freq, nx, ny)
    # (time axis replaced by frequency axis at position 0).
    # Move frequency to the last axis → (nx, ny, n_freq).
    pxx_map = np.moveaxis(Pxx, 0, -1)

    return xr.DataArray(
        pxx_map,
        dims=("x", "y", "frequency"),
        coords={
            "x": da.coords["x"],
            "y": da.coords["y"],
            "frequency": f,
        },
    )