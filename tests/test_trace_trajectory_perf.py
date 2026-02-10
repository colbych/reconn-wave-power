"""
Performance comparison: old vs new trace_trajectory implementation.

The old implementation creates two RegularGridInterpolator objects per timestep
and uses xarray indexing in the inner loop. The new implementation uses pure
numpy with manual bilinear interpolation.

Run with:
    python -m pytest tests/test_trace_trajectory_perf.py -v -s
"""
import time
from typing import Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from reconn_wave_power.lagrangian import (
    compute_exb_velocity,
    trace_trajectory,
)


def _make_dataset(nx=128, ny=64, nt=200, dx=1.0, dy=1.0, dt=0.1):
    """Build a dataset with spatially-varying fields for a realistic benchmark."""
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    t = np.arange(nt) * dt

    rng = np.random.default_rng(42)
    shape = (nt, nx, ny)

    # Background Bz=1 plus small fluctuations; other components are small
    ds = xr.Dataset(
        {
            "Ex": (("time", "x", "y"), 0.1 * rng.standard_normal(shape)),
            "Ey": (("time", "x", "y"), 0.5 + 0.1 * rng.standard_normal(shape)),
            "Ez": (("time", "x", "y"), 0.1 * rng.standard_normal(shape)),
            "Bx": (("time", "x", "y"), 0.1 * rng.standard_normal(shape)),
            "By": (("time", "x", "y"), 0.1 * rng.standard_normal(shape)),
            "Bz": (("time", "x", "y"), 1.0 + 0.1 * rng.standard_normal(shape)),
        },
        coords={"time": t, "x": x, "y": y},
    )
    return ds


def _trace_trajectory_old(
    ds: xr.Dataset,
    x0: float,
    y0: float,
    t0_idx: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original implementation using RegularGridInterpolator per timestep."""
    x_vals = ds["x"].values
    y_vals = ds["y"].values
    times = ds["time"].values
    nt = len(times)

    x_min, x_max = float(x_vals[0]), float(x_vals[-1])
    y_min, y_max = float(y_vals[0]), float(y_vals[-1])

    def _in_domain(x, y):
        return x_min <= x <= x_max and y_min <= y <= y_max

    vx_all, vy_all = compute_exb_velocity(ds)

    def _interpolated_velocity(ti, x, y):
        vx_2d = vx_all.isel(time=ti).values
        vy_2d = vy_all.isel(time=ti).values
        interp_vx = RegularGridInterpolator(
            (x_vals, y_vals), vx_2d, method="linear",
            bounds_error=False, fill_value=None,
        )
        interp_vy = RegularGridInterpolator(
            (x_vals, y_vals), vy_2d, method="linear",
            bounds_error=False, fill_value=None,
        )
        pt = np.array([[x, y]])
        return float(interp_vx(pt)[0]), float(interp_vy(pt)[0])

    fwd_x, fwd_y, fwd_t = [x0], [y0], [float(times[t0_idx])]
    cx, cy = x0, y0
    for ti in range(t0_idx, nt - 1):
        vx_loc, vy_loc = _interpolated_velocity(ti, cx, cy)
        cx_new = cx + vx_loc * dt
        cy_new = cy + vy_loc * dt
        if not _in_domain(cx_new, cy_new):
            break
        cx, cy = cx_new, cy_new
        fwd_x.append(cx)
        fwd_y.append(cy)
        fwd_t.append(float(times[ti + 1]))

    bwd_x, bwd_y, bwd_t = [], [], []
    cx, cy = x0, y0
    for ti in range(t0_idx, 0, -1):
        vx_loc, vy_loc = _interpolated_velocity(ti, cx, cy)
        cx_new = cx - vx_loc * dt
        cy_new = cy - vy_loc * dt
        if not _in_domain(cx_new, cy_new):
            break
        cx, cy = cx_new, cy_new
        bwd_x.append(cx)
        bwd_y.append(cy)
        bwd_t.append(float(times[ti - 1]))

    bwd_x.reverse()
    bwd_y.reverse()
    bwd_t.reverse()

    return np.array(bwd_x + fwd_x), np.array(bwd_y + fwd_y), np.array(bwd_t + fwd_t)


class TestTraceTrajectoryPerformance:
    """Benchmark old vs new trace_trajectory and verify identical results."""

    def test_speedup(self):
        ds = _make_dataset(nx=128, ny=64, nt=200, dt=0.1)
        x0, y0, t0_idx, dt = 60.0, 30.0, 100, 0.1

        # Warm up (first call may include import overhead, etc.)
        trace_trajectory(ds, x0, y0, t0_idx, dt)

        # --- Time the old implementation ---
        n_runs = 3
        old_times = []
        for _ in range(n_runs):
            t_start = time.perf_counter()
            x_old, y_old, t_old = _trace_trajectory_old(ds, x0, y0, t0_idx, dt)
            old_times.append(time.perf_counter() - t_start)
        old_median = sorted(old_times)[n_runs // 2]

        # --- Time the new implementation ---
        new_times = []
        for _ in range(n_runs):
            t_start = time.perf_counter()
            x_new, y_new, t_new = trace_trajectory(ds, x0, y0, t0_idx, dt)
            new_times.append(time.perf_counter() - t_start)
        new_median = sorted(new_times)[n_runs // 2]

        # --- Verify identical results ---
        np.testing.assert_allclose(x_new, x_old, atol=1e-12,
                                   err_msg="x trajectories differ")
        np.testing.assert_allclose(y_new, y_old, atol=1e-12,
                                   err_msg="y trajectories differ")
        np.testing.assert_allclose(t_new, t_old, atol=1e-12,
                                   err_msg="t trajectories differ")

        speedup = old_median / new_median

        print(f"\n{'='*60}")
        print(f"  trace_trajectory performance comparison")
        print(f"  Grid: 128×64, 200 timesteps, start at t0_idx=100")
        print(f"  Trajectory length: {len(t_new)} points")
        print(f"{'='*60}")
        print(f"  Old (RegularGridInterpolator): {old_median*1000:8.1f} ms")
        print(f"  New (bilinear numpy):          {new_median*1000:8.1f} ms")
        print(f"  Speedup:                       {speedup:8.1f}x")
        print(f"{'='*60}")

        # The new implementation should be consistently faster
        assert speedup > 1.2, f"Expected >1.2x speedup, got {speedup:.1f}x"
