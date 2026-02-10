"""
Performance comparison: three trace_trajectory implementations.

1. Original: RegularGridInterpolator constructed per timestep + xarray indexing
2. Full-grid numpy: pre-compute ExB over entire grid, bilinear interp in loop
3. Lazy local (current): compute ExB only at the 4 surrounding cells per step

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


# ---------------------------------------------------------------------------
# V1: Original (RegularGridInterpolator per timestep + xarray .isel)
# ---------------------------------------------------------------------------
def _trace_v1_scipy(ds, x0, y0, t0_idx, dt):
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
        fwd_x.append(cx); fwd_y.append(cy); fwd_t.append(float(times[ti + 1]))

    bwd_x, bwd_y, bwd_t = [], [], []
    cx, cy = x0, y0
    for ti in range(t0_idx, 0, -1):
        vx_loc, vy_loc = _interpolated_velocity(ti, cx, cy)
        cx_new = cx - vx_loc * dt
        cy_new = cy - vy_loc * dt
        if not _in_domain(cx_new, cy_new):
            break
        cx, cy = cx_new, cy_new
        bwd_x.append(cx); bwd_y.append(cy); bwd_t.append(float(times[ti - 1]))

    bwd_x.reverse(); bwd_y.reverse(); bwd_t.reverse()
    return np.array(bwd_x + fwd_x), np.array(bwd_y + fwd_y), np.array(bwd_t + fwd_t)


# ---------------------------------------------------------------------------
# V2: Full-grid numpy precompute + bilinear interp
# ---------------------------------------------------------------------------
def _trace_v2_fullgrid(ds, x0, y0, t0_idx, dt):
    x_vals = ds["x"].values
    y_vals = ds["y"].values
    times = ds["time"].values
    nt = len(times)
    x_min, x_max = float(x_vals[0]), float(x_vals[-1])
    y_min, y_max = float(y_vals[0]), float(y_vals[-1])

    def _in_domain(x, y):
        return x_min <= x <= x_max and y_min <= y <= y_max

    Ex = ds["Ex"].values; Ey = ds["Ey"].values; Ez = ds["Ez"].values
    Bx = ds["Bx"].values; By = ds["By"].values; Bz = ds["Bz"].values
    B2 = Bx**2 + By**2 + Bz**2
    vx_all = (Ey * Bz - Ez * By) / B2
    vy_all = (Ez * Bx - Ex * Bz) / B2

    grid_dx = float(x_vals[1] - x_vals[0])
    grid_dy = float(y_vals[1] - y_vals[0])
    grid_x0 = float(x_vals[0])
    grid_y0 = float(y_vals[0])
    gnx, gny = len(x_vals), len(y_vals)

    def _bilinear(field, x, y):
        fi = max(0.0, min((x - grid_x0) / grid_dx, gnx - 1.0))
        fj = max(0.0, min((y - grid_y0) / grid_dy, gny - 1.0))
        i0 = min(int(fi), gnx - 2); j0 = min(int(fj), gny - 2)
        wi = fi - i0; wj = fj - j0
        return (field[i0, j0] * (1 - wi) * (1 - wj) +
                field[i0+1, j0] * wi * (1 - wj) +
                field[i0, j0+1] * (1 - wi) * wj +
                field[i0+1, j0+1] * wi * wj)

    def _interpolated_velocity(ti, x, y):
        return float(_bilinear(vx_all[ti], x, y)), float(_bilinear(vy_all[ti], x, y))

    fwd_x, fwd_y, fwd_t = [x0], [y0], [float(times[t0_idx])]
    cx, cy = x0, y0
    for ti in range(t0_idx, nt - 1):
        vx_loc, vy_loc = _interpolated_velocity(ti, cx, cy)
        cx_new = cx + vx_loc * dt
        cy_new = cy + vy_loc * dt
        if not _in_domain(cx_new, cy_new):
            break
        cx, cy = cx_new, cy_new
        fwd_x.append(cx); fwd_y.append(cy); fwd_t.append(float(times[ti + 1]))

    bwd_x, bwd_y, bwd_t = [], [], []
    cx, cy = x0, y0
    for ti in range(t0_idx, 0, -1):
        vx_loc, vy_loc = _interpolated_velocity(ti, cx, cy)
        cx_new = cx - vx_loc * dt
        cy_new = cy - vy_loc * dt
        if not _in_domain(cx_new, cy_new):
            break
        cx, cy = cx_new, cy_new
        bwd_x.append(cx); bwd_y.append(cy); bwd_t.append(float(times[ti - 1]))

    bwd_x.reverse(); bwd_y.reverse(); bwd_t.reverse()
    return np.array(bwd_x + fwd_x), np.array(bwd_y + fwd_y), np.array(bwd_t + fwd_t)


# ---------------------------------------------------------------------------
# V3: Current implementation (lazy local ExB, imported trace_trajectory)
# ---------------------------------------------------------------------------
# Already imported as `trace_trajectory`


def _time_fn(fn, *args, n_runs=3):
    """Return median wall-clock time over n_runs."""
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    median = sorted(times)[n_runs // 2]
    return median, result


class TestTraceTrajectoryPerformance:
    """Benchmark three implementations and verify identical results."""

    def test_speedup_small_grid(self):
        """128x64 grid, 200 timesteps — the loop dominates."""
        ds = _make_dataset(nx=128, ny=64, nt=200, dt=0.1)
        x0, y0, t0_idx, dt = 60.0, 30.0, 100, 0.1

        # Warm up
        trace_trajectory(ds, x0, y0, t0_idx, dt)

        t_v1, (x1, y1, t1) = _time_fn(_trace_v1_scipy, ds, x0, y0, t0_idx, dt)
        t_v2, (x2, y2, t2) = _time_fn(_trace_v2_fullgrid, ds, x0, y0, t0_idx, dt)
        t_v3, (x3, y3, t3) = _time_fn(trace_trajectory, ds, x0, y0, t0_idx, dt)

        # All three must produce the same trajectory
        np.testing.assert_allclose(x3, x1, atol=1e-12)
        np.testing.assert_allclose(y3, y1, atol=1e-12)
        np.testing.assert_allclose(t3, t1, atol=1e-12)
        np.testing.assert_allclose(x3, x2, atol=1e-12)
        np.testing.assert_allclose(y3, y2, atol=1e-12)

        print(f"\n{'='*60}")
        print(f"  Small grid: 128x64, 200 timesteps")
        print(f"  Trajectory length: {len(t3)} points")
        print(f"{'='*60}")
        print(f"  V1 scipy/xarray:       {t_v1*1000:8.1f} ms")
        print(f"  V2 full-grid numpy:    {t_v2*1000:8.1f} ms")
        print(f"  V3 lazy local (curr):  {t_v3*1000:8.1f} ms")
        print(f"  V1->V3 speedup:        {t_v1/t_v3:8.1f}x")
        print(f"  V2->V3 speedup:        {t_v2/t_v3:8.1f}x")
        print(f"{'='*60}")

        assert t_v3 < t_v1, "V3 should be faster than V1"

    def test_speedup_large_grid(self):
        """800x200 grid, 500 timesteps — simulates real simulation scale.

        At this size the full-grid ExB precomputation is expensive,
        so V3 (lazy local) should show a large advantage over V2.
        """
        ds = _make_dataset(nx=800, ny=200, nt=500, dt=0.1)
        x0, y0, t0_idx, dt = 400.0, 100.0, 250, 0.1

        # Warm up
        trace_trajectory(ds, x0, y0, t0_idx, dt)

        # Skip V1 on the large grid — it's very slow
        t_v2, (x2, y2, t2) = _time_fn(_trace_v2_fullgrid, ds, x0, y0, t0_idx, dt)
        t_v3, (x3, y3, t3) = _time_fn(trace_trajectory, ds, x0, y0, t0_idx, dt)

        # Results must match
        np.testing.assert_allclose(x3, x2, atol=1e-12)
        np.testing.assert_allclose(y3, y2, atol=1e-12)
        np.testing.assert_allclose(t3, t2, atol=1e-12)

        speedup = t_v2 / t_v3

        print(f"\n{'='*60}")
        print(f"  Large grid: 800x200, 500 timesteps")
        print(f"  Trajectory length: {len(t3)} points")
        print(f"{'='*60}")
        print(f"  V2 full-grid numpy:    {t_v2*1000:8.1f} ms")
        print(f"  V3 lazy local (curr):  {t_v3*1000:8.1f} ms")
        print(f"  Speedup:               {speedup:8.1f}x")
        print(f"{'='*60}")

        assert speedup > 2.0, f"Expected >2x speedup on large grid, got {speedup:.1f}x"
