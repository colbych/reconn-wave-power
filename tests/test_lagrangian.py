import numpy as np
import xarray as xr
import pytest

from reconn_wave_power.lagrangian import (
    compute_exb_velocity,
    trace_trajectory,
    sample_along_trajectory,
    lagrangian_psd,
)


def _uniform_eb_dataset(
    nx: int = 32,
    ny: int = 16,
    nt: int = 20,
    dx: float = 1.0,
    dy: float = 1.0,
    dt: float = 0.1,
    Bz0: float = 1.0,
    Ey0: float = 0.5,
) -> xr.Dataset:
    """
    Build a synthetic dataset with a uniform magnetic field along z
    and a uniform electric field along y.

    This gives a uniform E×B drift in the x-direction:
        vx = Ey * Bz / |B|^2 = Ey0 / Bz0
        vy = 0
    """
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    t = np.arange(nt) * dt

    zeros = np.zeros((nt, nx, ny))
    bz_arr = np.full((nt, nx, ny), Bz0)
    ey_arr = np.full((nt, nx, ny), Ey0)

    ds = xr.Dataset(
        {
            "Ex": (("time", "x", "y"), zeros.copy()),
            "Ey": (("time", "x", "y"), ey_arr),
            "Ez": (("time", "x", "y"), zeros.copy()),
            "Bx": (("time", "x", "y"), zeros.copy()),
            "By": (("time", "x", "y"), zeros.copy()),
            "Bz": (("time", "x", "y"), bz_arr),
        },
        coords={"time": t, "x": x, "y": y},
    )
    return ds


class TestComputeExbVelocity:
    def test_uniform_field(self):
        ds = _uniform_eb_dataset(Bz0=2.0, Ey0=1.0)
        vx, vy = compute_exb_velocity(ds)
        # vx = Ey*Bz / |B|^2 = 1.0*2.0 / 4.0 = 0.5
        np.testing.assert_allclose(vx.values, 0.5, atol=1e-12)
        np.testing.assert_allclose(vy.values, 0.0, atol=1e-12)

    def test_single_timestep(self):
        ds = _uniform_eb_dataset()
        vx, vy = compute_exb_velocity(ds, time_idx=0)
        assert "time" not in vx.dims

    def test_missing_field_raises(self):
        ds = _uniform_eb_dataset().drop_vars("Bz")
        with pytest.raises(KeyError, match="Bz"):
            compute_exb_velocity(ds)


class TestTraceTrajectory:
    def test_straight_line_drift(self):
        """Uniform E×B drift should produce a straight-line trajectory in x."""
        Bz0, Ey0, dt = 1.0, 0.5, 0.1
        vx_expected = Ey0 / Bz0  # = 0.5

        ds = _uniform_eb_dataset(nx=64, nt=30, dt=dt, Bz0=Bz0, Ey0=Ey0)
        x0, y0 = 10.0, 5.0
        t0_idx = 10

        x_traj, y_traj, t_traj = trace_trajectory(ds, x0, y0, t0_idx, dt)

        # y should stay constant
        np.testing.assert_allclose(y_traj, y0, atol=1e-10)

        # x should advance linearly at vx_expected
        t0 = ds["time"].values[t0_idx]
        expected_x = x0 + vx_expected * (t_traj - t0)
        np.testing.assert_allclose(x_traj, expected_x, atol=1e-10)

    def test_dt_mismatch_uses_actual_time_spacing(self):
        """Trajectory should use actual time-coordinate spacing, not a user-supplied dt.

        Reproduces the real-data bug: output cadence (spacing=2.0) differs
        from simulation internal dt (0.05).  The Euler step must use the
        output cadence, otherwise the displacement is ~40x too small.
        """
        Bz0, Ey0 = 1.0, 1.0
        vx_expected = Ey0 / Bz0  # = 1.0

        nx, ny = 256, 16
        # Output cadence of 2.0 — completely different from internal dt
        time_coords = np.arange(0, 20, 2.0)  # [0, 2, 4, ..., 18]
        nt = len(time_coords)

        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)
        zeros = np.zeros((nt, nx, ny))

        ds = xr.Dataset(
            {
                "Ex": (("time", "x", "y"), zeros.copy()),
                "Ey": (("time", "x", "y"), np.full((nt, nx, ny), Ey0)),
                "Ez": (("time", "x", "y"), zeros.copy()),
                "Bx": (("time", "x", "y"), zeros.copy()),
                "By": (("time", "x", "y"), zeros.copy()),
                "Bz": (("time", "x", "y"), np.full((nt, nx, ny), Bz0)),
            },
            coords={"time": time_coords, "x": x, "y": y},
        )

        x0, y0 = 100.0, 5.0
        t0_idx = 0

        # Pass a wrong dt (simulation internal timestep) — should be ignored
        x_traj, y_traj, t_traj = trace_trajectory(ds, x0, y0, t0_idx, dt=0.05)

        # x should advance by vx_expected * elapsed_time
        expected_x = x0 + vx_expected * (t_traj - t_traj[0])
        np.testing.assert_allclose(x_traj, expected_x, atol=1e-10)

    def test_stops_at_boundary(self):
        """Trajectory should stop when it exits the domain."""
        ds = _uniform_eb_dataset(nx=16, nt=100, dx=1.0, dt=0.1, Bz0=1.0, Ey0=2.0)
        # vx = 2.0, domain x in [0, 15] => from x0=7 forward reaches boundary quickly
        x_traj, y_traj, t_traj = trace_trajectory(ds, 7.0, 5.0, 0, 0.1)
        assert len(t_traj) < 100
        assert np.all(x_traj >= 0) and np.all(x_traj <= 15.0)


class TestSampleAlongTrajectory:
    def test_sample_known_field(self):
        ds = _uniform_eb_dataset(nt=10, dt=0.1)
        # Bz is uniform = 1.0 everywhere
        x_traj = np.array([5.0, 5.5, 6.0])
        y_traj = np.array([3.0, 3.0, 3.0])
        t_traj = np.array([0.0, 0.1, 0.2])
        ts = sample_along_trajectory(ds, "Bz", x_traj, y_traj, t_traj)
        np.testing.assert_allclose(ts.values, 1.0, atol=1e-10)

    def test_missing_field_raises(self):
        ds = _uniform_eb_dataset()
        with pytest.raises(KeyError, match="Foo"):
            sample_along_trajectory(ds, "Foo", np.array([0.0]), np.array([0.0]), np.array([0.0]))


class TestLagrangianPsd:
    def test_returns_correct_shapes(self):
        ds = _uniform_eb_dataset(nx=64, ny=16, nt=50, dt=0.1, Bz0=1.0, Ey0=0.2)
        f, Pxx, traj = lagrangian_psd(ds, "Bz", 20.0, 5.0, 25, dt=0.1, method="fft")
        assert len(f) == len(Pxx)
        assert "x_traj" in traj and "y_traj" in traj and "t_traj" in traj
        assert len(traj["x_traj"]) == len(traj["t_traj"])
