"""Integration tests using real simulation data."""
import pytest
import os

# Path to test simulation data (outside the repo)
TEST_DATA_PATH = "/Users/colby/Research/Projects/My_Projects/2026.Reconn_Wave_Power/test_sim_data/test00"
INPUT_FILE = os.path.join(TEST_DATA_PATH, "input", "input")
OUTPUT_FOLDER = os.path.join(TEST_DATA_PATH, "Output", "Fields")

# Skip all tests if dhybridrpy is not installed or test data is missing
pytestmark = pytest.mark.skipif(
    not os.path.exists(TEST_DATA_PATH),
    reason="Test simulation data not found"
)

try:
    from dhybridrpy import DHybridrpy
    HAS_DHYBRIDRPY = True
except ImportError:
    HAS_DHYBRIDRPY = False


@pytest.mark.skipif(not HAS_DHYBRIDRPY, reason="dhybridrpy not installed")
def test_read_magnetic_field_last_timestep():
    """Read magnetic field data at the last timestep."""
    from reconn_wave_power.io import read_simulation

    # Initialize dhybridrpy to get available timesteps
    dpy = DHybridrpy(input_file=INPUT_FILE, output_folder=OUTPUT_FOLDER, lazy=True)
    all_timesteps = dpy.timesteps()
    last_timestep = max(all_timesteps)

    # Read only the magnetic field at the last timestep
    ds = read_simulation(
        dpy,
        timesteps=[last_timestep],
        fields=("B",),
    )

    # Verify all magnetic field components are present
    assert "Bx" in ds
    assert "By" in ds
    assert "Bz" in ds

    # Verify time coordinate has exactly one timestep
    assert ds["Bx"].coords["time"].size == 1

    # Verify spatial coordinates exist
    assert "x" in ds["Bx"].coords

    # Verify data is not empty
    assert ds["Bx"].size > 0
    assert ds["By"].size > 0
    assert ds["Bz"].size > 0
