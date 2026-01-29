import sys
import types
import numpy as np
import xarray as xr

# Inject a fake dhybridrpy module before importing the io module
fake = types.ModuleType("dhybridrpy")

class FakeField:
    def __init__(self, name, data, xdata=None, ydata=None, time=None, timestep=None, ftype="Total"):
        self.name = name
        self.data = data
        self.xdata = xdata
        self.ydata = ydata
        self.time = time
        self.timestep = timestep
        self.type = ftype

class FakeFieldsAPI:
    def __init__(self, fields):
        self._fields = fields
    def Bx(self, type="Total"):
        return self._fields["Bx"]
    def By(self, type="Total"):
        return self._fields["By"]
    def Bz(self, type="Total"):
        return self._fields["Bz"]
    def Ex(self, type="Total"):
        return self._fields["Ex"]
    def Ey(self, type="Total"):
        return self._fields["Ey"]
    def Ez(self, type="Total"):
        return self._fields["Ez"]

class FakeTimestep:
    def __init__(self, tsnum, fields):
        self._fields = fields
        self.fields = FakeFieldsAPI(fields)
        self.time = 0.1 * tsnum

class FakeDHybridrpy:
    def __init__(self, *args, **kwargs):
        # simple inputs dict
        self.inputs = {"time": {"dt": 0.1}, "grid": {"dx": 0.5}}
        # prepare two timesteps
        self._timesteps = [1, 2]
    def timesteps(self):
        return list(self._timesteps)
    def timestep(self, n):
        # return a timestep-like object with fields
        # use small 1D arrays for test
        x = np.linspace(0, 1, 10)
        fields = {
            "Bx": FakeField("Bx", np.sin(2 * np.pi * x), xdata=x, time=0.1 * n, timestep=n),
            "By": FakeField("By", np.zeros_like(x), xdata=x, time=0.1 * n, timestep=n),
            "Bz": FakeField("Bz", np.zeros_like(x), xdata=x, time=0.1 * n, timestep=n),
            "Ex": FakeField("Ex", np.zeros_like(x), xdata=x, time=0.1 * n, timestep=n),
            "Ey": FakeField("Ey", np.zeros_like(x), xdata=x, time=0.1 * n, timestep=n),
            "Ez": FakeField("Ez", np.zeros_like(x), xdata=x, time=0.1 * n, timestep=n),
        }
        return FakeTimestep(n, fields)

# install into sys.modules
fake.DHybridrpy = FakeDHybridrpy
sys.modules["dhybridrpy"] = fake

from reconn_wave_power.io import read_simulation

def test_read_simulation_basic():
    dpy = FakeDHybridrpy()
    ds = read_simulation(dpy, timesteps=[1,2], fields=("B",))
    # Expect components Bx, By, Bz present
    assert "Bx" in ds
    assert "By" in ds
    assert "Bz" in ds
    # time coordinate length matches timesteps
    assert ds["Bx"].coords["time"].size == 2
    # x coordinate exists
    assert "x" in ds["Bx"].coords
    # check dt in global attrs
    assert "dt" in ds.attrs