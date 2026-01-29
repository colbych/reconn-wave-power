"""
Minimal usage example (fill in actual I/O after wiring to dhybridrpy)
"""
from reconn_wave_power.io import read_simulation
from reconn_wave_power.spectrum import compute_psd_time
from reconn_wave_power.plotting import plot_psd

# path = "path/to/sim/output"
# ds = read_simulation(path)
# E_time = ds.E.sel(x=0, y=0)  # example: pick location, components
# f, P = compute_psd_time(E_time, dt=ds.attrs['dt'], nperseg=1024)
# plot_psd(f, P)