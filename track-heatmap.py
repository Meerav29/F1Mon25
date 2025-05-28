import os
import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from matplotlib.colors import Normalize


cache_dir = os.path.join(os.getcwd(), 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)
plotting.setup_mpl()

session = fastf1.get_session(2025, 'Monaco', 'R')
session.load()

top3 = (
    session.results
           .sort_values('Position')
           ['Abbreviation']
           .head(3)
           .tolist()
)

fastest_laps = [
    session.laps
           .pick_drivers([drv])
           .pick_fastest()
    for drv in top3
]

telemetries = [
    lap.get_telemetry()      # full telemetry with X, Y, Time, Speed, etc.
       .add_distance()       # computes & inserts a 'Distance' column
    for lap in fastest_laps
]

# 1) Compute a global speed range for uniform coloring
all_speeds = np.hstack([tel['Speed'].values for tel in telemetries])
global_norm = Normalize(vmin=all_speeds.min(), vmax=all_speeds.max())

# 2) Set up white‐background figure
plt.style.use('default')
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
ax.set_facecolor('white')

# 3) Add each driver as a colored line
for tel, drv in zip(telemetries, top3):
    pts = np.column_stack((tel['X'], tel['Y']))
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(
        segs,
        cmap='turbo',
        norm=global_norm,
        linewidth=3,
        alpha=0.8
    )
    lc.set_array(tel['Speed'])
    ax.add_collection(lc)

# 4) Autoscale to show all data
ax.autoscale(enable=True, axis='both', tight=True)


# 5) Final styling
ax.set_aspect('equal')
ax.axis('off')
cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Speed (km/h)')
plt.title('Top-3 Fastest Laps — Speed Profile', pad=12)
plt.tight_layout()
plt.show()