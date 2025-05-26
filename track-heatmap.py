import os
import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


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

for tel, drv in zip(telemetries, top3):
    pts = np.array([tel['X'], tel['Y']]).T.reshape(-1,1,2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap='plasma', norm=plt.Normalize(50, 300))
    lc.set_array(tel['Speed'].values)
    lc.set_linewidth(2)
    plt.gca().add_collection(lc)

plt.axis('equal'); plt.axis('off')
plt.title('Top-3 Fastest Laps as Color-mapped Lines')
plt.colorbar(lc, label='Speed (km/h)')
plt.show()