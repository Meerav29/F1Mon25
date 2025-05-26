import os
import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt


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
    plt.figure(figsize=(8, 6))
    plt.scatter(
        tel['X'], tel['Y'],
        c=tel['Speed'],
        s=1,
        cmap='plasma'
    )
    plt.axis('off')
    plt.title(f'{drv} — Speed Heatmap (Fastest Lap)')
    plt.colorbar(label='Speed (km/h)')
    plt.show()