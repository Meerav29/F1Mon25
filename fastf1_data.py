import fastf1
from fastf1.core import DataNotLoadedError
import pandas as pd
from pathlib import Path

_DEFAULT_CACHE = Path("cache")


def enable_cache(cache_dir=_DEFAULT_CACHE):
    """Ensure FastF1 caching is enabled."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def fetch_event_sessions(year: int, event: str, sessions=("FP1", "FP2", "FP3", "Q")):
    """Return concatenated lap data for the given event sessions.

    Parameters
    ----------
    year : int
        Championship year.
    event : str
        Name of the Grand Prix (e.g. "Monaco").
    sessions : iterable[str]
        Session types to fetch. Defaults to practice and qualifying.
    """
    enable_cache()
    dfs = []
    for sess_type in sessions:
        print(f"→ Loading {year} {event} {sess_type}…")
        sess = fastf1.get_session(year, event, sess_type)
        try:
            sess.load()
            laps = sess.laps
        except DataNotLoadedError:
            print(f"   • No lap data for {year} {sess_type}, skipping")
            continue
        if laps.empty:
            print(f"   • {year} {sess_type} returned 0 laps, skipping")
            continue
        df = laps.copy()
        df["Session"] = sess_type
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No sessions fetched for {event} {year}")
    return pd.concat(dfs, ignore_index=True)


def fetch_race_results(year: int, event: str) -> pd.DataFrame:
    """Return classification results for the race session."""
    enable_cache()
    sess = fastf1.get_session(year, event, "R")
    sess.load()
    results = sess.results
    results = results[results["Position"].notna()]
    return results.copy()
