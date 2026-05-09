from pathlib import Path
import csv
import numpy as np
from numpy.typing import NDArray


def resample_tracks(
    path: Path, points: int = 32, max_gap: float = 120
) -> dict[str, NDArray[np.float64]]:
    """Return local east/north kilometers on each track's normalized time axis."""
    if points < 2 or max_gap <= 0:
        raise ValueError("invalid resampling parameters")
    trajectories: dict[str, dict[float, tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for report in csv.DictReader(stream):
            timestamp, latitude, longitude = (
                float(report[key]) for key in ("timestamp_s", "latitude", "longitude")
            )
            if not np.isfinite([timestamp, latitude, longitude]).all() or not (
                -90 <= latitude <= 90 and -180 <= longitude <= 180
            ):
                raise ValueError("invalid ADS-B coordinate or timestamp")
            identifier = report["flight_id"].strip()
            if not identifier:
                raise ValueError("empty flight_id")
            reports = trajectories.setdefault(identifier, {})
            if timestamp in reports and reports[timestamp] != (latitude, longitude):
                raise ValueError("conflicting duplicate timestamp")
            reports[timestamp] = (latitude, longitude)
    resampled = {}
    for identifier, reports in sorted(trajectories.items()):
        timestamps = np.array(sorted(reports))
        if len(timestamps) < 4 or np.diff(timestamps).max() > max_gap:
            continue
        coordinates = np.array([reports[timestamp] for timestamp in timestamps])
        local_xy = np.column_stack(
            (
                (coordinates[:, 1] + 80.65) * 111.32 * np.cos(np.deg2rad(28.3)),
                (coordinates[:, 0] - 28.3) * 111.32,
            )
        )
        grid = np.linspace(timestamps[0], timestamps[-1], points)
        resampled[identifier] = np.column_stack(
            [np.interp(grid, timestamps, local_xy[:, axis]) for axis in (0, 1)]
        )
    if not resampled:
        raise ValueError("no continuous tracks with at least four reports")
    return resampled
