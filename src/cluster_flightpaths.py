import argparse
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from .preprocess import resample_tracks


def dtw_distance(first: NDArray[np.float64], second: NDArray[np.float64], window: int = 8) -> float:
    """Symmetric banded DTW cost in km, normalized by total sequence length."""
    if window < 0 or len(first) == 0 or len(second) == 0:
        raise ValueError("invalid DTW input")
    width = max(window, abs(len(first) - len(second)))
    previous = np.full(len(second) + 1, np.inf)
    previous[0] = 0
    for row, position in enumerate(first, 1):
        current = np.full(len(second) + 1, np.inf)
        for column in range(max(1, row - width), min(len(second), row + width) + 1):
            current[column] = np.linalg.norm(position - second[column - 1]) + min(
                previous[column], current[column - 1], previous[column - 1]
            )
        previous = current
    return float(previous[-1] / (len(first) + len(second)))


def cluster(path: Path, epsilon_km: float = 2, min_samples: int = 3) -> list[dict[str, object]]:
    trajectories = resample_tracks(path)
    identifiers = list(trajectories)
    if len(identifiers) > 500:
        raise ValueError("pairwise DTW batch limit is 500 flights")
    distances = np.zeros((len(identifiers), len(identifiers)))
    for row in range(len(identifiers)):
        for column in range(row):
            distances[row, column] = distances[column, row] = dtw_distance(
                trajectories[identifiers[row]], trajectories[identifiers[column]]
            )
    labels = DBSCAN(eps=epsilon_km, min_samples=min_samples, metric="precomputed").fit_predict(
        distances
    )
    results = []
    for identifier, label in zip(identifiers, labels):
        positions = trajectories[identifier]
        legs = np.diff(positions, axis=0)
        traveled = float(np.linalg.norm(legs, axis=1).sum())
        closure = float(np.linalg.norm(positions[-1] - positions[0])) / max(traveled, 1e-12)
        heading = np.unwrap(np.arctan2(legs[:, 1], legs[:, 0]))
        rotation = abs(float(np.diff(heading).sum()))
        results.append(
            {
                "flight_id": identifier,
                "cluster": int(label),
                "corridor_outlier": bool(label == -1),
                "holding_candidate": bool(closure < 0.2 and rotation > 4),
                "distance_km": round(traveled, 3),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster flight-disjoint trajectories using DTW")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--epsilon-km", type=float, default=2)
    parser.add_argument("--min-samples", type=int, default=3)
    arguments = parser.parse_args()
    print(
        json.dumps(
            cluster(arguments.capture, arguments.epsilon_km, arguments.min_samples), indent=2
        )
    )


if __name__ == "__main__":
    main()
