from pathlib import Path
import numpy as np
import pytest
from adsb_clustering.cluster_flightpaths import cluster, dtw_distance
from adsb_clustering.preprocess import resample_tracks


@pytest.mark.parametrize(
    "invalid",
    [np.zeros(4), np.zeros((2, 3)), np.full((2, 2), np.nan), np.empty((0, 2))],
)
def test_invalid_dtw_geometry(invalid):
    with pytest.raises(ValueError, match="DTW"):
        dtw_distance(invalid, np.zeros((2, 2)))


def test_dtw_symmetry():
    first = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    second = first + 1
    assert dtw_distance(first, first) == 0
    assert dtw_distance(first, second) == pytest.approx(dtw_distance(second, first))


def test_fixture_clusters():
    results = cluster(Path("data/orlando_melbourne_adsb.csv"))
    assert len(results) == 13
    assert len({item["cluster"] for item in results}) == 3
    assert sum(item["holding_candidate"] for item in results) == 6
    assert results[-1]["corridor_outlier"]


def test_conflicting_time(tmp_path):
    capture = tmp_path / "tracks.csv"
    capture.write_text(
        "flight_id,timestamp_s,latitude,longitude\na,0,28,-80\na,0,29,-80\n"
    )
    with pytest.raises(ValueError, match="conflicting"):
        resample_tracks(capture)
