# ADS-B trajectory clustering

[![Verification](https://github.com/cankaan120/adsb-trajectory-clustering-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/cankaan120/adsb-trajectory-clustering-ml/actions/workflows/ci.yml)

Cluster whole flights with band-limited dynamic time warping and DBSCAN.
The distance matrix preserves corridor location; a separate closure/turning test
marks holding candidates without treating a cluster label as an operational verdict.

```sh
pip install -e '.[test]'
adsb-cluster data/orlando_melbourne_adsb.csv --epsilon-km 2
pytest -q
```

| Fixture result | Count |
|---|---:|
| Linear corridor members | 6 |
| Holding candidates | 6 |
| Corridor outliers | 1 |

Input CSV requires flight_id,timestamp_s,latitude,longitude. Duplicate identical
reports collapse; conflicting timestamps fail. Tracks with fewer than four points
or gaps above 120 seconds are omitted. Each flight resamples to 32 points.
DTW uses rolling rows and an eight-point band; batch size is capped at 500 flights.
The bundled seeded tracks describe two patterns and an outlier around Central
Florida. These labels test geometry, not performance on an operational receiver.
Local tangent coordinates are intended for this regional extent; use a suitable
projection for wider areas. MIT.

## Development

```sh
git clone https://github.com/cankaan120/adsb-trajectory-clustering-ml.git
cd adsb-trajectory-clustering-ml
python -m venv .venv
# Activate .venv with your shell, then:
python -m pip install -e ".[test]" ruff==0.17.3 build
python -m pytest -q
python -m ruff check --select E9,F63,F7,F82 src tests
python -m ruff format --check src tests
python -m build
```

Report defects with the command, input shape or minimal fixture, and runtime version.
Regression tests should reproduce the failure and preserve the documented units.
