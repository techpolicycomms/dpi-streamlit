# DPI Streamlit (Parallel Migration Repo)

This repo is a parallel Python-first migration track from the existing Shiny project at `../DPI`.

The immediate objective is **result parity**: serve the same KPI tables/charts from the same pipeline outputs, now in Streamlit.

## What This Repo Does Today

- Runs a Streamlit dashboard from Python only.
- Reads existing DPI outputs from the legacy project (`../DPI/outputs`).
- Keeps metric parity by consuming the same output artifacts already used by Shiny.
- Supports optional snapshotting of legacy outputs into this repo.

## Local Setup

```bash
cd "/Users/rahuljha/Desktop/coding projects/DPI-streamlit"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Dashboard

```bash
streamlit run app.py
```

By default, the app reads from:

- `../DPI/outputs`

You can override this path with:

```bash
export DPI_LEGACY_ROOT="/absolute/path/to/DPI"
```

## Optional: Snapshot Outputs Into This Repo

```bash
python scripts/snapshot_legacy_outputs.py
```

This copies selected output CSVs from the legacy repo into:

- `data/legacy_snapshot/outputs`

## Migration Strategy

1. **Parity phase (current):** consume existing output CSVs in Python dashboard.
2. **Computation phase:** port R scoring scripts to Python modules and regenerate the same CSV outputs.
3. **Validation phase:** run side-by-side parity checks (file hashes, metric diffs, chart-level sanity checks).
4. **Cutover phase:** deploy Streamlit app and retire Shiny deployment.

## Python Core Pipeline (Now Added)

Run the ported core scoring path:

```bash
python scripts/run_core_pipeline.py
```

This writes Python-generated outputs to:

- `outputs/`

Core scripts:

- `scripts/build_dpi_ready_index.py`
- `scripts/build_doughnut_subindex.py`
- `scripts/build_trend_analysis.py`

Run a parity check vs legacy outputs:

```bash
python scripts/parity_check_core.py
```

Report path:

- `outputs/parity_check_core_report.csv`
