# MLOps Final — California Housing Prices

Predict `median_house_value` from California housing features. Train with FLAML + MLflow, serve via FastAPI, monitor drift with Evidently Cloud, and version production batches with GitHub + DVC + DagsHub.

## Architecture

```text
Streamlit (8501) → FastAPI (8000) → predict + RMSE
                         ├─► Evidently Cloud (drift report)
                         └─► GitHub production_batches/
                                  └─► GitHub Action → DVC → DagsHub
```

## Quick start (Docker — recommended)

**Prerequisites:** Docker and Docker Compose

1. Copy and configure environment variables:

   ```bash
   cp .env.example .env
   ```

2. Build and run:

   ```bash
   docker compose up --build
   ```

3. Open:

   - **Streamlit UI:** http://localhost:8501
   - **API docs:** http://localhost:8000/docs

## Local development (without Docker)

**Prerequisites:** [uv](https://docs.astral.sh/uv/)

```bash
uv sync
cp .env.example .env

# Terminal 1 — API
cd backend
uv run uvicorn main:app --reload

# Terminal 2 — Streamlit (from repo root)
uv run streamlit run frontend/app.py
```

## Demo files for presentation

| File | Use |
|------|-----|
| `splits/test.csv` | Clean holdout batch (~2,064 rows) — rubric step #7 |
| `extreme_drift_test.csv` | Heavily corrupted drift demo (~399 rows) — rubric step #9 |
| `production_batches/batch2_corrupted.csv` | Notebook-generated corrupted batch (may require `dvc pull`) |

**CSV format for upload:** same encoded columns as `reference_data.csv`, plus `median_house_value`. Do not include `prediction`. The API adds predictions. Extra `TARGET` column is ignored if present.

## Project layout

```text
model.ipynb          Training pipeline (FLAML, MLflow, batch creation)
backend/             FastAPI service (predict, GitHub, Evidently)
frontend/            Streamlit UI
models/              Champion model pickle
reference_data.csv   Baseline for Evidently drift
production_batches/  Versioned inference batches (DVC)
splits/              Train / val / test CSVs
```

## Training (first time or retrain)

Run `model.ipynb` end-to-end. It produces:

- `models/champion_model.pkl`
- `splits/*.csv`
- `reference_data.csv`
- `production_batches/batch1_clean.csv`, `batch2_corrupted.csv`, `batch3_mixed.csv`
- MLflow runs under `mlruns/`

## Verify monitoring

After uploading in Streamlit:

```bash
docker compose logs api --tail 30
```

Look for `Evidently: uploaded report for batch_...`, then check [Evidently Cloud](https://app.evidently.cloud).

## Team ownership

| Area | Owner | Location |
|------|-------|----------|
| Model building + monitoring setup | Adi | `model.ipynb`, `Model_Monitoring.py` |
| FastAPI + versioning + Evidently/GitHub | Lam | `backend/` |
| Streamlit + Docker | Minhao | `frontend/`, `Dockerfile.*`, `docker-compose.yml` |

## Notes

- `.env` is gitignored — never commit secrets.
- Swagger UI at `/docs` is for API testing; Streamlit is the presentation UI.
