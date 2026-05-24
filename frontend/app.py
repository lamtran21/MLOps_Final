import io
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")
EVIDENTLY_CLOUD_URL = os.getenv(
    "EVIDENTLY_CLOUD_URL",
    "https://app.evidently.cloud",
)

DROP_COLUMNS = {"TARGET", "prediction"}


def prepare_upload(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    cleaned = df.copy()

    if "median_house_value" not in cleaned.columns:
        raise ValueError(
            "CSV must include a `median_house_value` column for RMSE evaluation."
        )

    dropped = [col for col in DROP_COLUMNS if col in cleaned.columns]
    if dropped:
        cleaned = cleaned.drop(columns=dropped)
        warnings.append(
            f"Dropped non-feature columns before upload: {', '.join(dropped)}"
        )

    return cleaned, warnings


def check_api_health() -> tuple[bool, str]:
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        response.raise_for_status()
        return True, response.json().get("status", "ok")
    except requests.RequestException as exc:
        return False, str(exc)


def call_predict_api(df: pd.DataFrame) -> dict:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    response = requests.post(
        f"{FASTAPI_URL}/predict",
        files={"file": ("upload.csv", buffer.getvalue(), "text/csv")},
        timeout=120,
    )
    if not response.ok:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise RuntimeError(f"API error ({response.status_code}): {detail}")

    return response.json()


st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
)

st.title("California Housing Price Predictor")
st.caption(
    "Upload a batch CSV to score houses, compute RMSE, version the batch, "
    "and trigger Evidently drift monitoring."
)

with st.sidebar:
    st.subheader("Service status")
    healthy, detail = check_api_health()
    if healthy:
        st.success(f"API online ({FASTAPI_URL})")
    else:
        st.error(f"API unreachable: {detail}")

    st.subheader("Expected CSV format")
    st.markdown(
        """
        Match `reference_data.csv` feature columns, plus ground truth:

        - Required: `median_house_value`
        - Features: encoded `ocean_*` columns and numeric housing fields
        - Do **not** include `prediction`
        - `TARGET` is ignored if present
        """
    )
    st.link_button("Open Evidently Cloud", EVIDENTLY_CLOUD_URL)

uploaded = st.file_uploader("Upload batch CSV", type=["csv"])

if uploaded is not None:
    raw_df = pd.read_csv(uploaded)

    st.subheader("Uploaded preview")
    st.dataframe(raw_df.head(10), use_container_width=True)
    st.caption(f"{len(raw_df):,} rows x {len(raw_df.columns)} columns")

    try:
        upload_df, prep_warnings = prepare_upload(raw_df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    for warning in prep_warnings:
        st.warning(warning)

    if st.button("Run prediction", type="primary", disabled=not healthy):
        with st.spinner("Calling FastAPI and triggering monitoring..."):
            try:
                result = call_predict_api(upload_df)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        predictions = result["predictions"]
        rmse = result["rmse"]
        batch_name = result["batch_name"]

        results_df = upload_df.copy()
        results_df["prediction"] = predictions
        if "median_house_value" in results_df.columns:
            results_df["error"] = (
                results_df["prediction"] - results_df["median_house_value"]
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Batch RMSE", f"{rmse:,.2f}")
        col2.metric("Rows scored", f"{len(predictions):,}")
        col3.metric("Batch name", batch_name)

        st.subheader("Predictions")
        st.dataframe(results_df.head(50), use_container_width=True)

        st.success(
            "Batch sent to the backend. GitHub versioning, DVC, and Evidently "
            "monitoring run in the background when credentials are configured."
        )
        st.info(
            f"Open Evidently Cloud to inspect the drift report for `{batch_name}`."
        )

else:
    st.info(
        "Try `splits/test.csv` for a clean batch or `extreme_drift_test.csv` for a corrupted "
        "drift demo. Feature columns must match the trained model format."
    )
