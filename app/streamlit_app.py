"""Streamlit frontend for the defect prediction service.

Communicates with the FastAPI backend over HTTP only. It has no knowledge of
models, PyTorch, or the `sdp` package — swapping this for a React app would
require no backend changes.
"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("SDP_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 30

st.set_page_config(page_title="Software Defect Prediction", page_icon="🔍", layout="wide")


def check_health() -> dict | None:
    """Return backend health, or None if unreachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def request_prediction(code: str) -> tuple[dict | None, str | None]:
    """Return (result, error_message)."""
    try:
        response = requests.post(f"{API_URL}/predict", json={"code": code}, timeout=REQUEST_TIMEOUT)
        if response.status_code == 422:
            return None, "The API rejected this input. Is the snippet empty or too long?"
        response.raise_for_status()
        return response.json(), None
    except requests.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Is the server running?"
    except requests.Timeout:
        return None, "The request timed out."
    except requests.RequestException as exc:
        return None, f"Request failed: {exc}"


st.title("🔍 Software Defect Prediction")
st.caption("Multiclass defect classification for C++ source code")

health = check_health()

with st.sidebar:
    st.header("Service status")
    if health is None:
        st.error("Backend unreachable")
        st.code("uvicorn sdp.api.main:app --reload --port 8000", language="bash")
    else:
        st.success("Backend connected")
        st.metric("Model", health["model_name"])
        st.metric("Classes", health["num_classes"])

SAMPLE = """#include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n
    cout << n * 2 << endl;
    return 0;
}"""

code = st.text_area("Source code", value=SAMPLE, height=280)

if st.button("Analyse", type="primary", disabled=health is None):
    if not code.strip():
        st.warning("Please enter some code.")
    else:
        with st.spinner("Analysing..."):
            result, error = request_prediction(code)

        if error:
            st.error(error)
        else:
            if result["is_placeholder"]:
                st.warning(
                    "⚠️ **Placeholder model.** These predictions are randomly "
                    "generated and carry no meaning. A trained model has not yet "
                    "been integrated."
                )

            left, right = st.columns([1, 2])
            with left:
                st.metric("Predicted class", result["predicted_class"])
                st.metric("Confidence", f"{result['confidence']:.1%}")
            with right:
                df = pd.DataFrame(result["probabilities"])
                chart = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x=alt.X("label:N", sort=None, title=None),
                        y=alt.Y("probability:Q", title="Probability"),
                    )
                )
                st.altair_chart(chart, use_container_width=True)

            with st.expander("Raw response"):
                st.json(result)
