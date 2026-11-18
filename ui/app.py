"""PathSnap evaluator UI. The browser receives predictions, never reference labels."""
import os

import requests
import streamlit as st

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PathSnap Evaluator", page_icon="🗺️", layout="wide")
st.title("PathSnap – Route Reconstruction Evaluator")
st.caption("Select a trip to run the connected classifier and route decoder.")

@st.cache_data(ttl=10)
def get_trips():
    response = requests.get(f"{API_URL}/trips", timeout=10)
    response.raise_for_status()
    return response.json()

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
    st.success(f"API connected · partition: {health.json().get('partition', 'unknown')}")
except requests.RequestException as exc:
    st.error(f"Cannot reach the PathSnap API at {API_URL}: {exc}")
    st.stop()

try:
    trip_data = get_trips()
    trip_ids = trip_data.get("trip_ids", [])
except (requests.RequestException, ValueError) as exc:
    st.error(f"Could not load trip options: {exc}")
    st.stop()

if not trip_ids:
    st.warning("No trips were found in the configured dataset partition.")
    st.stop()

trip_id = st.selectbox("Choose a trip", trip_ids)
if st.button("Reconstruct route", type="primary", use_container_width=True):
    with st.spinner("Running classifier and route decoder…"):
        try:
            response = requests.post(
                f"{API_URL}/predict", json={"trip_id": trip_id}, timeout=180
            )
            response.raise_for_status()
            result = response.json()
            st.session_state["last_result"] = result
        except requests.HTTPError:
            try:
                st.error(response.json().get("detail", response.text))
            except ValueError:
                st.error(response.text)
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")

result = st.session_state.get("last_result")
if result:
    st.subheader(f"Prediction for trip {result.get('trip_id', '')}")
    metrics = result.get("metrics", {})
    if metrics:
        cols = st.columns(min(4, len(metrics)))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name.replace("_", " ").title(), value)
    route = result.get("predicted_route")
    if route is not None:
        st.write("Predicted road-segment sequence")
        st.json(route)
    map_html = result.get("map_html")
    if map_html:
        st.info("The API returned map HTML. Serve it through an approved static endpoint or provide a map URL in the integration adapter.")
    if not metrics and route is None:
        st.json(result)
    st.caption("Reference labels are not included in this response.")
