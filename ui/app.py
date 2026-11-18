"""PathSnap route reconstruction map UI."""
import os

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PathSnap | Route Map", page_icon="🗺️", layout="wide")
st.title("PathSnap Route Explorer")
st.caption("Inspect GPS observations, the nearby directed road network, classifier choices, and the HMM route as separate map layers.")


@st.cache_data(ttl=10)
def get_trips():
    response = requests.get(f"{API_URL}/trips", timeout=10)
    response.raise_for_status()
    return response.json()


try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
    health_data = health.json()
    if health_data.get("pipeline") == "configured":
        st.success(f"API connected · {health_data.get('partition', 'unknown')} trips")
    else:
        st.warning("API is connected, but the classifier and HMM pipeline is not configured yet.")
    trip_ids = get_trips().get("trip_ids", [])
except (requests.RequestException, ValueError) as exc:
    st.error(f"Cannot load trips from {API_URL}: {exc}")
    st.stop()

if not trip_ids:
    st.warning("No trips found. Check PATHSNAP_DATA_ROOT and PATHSNAP_UI_PARTITION on the API server.")
    st.stop()

trip_id = st.selectbox("Trip", trip_ids)
if st.button("Reconstruct and show map", type="primary"):
    st.session_state.pop("map_result", None)
    try:
        with st.spinner("Loading trip and running classifier + HMM decoder…"):
            map_response = requests.get(f"{API_URL}/trips/{trip_id}/map-data", timeout=60)
            map_response.raise_for_status()
            prediction_response = requests.post(f"{API_URL}/predict", json={"trip_id": trip_id}, timeout=300)
            prediction_response.raise_for_status()
            st.session_state["map_result"] = (map_response.json(), prediction_response.json())
    except requests.HTTPError as exc:
        try:
            message = exc.response.json().get("detail", str(exc))
        except ValueError:
            message = str(exc)
        st.error(message)
    except requests.RequestException as exc:
        st.error(f"Could not build the trip map: {exc}")


stored = st.session_state.get("map_result")
if stored:
    map_data, prediction = stored
    gps = map_data["gps"]
    nodes = map_data["nodes"]
    arc_lookup = {int(i): tuple(map(int, arc)) for i, arc in zip(map_data["arc_ids"], map_data["arcs"])}
    center = [sum(p[1] for p in gps) / len(gps), sum(p[0] for p in gps) / len(gps)]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    roads = folium.FeatureGroup(name="Nearby road network", show=True)
    for arc_id, (u, v) in arc_lookup.items():
        if 0 <= u < len(nodes) and 0 <= v < len(nodes):
            folium.PolyLine([[nodes[u][1], nodes[u][0]], [nodes[v][1], nodes[v][0]]],
                            color="#8993a4", weight=2, opacity=0.42,
                            tooltip=f"Road arc {arc_id}").add_to(roads)
    roads.add_to(fmap)

    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    stride = max(1, len(gps) // 1200)
    for i in range(0, len(gps), stride):
        folium.CircleMarker([gps[i][1], gps[i][0]], radius=3, color="#1688e8",
                            fill=True, fill_opacity=0.85,
                            tooltip=f"GPS point {i} · t={map_data['timestamps'][i]:.1f}s").add_to(gps_layer)
    gps_layer.add_to(fmap)

    top1 = folium.FeatureGroup(name="Classifier top candidate", show=False)
    for i, segment in enumerate(prediction.get("classifier_top1_segments", [])):
        folium.PolyLine([[segment[0][1], segment[0][0]], [segment[1][1], segment[1][0]]],
                        color="#f39c36", weight=4, opacity=0.8,
                        tooltip=f"Classifier · observation {i}").add_to(top1)
    top1.add_to(fmap)

    hmm = folium.FeatureGroup(name="HMM decoded route", show=True)
    for i, segment in enumerate(prediction.get("route_segments", [])):
        folium.PolyLine([[segment[0][1], segment[0][0]], [segment[1][1], segment[1][0]]],
                        color="#8c52c7", weight=5, opacity=0.92,
                        tooltip=f"HMM · observation {i}").add_to(hmm)
    hmm.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    st.subheader(f"Trip {trip_id}")
    metrics = prediction.get("metrics", {})
    if metrics:
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name.replace("_", " ").title(), f"{value:.3f}" if isinstance(value, (int, float)) else value)
    st.caption("Toggle map layers from the control in the upper right.")
    components.html(fmap.get_root().render(), height=680, scrolling=False)
