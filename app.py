"""Streamlit UI:  streamlit run app.py"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from roomcheck import Params, Zone, check_room, zones_from_json
from roomcheck import demo_data

st.set_page_config(page_title="Hostel Room Inventory Checker", layout="wide")
st.title("Hostel Room Inventory - Missing Item Detection")
st.caption("SIFT/ORB keypoints -> homography alignment -> zone-wise change detection. No model training.")


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_upload(f):
    return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Feature detector", ["sift", "orb"], help="SIFT is more robust; ORB is faster.")
    ratio = st.slider("Lowe ratio test", 0.5, 0.95, 0.75, 0.01)
    ransac = st.slider("RANSAC threshold (px)", 1.0, 10.0, 4.0, 0.5)
    color_thr = st.slider("Pixel colour-change threshold", 8, 60, 22)
    zone_thr = st.slider("Zone changed-area threshold", 0.02, 0.50, 0.10, 0.01)
    edge_thr = st.slider("Zone edge-change threshold", 0.05, 0.80, 0.25, 0.01)

use_demo = st.checkbox("Use built-in demo images", value=True)
if use_demo:
    scenario = st.selectbox("Demo scenario", list(demo_data.SCENARIOS))
    base, new = demo_data.make_pair(scenario)
    default_zones = [dict(name=n, item=i, x=r[0], y=r[1], w=r[2], h=r[3]) for n, i, r in demo_data.DEFAULT_ZONES]
else:
    c1, c2 = st.columns(2)
    fb = c1.file_uploader("Baseline photo (room as it should be)", type=["jpg", "jpeg", "png"])
    fn = c2.file_uploader("New photo (room now)", type=["jpg", "jpeg", "png"])
    base = read_upload(fb) if fb else None
    new = read_upload(fn) if fn else None
    default_zones = [dict(name="Study table", item="laptop", x=0.5, y=0.4, w=0.3, h=0.25)]

st.subheader("Furniture zones (normalised 0-1 on the baseline photo)")
zdf = st.data_editor(pd.DataFrame(default_zones), num_rows="dynamic", width="stretch", key=f"z{use_demo}")

if base is None or new is None:
    st.info("Upload both photos to continue.")
    st.stop()

zones = []
for _, r in zdf.dropna().iterrows():
    zones.append(Zone.from_rect(str(r["name"]), str(r["item"]), float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])))

if st.button("Run check", type="primary"):
    rep = check_room(base, new, zones, Params(method=method, ratio=ratio, ransac_thresh=ransac,
                                              color_thresh=float(color_thr), zone_change_thresh=zone_thr,
                                              edge_change_thresh=edge_thr))
    a = rep.alignment
    if not a.ok:
        st.error(f"Alignment failed: {a.reason}. Retake the photo so it overlaps the baseline more.")
    elif rep.missing:
        st.error("Missing: " + ", ".join(r.item for r in rep.missing))
    elif rep.alerts:
        st.warning("Changes found: " + ", ".join(f"{r.status.lower()} {r.item}" for r in rep.alerts))
    else:
        st.success("All checked zones match the baseline.")

    st.image(rgb(rep.images["panel"]), caption="Left: baseline zones | Right: new photo with alerts", width="stretch")
    st.dataframe(pd.DataFrame([r.to_dict() for r in rep.results]), width="stretch")
    with st.expander("How the decision was made (debug views)"):
        cols = st.columns(3)
        if a.ok:
            cols[0].image(rgb(rep.images["aligned_new"]), caption="New photo warped to baseline view")
            cols[1].image(rgb(rep.images["change_heatmap"]), caption="Colour-difference heatmap")
            cols[2].image(rgb(rep.images["edge_change"]), caption="Edges: blue=lost, yellow=gained")
            st.image(rgb(rep.images["matches"]), caption=f"Inlier keypoint matches ({a.n_inliers} inliers / {a.n_good} good)")
