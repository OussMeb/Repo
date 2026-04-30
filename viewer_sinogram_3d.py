# viewer_sinogram_3d.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Sinogram 3D Viewer", layout="wide")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def list_patient_dirs(data_root: str) -> list[str]:
    root = Path(data_root)
    if not root.exists():
        return []
    dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "sino.npy").exists():
            dirs.append(p.name)
    return dirs


@st.cache_data(show_spinner=False)
def load_patient_data(data_root: str, patient_id: str) -> dict[str, Any]:
    patient_dir = Path(data_root) / patient_id
    data: dict[str, Any] = {
        "patient_dir": patient_dir,
        "metadata": load_json(patient_dir / "metadata.json"),
        "target_sino": np.load(patient_dir / "sino.npy"),
    }

    pred_path = patient_dir / "pred_sino.npy"
    if pred_path.exists():
        data["pred_sino"] = np.load(pred_path)

    cp_views_path = patient_dir / "cp_views.npy"
    if cp_views_path.exists():
        data["cp_views"] = np.load(cp_views_path)

    cp_indices_path = patient_dir / "cp_indices.npy"
    if cp_indices_path.exists():
        data["cp_indices"] = np.load(cp_indices_path)
    else:
        data["cp_indices"] = np.arange(data["target_sino"].shape[0], dtype=np.int32)

    cp_features_path = patient_dir / "cp_features.npy"
    if cp_features_path.exists():
        data["cp_features"] = np.load(cp_features_path)

    angles_path = patient_dir / "angles_deg.npy"
    if angles_path.exists():
        data["angles_deg"] = np.load(angles_path)

    table_path = patient_dir / "table_mm.npy"
    if table_path.exists():
        data["table_mm"] = np.load(table_path)

    duration_path = patient_dir / "cp_duration_sec.npy"
    if duration_path.exists():
        data["cp_duration_sec"] = np.load(duration_path)

    return data


def get_available_sources(data: dict[str, Any]) -> list[str]:
    sources = ["target"]
    if "pred_sino" in data:
        sources.append("prediction")
        sources.append("target_vs_prediction")
    return sources


def select_sino(data: dict[str, Any], source: str) -> np.ndarray:
    if source == "prediction":
        return data["pred_sino"]
    return data["target_sino"]


def make_heatmap(sino: np.ndarray, selected_cp: int, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=sino,
                x=np.arange(sino.shape[1]),
                y=np.arange(sino.shape[0]),
                colorscale="Viridis",
                colorbar=dict(title="Opening"),
            )
        ]
    )
    fig.add_hline(y=selected_cp, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(
        title=title,
        xaxis_title="Leaf index",
        yaxis_title="Control point",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_surface(
    sino: np.ndarray,
    cp_stride: int,
    z_scale: float,
    title: str,
) -> go.Figure:
    sino_ds = sino[::cp_stride]
    x = np.arange(sino_ds.shape[1], dtype=np.float32)
    y = np.arange(0, sino.shape[0], cp_stride, dtype=np.float32)

    fig = go.Figure(
        data=[
            go.Surface(
                z=sino_ds * float(z_scale),
                x=x,
                y=y,
                colorscale="Viridis",
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Leaf index",
            yaxis_title="Control point",
            zaxis_title="Opening",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=2.2, z=0.6),
        ),
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def make_row_plot(
    target_row: np.ndarray,
    pred_row: np.ndarray | None,
    cp_idx: int,
) -> go.Figure:
    x = np.arange(target_row.shape[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=target_row, mode="lines", name="Target"))
    if pred_row is not None:
        fig.add_trace(go.Scatter(x=x, y=pred_row, mode="lines", name="Prediction"))
    fig.update_layout(
        title=f"Sinogram row at CP {cp_idx}",
        xaxis_title="Leaf index",
        yaxis_title="Opening",
        yaxis_range=[-0.02, 1.02],
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def find_local_cp_index(cp_indices: np.ndarray, cp: int) -> tuple[int | None, int | None]:
    matches = np.where(cp_indices == cp)[0]
    if len(matches) > 0:
        local_idx = int(matches[0])
        return local_idx, cp

    if len(cp_indices) == 0:
        return None, None

    nearest_idx = int(np.argmin(np.abs(cp_indices - cp)))
    nearest_cp = int(cp_indices[nearest_idx])
    return nearest_idx, nearest_cp


def image_figure(img: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=img,
                colorscale="Viridis",
                showscale=False,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Detector / projected width",
        yaxis_title="Projected height",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def summarize_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(target - pred)))
    mse = float(np.mean((target - pred) ** 2))
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "RMSE": rmse}


def main() -> None:
    st.title("Interactive 3D Sinogram Viewer")

    with st.sidebar:
        st.header("Controls")
        data_root = st.text_input("Processed dataset root", value="processed_cp_sino")
        patient_ids = list_patient_dirs(data_root)

        if not patient_ids:
            st.error(f"No patient folders with sino.npy found under: {data_root}")
            st.stop()

        patient_id = st.selectbox("Patient / plan", patient_ids)
        data = load_patient_data(data_root, patient_id)

        sources = get_available_sources(data)
        source = st.selectbox("Source", sources, index=0)

        sino_for_slider = data["target_sino"]
        max_cp = int(sino_for_slider.shape[0] - 1)
        cp_idx = st.slider("Control point", min_value=0, max_value=max_cp, value=0, step=1)

        surface_stride = st.number_input("3D surface CP stride", min_value=1, max_value=50, value=4, step=1)
        z_scale = st.slider("3D vertical scale", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    target_sino = data["target_sino"]
    pred_sino = data.get("pred_sino")
    cp_indices = data["cp_indices"]

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.metric("Patient / plan", patient_id)
    info_col2.metric("N_CP", int(target_sino.shape[0]))
    info_col3.metric("Leaves", int(target_sino.shape[1]))
    info_col4.metric("Prediction loaded", "Yes" if pred_sino is not None else "No")

    with st.expander("Metadata", expanded=False):
        st.json(data["metadata"] if data["metadata"] else {"metadata": "not found"})

    if source == "target_vs_prediction" and pred_sino is not None:
        metrics = summarize_metrics(target_sino, pred_sino)
        m1, m2 = st.columns(2)
        m1.metric("Global MAE", f"{metrics['MAE']:.6f}")
        m2.metric("Global RMSE", f"{metrics['RMSE']:.6f}")

        left, right = st.columns(2)
        left.plotly_chart(
            make_heatmap(target_sino, cp_idx, "Target sinogram"),
            use_container_width=True,
        )
        right.plotly_chart(
            make_heatmap(pred_sino, cp_idx, "Prediction sinogram"),
            use_container_width=True,
        )

        surface_left, surface_right = st.columns(2)
        surface_left.plotly_chart(
            make_surface(target_sino, int(surface_stride), float(z_scale), "3D target sinogram"),
            use_container_width=True,
        )
        surface_right.plotly_chart(
            make_surface(pred_sino, int(surface_stride), float(z_scale), "3D prediction sinogram"),
            use_container_width=True,
        )

        st.plotly_chart(
            make_row_plot(target_sino[cp_idx], pred_sino[cp_idx], cp_idx),
            use_container_width=True,
        )
    else:
        sino = select_sino(data, source)
        left, right = st.columns([1, 1.1])

        with left:
            st.plotly_chart(
                make_heatmap(sino, cp_idx, f"2D {source} sinogram"),
                use_container_width=True,
            )

        with right:
            st.plotly_chart(
                make_surface(sino, int(surface_stride), float(z_scale), f"3D {source} sinogram"),
                use_container_width=True,
            )

        pred_row = pred_sino[cp_idx] if (pred_sino is not None and source != "prediction") else None
        st.plotly_chart(
            make_row_plot(target_sino[cp_idx], pred_row, cp_idx),
            use_container_width=True,
        )

    cp_views = data.get("cp_views")
    if cp_views is not None:
        local_idx, shown_cp = find_local_cp_index(cp_indices, cp_idx)
        st.subheader("CP-local views")

        if local_idx is None:
            st.warning("No CP views available.")
        else:
            if shown_cp != cp_idx:
                st.info(f"Requested CP {cp_idx} not precomputed. Showing nearest available CP {shown_cp}.")

            ct_img = cp_views[local_idx, 0]
            dose_img = cp_views[local_idx, 1]

            c1, c2 = st.columns(2)
            c1.plotly_chart(
                image_figure(ct_img, f"CT CP-view #{shown_cp}"),
                use_container_width=True,
            )
            c2.plotly_chart(
                image_figure(dose_img, f"Dose CP-view #{shown_cp}"),
                use_container_width=True,
            )

            detail_cols = st.columns(4)
            if "angles_deg" in data:
                detail_cols[0].metric("Angle (deg)", f"{float(data['angles_deg'][shown_cp]):.2f}")
            if "table_mm" in data:
                detail_cols[1].metric("Table (mm)", f"{float(data['table_mm'][shown_cp]):.2f}")
            if "cp_duration_sec" in data:
                detail_cols[2].metric("CP duration (s)", f"{float(data['cp_duration_sec'][shown_cp]):.4f}")
            detail_cols[3].metric("CP sampled", "Yes" if shown_cp in set(cp_indices.tolist()) else "No")

    st.caption("Viewer reads the saved .npy outputs directly. No reprocessing is needed.")


if __name__ == "__main__":
    main()