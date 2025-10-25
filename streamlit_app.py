"""Interactive interface for the Coronary Heart Disease detection pipeline."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from app import config, data_access, model_runner

st.set_page_config(page_title="Coronary Heart Disease UI", layout="wide")
st.title("Coronary Heart Disease Detection Dashboard")
st.caption(
    "Khám phá dữ liệu ECG, huấn luyện nhanh KSOM và trực quan hóa log nghiên cứu trong cùng một nơi."
)


@st.cache_data(show_spinner=False)
def cached_samples(env: str, sample_size: int | None, seed: int | None):
    split = data_access.load_numpy_samples(env, sample_size=sample_size, seed=seed)
    return split.data, split.labels


@st.cache_data(show_spinner=False)
def cached_log_dataframe(path: Path) -> pd.DataFrame:
    entries = data_access.load_log_file(path)
    return pd.DataFrame(entries)


def render_dataset_overview(selected_envs: List[str]) -> None:
    st.subheader("Tổng quan tập dữ liệu")
    if not selected_envs:
        st.info("Chọn ít nhất một môi trường ở thanh bên để xem thống kê.")
        return

    cols = st.columns(len(selected_envs))
    for col, env in zip(cols, selected_envs, strict=False):
        stats = data_access.summarize_environment(env)
        with col:
            st.metric(label=f"{env} - Số mẫu", value=stats["total"])
            distribution = (
                pd.Series(stats["label_distribution"])
                .sort_index()
                .rename(index=lambda k: f"Label {k}")
            )
            st.bar_chart(distribution)


def render_model_playground(available_envs: List[str]) -> None:
    st.subheader("Thử nghiệm KSOM nhanh")
    if not available_envs:
        st.warning("Không tìm thấy môi trường dữ liệu nào.")
        return

    default_env = next((env for env in available_envs if env != config.EVAL_ENV), available_envs[0])
    with st.form("ksom-train-form"):
        env = st.selectbox("Chọn môi trường huấn luyện", options=available_envs, index=available_envs.index(default_env))
        stats = data_access.summarize_environment(env)
        max_samples = stats["total"]
        sample_size = st.slider("Số mẫu dùng để huấn luyện", min_value=50, max_value=max_samples, value=min(200, max_samples))
        epochs = st.slider("Số epoch", min_value=1, max_value=15, value=3)
        grid_size = st.slider("Kích thước lưới", min_value=5, max_value=20, value=10)
        learning_rate = st.slider("Learning rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        radius = st.slider("Bán kính lân cận", min_value=0.1, max_value=5.0, value=1.0)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        submitted = st.form_submit_button("Huấn luyện KSOM")

    if submitted:
        with st.spinner("Đang huấn luyện mô hình KSOM..."):
            data, labels = cached_samples(env, sample_size=sample_size, seed=int(seed))
            result = model_runner.train_ksom(
                data,
                labels,
                grid_size=grid_size,
                learning_rate=learning_rate,
                radius=radius,
                epochs=epochs,
                dataset_name=env,
                seed=int(seed),
            )
        st.session_state["ksom_model"] = result["model"]
        st.session_state["ksom_train_report"] = result["report"]
        report = result["report"]
        st.success(
            f"Hoàn tất huấn luyện sau {report.duration_s:.2f}s – success rate: {report.success_rate:.2%}"
        )

    if "ksom_model" not in st.session_state:
        st.info("Huấn luyện một mô hình KSOM để mở khóa các tiện ích đánh giá phía dưới.")
        return

    model = st.session_state["ksom_model"]
    report = st.session_state.get("ksom_train_report")

    with st.expander("Chi tiết quá trình huấn luyện", expanded=False):
        if report:
            st.json(asdict(report))

    evaluation_env = st.selectbox(
        "Chọn môi trường đánh giá",
        options=available_envs,
        index=available_envs.index(config.EVAL_ENV) if config.EVAL_ENV in available_envs else 0,
        key="ksom-eval-env",
    )

    eval_stats = data_access.summarize_environment(evaluation_env)
    eval_sample_size = st.slider(
        "Số mẫu dùng để đánh giá",
        min_value=50,
        max_value=eval_stats["total"],
        value=min(200, eval_stats["total"]),
        key="ksom-eval-sample",
    )

    if st.button("Đánh giá KSOM"):
        with st.spinner("Đang đánh giá..."):
            eval_data, eval_labels = cached_samples(evaluation_env, sample_size=eval_sample_size, seed=7)
            metrics = model_runner.evaluate_ksom(model, eval_data, eval_labels)
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        preview_df = (
            pd.DataFrame({
                "true_label": eval_labels,
                "predicted": metrics["predictions"],
            })
            .reset_index()
            .rename(columns={"index": "sample"})
        )
        st.dataframe(preview_df.head(20))

    uploaded = st.file_uploader("Upload file .npy để dự đoán", type=["npy"])
    if uploaded:
        try:
            arr = np.load(uploaded)
            if arr.ndim == 2:
                vector = arr.flatten()
            elif arr.ndim == 1:
                vector = arr
            else:
                raise ValueError("Định dạng numpy không hợp lệ – cần 1D hoặc 2D.")
            label = model.predict(vector)
            st.success(f"Dự đoán nhãn: {label}")
        except Exception as exc:  # pragma: no cover - safe guard for UI
            st.error(f"Không thể đọc tệp được tải lên: {exc}")


def render_log_explorer() -> None:
    st.subheader("Khám phá log thí nghiệm")
    runs = data_access.list_log_runs()
    if not runs:
        st.info("Chưa có log nào trong thư mục Thesis_log.")
        return

    experiments = sorted({run.experiment for run in runs})
    experiment = st.selectbox("Chọn nhóm thí nghiệm", options=experiments)

    filtered_runs = [run for run in runs if run.experiment == experiment]
    run_id = st.selectbox("Chọn phiên chạy", options=[run.run_id for run in filtered_runs])
    current_run = next(run for run in filtered_runs if run.run_id == run_id)

    log_file = st.selectbox(
        "Chọn snapshot",
        options=current_run.step_files,
        format_func=lambda p: p.stem,
    )
    df = cached_log_dataframe(log_file)

    metric_options = [col for col in df.columns if col != "step"]
    selected_metrics = st.multiselect(
        "Chọn chỉ số để vẽ",
        options=metric_options,
        default=["num_nodes", "num_edges"],
    )

    if not selected_metrics:
        st.warning("Chọn ít nhất một chỉ số để trực quan hóa.")
        return

    chart_df = df[["step", *selected_metrics]].set_index("step")
    st.line_chart(chart_df)


# Sidebar controls
available_envs = data_access.list_environments()
selected_overview_envs = st.sidebar.multiselect(
    "Môi trường hiển thị ở phần tổng quan",
    options=available_envs,
    default=available_envs,
)

# Tabs
overview_tab, model_tab, log_tab = st.tabs([
    "Dataset Overview",
    "Model Playground",
    "Log Explorer",
])

with overview_tab:
    render_dataset_overview(selected_overview_envs)

with model_tab:
    render_model_playground(available_envs)

with log_tab:
    render_log_explorer()

st.sidebar.caption(f"Dataset path: {config.DATA_ROOT}")
