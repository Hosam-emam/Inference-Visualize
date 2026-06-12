from pathlib import Path
import tempfile
import torch
import streamlit as st
import pandas as pd
import numpy as np



model = model.load()
# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="Video Summarization Research Platform",
    page_icon="🎬",
    layout="wide",
)

# =====================================================
# Mock Discovery Functions
# =====================================================

def discover_models():
    """
    Later:
        scan src/models/classes
        load available architectures dynamically
    """
    return [
        "CA-SUM",
        "VASNet",
        "PGL-SUM",
        "DSNet",
        "MSVA",
    ]


def discover_feature_extractors():
    """
    Later:
        scan src/feature_extractors
    """
    return [
        "GoogleNet",
        "ResNet50",
        "CLIP",
        "ViT",
    ]


# =====================================================
# Session State
# =====================================================

if "results" not in st.session_state:
    st.session_state.results = None


# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("Configuration")

uploaded_video = st.sidebar.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov", "mkv"],
)

selected_model = st.sidebar.selectbox(
    "Model Architecture",
    discover_models(),
)

selected_extractor = st.sidebar.selectbox(
    "Feature Extractor",
    discover_feature_extractors(),
)

summary_ratio = st.sidebar.slider(
    "Summary Length (%)",
    min_value=5,
    max_value=30,
    value=15,
)

run_btn = st.sidebar.button(
    "🚀 Run Inference",
    use_container_width=True,
)

# =====================================================
# Header
# =====================================================

st.title("🎬 Video Summarization Research Platform")
st.caption(
    "Interactive visualization and comparison of state-of-the-art video summarization models."
)

# =====================================================
# Video Preview
# =====================================================
video_path = ""

if uploaded_video:

    st.subheader("Input Video")

    st.video(uploaded_video)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

# =====================================================
# Inference Placeholder
# =====================================================

if run_btn and uploaded_video:

    with st.spinner("Running inference..."):

        # -------------------------------------------------
        # TODO:
        # call src/run_inference.py
        # -------------------------------------------------

        st.session_state.results = {
            "num_frames": 1200,
            "num_shots": 35,
            "summary_frames": 180,
            "importance_scores": np.random.rand(1200),
            "shot_scores": np.random.rand(35),
        }

# =====================================================
# Tabs
# =====================================================

tabs = st.tabs(
    [
        "🔄 Pipeline",
        "🖼️ Frames",
        "📈 Importance Scores",
        "🎞️ Shots",
        "✂️ Summary",
        "⚖️ Model Comparison",
    ]
)

# =====================================================
# Pipeline Tab
# =====================================================

with tabs[0]:

    st.header("Pipeline Overview")

    st.markdown("""
    1. Sample video frames
    2. Extract visual features
    3. Segment frames into shots (KTS)
    4. Predict frame importance
    5. Select summary shots (Knapsack)
    6. Generate final summarized video
    """)

# =====================================================
# Frames Tab
# =====================================================

with tabs[1]:

    st.header("Extracted Frames")

    st.info(
        "Future: show sampled frames, frame indices, and extracted features."
    )

# =====================================================
# Importance Scores Tab
# =====================================================

with tabs[2]:

    st.header("Importance Score Visualization")

    if st.session_state.results:

        scores = st.session_state.results["importance_scores"]

        df = pd.DataFrame({
            "frame": np.arange(len(scores)),
            "score": scores
        })

        st.line_chart(
            df,
            x="frame",
            y="score",
        )

    else:
        st.info("Run inference first.")

# =====================================================
# Shot Segmentation Tab
# =====================================================

with tabs[3]:

    st.header("Shot Segmentation")

    if st.session_state.results:

        st.metric(
            "Detected Shots",
            st.session_state.results["num_shots"],
        )

        st.info(
            "Future: visualize KTS boundaries and shot scores."
        )

    else:
        st.info("Run inference first.")

# =====================================================
# Summary Tab
# =====================================================

with tabs[4]:

    st.header("Generated Summary")

    if st.session_state.results:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Frames",
                st.session_state.results["num_frames"]
            )

        with col2:
            st.metric(
                "Summary Frames",
                st.session_state.results["summary_frames"]
            )

        with col3:
            compression = (
                st.session_state.results["summary_frames"]
                / st.session_state.results["num_frames"]
            ) * 100

            st.metric(
                "Compression",
                f"{compression:.2f}%"
            )

        st.info(
            "Future: display generated summary video."
        )

    else:
        st.info("Run inference first.")

# =====================================================
# Model Comparison Tab
# =====================================================

with tabs[5]:

    st.header("Model Benchmarking")

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "CA-SUM",
                "VASNet",
                "PGL-SUM",
            ],
            "F-Score": [
                0.0,
                0.0,
                0.0,
            ],
            "Inference Time (s)": [
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
    )