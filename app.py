from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from highjump.inference import load_pose_model, process_video


st.set_page_config(page_title="High-jump form review", layout="wide")
st.title("High-jump form review")
st.write(
    "Analyze one attempt per video. The first run downloads pretrained YOLO11n-pose "
    "weights; processing is local and may take a few minutes on CPU."
)


@st.cache_resource
def get_model():
    return load_pose_model()


video_file = st.file_uploader("Upload a video of one attempt", type=["mp4", "mov", "avi", "mkv"])

with st.expander("Analysis settings"):
    confidence = st.slider("Person detection confidence", min_value=0.1, max_value=0.8, value=0.25, step=0.05)
    sample_rate = st.slider("Analysis sampling rate (frames per second)", min_value=4, max_value=20, value=12)

if video_file is not None and st.button("Analyze video", type="primary"):
    input_path: Path | None = None
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(video_file.name).suffix, delete=False) as temp_input:
            temp_input.write(video_file.getvalue())
            input_path = Path(temp_input.name)

        with st.spinner("Loading the pose model and analyzing the video locally..."):
            result = process_video(
                input_path,
                model=get_model(),
                confidence=confidence,
                sample_rate=sample_rate,
            )
        output_path = result.annotated_video

        st.success("Analysis complete.")
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Pose-tracked video")
            st.video(output_path.read_bytes())
        with right:
            st.subheader("Automatic event estimates")
            st.metric(
                "Takeoff",
                result.takeoff.display_time,
                delta=f"confidence: {result.takeoff.confidence}",
                delta_color="off",
            )
            st.caption(result.takeoff.explanation)
            st.metric(
                "Bar clearance",
                result.clearance.display_time,
                delta=f"confidence: {result.clearance.confidence}",
                delta_color="off",
            )
            st.caption(result.clearance.explanation)
            st.metric("Candidate bar line", f"{result.bar_y_px:.0f} px" if result.bar_y_px is not None else "Not detected")
            st.caption(
                "Event times and the bar position are estimates. Review the source video, "
                "especially when confidence is low."
            )

        st.subheader("Observable form cues")
        if result.cues:
            for cue in result.cues:
                st.markdown(f"**{cue.title}** — {cue.observation}")
                st.caption(f"{cue.guidance} · Confidence: {cue.confidence}")
        else:
            st.info("Not enough reliable pose data was available to calculate coaching cues.")

        st.warning(
            "These are 2D screening measurements, not validated coaching diagnoses. "
            "A single camera cannot establish 3D body alignment, and standard pose landmarks "
            "do not identify toe direction; plant-foot angle is not measured."
        )
        st.download_button(
            "Download analysis summary",
            data=result.summary_csv,
            file_name="high-jump-analysis.csv",
            mime="text/csv",
        )
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        for temporary_path in (input_path, output_path):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

st.caption(
    "The app runs on your laptop. The first launch needs internet access to download model weights; "
    "subsequent inference uses the local weights."
)
