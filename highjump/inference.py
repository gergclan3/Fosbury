from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from highjump.analysis import (
    Analysis,
    CoachingCue,
    EventEstimate,
    PoseFrame,
    analyze_poses,
    build_summary_csv,
)

DEFAULT_MODEL = "yolo11n-pose.pt"


@dataclass(frozen=True)
class VideoAnalysis:
    annotated_video: Path
    analysis: Analysis
    bar_y_px: float | None
    summary_csv: str

    @property
    def takeoff(self) -> EventEstimate:
        return self.analysis.takeoff

    @property
    def clearance(self) -> EventEstimate:
        return self.analysis.clearance

    @property
    def cues(self) -> list[CoachingCue]:
        return self.analysis.cues


def load_pose_model(model_name: str = DEFAULT_MODEL) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Install the requirements first.") from exc
    return YOLO(model_name)


def _largest_person_keypoints(result: Any) -> np.ndarray | None:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keypoints = result.keypoints.data.cpu().numpy()
    return keypoints[int(np.argmax(areas))]


def _bar_line_y(frames: list[np.ndarray], frame_width: int, frame_height: int) -> float | None:
    votes: dict[int, int] = {}
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, frame_width // 8),
            minLineLength=int(frame_width * 0.35),
            maxLineGap=max(12, frame_width // 40),
        )
        if lines is None:
            continue
        per_frame: set[int] = set()
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = (int(value) for value in line)
            if abs(y2 - y1) <= max(4, frame_height * 0.008) and frame_height * 0.12 < (y1 + y2) / 2 < frame_height * 0.82:
                per_frame.add(round((y1 + y2) / 2 / 5) * 5)
        for y_bin in per_frame:
            votes[y_bin] = votes.get(y_bin, 0) + 1
    if not votes:
        return None
    y_bin, support = max(votes.items(), key=lambda vote: vote[1])
    required_support = max(2, int(len(frames) * 0.2))
    return float(y_bin) if support >= required_support else None


def process_video(
    video_path: str | Path,
    model: Any,
    confidence: float = 0.25,
    sample_rate: int = 12,
    image_size: int = 640,
) -> VideoAnalysis:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The video is missing valid frame-rate or image-size metadata.")

    stride = max(1, round(fps / sample_rate))
    sampled_fps = fps / stride
    descriptor, output_name = tempfile.mkstemp(suffix=".mp4", prefix="highjump-annotated-")
    os.close(descriptor)
    output_path = Path(output_name)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        sampled_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Could not initialize the annotated-video encoder.")

    poses: list[PoseFrame] = []
    bar_samples: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                results = model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False,
                )
                if not results:
                    raise RuntimeError("The pose model returned no inference result.")
                result = results[0]
                points = _largest_person_keypoints(result)
                if points is None:
                    keypoints = [(0.0, 0.0, 0.0)] * 17
                else:
                    keypoints = [tuple(float(item) for item in point[:3]) for point in points[:17]]
                poses.append(PoseFrame(frame_index, frame_index / fps, keypoints))
                annotated = result.plot()
                writer.write(annotated)
                if len(bar_samples) < 24 and frame_index % (stride * 3) == 0:
                    bar_samples.append(frame.copy())
            frame_index += 1
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        writer.release()

    if not poses:
        output_path.unlink(missing_ok=True)
        raise ValueError("No frames could be sampled from the video.")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("The annotated video could not be written.")
    if not any(any(point[2] > 0 for point in pose.keypoints) for pose in poses):
        output_path.unlink(missing_ok=True)
        raise ValueError("No person pose was detected. Try a clip with the athlete larger and unobstructed in frame.")

    try:
        bar_y = _bar_line_y(bar_samples, width, height)
        analysis = analyze_poses(poses, height, bar_y)
        return VideoAnalysis(output_path, analysis, bar_y, build_summary_csv(analysis, bar_y))
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
