from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from statistics import median
from typing import Sequence


LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    timestamp: float
    keypoints: Sequence[Sequence[float]]


@dataclass(frozen=True)
class EventEstimate:
    timestamp: float | None
    confidence: str
    explanation: str

    @property
    def display_time(self) -> str:
        return f"{self.timestamp:.2f} s" if self.timestamp is not None else "Not detected"


@dataclass(frozen=True)
class CoachingCue:
    title: str
    observation: str
    guidance: str
    confidence: str


@dataclass(frozen=True)
class Analysis:
    takeoff: EventEstimate
    clearance: EventEstimate
    knee_drive_duration: float | None
    cues: list[CoachingCue]


def smooth_poses(frames: Sequence[PoseFrame], radius: int = 1) -> list[PoseFrame]:
    smoothed: list[PoseFrame] = []
    for index, frame in enumerate(frames):
        points: list[tuple[float, float, float]] = []
        for point_index, point in enumerate(frame.keypoints):
            nearby = [
                neighbor.keypoints[point_index]
                for neighbor in frames[max(0, index - radius) : index + radius + 1]
                if point_index < len(neighbor.keypoints)
                and len(neighbor.keypoints[point_index]) >= 3
                and neighbor.keypoints[point_index][2] >= 0.2
            ]
            if not nearby:
                points.append((0.0, 0.0, 0.0))
                continue
            weight = sum(sample[2] for sample in nearby)
            x = sum(sample[0] * sample[2] for sample in nearby) / weight
            y = sum(sample[1] * sample[2] for sample in nearby) / weight
            points.append((x, y, max(sample[2] for sample in nearby)))
        smoothed.append(PoseFrame(frame.frame_index, frame.timestamp, points))
    return smoothed


def _point(frame: PoseFrame, index: int, min_confidence: float = 0.2) -> tuple[float, float] | None:
    if index >= len(frame.keypoints):
        return None
    point = frame.keypoints[index]
    if len(point) < 3 or point[2] < min_confidence:
        return None
    return float(point[0]), float(point[1])


def _midpoint(
    frame: PoseFrame, first_index: int, second_index: int
) -> tuple[float, float] | None:
    first = _point(frame, first_index)
    second = _point(frame, second_index)
    if first is None or second is None:
        return None
    return (first[0] + second[0]) / 2, (first[1] + second[1]) / 2


def _airborne_intervals(
    frames: Sequence[PoseFrame], frame_height: int
) -> tuple[list[tuple[int, int]], float]:
    ankle_samples = [
        ankle[1]
        for frame in frames
        for ankle in (_point(frame, LEFT_ANKLE), _point(frame, RIGHT_ANKLE))
        if ankle is not None
    ]
    if len(ankle_samples) < max(6, len(frames) // 4):
        return [], 0.0

    ground_y = sorted(ankle_samples)[int((len(ankle_samples) - 1) * 0.9)]
    threshold = max(18.0, frame_height * 0.055)
    is_airborne: list[bool] = []
    for frame in frames:
        left = _point(frame, LEFT_ANKLE)
        right = _point(frame, RIGHT_ANKLE)
        is_airborne.append(
            left is not None
            and right is not None
            and left[1] < ground_y - threshold
            and right[1] < ground_y - threshold
        )

    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, airborne in enumerate(is_airborne + [False]):
        if airborne and start is None:
            start = index
        elif not airborne and start is not None:
            duration = frames[index - 1].timestamp - frames[start].timestamp
            if duration >= 0.16:
                intervals.append((start, index - 1))
            start = None
    return intervals, ground_y


def estimate_events(
    frames: Sequence[PoseFrame],
    frame_height: int,
    bar_y_px: float | None,
) -> tuple[EventEstimate, EventEstimate]:
    intervals, _ = _airborne_intervals(frames, frame_height)
    if not intervals:
        missing = EventEstimate(
            None,
            "low",
            "No sustained two-foot airborne phase was found. Check that the athlete is visible and the clip contains the full attempt.",
        )
        return missing, EventEstimate(None, "low", "Clearance cannot be estimated without a detected flight phase.")

    start, end = max(intervals, key=lambda interval: interval[1] - interval[0])
    takeoff_time = frames[start].timestamp
    takeoff = EventEstimate(
        takeoff_time,
        "low",
        "Estimated from the start of a sustained two-foot airborne pose; running strides or occlusion can shift this time.",
    )

    if bar_y_px is not None:
        candidates = [
            (index, _midpoint(frames[index], LEFT_HIP, RIGHT_HIP))
            for index in range(start, end + 1)
        ]
        candidates = [(index, hip) for index, hip in candidates if hip is not None]
        if candidates:
            closest_index, _ = min(candidates, key=lambda candidate: abs(candidate[1][1] - bar_y_px))
            clearance = EventEstimate(
                frames[closest_index].timestamp,
                "low",
                "Estimated where the tracked hip center is closest to the visually detected bar line; review against the footage.",
            )
            return takeoff, clearance

    apex_index = min(
        range(start, end + 1),
        key=lambda index: (
            _midpoint(frames[index], LEFT_HIP, RIGHT_HIP) or (0.0, float("inf"))
        )[1],
    )
    clearance = EventEstimate(
        frames[apex_index].timestamp,
        "low",
        "A usable bar/hip signal was unavailable; this is the flight-apex proxy, not a verified bar-clearance time.",
    )
    return takeoff, clearance


def _knee_drive_duration(
    frames: Sequence[PoseFrame], takeoff_time: float, frame_height: int
) -> tuple[float | None, bool]:
    window = [
        frame
        for frame in frames
        if takeoff_time - 0.6 <= frame.timestamp <= takeoff_time
    ]
    if len(window) < 3:
        return None, False
    above: list[bool] = []
    for frame in window:
        hip = _midpoint(frame, LEFT_HIP, RIGHT_HIP)
        knees = [_point(frame, LEFT_KNEE), _point(frame, RIGHT_KNEE)]
        above.append(
            hip is not None
            and any(knee is not None and knee[1] < hip[1] - frame_height * 0.025 for knee in knees)
        )
    longest = 0
    run = 0
    for value in above:
        run = run + 1 if value else 0
        longest = max(longest, run)
    sample_interval = median(
        [right.timestamp - left.timestamp for left, right in zip(window, window[1:])]
    )
    return max(0, longest - 1) * sample_interval, True


def analyze_poses(
    frames: Sequence[PoseFrame], frame_height: int, bar_y_px: float | None
) -> Analysis:
    frames = smooth_poses(frames)
    takeoff, clearance = estimate_events(frames, frame_height, bar_y_px)
    cues: list[CoachingCue] = []
    duration: float | None = None
    if takeoff.timestamp is not None:
        duration, enough_samples = _knee_drive_duration(frames, takeoff.timestamp, frame_height)
        if duration is not None and enough_samples:
            cues.append(
                CoachingCue(
                    title="Knee-drive hold (2D proxy)",
                    observation=f"A knee stayed above the hip for approximately {duration:.2f} s in the 0.6 s before estimated takeoff.",
                    guidance=(
                        "If a sustained knee drive is part of your technique goal, review whether to hold it longer. "
                        "The 0.15 s comparison point is a provisional heuristic, not a validated standard."
                        if duration < 0.15
                        else "Compare this duration with your coach's target; the app cannot judge technique intent."
                    ),
                    confidence="low — camera angle and occlusion affect the measurement",
                )
            )

    return Analysis(takeoff, clearance, duration, cues)


def build_summary_csv(analysis: Analysis, bar_y_px: float | None) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["measurement", "value", "confidence", "notes"])
    writer.writerow(
        [
            "takeoff_time_seconds",
            analysis.takeoff.timestamp if analysis.takeoff.timestamp is not None else "",
            analysis.takeoff.confidence,
            analysis.takeoff.explanation,
        ]
    )
    writer.writerow(
        [
            "clearance_time_seconds",
            analysis.clearance.timestamp if analysis.clearance.timestamp is not None else "",
            analysis.clearance.confidence,
            analysis.clearance.explanation,
        ]
    )
    writer.writerow(["detected_bar_y_pixels", bar_y_px if bar_y_px is not None else "", "low", "Image-coordinate estimate"])
    if analysis.knee_drive_duration is not None:
        writer.writerow(
            [
                "knee_above_hip_duration_seconds",
                f"{analysis.knee_drive_duration:.3f}",
                "low",
                "2D pose proxy; provisional comparison point",
            ]
        )
    for cue in analysis.cues:
        writer.writerow([cue.title, cue.observation, cue.confidence, cue.guidance])
    return output.getvalue()
