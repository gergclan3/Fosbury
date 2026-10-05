from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from highjump.inference import _bar_line_y, process_video


class ArrayLike:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def cpu(self) -> ArrayLike:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = ArrayLike(np.array([[100.0, 80.0, 500.0, 430.0]]))

    def __len__(self) -> int:
        return 1


class FakeResult:
    def __init__(self, frame: np.ndarray, points: np.ndarray) -> None:
        self.boxes = FakeBoxes()
        self.keypoints = type("Keypoints", (), {"data": ArrayLike(points[None, :, :])})()
        self.frame = frame

    def plot(self) -> np.ndarray:
        return self.frame.copy()


class FakePoseModel:
    def __init__(self) -> None:
        self.index = 0

    def predict(self, source: np.ndarray, **_kwargs: object) -> list[FakeResult]:
        ankle_y = 400.0 if self.index < 10 or self.index >= 20 else 270.0
        hip_y = 280.0 if self.index < 10 or self.index >= 20 else 260.0 - (self.index - 10) * 4
        points = np.zeros((17, 3), dtype=np.float32)
        points[:, 2] = 0.9
        points[5] = (290, hip_y - 80, 0.9)
        points[6] = (350, hip_y - 80, 0.9)
        points[11] = (300, hip_y, 0.9)
        points[12] = (340, hip_y, 0.9)
        points[13] = (290, hip_y + 50, 0.9)
        points[14] = (350, hip_y + 50, 0.9)
        points[15] = (280, ankle_y, 0.9)
        points[16] = (360, ankle_y, 0.9)
        self.index += 1
        return [FakeResult(source, points)]


class BarLineDetectionTests(unittest.TestCase):
    def test_finds_a_persistent_horizontal_line(self) -> None:
        frames = [
            cv2.line(
                np.zeros((480, 640, 3), dtype=np.uint8),
                (40, 220),
                (600, 220),
                (255, 255, 255),
                3,
            )
            for _ in range(5)
        ]
        self.assertEqual(_bar_line_y(frames, 640, 480), 220.0)

    def test_does_not_infer_a_bar_without_line_evidence(self) -> None:
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        self.assertIsNone(_bar_line_y(frames, 640, 480))

    def test_processes_video_and_writes_annotated_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "attempt.mp4"
            writer = cv2.VideoWriter(
                str(source_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (640, 480),
            )
            self.assertTrue(writer.isOpened())
            for _ in range(30):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.line(frame, (40, 220), (600, 220), (255, 255, 255), 3)
                writer.write(frame)
            writer.release()

            result = process_video(source_path, FakePoseModel(), sample_rate=10)
            try:
                self.assertTrue(result.annotated_video.is_file())
                self.assertIsNotNone(result.takeoff.timestamp)
                self.assertIsNotNone(result.clearance.timestamp)
                self.assertIn("takeoff_time_seconds", result.summary_csv)
                output = cv2.VideoCapture(str(result.annotated_video))
                self.assertTrue(output.isOpened())
                self.assertEqual(int(output.get(cv2.CAP_PROP_FRAME_COUNT)), 30)
                output.release()
            finally:
                result.annotated_video.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
