from __future__ import annotations

import unittest

from highjump.analysis import PoseFrame, analyze_poses, build_summary_csv, estimate_events


def make_pose(timestamp: float, ankle_y: float, hip_y: float) -> PoseFrame:
    points = [(320.0, 200.0, 0.9) for _ in range(17)]
    points[5] = (300.0, hip_y - 90.0, 0.9)
    points[6] = (340.0, hip_y - 90.0, 0.9)
    points[11] = (305.0, hip_y, 0.9)
    points[12] = (335.0, hip_y, 0.9)
    points[13] = (290.0, hip_y + 55.0, 0.9)
    points[14] = (350.0, hip_y + 55.0, 0.9)
    points[15] = (285.0, ankle_y, 0.9)
    points[16] = (355.0, ankle_y, 0.9)
    return PoseFrame(round(timestamp * 10), timestamp, points)


class EventAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = [
            make_pose(index / 10, 500.0, 350.0)
            for index in range(10)
        ] + [
            make_pose(index / 10, 360.0, 310.0 - (index - 10) * 12.0)
            for index in range(10, 19)
        ] + [
            make_pose(index / 10, 500.0, 350.0)
            for index in range(19, 26)
        ]

    def test_detects_airborne_transition_and_bar_proximity(self) -> None:
        takeoff, clearance = estimate_events(self.frames, 720, 220.0)
        self.assertEqual(takeoff.timestamp, 1.0)
        self.assertIsNotNone(clearance.timestamp)
        self.assertEqual(takeoff.confidence, "low")

    def test_uses_apex_proxy_when_bar_line_is_missing(self) -> None:
        _, clearance = estimate_events(self.frames, 720, None)
        self.assertIsNotNone(clearance.timestamp)
        self.assertEqual(clearance.confidence, "low")
        self.assertIn("apex proxy", clearance.explanation)

    def test_reports_no_event_when_flight_is_not_visible(self) -> None:
        frames = [make_pose(index / 10, 500.0, 350.0) for index in range(20)]
        takeoff, clearance = estimate_events(frames, 720, None)
        self.assertIsNone(takeoff.timestamp)
        self.assertIsNone(clearance.timestamp)

    def test_summary_has_machine_readable_measurements(self) -> None:
        analysis = analyze_poses(self.frames, 720, None)
        self.assertIsNotNone(analysis.takeoff.timestamp)
        summary = build_summary_csv(analysis, None)
        self.assertIn("takeoff_time_seconds", summary)
        self.assertIn("clearance_time_seconds", summary)
        self.assertTrue(analysis.cues)
        self.assertIn(analysis.cues[0].title, summary)


if __name__ == "__main__":
    unittest.main()
