"""
Unit tests for the geometry, tracking, and analytics core.
Run with:  PYTHONPATH=. pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.detector import Detection
from app.core.tracker import CentroidTracker
from app.core.zones import BorderLine, Zone, default_demo_geometry
from app.core.analytics import GroupBehaviorAnalyzer


def test_zone_contains_point_inside_and_outside():
    zone = Zone(1, "Test Zone", polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
    assert zone.contains((5, 5)) is True
    assert zone.contains((15, 5)) is False


def test_border_line_side_flip():
    line = BorderLine(1, "Fence", p1=(5, 0), p2=(5, 10))
    assert line.side_of((0, 5)) != line.side_of((10, 5))


def test_tracker_assigns_stable_id_across_frames():
    tracker = CentroidTracker()
    d1 = Detection(x1=10, y1=10, x2=30, y2=50, confidence=0.9)
    tracks = tracker.update([d1], timestamp=0.0)
    tid = tracks[0].track_id

    # small movement next frame -> same id should persist
    d2 = Detection(x1=13, y1=10, x2=33, y2=50, confidence=0.9)
    tracks2 = tracker.update([d2], timestamp=0.1)
    assert tracks2[0].track_id == tid


def test_tracker_drops_track_after_max_misses():
    tracker = CentroidTracker(max_misses=2)
    d1 = Detection(x1=10, y1=10, x2=30, y2=50, confidence=0.9)
    tracker.update([d1], timestamp=0.0)
    for i in range(1, 5):
        tracker.update([], timestamp=float(i))
    assert len(tracker.tracks) == 0


def test_loitering_alert_fires_after_threshold():
    zones, lines = default_demo_geometry(960, 540)
    analyzer = GroupBehaviorAnalyzer(loiter_seconds=2.0)
    tracker = CentroidTracker()

    # a "person" sitting inside the restricted zone at x=800 (zone starts ~528)
    all_alerts = []
    for t in range(0, 5):
        det = Detection(x1=790, y1=200, x2=810, y2=240, confidence=0.9)
        tracks = tracker.update([det], timestamp=float(t))
        all_alerts += analyzer.analyze(tracks, zones, lines, timestamp=float(t))

    # loitering fires once dwell crosses the threshold, then respects a
    # cooldown so it doesn't spam every frame -- so we check it fired at
    # least once across the whole dwell, not on any single frame.
    types = [a.alert_type for a in all_alerts]
    assert "loitering" in types


def test_group_formation_requires_min_size():
    zones, lines = default_demo_geometry(960, 540)
    analyzer = GroupBehaviorAnalyzer(group_min_size=3, group_persist_seconds=0.0)
    tracker = CentroidTracker()

    # only 2 people close together -> should NOT trigger group_formation
    dets = [
        Detection(x1=100, y1=100, x2=120, y2=140, confidence=0.9),
        Detection(x1=110, y1=100, x2=130, y2=140, confidence=0.9),
    ]
    tracks = tracker.update(dets, timestamp=0.0)
    alerts = analyzer.analyze(tracks, zones, lines, timestamp=0.0)
    assert "group_formation" not in [a.alert_type for a in alerts]


def test_group_formation_fires_for_three_or_more():
    zones, lines = default_demo_geometry(960, 540)
    analyzer = GroupBehaviorAnalyzer(group_min_size=3, group_persist_seconds=0.0)
    tracker = CentroidTracker()

    dets = [
        Detection(x1=100, y1=100, x2=120, y2=140, confidence=0.9),
        Detection(x1=110, y1=105, x2=130, y2=145, confidence=0.9),
        Detection(x1=120, y1=95, x2=140, y2=135, confidence=0.9),
    ]
    tracks = tracker.update(dets, timestamp=0.0)
    alerts = analyzer.analyze(tracks, zones, lines, timestamp=0.0)
    assert "group_formation" in [a.alert_type for a in alerts]


def test_unauthorized_crossing_detected_on_side_change():
    zones, lines = default_demo_geometry(960, 540)
    analyzer = GroupBehaviorAnalyzer()
    tracker = CentroidTracker()
    fence_x = 528  # matches default_demo_geometry

    # walk a single track from left (a-side) to right (b-side) across the fence
    xs = [fence_x - 40, fence_x - 10, fence_x + 10, fence_x + 40]
    all_alerts = []
    for i, x in enumerate(xs):
        det = Detection(x1=int(x - 10), y1=200, x2=int(x + 10), y2=240, confidence=0.9)
        tracks = tracker.update([det], timestamp=float(i))
        all_alerts += analyzer.analyze(tracks, zones, lines, timestamp=float(i))

    assert any(a.alert_type == "unauthorized_crossing" for a in all_alerts)
