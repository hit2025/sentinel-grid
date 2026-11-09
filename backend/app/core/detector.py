"""
detector.py
------------
Pluggable person-detection backend for SentinelGrid.

Two backends are supported behind one interface (`PersonDetector.detect`):

  1. HOGDetector   - OpenCV's built-in HOG + Linear SVM pedestrian detector.
                      Zero external downloads, works fully offline, CPU-only.
                      Used as the default so the prototype runs anywhere
                      out of the box (judge's laptop, CI runner, no GPU).

  2. YOLODetector   - Wraps an Ultralytics YOLOv8 model (yolov8n by default).
                      Higher accuracy + gives us class labels (person, car,
                      bag, etc.) which matters for a real border-surveillance
                      deployment. Auto-selected if `ultralytics` + weights
                      are available; the system transparently falls back to
                      HOG otherwise, so it never crashes a demo.

This split is the actual production pattern: swap the backend without
touching a single line of the tracking/analytics/alerting code above it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Protocol

import cv2
import numpy as np

logger = logging.getLogger("sentinelgrid.detector")


@dataclass
class Detection:
    """A single detected object in one frame."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str = "person"

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class DetectorBackend(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...


class HOGDetector:
    """OpenCV HOG + SVM pedestrian detector. No downloads, no GPU, no
    external calls -- guaranteed to run in any offline judging environment.
    """

    def __init__(self, hit_threshold: float = 0.0, scale: float = 1.05):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hit_threshold = hit_threshold
        self.scale = scale

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        # Downscale very large frames for real-time throughput; HOG is CPU-bound.
        scale_factor = 1.0
        work = frame
        max_dim = 720
        if max(h, w) > max_dim:
            scale_factor = max_dim / max(h, w)
            work = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))

        rects, weights = self.hog.detectMultiScale(
            work,
            winStride=(8, 8),
            padding=(8, 8),
            scale=self.scale,
            hitThreshold=self.hit_threshold,
        )

        detections: List[Detection] = []
        for (x, y, bw, bh), weight in zip(rects, weights):
            x, y, bw, bh = [int(v / scale_factor) for v in (x, y, bw, bh)]
            detections.append(
                Detection(x1=x, y1=y, x2=x + bw, y2=y + bh,
                           confidence=float(weight), label="person")
            )
        return _nms(detections, iou_threshold=0.4)


class YOLODetector:
    """Ultralytics YOLOv8 backend -- higher accuracy, multi-class, GPU-ready.
    Only instantiated if `ultralytics` is importable and weights resolve;
    the factory below falls back to HOG on any failure.
    """

    PERSON_CLASS_ID = 0  # COCO class id for "person"

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35,
                 classes: List[int] | None = None):
        from ultralytics import YOLO  # local import: optional dependency

        self.model = YOLO(weights)
        self.conf = conf
        self.classes = classes or [self.PERSON_CLASS_ID]

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame, conf=self.conf, classes=self.classes, verbose=False
        )
        detections: List[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append(
                    Detection(x1=x1, y1=y1, x2=x2, y2=y2,
                               confidence=conf, label=names.get(cls_id, "object"))
                )
        return detections


def _nms(detections: List[Detection], iou_threshold: float = 0.4) -> List[Detection]:
    """Simple greedy non-max suppression to collapse duplicate HOG boxes.

    score_threshold must be >= 0 on newer OpenCV builds (an assertion in
    cv2.dnn.NMSBoxes, not a filter we want) -- HOG's own hitThreshold=0.0
    already drops negative-score detections before this runs, so 0.0 here
    is a safe no-op floor, not an actual score filter.
    """
    if not detections:
        return []
    boxes = np.array([d.bbox for d in detections], dtype=np.float32)
    scores = np.array([d.confidence for d in detections], dtype=np.float32)
    idxs = cv2.dnn.NMSBoxes(
        bboxes=[[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes],
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten()
    return [detections[i] for i in idxs]


def build_detector(prefer: str = "auto") -> DetectorBackend:
    """Factory: try YOLO if requested/available, else guarantee HOG.

    prefer: "auto" | "yolo" | "hog"
    """
    if prefer in ("auto", "yolo"):
        try:
            det = YOLODetector()
            logger.info("Detector backend: YOLOv8 (production mode)")
            return det
        except Exception as e:  # noqa: BLE001 - intentional graceful fallback
            if prefer == "yolo":
                logger.warning("YOLO requested but unavailable (%s); falling back to HOG", e)
            else:
                logger.info("ultralytics not available (%s) -> using HOG backend", e)
    logger.info("Detector backend: OpenCV HOG (offline/demo mode)")
    return HOGDetector()
