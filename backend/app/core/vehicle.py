"""
vehicle.py
----------
Vehicle detection and classification -- required capability #2 in the
official spec. Same swap-in pattern as detector.py:

  * production backend: YOLOv8 restricted to COCO vehicle classes
    (car, motorcycle, bus, truck) -- gives an actual class label, which is
    what "classification" means here, not just "vehicle vs not".
  * offline fallback: background-subtraction blob detector (MOG2). No
    downloads, no GPU. It can't tell a truck from a car, so it labels
    everything "vehicle" and we say that plainly -- but it correctly
    distinguishes vehicle-shaped blobs (wide, fast-moving) from
    person-shaped ones (tall, narrow) using aspect ratio + size, which is
    enough to drive the demo and the intrusion/ANPR pipeline honestly.

The MOG2 fallback needs per-camera state (it learns a background model
frame by frame), so one instance lives on the CameraSession for the whole
session -- it is NOT stateless like the HOG person detector.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

import cv2
import numpy as np

from app.core.detector import Detection

VEHICLE_COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetectorBackend(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...


class YOLOVehicleDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35):
        from ultralytics import YOLO  # optional dependency, same as detector.py

        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame, conf=self.conf, classes=list(VEHICLE_COCO_CLASSES), verbose=False
        )
        dets: List[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                dets.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(box.conf[0]),
                    label=VEHICLE_COCO_CLASSES.get(cls_id, "vehicle"),
                ))
        return dets


class BlobVehicleDetector:
    """Offline fallback: MOG2 background subtraction + shape heuristics.

    Vehicles (in typical CCTV side/front views) present as blobs that are
    wider than they are tall and larger in area than a pedestrian; people
    present as taller-than-wide blobs. This won't classify make/model, but
    it correctly separates "vehicle" from "person" traffic without any
    external weights -- exactly the constraint the problem statement itself
    calls out (no dedicated hardware, deployable anywhere).

    Tuned to reject wind-blown foliage and lighting/shadow changes, which
    otherwise get misread as "vehicles" on real outdoor footage:
      - detectShadows=True + dropping the shadow-gray pixel value removes
        the diffuse blobs that tree/cloud shadows produce.
      - a higher varThreshold makes MOG2 less trigger-happy on small
        lighting shifts.
      - min/max area are fractions of the frame, not fixed pixel counts,
        so this behaves consistently at both 960x540 (simulation) and
        1920x1080+ (real camera) resolutions.
      - a solidity check (contour area vs. its bounding box area) rejects
        the sparse, ragged blobs foliage produces -- a real vehicle
        silhouette fills most of its bounding box; leaves and branches don't.
      - a longer warmup lets the background model settle on a busy real
        scene before it starts reporting anything.
    """

    def __init__(self, min_area_frac: float = 0.004, max_area_frac: float = 0.12,
                 min_aspect: float = 1.3, max_aspect: float = 4.0, min_solidity: float = 0.45):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=70, detectShadows=True
        )
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_solidity = min_solidity
        self._warmup = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = self.min_area_frac * frame_area
        max_area = self.max_area_frac * frame_area

        fg = self.bg_sub.apply(frame)
        self._warmup += 1
        if self._warmup < 25:
            return []  # let the background model stabilize on real, textured scenes

        # detectShadows marks shadow pixels as 127 (gray) and real
        # foreground as 255 -- keep only the latter so tree/cloud shadows
        # don't get counted as moving objects.
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets: List[Detection] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area or area > max_area:
                continue
            aspect = w / max(1, h)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            contour_area = cv2.contourArea(c)
            solidity = contour_area / max(1, area)
            if solidity < self.min_solidity:
                continue  # ragged/sparse blob -- foliage, not a solid vehicle silhouette
            dets.append(Detection(x1=x, y1=y, x2=x + w, y2=y + h,
                                   confidence=0.5, label="vehicle"))
        return dets


def build_vehicle_detector(prefer: str = "auto") -> VehicleDetectorBackend:
    if prefer in ("auto", "yolo"):
        try:
            return YOLOVehicleDetector()
        except Exception:
            pass
    return BlobVehicleDetector()
