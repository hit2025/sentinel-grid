"""
face.py
-------
Face detection -- one of the eight explicitly required capabilities in the
official SIH26187 spec ("Support facial recognition... through software").

We implement *detection* (localizing faces in frame), not full recognition
(matching against a watchlist database) -- recognition needs an enrolled
face database and a legal/privacy framework for a border security deployment
that is out of scope for a hackathon prototype, and we say so plainly rather
than overclaim. Detection is the correct, honest scope: it's what feeds a
recognition system, and it's independently useful (face count in frame,
face-detected event logging).

Uses OpenCV's bundled Haar cascade (haarcascade_frontalface_default.xml) --
ships with opencv-python, zero downloads, zero GPU. This only runs on real
imagery (uploaded video / real camera frames); synthetic simulation frames
have no real face texture for a cascade to find, so we don't run it there
-- see pipeline.py for that decision.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from app.core.detector import Detection


class FaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: tuple[int, int] = (24, 24)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame: np.ndarray, offset: tuple[int, int] = (0, 0)) -> List[Detection]:
        """Detect faces in `frame` (typically a cropped person ROI, but
        works on a full frame too). `offset` translates local (crop)
        coordinates back into full-frame coordinates when called on a crop.
        """
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size,
        )
        ox, oy = offset
        return [
            Detection(x1=int(x + ox), y1=int(y + oy), x2=int(x + w + ox), y2=int(y + h + oy),
                      confidence=0.6, label="face")
            for (x, y, w, h) in faces
        ]
