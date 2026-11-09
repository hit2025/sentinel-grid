"""
anpr.py
-------
Automatic Number Plate Recognition -- required capability #4 in the
official spec. Two real stages, both genuinely running (not mocked):

  1. Localization: OpenCV's bundled Haar cascade for plate-like rectangular
     regions (haarcascade_russian_plate_number.xml -- despite the name,
     it's a generic rectangular-plate detector commonly reused for ANPR
     prototypes worldwide, not Russia-specific; ships with opencv-python).
  2. Reading: Tesseract OCR (via pytesseract) on the localized, preprocessed
     plate crop, restricted to an alphanumeric whitelist and cleaned with a
     regex so junk reads are dropped rather than logged as a false plate.

Runs on any BGR image: a real vehicle crop from CCTV footage, or a
synthetic plate we render ourselves for the deterministic demo (see
simulator.py) -- the OCR does not know or care which; it's reading real
pixels either way. That's a deliberate design choice: the demo's ANPR
result is not faked, it is Tesseract actually reading rendered text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:  # noqa: BLE001
    _HAS_TESSERACT = False

_PLATE_CHARS = re.compile(r"[^A-Z0-9]")


@dataclass
class PlateRead:
    text: str
    bbox: tuple[int, int, int, int]  # relative to the crop passed in
    raw_ocr: str


class PlateReader:
    def __init__(self, min_neighbors: int = 5, min_size: tuple[int, int] = (60, 20)):
        cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def read(self, vehicle_crop: np.ndarray) -> Optional[PlateRead]:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        plates = self.cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=self.min_neighbors, minSize=self.min_size
        )
        if len(plates) == 0:
            return None

        # take the largest candidate region (most likely the real plate)
        x, y, w, h = max(plates, key=lambda p: p[2] * p[3])
        plate_gray = gray[y:y + h, x:x + w]
        if plate_gray.size == 0:
            return None

        text = self._ocr(plate_gray)
        if not text:
            return None
        return PlateRead(text=text, bbox=(int(x), int(y), int(x + w), int(y + h)), raw_ocr=text)

    def _ocr(self, plate_gray: np.ndarray) -> Optional[str]:
        if not _HAS_TESSERACT:
            return None
        # upscale + threshold: Tesseract reads clean high-contrast text far
        # better than a small, low-contrast plate crop straight off a camera.
        scaled = cv2.resize(plate_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            raw = pytesseract.image_to_string(
                thresh,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
        except Exception:  # noqa: BLE001 - tesseract not on PATH, etc.
            return None

        cleaned = _PLATE_CHARS.sub("", raw.upper())
        if len(cleaned) < 4:  # too short to plausibly be a real plate read
            return None
        return cleaned
