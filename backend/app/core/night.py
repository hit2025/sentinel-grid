"""
night.py
--------
Night-time movement detection -- required capability #7 in the official
spec. Two parts:

  1. `is_night`: classify a frame as low-light using mean luminance in the
     LAB color space (the L channel), which is a standard, cheap proxy for
     "is this frame dark" without needing a timestamp or external light
     sensor -- important because a camera feed alone doesn't carry
     wall-clock time, and dusk/dawn/indoor-shadow conditions matter more
     than the literal clock time anyway.
  2. `enhance_for_detection`: CLAHE (Contrast-Limited Adaptive Histogram
     Equalization) on the L channel before running detection on a dark
     frame. Detectors trained/tuned on daylight-ish contrast miss a lot in
     genuinely dark footage; this is the standard, cheap fix that doesn't
     need a separate low-light-trained model.

Border security posts operate around the clock, and night is when this
capability matters most operationally -- so we treat "detected while dark"
as itself alert-worthy (see analytics.py's night_time_movement check),
not just a rendering nicety.
"""

from __future__ import annotations

import cv2
import numpy as np


def frame_brightness(frame: np.ndarray) -> float:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return float(np.mean(lab[:, :, 0]))


def is_night(frame: np.ndarray, threshold: float = 40.0) -> bool:
    """Threshold calibrated against real footage, not guessed: a genuinely
    well-lit real street scene (colorful, detailed, evening string-lights,
    clearly not needing low-light enhancement) measured 58-96 on this
    scale and was incorrectly flagged as 'night' at the previous
    threshold of 70 -- that was tuned on nothing but intuition. 40 keeps
    that scene correctly classified as daytime while still catching
    genuinely dark frames (a real black/corrupted frame measures ~0).

    This is still a single global-brightness heuristic, not a trained
    day/night classifier -- a scene that's mostly dark with one bright
    light source (or the reverse: mostly lit with deep shadows) can still
    fool it. If false triggers keep showing up across different real
    cameras, the next real fix is a percentile-based check (e.g. is there
    enough bright content anywhere in frame to imply ambient light)
    rather than pushing this one number further -- that's a data problem
    a single threshold can't fully solve."""
    return frame_brightness(frame) < threshold


def enhance_for_detection(frame: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
