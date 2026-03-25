#!/usr/bin/env python3
"""Locate circular battery anode coordinates using Sobel edge detection.

This script intentionally focuses on one task only:
- find the circular anode center for provided images

No rotation/alignment pipeline is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class DetectionResult:
    image: str
    ok: bool
    message: str
    anode_x: Optional[float] = None
    anode_y: Optional[float] = None
    anode_r: Optional[float] = None


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag_u8


def battery_roi(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return tight ROI around battery-like region and ROI bounds (x1,y1,x2,y2)."""
    mag = sobel_magnitude(gray)

    # Adaptive threshold tuned to keep stronger Sobel edges
    thr = int(np.percentile(mag, 85))
    thr = max(30, min(180, thr))
    edge = (mag >= thr).astype(np.uint8) * 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k_close, iterations=2)
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    if not contours:
        return gray.copy(), (0, 0, w, h)

    # Select contour by area and centrality
    cx0, cy0 = w / 2.0, h / 2.0
    best = None
    best_score = -1.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 0.002 * (w * h):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        cx, cy = x + cw / 2.0, y + ch / 2.0
        dist = np.hypot(cx - cx0, cy - cy0)
        dist_penalty = 1.0 - min(1.0, dist / (0.8 * np.hypot(w, h)))
        score = area * (0.6 + 0.4 * dist_penalty)
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        return gray.copy(), (0, 0, w, h)

    x, y, cw, ch = best
    pad_x = int(0.15 * cw)
    pad_y = int(0.20 * ch)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)

    roi = gray[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def find_anode_circle(roi_gray: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Detect most plausible anode circle inside ROI."""
    mag = sobel_magnitude(roi_gray)

    # Threshold Sobel map for edge concentration
    t = int(np.percentile(mag, 82))
    t = max(25, min(170, t))
    edge = (mag >= t).astype(np.uint8) * 255
    edge = cv2.morphologyEx(
        edge, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    h, w = roi_gray.shape[:2]
    rmin = max(6, int(0.06 * min(w, h)))
    rmax = max(rmin + 2, int(0.35 * min(w, h)))

    circles = cv2.HoughCircles(
        edge,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, int(0.18 * min(w, h))),
        param1=120,
        param2=12,
        minRadius=rmin,
        maxRadius=rmax,
    )
    if circles is None:
        return None

    # Choose circle with strongest Sobel support near circumference
    cand = np.round(circles[0]).astype(np.int32)
    best = None
    best_score = -1.0
    for x, y, r in cand:
        if r <= 0:
            continue
        mask = np.zeros_like(mag, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(r), 255, 2)
        support = float(mag[mask > 0].mean()) if np.any(mask > 0) else 0.0
        if support > best_score:
            best_score = support
            best = (float(x), float(y), float(r))

    return best


def detect_anode(path: Path, debug_dir: Optional[Path]) -> DetectionResult:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return DetectionResult(image=str(path), ok=False, message="failed to read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi, (x1, y1, x2, y2) = battery_roi(gray)
    circ = find_anode_circle(roi)

    if circ is None:
        return DetectionResult(image=str(path), ok=False, message="no circular anode detected")

    cx, cy, r = circ
    gx, gy = cx + x1, cy + y1

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

        dbg = img.copy()
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.circle(dbg, (int(round(gx)), int(round(gy))), int(round(r)), (0, 0, 255), 2)
        cv2.circle(dbg, (int(round(gx)), int(round(gy))), 3, (0, 255, 255), -1)
        cv2.putText(
            dbg,
            f"anode=({gx:.1f},{gy:.1f}) r={r:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(debug_dir / f"{path.stem}_anode_debug.jpg"), dbg)

    return DetectionResult(
        image=str(path),
        ok=True,
        message="ok",
        anode_x=float(gx),
        anode_y=float(gy),
        anode_r=float(r),
    )


def collect_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file():
            out.append(p)
        else:
            out.extend(sorted(Path(".").glob(item)))
    uniq: list[Path] = []
    seen = set()
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect circular battery anode coordinates")
    parser.add_argument("inputs", nargs="+", help="image files or glob patterns")
    parser.add_argument("--debug-dir", default=None, help="save debug images here")
    parser.add_argument("--json", action="store_true", help="print JSON one object per line")
    args = parser.parse_args()

    images = collect_inputs(args.inputs)
    if not images:
        print("No matching images found")
        return 1

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    rc = 0
    for p in images:
        res = detect_anode(p, debug_dir)
        if args.json:
            print(json.dumps(res.__dict__, ensure_ascii=True))
        else:
            if res.ok:
                print(f"{res.image}: anode=({res.anode_x:.1f}, {res.anode_y:.1f}) r={res.anode_r:.1f}")
            else:
                print(f"{res.image}: ERROR - {res.message}")
                rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
