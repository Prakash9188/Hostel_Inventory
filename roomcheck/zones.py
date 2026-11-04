"""Furniture zones: where on the baseline photo should we look for items?

Coordinates are stored NORMALISED (0..1) relative to the baseline image, so a zone file keeps
working when photos are resized or re-taken at another resolution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Zone:
    name: str
    item: str
    poly: list  # [[x, y], ...] normalised 0..1

    @staticmethod
    def from_rect(name, item, x, y, w, h) -> "Zone":
        return Zone(name, item, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    def pixel_poly(self, width: int, height: int) -> np.ndarray:
        pts = np.array(self.poly, dtype=np.float32) * np.array([width, height], dtype=np.float32)
        return np.round(pts).astype(np.int32)

    def mask(self, width: int, height: int) -> np.ndarray:
        m = np.zeros((height, width), np.uint8)
        cv2.fillPoly(m, [self.pixel_poly(width, height)], 255)
        return m

    def is_rect(self) -> bool:
        a = np.array(self.poly)
        return len(a) == 4 and len(set(np.round(a[:, 0], 6))) == 2 and len(set(np.round(a[:, 1], 6))) == 2

    def to_json(self) -> dict:
        if self.is_rect():
            a = np.array(self.poly)
            x0, y0, x1, y1 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
            return {"name": self.name, "item": self.item,
                    "rect": [round(float(v), 4) for v in (x0, y0, x1 - x0, y1 - y0)]}
        return {"name": self.name, "item": self.item,
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in self.poly]}


def zones_from_json(data: dict) -> list:
    out = []
    for z in data["zones"]:
        if "rect" in z:
            out.append(Zone.from_rect(z["name"], z.get("item", z["name"]), *z["rect"]))
        else:
            out.append(Zone(z["name"], z.get("item", z["name"]), z["polygon"]))
    return out


def load_zones(path: str) -> list:
    with open(path) as f:
        return zones_from_json(json.load(f))


def save_zones(zones, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"zones": [z.to_json() for z in zones]}, f, indent=2)


def select_zones_interactive(image) -> list:
    """Draw boxes on the baseline photo with the mouse (needs a desktop OpenCV window)."""
    print("Draw a box around each zone, press ENTER/SPACE to confirm it, ESC when finished.")
    rois = cv2.selectROIs("Select furniture zones", image, showCrosshair=True)
    cv2.destroyAllWindows()
    h, w = image.shape[:2]
    zones = []
    for i, (x, y, rw, rh) in enumerate(rois, 1):
        name = input(f"Zone {i} name (e.g. 'Study table'): ").strip() or f"zone{i}"
        item = input(f"  Expected item in '{name}' (e.g. 'laptop'): ").strip() or name
        zones.append(Zone.from_rect(name, item, x / w, y / h, rw / w, rh / h))
    return zones
