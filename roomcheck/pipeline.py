"""End-to-end pipeline: baseline + new photo + zones  ->  report with red alert boxes."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from .align import align_images, AlignResult
from .change import Params, ZoneResult, normalize_lighting, compute_change, score_zone
from . import viz


def _resize(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img.copy()
    s = max_w / w
    return cv2.resize(img, (max_w, int(round(h * s))), interpolation=cv2.INTER_AREA)


@dataclass
class Report:
    alignment: AlignResult
    results: list
    images: dict = field(default_factory=dict)

    @property
    def alerts(self):
        return [r for r in self.results if r.status in ("MISSING", "ADDED", "CHANGED")]

    @property
    def missing(self):
        return [r for r in self.results if r.status == "MISSING"]

    def to_dict(self):
        a = self.alignment
        return {
            "alignment": {"ok": a.ok, "reason": a.reason, "keypoints_baseline": a.n_kp_base,
                          "keypoints_new": a.n_kp_new, "good_matches": a.n_good,
                          "ransac_inliers": a.n_inliers, "inlier_ratio": round(a.inlier_ratio, 3)},
            "zones": [r.to_dict() for r in self.results],
            "missing_items": [r.item for r in self.missing],
        }


def check_room(baseline, new, zones, params: Params | None = None) -> Report:
    p = params or Params()
    base, new0 = _resize(baseline, p.max_w), _resize(new, p.max_w)
    H, W = base.shape[:2]
    al = align_images(base, new0, p.method, p.nfeatures, p.ratio, p.ransac_thresh, p.min_inliers)

    base_vis = base.copy()
    for z in zones:
        viz.draw_zone(base_vis, z.pixel_poly(W, H), viz.GREEN, z.name, 2)

    if not al.ok:
        res = [ZoneResult(z.name, z.item, "UNCERTAIN", note=f"alignment failed: {al.reason}") for z in zones]
        fail = viz.banner(new0, f"ALIGNMENT FAILED: {al.reason}", viz.GRAY)
        return Report(al, res, {"baseline_zones": base_vis, "annotated_original": fail,
                                "panel": _side_by_side(base_vis, fail)})

    aligned = normalize_lighting(base, al.warped, al.valid)
    maps = compute_change(base, aligned, al.valid, p)
    results = [score_zone(z, maps, al.valid, p, (W, H)) for z in zones]

    # --- visual outputs --------------------------------------------------------------
    flagged_mask = np.zeros((H, W), bool)
    for z, r in zip(zones, results):
        if r.status in ("MISSING", "ADDED", "CHANGED"):
            flagged_mask |= (z.mask(W, H) > 0) & maps.changed
    aligned_vis = viz.tint_mask(al.warped, flagged_mask)
    orig_vis = new0.copy()
    Hinv = np.linalg.inv(al.H)
    for z, r in zip(zones, results):
        col, label = viz.STATUS_COLOR[r.status], viz.zone_label(r)
        viz.draw_zone(aligned_vis, z.pixel_poly(W, H), col, label)
        pts = z.pixel_poly(W, H).astype(np.float32).reshape(-1, 1, 2)
        back = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)      # project zone onto the ORIGINAL view
        viz.draw_zone(orig_vis, np.round(back).astype(np.int32), col, label)

    edges = np.zeros((H, W, 3), np.uint8)
    edges[maps.lost] = (255, 120, 0)      # blue-ish: lost from baseline
    edges[maps.gained] = (0, 200, 255)    # yellow: new in photo
    edges[maps.edges_base & ~maps.lost] = (90, 90, 90)

    images = {
        "baseline_zones": base_vis,
        "aligned_new": aligned_vis,
        "annotated_original": orig_vis,
        "change_heatmap": viz.heatmap(maps.diff, al.valid),
        "change_mask": (maps.changed.astype(np.uint8) * 255),
        "edge_change": edges,
        "matches": al.match_vis,
        "panel": _side_by_side(base_vis, orig_vis),
    }
    return Report(al, results, images)


def _side_by_side(a, b):
    h = a.shape[0]
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    return np.hstack([a, np.full((h, 8, 3), 255, np.uint8), b])
