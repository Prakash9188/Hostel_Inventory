"""Steps 2-3 - photometric normalisation, change maps, per-zone decision."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class Params:
    # alignment
    method: str = "sift"           # "sift" (robust) or "orb" (fast)
    nfeatures: int = 5000
    ratio: float = 0.75            # Lowe ratio test
    ransac_thresh: float = 4.0     # px
    min_inliers: int = 20
    # change detection
    color_thresh: float = 22.0     # LAB distance that counts as "pixel changed"
    edge_tol: int = 3              # px of misalignment tolerated when comparing edges
    min_blob: int = 120            # ignore changed blobs smaller than this (px)
    zone_change_thresh: float = 0.10   # fraction of zone pixels changed
    edge_change_thresh: float = 0.25   # fraction of zone edges lost/gained
    min_coverage: float = 0.60     # zone must be at least this visible in the new photo
    max_w: int = 1100


@dataclass
class Maps:
    diff: np.ndarray
    changed: np.ndarray       # bool
    edges_base: np.ndarray    # bool
    edges_new: np.ndarray     # bool
    lost: np.ndarray          # bool: baseline edge with no counterpart in new photo
    gained: np.ndarray        # bool: new edge with no counterpart in baseline


@dataclass
class ZoneResult:
    name: str
    item: str
    status: str               # OK | MISSING | ADDED | CHANGED | UNCERTAIN
    score: float = 0.0        # >= 1.0 means alert
    change_ratio: float = 0.0
    lost_edges: float = 0.0
    gained_edges: float = 0.0
    coverage: float = 0.0
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        for k in ("score", "change_ratio", "lost_edges", "gained_edges", "coverage"):
            d[k] = round(float(d[k]), 3)
        return d


def normalize_lighting(base, aligned, valid):
    """Match per-channel mean/std (in LAB) of the aligned photo to the baseline,
    using only pixels that exist in both. Removes global brightness / colour-cast changes."""
    b = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
    if valid.sum() < 500:
        return aligned
    out = a.copy()
    for c in range(3):
        bv, av = b[..., c][valid], a[..., c][valid]
        scale = np.clip(bv.std() / (av.std() + 1e-6), 0.5, 2.0)
        out[..., c] = (a[..., c] - av.mean()) * scale + bv.mean()
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _drop_small(mask_u8, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros_like(mask_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def compute_change(base, aligned, valid, p: Params) -> Maps:
    # (a) colour difference in LAB (lightness down-weighted so shadows matter less)
    lb = cv2.cvtColor(cv2.GaussianBlur(base, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    la = cv2.cvtColor(cv2.GaussianBlur(aligned, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = lb - la
    diff = np.sqrt((0.5 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    changed = ((diff > p.color_thresh) & valid).astype(np.uint8) * 255
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    changed = _drop_small(changed, p.min_blob) > 0

    # (b) structural difference: edges with a small tolerance for residual misalignment
    gb = cv2.GaussianBlur(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
    ga = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
    eb, ea = cv2.Canny(gb, 40, 120) > 0, cv2.Canny(ga, 40, 120) > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.edge_tol + 1,) * 2)
    near_b = cv2.dilate(eb.astype(np.uint8), k) > 0
    near_a = cv2.dilate(ea.astype(np.uint8), k) > 0
    return Maps(diff, changed, eb & valid, ea & valid, eb & ~near_a & valid, ea & ~near_b & valid)


def score_zone(zone, maps: Maps, valid, p: Params, size) -> ZoneResult:
    W, H = size
    zmask = zone.mask(W, H) > 0
    area = int(zmask.sum())
    zv = zmask & valid
    coverage = zv.sum() / max(area, 1)
    r = ZoneResult(zone.name, zone.item, "OK", coverage=coverage)
    if coverage < p.min_coverage:
        r.status, r.note = "UNCERTAIN", "zone not sufficiently visible in the new photo"
        return r

    n = int(zv.sum())
    r.change_ratio = float(maps.changed[zv].mean())
    floor = max(30.0, 0.004 * n)                          # avoid dividing by ~0 on blank surfaces
    lost_n, gained_n = int(maps.lost[zv].sum()), int(maps.gained[zv].sum())
    r.lost_edges = lost_n / max(float(maps.edges_base[zv].sum()), floor)
    r.gained_edges = gained_n / max(float(maps.edges_new[zv].sum()), floor)
    edge_change = max(r.lost_edges, r.gained_edges)
    r.score = 0.5 * r.change_ratio / p.zone_change_thresh + 0.5 * edge_change / p.edge_change_thresh

    # Two independent pieces of evidence are required: colour AND structure changed.
    if r.change_ratio >= p.zone_change_thresh and edge_change >= p.edge_change_thresh:
        if lost_n > 1.25 * gained_n:
            r.status, r.note = "MISSING", "structure present in baseline has disappeared"
        elif gained_n > 1.25 * lost_n:
            r.status, r.note = "ADDED", "new structure appeared that was not in baseline"
        else:
            r.status, r.note = "CHANGED", "item moved / replaced / different"
    return r
