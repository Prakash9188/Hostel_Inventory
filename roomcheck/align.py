"""Step 1 - align the new photo to the baseline photo (keypoints + homography)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AlignResult:
    ok: bool
    reason: str = ""
    H: np.ndarray | None = None          # maps NEW image -> BASELINE image
    warped: np.ndarray | None = None     # new photo seen from the baseline viewpoint
    valid: np.ndarray | None = None      # bool mask: True where warped has real pixels
    n_kp_base: int = 0
    n_kp_new: int = 0
    n_good: int = 0                      # matches surviving Lowe's ratio test
    n_inliers: int = 0                   # matches surviving RANSAC
    match_vis: np.ndarray | None = None

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_good if self.n_good else 0.0


def _detector(method: str, nfeatures: int):
    method = method.lower()
    if method == "sift":
        return cv2.SIFT_create(nfeatures=nfeatures), cv2.NORM_L2
    if method == "orb":
        return cv2.ORB_create(nfeatures=nfeatures, fastThreshold=10), cv2.NORM_HAMMING
    raise ValueError("method must be 'sift' or 'orb'")


def _plausible(H: np.ndarray, new_shape, base_shape) -> bool:
    """Reject absurd homographies (flipped, collapsed or exploded warps)."""
    h, w = new_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(corners, H).reshape(-1, 2).astype(np.float32)
    if not np.isfinite(q).all() or not cv2.isContourConvex(q):
        return False
    ratio = cv2.contourArea(q) / float(base_shape[0] * base_shape[1])
    return 0.2 < ratio < 5.0


def align_images(base, new, method="sift", nfeatures=5000, ratio=0.75,
                 ransac_thresh=4.0, min_inliers=20) -> AlignResult:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # helps under lighting change
    g_base = clahe.apply(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY))
    g_new = clahe.apply(cv2.cvtColor(new, cv2.COLOR_BGR2GRAY))

    det, norm = _detector(method, nfeatures)
    k_base, d_base = det.detectAndCompute(g_base, None)
    k_new, d_new = det.detectAndCompute(g_new, None)
    res = AlignResult(False, n_kp_base=len(k_base or []), n_kp_new=len(k_new or []))
    if d_base is None or d_new is None or len(k_base) < 8 or len(k_new) < 8:
        res.reason = "too few keypoints (blurry / dark / featureless photo)"
        return res

    # query = new photo, train = baseline photo.  Lowe's ratio test removes ambiguous matches.
    knn = cv2.BFMatcher(norm).knnMatch(d_new, d_base, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    res.n_good = len(good)
    if len(good) < max(8, min_inliers):
        res.reason = f"only {len(good)} good matches - photos may show different rooms/areas"
        return res

    src = np.float32([k_new[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k_base[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None:
        res.reason = "homography estimation failed"
        return res
    inl = inl.ravel().astype(bool)
    res.n_inliers = int(inl.sum())
    if res.n_inliers < min_inliers:
        res.reason = f"only {res.n_inliers} RANSAC inliers (need {min_inliers})"
        return res
    if not _plausible(H, new.shape, base.shape):
        res.reason = "estimated homography is geometrically implausible"
        return res

    h, w = base.shape[:2]
    res.H = H
    res.warped = cv2.warpPerspective(new, H, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    ones = np.full(new.shape[:2], 255, np.uint8)
    valid = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))      # drop the unreliable border pixels
    res.valid = valid > 0
    inlier_matches = [m for m, keep in zip(good, inl) if keep]
    inlier_matches = sorted(inlier_matches, key=lambda m: m.distance)[:60]
    res.match_vis = cv2.drawMatches(new, k_new, base, k_base, inlier_matches, None,
                                    matchColor=(0, 200, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    res.ok = True
    res.reason = "ok"
    return res
