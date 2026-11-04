"""Drawing helpers (red alert boxes etc.)."""
import cv2
import numpy as np

RED, GREEN, AMBER, GRAY = (0, 0, 255), (0, 170, 0), (0, 165, 255), (128, 128, 128)
STATUS_COLOR = {"MISSING": RED, "ADDED": AMBER, "CHANGED": AMBER, "OK": GREEN, "UNCERTAIN": GRAY}


def draw_zone(img, poly_px, color, label, thick=3):
    poly = poly_px.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [poly], True, color, thick, cv2.LINE_AA)
    x, y = int(poly[:, 0, 0].min()), int(poly[:, 0, 1].min())
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y0 = max(y - th - bl - 4, 0)
    cv2.rectangle(img, (x, y0), (x + tw + 8, y0 + th + bl + 4), color, -1)
    cv2.putText(img, label, (x + 4, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def zone_label(r):
    if r.status == "MISSING":
        return f"MISSING: {r.item}"
    if r.status in ("ADDED", "CHANGED"):
        return f"{r.status}: {r.item}"
    if r.status == "UNCERTAIN":
        return f"{r.name}: not visible"
    return f"{r.name}: OK"


def tint_mask(img, mask, color=(0, 0, 255), alpha=0.45):
    out = img.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def heatmap(diff, valid, vmax=60):
    h = np.clip(diff / vmax * 255, 0, 255).astype(np.uint8)
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    h[~valid] = (60, 60, 60)
    return h


def banner(img, text, color=RED):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), color, -1)
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out
