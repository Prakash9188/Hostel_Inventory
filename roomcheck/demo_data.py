"""Synthetic hostel-room photos so the project can be tried without real data."""
import cv2
import numpy as np

W, H = 900, 600

# name, expected item, (x, y, w, h) normalised on the baseline
DEFAULT_ZONES = [
    ("Desk - laptop spot", "laptop", (0.565, 0.425, 0.155, 0.165)),
    ("Desk - bottle spot", "water bottle", (0.765, 0.420, 0.050, 0.165)),
    ("Chair", "chair", (0.35, 0.52, 0.14, 0.26)),
    ("Bed", "bed / bedding", (0.02, 0.47, 0.33, 0.35)),
]

SCENARIOS = {
    "Control - nothing removed (new angle + dimmer light)": {},
    "Laptop removed": {"laptop": False},
    "Chair removed": {"chair": False},
    "Laptop + bottle removed": {"laptop": False, "bottle": False},
}


def _background(seed=11):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (205, 200, 190), np.uint8)
    floor_y = int(H * 0.62)
    img[floor_y:] = (120, 140, 160)
    for x in range(-200, W + 200, 70):                       # floor planks
        cv2.line(img, (x, H), (x + (x - W // 2) // 6, floor_y), (95, 115, 135), 2)
    for _ in range(450):                                     # low-contrast clutter -> keypoints
        c = tuple(int(v) for v in rng.integers(90, 230, 3))
        p = (int(rng.integers(0, W)), int(rng.integers(0, floor_y + 60)))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(2, 7)), c, -1)
        else:
            cv2.rectangle(img, p, (p[0] + int(rng.integers(4, 16)), p[1] + int(rng.integers(4, 16))), c, -1)
    posters = [((60, 40, 200, 150), "ROOM 204", (80, 120, 220)), ((300, 60, 430, 170), "NO SMOKING", (60, 60, 200)),
               ((520, 30, 690, 140), "WIFI: HOSTEL_B", (160, 90, 40)), ((730, 50, 850, 200), "RULES", (60, 160, 90))]
    for (x0, y0, x1, y1), txt, col in posters:
        cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (30, 30, 30), 2)
        cv2.putText(img, txt, (x0 + 8, (y0 + y1) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (580, 190), (700, 235), (230, 230, 235), -1)   # window sill / shelf
    cv2.rectangle(img, (580, 190), (700, 235), (60, 60, 60), 2)
    noise = rng.normal(0, 2.5, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _scene(bg, laptop=True, bottle=True, chair=True, book=True):
    img = bg.copy()
    # bed
    cv2.rectangle(img, (30, 300), (300, 480), (150, 90, 70), -1)
    cv2.rectangle(img, (30, 300), (300, 480), (60, 40, 30), 3)
    cv2.rectangle(img, (40, 305), (115, 345), (240, 240, 240), -1)
    for x in range(60, 300, 40):
        cv2.line(img, (x, 350), (x, 478), (120, 70, 50), 3)
    # desk
    cv2.rectangle(img, (450, 350), (800, 375), (60, 100, 150), -1)
    cv2.rectangle(img, (450, 350), (800, 375), (30, 50, 80), 2)
    cv2.rectangle(img, (460, 375), (476, 470), (40, 70, 110), -1)
    cv2.rectangle(img, (774, 375), (790, 470), (40, 70, 110), -1)
    if laptop:
        cv2.rectangle(img, (520, 335), (645, 350), (150, 150, 155), -1)
        cv2.rectangle(img, (520, 335), (645, 350), (50, 50, 50), 2)
        cv2.rectangle(img, (530, 265), (635, 335), (50, 50, 50), -1)
        cv2.rectangle(img, (538, 272), (627, 328), (200, 170, 120), -1)
        cv2.putText(img, "HP", (565, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    if bottle:
        cv2.rectangle(img, (700, 270), (725, 350), (200, 120, 30), -1)
        cv2.rectangle(img, (700, 270), (725, 350), (90, 50, 10), 2)
        cv2.rectangle(img, (705, 258), (720, 270), (30, 30, 30), -1)
    if book:
        cv2.rectangle(img, (740, 330), (792, 350), (40, 40, 190), -1)
        cv2.rectangle(img, (740, 330), (792, 350), (20, 20, 90), 2)
    if chair:
        cv2.rectangle(img, (340, 400), (432, 420), (40, 70, 110), -1)
        cv2.rectangle(img, (345, 320), (427, 400), (60, 90, 140), -1)
        cv2.rectangle(img, (345, 320), (427, 400), (20, 40, 70), 2)
        for x in (365, 386, 407):
            cv2.line(img, (x, 322), (x, 398), (30, 50, 90), 2)
        cv2.rectangle(img, (350, 420), (362, 472), (30, 50, 90), -1)
        cv2.rectangle(img, (410, 420), (422, 472), (30, 50, 90), -1)
    return img


def make_pair(scenario: str = "Laptop removed", seed: int = 3):
    """Return (baseline, new). `new` is also viewed from a different angle with different lighting."""
    opts = SCENARIOS[scenario]
    bg = _background()
    base = cv2.GaussianBlur(_scene(bg), (3, 3), 0)
    scene = _scene(bg, **opts)
    rng = np.random.default_rng(seed)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[45, 25], [W - 20, 0], [W - 55, H - 35], [15, H - 8]])
    M = cv2.getPerspectiveTransform(src, dst)
    new = cv2.warpPerspective(scene, M, (W, H), borderValue=(70, 70, 70))
    new = new.astype(np.float32) * np.array([0.72, 0.76, 0.82]) - 6          # dimmer, warm colour cast
    new += rng.normal(0, 3, new.shape)
    new = cv2.GaussianBlur(np.clip(new, 0, 255).astype(np.uint8), (3, 3), 0)
    return base, new


def default_zones():
    from .zones import Zone
    return [Zone.from_rect(n, i, *r) for n, i, r in DEFAULT_ZONES]
