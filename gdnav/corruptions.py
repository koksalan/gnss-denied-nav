"""Hard test scenarios for the onboard camera: weather, light, sensor, datalink and navigation-sensor errors.

Applied to the RAW camera frame (before north-up warping), so the whole pipeline is tested: retrieval,
matching and pose. Image effects are scaled to the frame size. Navigation errors (compass, barometer) are
returned as multipliers/offsets for the heading and height the localizer is given.

These are deliberately implemented differently from the training augmentations (scripts/train_finetune.py
`--aug robust`, which work on the 224 px patch with kornia), so a robust model cannot simply memorise them.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Scenario:
    name: str
    label: str                       # human-readable, for tables
    image: callable = None           # (img, rng) -> img
    heading_err_deg: float = 0.0
    height_scale: float = 1.0


def _lowfreq(h: int, w: int, rng: np.random.Generator, cells=(3, 6, 12, 24)) -> np.ndarray:
    """Fractal (multi-octave) smooth noise in [0, 1], like clouds / fog density."""
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for c in cells:
        g = rng.random((c, max(2, int(round(c * w / h))))).astype(np.float32)
        acc += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    acc /= tot
    return (acc - acc.min()) / max(float(acc.max() - acc.min()), 1e-6)


def _blend(img: np.ndarray, color, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return np.clip(img.astype(np.float32) * (1 - a) + np.asarray(color, np.float32) * a, 0, 255).astype(np.uint8)


def fog(img, rng, base=0.45, var=0.3):
    h, w = img.shape[:2]
    d = base + var * _lowfreq(h, w, rng)
    return _blend(img, (205, 208, 212), d)


def low_light(img, rng):
    """Dusk: ~6x less light, blue cast; auto-gain brings brightness back but amplifies sensor noise
    and quantisation (few photons -> few grey levels)."""
    x = (img.astype(np.float32) / 255) ** 1.4 * 0.16 * np.array([0.85, 0.95, 1.12], np.float32)
    x = x * 255
    x = x + rng.normal(0, 1.0, x.shape).astype(np.float32) * np.sqrt(np.maximum(x, 1.0)) * 0.8   # shot noise
    x = np.round(np.clip(x, 0, 255) / 4) * 4                                                     # low bit depth
    gain = 0.42 * 255 / max(float(x.mean()), 1.0)
    return np.clip(x * gain, 0, 255).astype(np.uint8)


def clouds(img, rng, cover=0.35):
    """Opaque cumulus between camera and ground (+ their shadows), ~`cover` of the frame."""
    h, w = img.shape[:2]
    c = _lowfreq(h, w, rng)
    thr = np.quantile(c, 1 - cover)
    m = np.clip((c - thr) / 0.08, 0, 1)
    sh = np.roll(m, (int(0.06 * h), int(0.08 * w)), (0, 1)) * 0.45
    out = _blend(img, (0, 0, 0), sh)
    return _blend(out, (244, 246, 248), m)


def motion_blur(img, rng, frac=0.012):
    """Vibration / fast motion: linear blur ~1.2 % of the frame width, random direction."""
    h, w = img.shape[:2]
    n = max(3, int(frac * w)) | 1
    k = np.zeros((n, n), np.float32)
    k[n // 2] = 1
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - .5, n / 2 - .5), float(rng.uniform(0, 180)), 1.0), (n, n))
    return cv2.filter2D(img, -1, k / k.sum())


def weak_link(img, rng, down=6, quality=20):
    """Low-bandwidth video link: 1/6 resolution, heavy JPEG, upsampled back."""
    h, w = img.shape[:2]
    s = cv2.resize(img, (w // down, h // down), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(s, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    s = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)


def overexposure(img, rng):
    """Auto-exposure failure / sun glare: x1.9 gain, clipped, plus a bright glare blob."""
    h, w = img.shape[:2]
    x = img.astype(np.float32) * 1.9
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.25 * w) ** 2)))
    x = x * (1 - 0.6 * g[..., None]) + 255 * 0.6 * g[..., None]
    return np.clip(x, 0, 255).astype(np.uint8)


def color_shift(img, rng):
    """Different sensor / season: hue rotated, washed-out colours, lower contrast."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + 10) % 180          # ~20 deg hue rotation
    hsv[..., 1] *= 0.35
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    return np.clip((out - 128) * 0.75 + 140, 0, 255).astype(np.uint8)


def combined(img, rng):
    return weak_link(fog(img, rng, base=0.3, var=0.25), rng, down=4, quality=30)


SCENARIOS = [
    Scenario("clean", "clean"),
    Scenario("fog", "fog / haze", fog),
    Scenario("low_light", "dusk, sensor noise", low_light),
    Scenario("clouds", "35 % cloud cover", clouds),
    Scenario("motion_blur", "vibration blur", motion_blur),
    Scenario("weak_link", "weak datalink (1/6 res, JPEG 20)", weak_link),
    Scenario("overexposure", "overexposure / glare", overexposure),
    Scenario("color_shift", "colour / season shift", color_shift),
    Scenario("heading_err", "compass error +10 deg", heading_err_deg=10.0),
    Scenario("alt_high", "baro altitude +20 %", height_scale=1.2),
    Scenario("alt_low", "baro altitude -20 %", height_scale=0.8),
    Scenario("combined", "fog + weak link + compass 5 deg + baro +10 %", combined, heading_err_deg=5.0, height_scale=1.1),
]
BY_NAME = {s.name: s for s in SCENARIOS}


def apply(s: Scenario, img: np.ndarray, height_m: float, heading_deg: float, seed: int):
    """Returns (image, height given to the localizer, heading given to the localizer)."""
    rng = np.random.default_rng(seed)
    out = s.image(img, rng) if s.image else img
    return out, height_m * s.height_scale, heading_deg + s.heading_err_deg
