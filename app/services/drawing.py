"""
Draws bounding boxes, labels, and confidence scores onto an image using Pillow.
Each unique label gets a stable, deterministic color so the same class always
renders the same way across an image and across requests.
"""

import colorsys
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _color_for_label(label: str) -> tuple[int, int, int]:
    """Deterministic, visually distinct color per label."""
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _load_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_detections(
    image_path: str,
    output_path: str,
    detections: list[dict],
) -> str:
    """
    detections: list of dicts with keys label, confidence, bbox (x1, y1, x2, y2)
    Returns output_path.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        confidence = det["confidence"]
        color = _color_for_label(label)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{label} {confidence * 100:.1f}%"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_bg = [x1, max(0, y1 - text_h - 6), x1 + text_w + 8, y1]
        draw.rectangle(label_bg, fill=color)
        draw.text((x1 + 4, max(0, y1 - text_h - 4)), text, fill=(255, 255, 255), font=font)

    image.save(output_path, quality=95)
    return output_path


def draw_detections_on_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    OpenCV variant of draw_detections, for video frames (BGR numpy arrays).
    Used instead of the Pillow path so per-frame annotation is fast enough
    to run across an entire video.
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        confidence = det["confidence"]

        r, g, b = _color_for_label(label)
        bgr_color = (b, g, r)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 2)

        text = f"{label} {confidence * 100:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(0, y1 - text_h - 8)
        cv2.rectangle(annotated, (x1, label_top), (x1 + text_w + 6, y1), bgr_color, -1)
        cv2.putText(
            annotated,
            text,
            (x1 + 3, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated
