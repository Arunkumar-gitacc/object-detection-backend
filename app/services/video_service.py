"""
Video object detection & counting.

Videos are processed frame-by-frame with OpenCV. To keep this tractable on
CPU, YOLO-World only actually runs on every Nth frame
(settings.VIDEO_FRAME_SAMPLE_RATE) -- frames in between reuse the most
recent detections, so the annotated output video still looks smooth.

Counting semantics for video: rather than a running total (which would
double-count the same object across many frames), each label's count is the
*maximum number seen at once* in any sampled frame -- e.g. "at most 3 people
in frame at the same time". CLIP verification is checked at most once per
label per frame (only when a new best-confidence example for that label
appears), to keep cost down.
"""

import os
from collections import defaultdict

import cv2
from PIL import Image

from app.config import settings
from app.services.clip_service import verifier as clip_verifier
from app.services.drawing import draw_detections_on_frame
from app.services.yolo_service import RawDetection, detector as yolo_detector


def _to_draw_dicts(detections: list[RawDetection]) -> list[dict]:
    return [{"label": d.label, "confidence": d.confidence, "bbox": d.bbox} for d in detections]


def process_video(
    input_path: str,
    output_path: str,
    classes: list[str],
    confidence_threshold: float,
    use_clip_verification: bool,
) -> tuple[list[dict], int]:
    """
    Runs detection across the video, writes an annotated copy to
    `output_path`, and returns (aggregated_detections, frames_sampled).

    Each entry in aggregated_detections is:
        {label, confidence, bbox, clip_verified, clip_score, max_concurrent_count}
    """
    if not os.path.exists(input_path):
        raise RuntimeError(f"Video file not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video dimensions -- is this a valid video file?")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    sample_rate = max(1, settings.VIDEO_FRAME_SAMPLE_RATE)

    # label -> best example seen so far: {confidence, bbox, clip_score}
    best_examples: dict[str, dict] = {}
    max_concurrent_count: dict[str, int] = defaultdict(int)

    last_detections: list[RawDetection] = []
    frame_index = 0
    frames_sampled = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_rate == 0:
                last_detections = yolo_detector.detect(
                    source=frame,
                    classes=classes,
                    confidence_threshold=confidence_threshold,
                )
                frames_sampled += 1

                counts_this_frame: dict[str, int] = defaultdict(int)
                for det in last_detections:
                    counts_this_frame[det.label] += 1

                    current_best = best_examples.get(det.label)
                    if current_best is None or det.confidence > current_best["confidence"]:
                        clip_score = None
                        if use_clip_verification:
                            x1, y1, x2, y2 = det.bbox
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            crop = Image.fromarray(rgb_frame).crop((x1, y1, x2, y2))
                            if crop.width > 0 and crop.height > 0:
                                clip_score = clip_verifier.verify(crop, det.label)

                        best_examples[det.label] = {
                            "confidence": det.confidence,
                            "bbox": det.bbox,
                            "clip_score": clip_score,
                        }

                for label, count in counts_this_frame.items():
                    max_concurrent_count[label] = max(max_concurrent_count[label], count)

            annotated_frame = draw_detections_on_frame(frame, _to_draw_dicts(last_detections))
            writer.write(annotated_frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    aggregated: list[dict] = []
    for label, example in best_examples.items():
        clip_score = example["clip_score"]
        aggregated.append(
            {
                "label": label,
                "confidence": example["confidence"],
                "bbox": example["bbox"],
                "clip_verified": (
                    clip_score >= settings.CLIP_VERIFICATION_THRESHOLD if clip_score is not None else None
                ),
                "clip_score": clip_score,
                "max_concurrent_count": max_concurrent_count[label],
            }
        )

    return aggregated, frames_sampled
