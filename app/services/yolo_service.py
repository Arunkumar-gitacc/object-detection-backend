"""
YOLO-World open-vocabulary detector.

YOLO-World lets us set an arbitrary list of text classes at inference time
(instead of being locked to the 80 COCO classes), which is what makes the
"detect whatever the user asks for" behaviour possible.

The model is loaded ONCE at application startup (see app.main.lifespan) and
reused across requests -- loading it per-request would be far too slow.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
from ultralytics import YOLOWorld

from app.config import settings

# Either a path to an image file, or an in-memory BGR frame (e.g. from OpenCV
# when reading a video). ultralytics' `predict(source=...)` accepts both.
DetectionSource = Union[str, np.ndarray]


@dataclass
class RawDetection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


class YoloWorldDetector:
    def __init__(self) -> None:
        self._model: YOLOWorld | None = None

    def load(self) -> None:
        self._model = YOLOWorld(settings.YOLO_WORLD_MODEL_PATH)

    def unload(self) -> None:
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(
        self,
        source: DetectionSource,
        classes: list[str],
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[RawDetection]:
        """
        Runs detection on either an image file path (str) or an in-memory
        BGR frame (numpy array, e.g. a single video frame from OpenCV).
        """
        if self._model is None:
            raise RuntimeError("YOLO-World model has not been loaded yet.")

        conf = confidence_threshold if confidence_threshold is not None else settings.DEFAULT_CONFIDENCE_THRESHOLD
        iou = iou_threshold if iou_threshold is not None else settings.DEFAULT_IOU_THRESHOLD

        # Restrict the open-vocabulary head to just the classes the user asked for
        self._model.set_classes(classes)

        results = self._model.predict(
            source=source,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        detections: list[RawDetection] = []
        for result in results:
            names = result.names  # {class_id: label}
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = names.get(cls_id, classes[cls_id] if cls_id < len(classes) else "object")
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(RawDetection(label=label, confidence=confidence, bbox=(x1, y1, x2, y2)))

        return detections


# Module-level singleton, populated during the app lifespan
detector = YoloWorldDetector()
