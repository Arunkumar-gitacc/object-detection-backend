"""
OpenAI CLIP is used as a second opinion on each YOLO-World detection:
we crop the detected bounding box, embed it, embed the candidate label
as text, and check their cosine similarity. This catches cases where
YOLO-World's open-vocabulary head is over-eager about a class name.

Like the YOLO-World model, CLIP is loaded once at startup and reused.
"""

import clip
import torch
from PIL import Image

from app.config import settings


class ClipVerifier:
    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        self._model, self._preprocess = clip.load(settings.CLIP_MODEL_NAME, device=self._device)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._preprocess = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @torch.no_grad()
    def verify(self, crop: Image.Image, label: str) -> float:
        """
        Returns the cosine similarity (roughly -1..1, in practice ~0..0.4)
        between the cropped region and the text prompt "a photo of a {label}".
        Higher = more confident the crop actually contains that label.
        """
        if self._model is None or self._preprocess is None:
            raise RuntimeError("CLIP model has not been loaded yet.")

        image_input = self._preprocess(crop).unsqueeze(0).to(self._device)
        text_input = clip.tokenize([f"a photo of a {label}"]).to(self._device)

        image_features = self._model.encode_image(image_input)
        text_features = self._model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()
        return similarity

    def is_verified(self, crop: Image.Image, label: str) -> tuple[bool, float]:
        score = self.verify(crop, label)
        return score >= settings.CLIP_VERIFICATION_THRESHOLD, score


# Module-level singleton, populated during the app lifespan
verifier = ClipVerifier()
