from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None


class UserRead(BaseModel):
    id: int
    email: EmailStr
    full_name: str | None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DetectedObjectRead(BaseModel):
    id: int
    label: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    clip_verified: bool | None
    clip_score: float | None
    occurrence_count: int | None  # video jobs only: max simultaneous count for this label

    model_config = ConfigDict(from_attributes=True)


class ObjectCount(BaseModel):
    label: str
    count: int


class DetectionJobRead(BaseModel):
    id: int
    user_id: int
    original_image_url: str
    original_filename: str
    processed_image_url: str | None
    file_type: str
    requested_classes: str
    confidence_threshold: float
    clip_verification_enabled: bool
    total_objects_detected: int
    status: str
    error_message: str | None
    created_at: datetime
    objects: list[DetectedObjectRead] = []
    counts: list[ObjectCount] = []

    model_config = ConfigDict(from_attributes=True)


class DetectionJobSummary(BaseModel):
    """Lightweight listing item (no full object list)."""

    id: int
    original_image_url: str
    processed_image_url: str | None
    file_type: str
    total_objects_detected: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
