from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    """Account that owns uploaded images/videos and their detection jobs."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    jobs: Mapped[list["DetectionJob"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class DetectionJob(Base):
    """
    One row per uploaded image that has gone through the detection pipeline.
    Mirrors the role of `Post` in the generic upload flow, but scoped to
    object-detection results.
    """

    __tablename__ = "detection_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Owner of this upload
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    user: Mapped["User"] = relationship(back_populates="jobs")

    # Original file as uploaded by the client, hosted on ImageKit
    original_image_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_file_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)

    # Image with bounding boxes + labels drawn on it, hosted on ImageKit
    processed_image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    processed_file_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    file_type: Mapped[str] = mapped_column(String(20), default="image")  # "image" | "video"

    # What the caller asked YOLO-World to look for, e.g. "person,car,dog"
    requested_classes: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.25)
    clip_verification_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    total_objects_detected: Mapped[int] = mapped_column(Integer, default=0)

    # pending -> processing -> completed | failed
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    objects: Mapped[list["DetectedObject"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class DetectedObject(Base):
    """One detected bounding box belonging to a DetectionJob."""

    __tablename__ = "detected_objects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("detection_jobs.id", ondelete="CASCADE"), index=True)

    label: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    # CLIP-based label verification
    clip_verified: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    clip_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # For video jobs: this row represents one label, with the max number of
    # simultaneous instances seen across sampled frames (e.g. "3 people at
    # once, at most"). NULL for image jobs, where each row is one instance
    # and counting is done by grouping rows in the /counts endpoint.
    occurrence_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    job: Mapped["DetectionJob"] = relationship(back_populates="objects")
