import os
import shutil
import uuid
from collections import Counter

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models import DetectedObject, DetectionJob, User
from app.schemas import DetectionJobRead, DetectionJobSummary, ObjectCount
from app.security import get_current_user
from app.services import imagekit_service, video_service
from app.services.clip_service import verifier as clip_verifier
from app.services.drawing import draw_detections
from app.services.yolo_service import detector as yolo_detector

router = APIRouter(prefix="/api/v1", tags=["detection"])

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _is_video(file: UploadFile, ext: str) -> bool:
    return (file.content_type or "").startswith("video/") or ext.lower() in VIDEO_EXTENSIONS


def _aggregate_counts(objects: list[DetectedObject]) -> list[ObjectCount]:
    """
    For image jobs, each DetectedObject row is one instance (occurrence_count
    is None -> counts as 1). For video jobs, each row already represents the
    max simultaneous count for that label, stored in occurrence_count.
    """
    counts: Counter[str] = Counter()
    for obj in objects:
        counts[obj.label] += obj.occurrence_count or 1
    return [ObjectCount(label=label, count=count) for label, count in counts.items()]


def _job_to_read(job: DetectionJob) -> DetectionJobRead:
    data = DetectionJobRead.model_validate(job)
    data.counts = _aggregate_counts(job.objects)
    return data


async def _run_image_pipeline(
    original_path: str,
    classes: list[str],
    confidence_threshold: float,
    use_clip_verification: bool,
) -> list[dict]:
    raw_detections = yolo_detector.detect(
        source=original_path,
        classes=classes,
        confidence_threshold=confidence_threshold,
    )

    source_image = Image.open(original_path).convert("RGB")
    enriched: list[dict] = []
    for det in raw_detections:
        x1, y1, x2, y2 = det.bbox
        clip_verified: bool | None = None
        clip_score: float | None = None

        if use_clip_verification:
            crop = source_image.crop((x1, y1, x2, y2))
            if crop.width > 0 and crop.height > 0:
                clip_verified, clip_score = clip_verifier.is_verified(crop, det.label)

        enriched.append(
            {
                "label": det.label,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "clip_verified": clip_verified,
                "clip_score": clip_score,
                "occurrence_count": None,
            }
        )
    return enriched


def _run_video_pipeline(
    original_path: str,
    processed_path: str,
    classes: list[str],
    confidence_threshold: float,
    use_clip_verification: bool,
) -> list[dict]:
    aggregated, _frames_sampled = video_service.process_video(
        input_path=original_path,
        output_path=processed_path,
        classes=classes,
        confidence_threshold=confidence_threshold,
        use_clip_verification=use_clip_verification,
    )
    for det in aggregated:
        det["occurrence_count"] = det.pop("max_concurrent_count")
    return aggregated


async def _process_upload(
    file: UploadFile,
    classes: list[str],
    confidence_threshold: float,
    use_clip_verification: bool,
    current_user: User,
    session: AsyncSession,
) -> DetectionJob:
    """
    1. Save to server temp file
    2. Run detection (image: YOLO-World + optional CLIP / video: sampled frames)
    3. Draw bounding boxes -> processed temp file
    4. Upload original + processed to ImageKit CDN
    5. Insert DetectionJob + DetectedObject rows into PostgreSQL
    6. Cleanup local temp files (finally block)
    """
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    is_video = _is_video(file, ext)
    processed_ext = ".mp4" if is_video else ext

    unique_id = uuid.uuid4().hex
    original_temp_path = os.path.join(settings.TEMP_DIR, f"{unique_id}_original{ext}")
    processed_temp_path = os.path.join(settings.TEMP_DIR, f"{unique_id}_processed{processed_ext}")

    job = DetectionJob(
        user_id=current_user.id,
        original_image_url="",
        original_filename=file.filename or f"{unique_id}{ext}",
        file_type="video" if is_video else "image",
        requested_classes=",".join(classes),
        confidence_threshold=confidence_threshold,
        clip_verification_enabled=use_clip_verification,
        status="processing",
    )

    try:
        # 1. Stream upload to local disk (low memory footprint even for large files)
        with open(original_temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2 + 3. Detect and draw, branching on media type
        if is_video:
            enriched_detections = _run_video_pipeline(
                original_temp_path, processed_temp_path, classes, confidence_threshold, use_clip_verification
            )
        else:
            enriched_detections = await _run_image_pipeline(
                original_temp_path, classes, confidence_threshold, use_clip_verification
            )
            draw_detections(original_temp_path, processed_temp_path, enriched_detections)

        # 4. Upload both original and processed files to ImageKit CDN
        original_upload = imagekit_service.upload_to_imagekit(
            original_temp_path, f"original_{job.original_filename}", tags=["backend-upload", "original"]
        )
        if not original_upload.success:
            raise RuntimeError(f"ImageKit upload failed for original file: {original_upload.raw_error}")

        processed_upload = imagekit_service.upload_to_imagekit(
            processed_temp_path, f"processed_{job.original_filename}", tags=["backend-upload", "processed"]
        )
        if not processed_upload.success:
            raise RuntimeError(f"ImageKit upload failed for processed file: {processed_upload.raw_error}")

        # 5. Insert DetectionJob + DetectedObject rows
        job.original_image_url = original_upload.url
        job.original_file_id = original_upload.file_id
        job.processed_image_url = processed_upload.url
        job.processed_file_id = processed_upload.file_id
        job.total_objects_detected = sum(det["occurrence_count"] or 1 for det in enriched_detections)
        job.status = "completed"

        session.add(job)
        await session.flush()  # get job.id before creating child rows

        for det in enriched_detections:
            x1, y1, x2, y2 = det["bbox"]
            session.add(
                DetectedObject(
                    job_id=job.id,
                    label=det["label"],
                    confidence=det["confidence"],
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    clip_verified=det["clip_verified"],
                    clip_score=det["clip_score"],
                    occurrence_count=det["occurrence_count"],
                )
            )

        await session.commit()
        await session.refresh(job)
        return job

    except Exception as exc:
        await session.rollback()
        job.status = "failed"
        job.error_message = str(exc)
        session.add(job)
        await session.commit()
        await session.refresh(job)
        raise HTTPException(status_code=500, detail=f"Detection failed for '{file.filename}': {exc}") from exc

    finally:
        # 6. Cleanup: always remove local temp files and close the upload stream
        for path in (original_temp_path, processed_temp_path):
            if os.path.exists(path):
                os.unlink(path)
        await file.close()


@router.post("/detect", response_model=list[DetectionJobRead])
async def detect_objects(
    files: list[UploadFile] = File(..., description="One or more images or videos to run detection on"),
    classes: str = Form(..., description="Comma-separated object classes to detect, e.g. 'person,car,dog'"),
    confidence_threshold: float = Form(default=settings.DEFAULT_CONFIDENCE_THRESHOLD),
    use_clip_verification: bool = Form(default=settings.ENABLE_CLIP_VERIFICATION_DEFAULT),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    Upload single or multiple images/videos and run YOLO-World detection +
    counting (+ optional CLIP verification) on each one.
    """
    class_list = [c.strip() for c in classes.split(",") if c.strip()]
    if not class_list:
        raise HTTPException(status_code=400, detail="Provide at least one class in 'classes'.")

    jobs = [
        await _process_upload(file, class_list, confidence_threshold, use_clip_verification, current_user, session)
        for file in files
    ]
    return [_job_to_read(job) for job in jobs]


@router.get("/jobs", response_model=list[DetectionJobSummary])
async def list_jobs(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(DetectionJob)
        .where(DetectionJob.user_id == current_user.id)
        .order_by(DetectionJob.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def _get_owned_job(job_id: int, current_user: User, session: AsyncSession) -> DetectionJob:
    job = await session.get(DetectionJob, job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Detection job not found.")
    return job


@router.get("/jobs/{job_id}", response_model=DetectionJobRead)
async def get_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    job = await _get_owned_job(job_id, current_user, session)
    return _job_to_read(job)


@router.get("/jobs/{job_id}/counts", response_model=list[ObjectCount])
async def get_job_counts(
    job_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    job = await _get_owned_job(job_id, current_user, session)
    return _aggregate_counts(job.objects)


@router.get("/jobs/{job_id}/download")
async def download_processed_file(
    job_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    Same-origin download proxy: streams the processed (annotated) image or
    video from ImageKit through this API, so the client can trigger a
    download without exposing/relying on the raw CDN URL.
    """
    job = await _get_owned_job(job_id, current_user, session)
    if not job.processed_image_url:
        raise HTTPException(status_code=404, detail="Processed file is not available yet.")

    client = httpx.AsyncClient()
    upstream_request = client.build_request("GET", job.processed_image_url)
    upstream_response = await client.send(upstream_request, stream=True)

    if upstream_response.status_code != 200:
        await upstream_response.aclose()
        await client.aclose()
        raise HTTPException(status_code=502, detail="Failed to fetch the processed file from the CDN.")

    async def _stream_body():
        try:
            async for chunk in upstream_response.aiter_bytes():
                yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()

    base_name = os.path.splitext(job.original_filename)[0]
    if job.file_type == "video":
        filename, media_type = f"processed_{base_name}.mp4", "video/mp4"
    else:
        filename, media_type = f"processed_{base_name}.jpg", "image/jpeg"

    return StreamingResponse(
        _stream_body(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    job = await _get_owned_job(job_id, current_user, session)

    # Best-effort cleanup of CDN assets
    if job.original_file_id:
        imagekit_service.delete_from_imagekit(job.original_file_id)
    if job.processed_file_id:
        imagekit_service.delete_from_imagekit(job.processed_file_id)

    await session.delete(job)
    await session.commit()
