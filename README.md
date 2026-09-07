# AI Object Detection & Counting API

FastAPI backend for an AI-powered object detection and counting app, using:

- **YOLO-World** (via `ultralytics`) — open-vocabulary detection, so users can ask for *any* object class by name (not just the 80 COCO classes)
- **OpenAI CLIP** — verifies each YOLO-World detection by comparing the cropped region against the label text, catching false positives
- **PostgreSQL** (async SQLAlchemy 2.0 + `asyncpg`, migrated with **Alembic**) — stores users, detection jobs, and per-object results
- **JWT auth** — every upload belongs to an authenticated user
- **ImageKit** — CDN storage for original and annotated (bounding-box) images/videos
- **Image + video** support — videos are sampled frame-by-frame and annotated end-to-end

## Project structure

```
app/
  main.py                # FastAPI app + lifespan (schema setup, model loading)
  config.py               # Settings (env vars)
  database.py              # Async engine/session, create_db_and_tables()
  security.py               # Password hashing, JWT issuing/verification, get_current_user
  models.py                  # User, DetectionJob, DetectedObject (SQLAlchemy)
  schemas.py                  # Pydantic response models
  services/
    imagekit_service.py        # Upload/delete on ImageKit CDN
    yolo_service.py              # YOLO-World detector (path OR in-memory frame), loaded once at startup
    clip_service.py               # CLIP label verification, loaded once at startup
    drawing.py                      # Draws bounding boxes (Pillow for images, OpenCV for video frames)
    video_service.py                 # Frame-sampled video detection, annotation, aggregation
  routers/
    auth.py                           # /register, /login, /me
    detection.py                       # /detect, /jobs, /jobs/{id}/download, etc.
alembic/
  env.py                                # Async-aware Alembic environment, wired to app models
  versions/0001_initial_schema.py        # users, detection_jobs, detected_objects tables
alembic.ini
requirements.txt
.env.example
```

## Flow

### 1. App startup (`lifespan`)

```
App Start ──► [AUTO_CREATE_TABLES?] create_db_and_tables() ──► load YOLO-World ──► load CLIP ──► Yield (App Runs) ──► Shutdown
```

In development, `AUTO_CREATE_TABLES=true` makes SQLAlchemy inspect `User`,
`DetectionJob`, and `DetectedObject` and create any missing tables on
startup. **In production, set `AUTO_CREATE_TABLES=false` and run `alembic
upgrade head` as a deploy step instead** — see [Migrations](#migrations)
below. The ML models are loaded once into memory (not per-request) since
loading them is expensive.

### 2. Detection flow (`POST /api/v1/detect`, requires a Bearer token)

```
Client (JWT + files + classes)
      │
      ▼
1. Stream each file to local temp file (shutil.copyfileobj)
      │
      ▼
2. Detect:
     image → YOLO-World once, full resolution
     video → YOLO-World every Nth frame (VIDEO_FRAME_SAMPLE_RATE), reused across frames in between
      │
      ▼
3. (Optional) CLIP verifies each detected crop / best example per label against its label
      │
      ▼
4. Draw bounding boxes + labels + confidence scores → processed temp file (image or annotated .mp4)
      │
      ▼
5. Upload original + processed files to ImageKit CDN
      │
      ▼
6. Insert DetectionJob (owned by current_user) + DetectedObject rows into PostgreSQL
      │
      ▼
7. Cleanup: delete local temp files, close upload stream (finally block)
```

**Counting semantics differ slightly by media type:**
- **Images**: each `DetectedObject` row is one instance; counts are a simple group-by on `label`.
- **Videos**: the same object appears across many frames, so summing rows would over-count. Instead, each label gets *one* row with `occurrence_count` = the maximum number of simultaneous instances seen in any sampled frame (e.g. "3 people in shot at once, at most").

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in: DATABASE_URL, ImageKit keys, and a real SECRET_KEY
# (generate one with: python -c "import secrets; print(secrets.token_hex(32))")

# Make sure PostgreSQL is running and the database in DATABASE_URL exists
alembic upgrade head            # create schema (see Migrations below)

uvicorn app.main:app --reload
```

Docs at `http://localhost:8000/docs`.

> First run will download the YOLO-World checkpoint (`yolov8s-worldv2.pt`) and the
> CLIP checkpoint (`ViT-B/32`) automatically — this can take a few minutes and
> needs internet access. A GPU is strongly recommended for real-time inference
> (especially for video); it will fall back to CPU otherwise.

## Migrations

Schema changes are managed with **Alembic** rather than relying on
`create_db_and_tables()` in production.

```bash
# Apply all migrations (creates users, detection_jobs, detected_objects)
alembic upgrade head

# After changing a model in app/models.py, autogenerate the next migration
alembic revision --autogenerate -m "describe your change"

# Review the generated file in alembic/versions/, then apply it
alembic upgrade head

# Roll back one revision
alembic downgrade -1
```

`alembic/env.py` reads `DATABASE_URL` from the same `app.config.settings`
the app uses, and targets `Base.metadata` from `app.database`, so
autogenerate diffs always compare against your actual SQLAlchemy models.

For local/dev convenience you can still leave `AUTO_CREATE_TABLES=true` and
skip `alembic upgrade head` — the app will create any missing tables on
startup. Switch to Alembic-only (`AUTO_CREATE_TABLES=false`) once you have
real data you don't want to risk, or more than one environment to keep in sync.

## Auth

All detection/job endpoints require a JWT Bearer token.

```bash
# 1. Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "a-strong-password", "full_name": "Your Name"}'

# 2. Log in (OAuth2 password flow — note the form fields, not JSON)
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=you@example.com&password=a-strong-password"
# -> {"access_token": "...", "token_type": "bearer"}

# 3. Use the token
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer <access_token>"
```

Every `DetectionJob` is tied to `user_id`; listing, viewing, downloading, or
deleting a job you don't own returns a 404 (not a 403, so job existence
isn't leaked to other users).

## Endpoints

| Method | Path                              | Auth | Description                                            |
|--------|------------------------------------|------|-----------------------------------------------------------|
| POST   | `/api/v1/auth/register`            | —    | Create an account                                            |
| POST   | `/api/v1/auth/login`               | —    | Exchange email + password for a JWT                             |
| GET    | `/api/v1/auth/me`                  | ✅   | Current user profile                                              |
| POST   | `/api/v1/detect`                   | ✅   | Upload 1+ images/videos, detect + count objects                     |
| GET    | `/api/v1/jobs`                     | ✅   | List your past detection jobs (paginated)                              |
| GET    | `/api/v1/jobs/{job_id}`            | ✅   | Full detail: objects, bboxes, counts                                     |
| GET    | `/api/v1/jobs/{job_id}/counts`     | ✅   | Per-label counts only                                                       |
| GET    | `/api/v1/jobs/{job_id}/download`   | ✅   | Same-origin download proxy for the processed file                            |
| DELETE | `/api/v1/jobs/{job_id}`            | ✅   | Delete a job (+ best-effort ImageKit cleanup)                                  |
| GET    | `/health`                          | —    | App + model load status                                                          |

### Example: detect

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Authorization: Bearer <access_token>" \
  -F "files=@street.jpg" \
  -F "classes=person,car,bicycle,dog" \
  -F "confidence_threshold=0.3" \
  -F "use_clip_verification=true"
```

Response (abridged):

```json
[
  {
    "id": 1,
    "user_id": 4,
    "original_image_url": "https://ik.imagekit.io/.../original_street.jpg",
    "processed_image_url": "https://ik.imagekit.io/.../processed_street.jpg",
    "file_type": "image",
    "total_objects_detected": 7,
    "status": "completed",
    "counts": [
      {"label": "person", "count": 4},
      {"label": "car", "count": 2},
      {"label": "bicycle", "count": 1}
    ],
    "objects": [
      {"label": "person", "confidence": 0.91, "bbox_x1": 120.3, "bbox_y1": 80.1, "bbox_x2": 210.7, "bbox_y2": 340.9, "clip_verified": true, "clip_score": 0.27, "occurrence_count": null}
    ]
  }
]
```

Video uploads (e.g. `.mp4`) go through the same `/api/v1/detect` endpoint —
`file_type` is auto-detected from the content type / extension, and
`processed_image_url` becomes an annotated `.mp4` on ImageKit.

### Example: download

```bash
curl -L http://localhost:8000/api/v1/jobs/1/download \
  -H "Authorization: Bearer <access_token>" \
  -o processed_street.jpg
```

This streams the file through the API rather than redirecting to the raw
ImageKit URL, so the client gets a proper `Content-Disposition: attachment`
download without needing to know about ImageKit at all.

## Notes

- **Video cost**: video processing runs YOLO-World on every `VIDEO_FRAME_SAMPLE_RATE`-th frame (default 5) and writes every frame to the output — on CPU this can still be slow for long clips. Tune the sample rate, or move this to a background worker (Celery/RQ) if uploads need to return immediately rather than blocking on processing.
- **Batch failure behavior**: `/api/v1/detect` processes multiple files sequentially in one request; if one file fails, its job is marked `failed` (and kept in the DB) but the request raises a 500 immediately, so any remaining files in that batch are not processed. Call `/detect` once per file from the client if you want independent failure handling.
