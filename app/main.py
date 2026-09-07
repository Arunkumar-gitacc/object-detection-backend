from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_db_and_tables
from app.routers import auth, detection
from app.services.clip_service import verifier as clip_verifier
from app.services.yolo_service import detector as yolo_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    # 1. Database schema setup. In development (AUTO_CREATE_TABLES=true),
    #    SQLAlchemy inspects all defined models (User, DetectionJob,
    #    DetectedObject) and creates any missing tables/indexes directly.
    #    In production, set AUTO_CREATE_TABLES=false and run
    #    `alembic upgrade head` as a deploy step instead -- see alembic/.
    if settings.AUTO_CREATE_TABLES:
        await create_db_and_tables()

    # 2. Load heavy ML models once, so requests don't pay model-load latency.
    yolo_detector.load()
    clip_verifier.load()

    yield  # --- Application runs, serving requests ---

    # --- Shutdown ---
    yolo_detector.unload()
    clip_verifier.unload()


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-powered object detection and counting API. Upload images, detect "
        "arbitrary objects with YOLO-World, verify labels with CLIP, and get "
        "back annotated images with bounding boxes and per-object counts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

origins = (
    ["*"] if settings.ALLOWED_ORIGINS.strip() == "*" else [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(detection.router)


@app.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "ok",
        "yolo_world_loaded": yolo_detector.is_loaded,
        "clip_loaded": clip_verifier.is_loaded,
    }
