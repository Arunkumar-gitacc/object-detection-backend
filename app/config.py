from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration, loaded from environment variables / .env file.
    """

    # --- Database ---
    # Example: postgresql+asyncpg://user:password@localhost:5432/object_detection_db
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/object_detection_db"

    # --- ImageKit CDN ---
    IMAGEKIT_PRIVATE_KEY: str = ""
    IMAGEKIT_PUBLIC_KEY: str = ""
    IMAGEKIT_URL_ENDPOINT: str = ""

    # --- YOLO-World ---
    # Any ultralytics YOLO-World checkpoint, e.g. yolov8s-worldv2.pt, yolov8m-worldv2.pt
    YOLO_WORLD_MODEL_PATH: str = "yolov8s-worldv2.pt"
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
    DEFAULT_IOU_THRESHOLD: float = 0.45

    # --- CLIP ---
    CLIP_MODEL_NAME: str = "ViT-B/32"
    CLIP_VERIFICATION_THRESHOLD: float = 0.20  # min cosine similarity to accept a label
    ENABLE_CLIP_VERIFICATION_DEFAULT: bool = True

    # --- Local temp storage (scratch space before upload to ImageKit) ---
    TEMP_DIR: str = "./tmp_uploads"

    # --- Auth (JWT) ---
    # Generate a real one with: python -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY: str = "CHANGE_ME_this_is_not_a_secure_default_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # --- Video support ---
    # Run YOLO-World on every Nth frame (rest of the output video reuses the
    # most recent detection) to keep CPU-bound processing tractable.
    VIDEO_FRAME_SAMPLE_RATE: int = 5

    # --- Schema management ---
    # Dev convenience: auto-create tables on startup. Set to False in
    # production and run `alembic upgrade head` instead.
    AUTO_CREATE_TABLES: bool = True

    # --- Misc ---
    APP_NAME: str = "AI Object Detection & Counting API"
    ALLOWED_ORIGINS: str = "*"  # comma separated list in production

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
