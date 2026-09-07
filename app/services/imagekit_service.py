"""
Thin wrapper around the ImageKit Python SDK.

Flow used by the /detect endpoint:
    local temp file -> imagekit.upload_file() -> CDN url (+ file_id for cleanup)
"""

from dataclasses import dataclass

from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions

from app.config import settings

_imagekit = ImageKit(
    private_key=settings.IMAGEKIT_PRIVATE_KEY,
    public_key=settings.IMAGEKIT_PUBLIC_KEY,
    url_endpoint=settings.IMAGEKIT_URL_ENDPOINT,
)


@dataclass
class ImageKitUploadResult:
    success: bool
    url: str | None = None
    file_id: str | None = None
    status_code: int | None = None
    raw_error: str | None = None


def upload_to_imagekit(file_path: str, file_name: str, tags: list[str] | None = None) -> ImageKitUploadResult:
    """
    Uploads a local file to ImageKit and returns its CDN URL.

    Mirrors the reference flow: open the temp file, call imagekit.upload_file()
    with a unique filename + tags, and check the HTTP status before trusting
    the response.
    """
    tags = tags or ["backend-upload"]

    try:
        with open(file_path, "rb") as f:
            options = UploadFileRequestOptions(
                tags=tags,
                use_unique_file_name=True,
                folder="/object-detection/",
            )
            result = _imagekit.upload_file(
                file=f,
                file_name=file_name,
                options=options,
            )
    except Exception as exc:  # network / SDK errors
        return ImageKitUploadResult(success=False, raw_error=str(exc))

    status_code = getattr(result.response_metadata, "http_status_code", None) if result.response_metadata else None

    if status_code == 200 and result.url:
        return ImageKitUploadResult(
            success=True,
            url=result.url,
            file_id=result.file_id,
            status_code=status_code,
        )

    return ImageKitUploadResult(
        success=False,
        status_code=status_code,
        raw_error=str(getattr(result, "response_metadata", "unknown ImageKit error")),
    )


def delete_from_imagekit(file_id: str) -> bool:
    """Best-effort cleanup of a previously uploaded file (e.g. on downstream failure)."""
    try:
        _imagekit.delete_file(file_id=file_id)
        return True
    except Exception:
        return False
