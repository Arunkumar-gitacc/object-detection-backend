"""initial schema: users, detection_jobs, detected_objects

Revision ID: 0001
Revises:
Create Date: 2026-09-07

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "detection_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("original_image_url", sa.String(length=1024), nullable=False),
        sa.Column("original_file_id", sa.String(length=255), nullable=True),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("processed_image_url", sa.String(length=1024), nullable=True),
        sa.Column("processed_file_id", sa.String(length=255), nullable=True),
        sa.Column("file_type", sa.String(length=20), nullable=False, server_default="image"),
        sa.Column("requested_classes", sa.Text(), nullable=False),
        sa.Column("confidence_threshold", sa.Float(), nullable=False, server_default="0.25"),
        sa.Column("clip_verification_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("total_objects_detected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_detection_jobs_id", "detection_jobs", ["id"])
    op.create_index("ix_detection_jobs_user_id", "detection_jobs", ["user_id"])

    op.create_table(
        "detected_objects",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.Integer(), sa.ForeignKey("detection_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox_x1", sa.Float(), nullable=False),
        sa.Column("bbox_y1", sa.Float(), nullable=False),
        sa.Column("bbox_x2", sa.Float(), nullable=False),
        sa.Column("bbox_y2", sa.Float(), nullable=False),
        sa.Column("clip_verified", sa.Boolean(), nullable=True),
        sa.Column("clip_score", sa.Float(), nullable=True),
        sa.Column("occurrence_count", sa.Integer(), nullable=True),
    )
    op.create_index("ix_detected_objects_id", "detected_objects", ["id"])
    op.create_index("ix_detected_objects_job_id", "detected_objects", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_detected_objects_job_id", table_name="detected_objects")
    op.drop_index("ix_detected_objects_id", table_name="detected_objects")
    op.drop_table("detected_objects")

    op.drop_index("ix_detection_jobs_user_id", table_name="detection_jobs")
    op.drop_index("ix_detection_jobs_id", table_name="detection_jobs")
    op.drop_table("detection_jobs")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_table("users")
