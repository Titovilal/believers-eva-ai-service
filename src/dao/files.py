from sqlalchemy import Column, Integer, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    s3_key = Column(Text, nullable=False, unique=True)
    filename = Column(Text, nullable=False)
    description = Column(Text)
    size = Column(Integer)  # Size in bytes
    content_type = Column(Text, nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"))
    folder_id = Column(Integer, ForeignKey("folders.id", ondelete="CASCADE"))
    uploaded_by = Column(
        Text, ForeignKey("profiles.user_id", ondelete="SET NULL"), nullable=False
    )

    # Document metadata fields
    document_name = Column(Text)
    scope = Column(Text)  # DocumentScope type
    scope_name = Column(Text)
    published_date = Column(DateTime(timezone=True))
    uploaded_date = Column(DateTime(timezone=True))
    end_validation_date = Column(DateTime(timezone=True))
    audit = Column(Text)  # Yes / No
    confidence_level = Column(Text)  # ConfidenceLevel type
    author = Column(Text)
    official_author = Column(Text)  # Yes / No
    language = Column(Text)  # ISO Code 2
    format = Column(Text)  # ppt, pdf, csv
    folder = Column(Text)  # AWS S3 folder name
    text_extraction_type = Column(Text)  # TextExtractionType

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self):
        return f"<File(id={self.id}, filename='{self.filename}')>"
