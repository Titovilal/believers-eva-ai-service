from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone

Base = declarative_base()


class FileChunk(Base):
    __tablename__ = "file_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(
        Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    company_id = Column(
        Integer, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    chunk = Column(Text, nullable=False)
    embedding = Column(Vector(1536))
    scope = Column(Text)
    is_audited = Column(Boolean, nullable=False, default=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self):
        return f"<FileChunk(id={self.id}, file_id={self.file_id}, chunk_preview='{self.chunk[:50]}...')>"
