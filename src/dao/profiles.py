from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Text, nullable=False, unique=True)
    name = Column(Text, nullable=False)
    mail = Column(Text, nullable=False)
    role = Column(Text, nullable=False, default="none")
    company_id = Column(Integer, ForeignKey("companies.id", ondelete="SET NULL"))

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<Profile(id={self.id}, user_id='{self.user_id}', name='{self.name}')>"
