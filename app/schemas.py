"""Pydantic models shared by the API and the job manager."""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=300)


class ResearchResult(BaseModel):
    draft: str
    approved: bool
    revision_count: int
    research_rounds: int
    critique: str


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus


class JobView(BaseModel):
    job_id: str
    status: JobStatus
    topic: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: ResearchResult | None = None
    error: str | None = None