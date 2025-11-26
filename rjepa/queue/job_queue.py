"""
R-JEPA Job Queue System - Queue Management

Persistent job queue with:
- JSON-based storage
- File locking for concurrent access
- Status tracking (pending, running, completed, failed)
- Auto-retry with exponential backoff
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import filelock

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class JobType(str, Enum):
    """Supported job types."""
    TRAIN_RJEPA = "train_rjepa"
    EXTRACT_LATENTS = "extract_latents"
    EVALUATE = "evaluate"
    CUSTOM = "custom"


@dataclass
class Job:
    """A job in the queue."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = JobType.TRAIN_RJEPA.value
    name: str = ""
    command: str = ""  # Full command to execute
    config: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = JobStatus.PENDING.value
    priority: int = 0  # Higher = more priority

    # Retry
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_base: int = 60  # Base delay in seconds (1 min)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Results
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    log_file: Optional[str] = None
    output_dir: Optional[str] = None

    # Process info
    pid: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        return cls(**data)

    def get_retry_delay(self) -> int:
        """Get delay before next retry (exponential backoff)."""
        return self.retry_delay_base * (2 ** self.retry_count)

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries

    def __repr__(self):
        return f"Job({self.id}, {self.type}, {self.status})"


class JobQueue:
    """
    Persistent job queue with file locking.

    Usage:
        queue = JobQueue("path/to/queue.json")

        # Add job
        job = queue.add_job(
            job_type=JobType.TRAIN_RJEPA,
            name="Train R-JEPA on validated data",
            command="python -m rjepa.pipeline.train_rjepa --config ...",
            config={"epochs": 100}
        )

        # Get next job
        job = queue.get_next_job()

        # Update status
        queue.update_job(job.id, status=JobStatus.COMPLETED)
    """

    def __init__(
        self,
        queue_file: Path = None,
        lock_timeout: int = 30,
    ):
        """
        Initialize job queue.

        Args:
            queue_file: Path to queue JSON file
            lock_timeout: Lock timeout in seconds
        """
        if queue_file is None:
            queue_file = Path("data/queue/jobs.json")

        self.queue_file = Path(queue_file)
        self.lock_file = self.queue_file.with_suffix(".lock")
        self.lock_timeout = lock_timeout

        # Create directory
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize queue file if not exists
        if not self.queue_file.exists():
            self._save_jobs([])

    def _get_lock(self) -> filelock.FileLock:
        """Get file lock."""
        return filelock.FileLock(self.lock_file, timeout=self.lock_timeout)

    def _load_jobs(self) -> List[Job]:
        """Load jobs from file."""
        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Job.from_dict(j) for j in data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_jobs(self, jobs: List[Job]) -> None:
        """Save jobs to file."""
        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump([j.to_dict() for j in jobs], f, indent=2, ensure_ascii=False)

    def add_job(
        self,
        job_type: JobType = JobType.TRAIN_RJEPA,
        name: str = "",
        command: str = "",
        config: Dict[str, Any] = None,
        priority: int = 0,
        max_retries: int = 3,
        output_dir: str = None,
    ) -> Job:
        """
        Add a job to the queue.

        Args:
            job_type: Type of job
            name: Human-readable name
            command: Command to execute
            config: Job configuration
            priority: Priority (higher = first)
            max_retries: Maximum retry attempts
            output_dir: Output directory for results

        Returns:
            Created job
        """
        job = Job(
            type=job_type.value if isinstance(job_type, JobType) else job_type,
            name=name or f"{job_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            command=command,
            config=config or {},
            priority=priority,
            max_retries=max_retries,
            output_dir=output_dir,
        )

        with self._get_lock():
            jobs = self._load_jobs()
            jobs.append(job)
            self._save_jobs(jobs)

        logger.info(f"Added job: {job.id} ({job.name})")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._get_lock():
            jobs = self._load_jobs()
            for job in jobs:
                if job.id == job_id:
                    return job
        return None

    def get_next_job(self) -> Optional[Job]:
        """
        Get next pending job (highest priority first).

        Returns:
            Next job or None if queue empty
        """
        with self._get_lock():
            jobs = self._load_jobs()

            # Filter pending jobs
            pending = [j for j in jobs if j.status == JobStatus.PENDING.value]

            if not pending:
                return None

            # Sort by priority (desc) then by created_at (asc)
            pending.sort(key=lambda j: (-j.priority, j.created_at))

            return pending[0]

    def update_job(
        self,
        job_id: str,
        status: JobStatus = None,
        exit_code: int = None,
        error_message: str = None,
        pid: int = None,
        log_file: str = None,
        **kwargs,
    ) -> Optional[Job]:
        """
        Update a job.

        Args:
            job_id: Job ID
            status: New status
            exit_code: Exit code if completed
            error_message: Error message if failed
            pid: Process ID if running
            log_file: Log file path
            **kwargs: Additional fields to update

        Returns:
            Updated job or None
        """
        with self._get_lock():
            jobs = self._load_jobs()

            for i, job in enumerate(jobs):
                if job.id == job_id:
                    # Update fields
                    if status:
                        job.status = status.value if isinstance(status, JobStatus) else status

                        # Set timestamps
                        if status == JobStatus.RUNNING:
                            job.started_at = datetime.now().isoformat()
                        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                            job.completed_at = datetime.now().isoformat()

                    if exit_code is not None:
                        job.exit_code = exit_code
                    if error_message is not None:
                        job.error_message = error_message
                    if pid is not None:
                        job.pid = pid
                    if log_file is not None:
                        job.log_file = log_file

                    # Update additional fields
                    for key, value in kwargs.items():
                        if hasattr(job, key):
                            setattr(job, key, value)

                    jobs[i] = job
                    self._save_jobs(jobs)

                    logger.info(f"Updated job {job_id}: status={job.status}")
                    return job

            return None

    def mark_for_retry(self, job_id: str, error_message: str = None) -> Optional[Job]:
        """
        Mark a failed job for retry.

        Args:
            job_id: Job ID
            error_message: Error message

        Returns:
            Updated job or None
        """
        with self._get_lock():
            jobs = self._load_jobs()

            for i, job in enumerate(jobs):
                if job.id == job_id:
                    job.retry_count += 1

                    if job.can_retry():
                        job.status = JobStatus.PENDING.value
                        job.error_message = error_message
                        logger.info(
                            f"Job {job_id} marked for retry "
                            f"({job.retry_count}/{job.max_retries}), "
                            f"delay={job.get_retry_delay()}s"
                        )
                    else:
                        job.status = JobStatus.FAILED.value
                        job.error_message = f"Max retries exceeded. Last error: {error_message}"
                        logger.error(f"Job {job_id} failed after {job.max_retries} retries")

                    jobs[i] = job
                    self._save_jobs(jobs)
                    return job

            return None

    def cancel_job(self, job_id: str) -> Optional[Job]:
        """Cancel a job."""
        return self.update_job(job_id, status=JobStatus.CANCELLED)

    def pause_job(self, job_id: str) -> Optional[Job]:
        """Pause a pending job."""
        job = self.get_job(job_id)
        if job and job.status == JobStatus.PENDING.value:
            return self.update_job(job_id, status=JobStatus.PAUSED)
        return None

    def resume_job(self, job_id: str) -> Optional[Job]:
        """Resume a paused job."""
        job = self.get_job(job_id)
        if job and job.status == JobStatus.PAUSED.value:
            return self.update_job(job_id, status=JobStatus.PENDING)
        return None

    def list_jobs(
        self,
        status: JobStatus = None,
        limit: int = None,
    ) -> List[Job]:
        """
        List jobs.

        Args:
            status: Filter by status
            limit: Maximum number of jobs

        Returns:
            List of jobs
        """
        with self._get_lock():
            jobs = self._load_jobs()

        if status:
            status_value = status.value if isinstance(status, JobStatus) else status
            jobs = [j for j in jobs if j.status == status_value]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        if limit:
            jobs = jobs[:limit]

        return jobs

    def clear_completed(self, keep_last: int = 10) -> int:
        """
        Clear completed jobs, keeping the last N.

        Args:
            keep_last: Number of completed jobs to keep

        Returns:
            Number of jobs removed
        """
        with self._get_lock():
            jobs = self._load_jobs()

            # Separate completed from others
            completed = [j for j in jobs if j.status == JobStatus.COMPLETED.value]
            others = [j for j in jobs if j.status != JobStatus.COMPLETED.value]

            # Keep last N completed
            completed.sort(key=lambda j: j.completed_at or "", reverse=True)
            to_remove = len(completed) - keep_last

            if to_remove > 0:
                completed = completed[:keep_last]
                jobs = others + completed
                self._save_jobs(jobs)
                logger.info(f"Cleared {to_remove} completed jobs")
                return to_remove

            return 0

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        jobs = self.list_jobs()

        stats = {
            "total": len(jobs),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "paused": 0,
        }

        for job in jobs:
            if job.status in stats:
                stats[job.status] += 1

        return stats


# Convenience functions
def create_training_job(
    config_path: str,
    output_dir: str,
    name: str = None,
    priority: int = 0,
    max_retries: int = 3,
    queue_file: Path = None,
) -> Job:
    """
    Create a training job.

    Args:
        config_path: Path to config YAML
        output_dir: Output directory
        name: Job name
        priority: Priority
        max_retries: Max retries
        queue_file: Queue file path

    Returns:
        Created job
    """
    queue = JobQueue(queue_file)

    command = (
        f"python -m rjepa.pipeline.train_rjepa "
        f"--config {config_path} "
        f"--output {output_dir}"
    )

    return queue.add_job(
        job_type=JobType.TRAIN_RJEPA,
        name=name or f"train_{Path(config_path).stem}",
        command=command,
        config={"config_path": config_path, "output_dir": output_dir},
        priority=priority,
        max_retries=max_retries,
        output_dir=output_dir,
    )


def create_extraction_job(
    output_dir: str,
    batch_size: int = 8,
    name: str = None,
    priority: int = 0,
    max_retries: int = 3,
    queue_file: Path = None,
) -> Job:
    """
    Create a latent extraction job.

    Args:
        output_dir: Output directory
        batch_size: Batch size
        name: Job name
        priority: Priority
        max_retries: Max retries
        queue_file: Queue file path

    Returns:
        Created job
    """
    queue = JobQueue(queue_file)

    command = (
        f"python scripts/extract_latents_validated.py "
        f"--output {output_dir} "
        f"--batch-size {batch_size}"
    )

    return queue.add_job(
        job_type=JobType.EXTRACT_LATENTS,
        name=name or f"extract_{datetime.now().strftime('%Y%m%d')}",
        command=command,
        config={"output_dir": output_dir, "batch_size": batch_size},
        priority=priority,
        max_retries=max_retries,
        output_dir=output_dir,
    )
