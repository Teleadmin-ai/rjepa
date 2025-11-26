"""
R-JEPA Job Queue Worker

Daemon that:
- Polls the queue for pending jobs
- Executes jobs with proper cleanup
- Auto-retries on failure
- Purges VRAM between jobs
- Logs everything
"""

import gc
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .job_queue import JobQueue, Job, JobStatus

logger = logging.getLogger(__name__)


def cleanup_cuda():
    """Clean up CUDA memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            logger.info("CUDA memory cleaned up")
            return True
    except ImportError:
        pass
    return False


def kill_python_processes(exclude_pid: int = None):
    """Kill other Python processes (Windows)."""
    try:
        import psutil
        current_pid = os.getpid()
        killed = 0

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    pid = proc.info['pid']
                    if pid != current_pid and pid != exclude_pid:
                        # Check if it's a training process
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any('rjepa' in str(c).lower() for c in cmdline):
                            proc.kill()
                            killed += 1
                            logger.info(f"Killed Python process {pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return killed
    except ImportError:
        logger.warning("psutil not installed, cannot kill processes")
        return 0


class QueueWorker:
    """
    Worker that processes jobs from the queue.

    Usage:
        worker = QueueWorker()
        worker.start()  # Blocks and processes jobs

        # Or run once
        worker.process_one()
    """

    def __init__(
        self,
        queue_file: Path = None,
        log_dir: Path = None,
        poll_interval: int = 10,
        cleanup_before_job: bool = True,
        max_job_duration: int = 86400,  # 24 hours
    ):
        """
        Initialize worker.

        Args:
            queue_file: Path to queue JSON
            log_dir: Directory for job logs
            poll_interval: Seconds between queue checks
            cleanup_before_job: Whether to cleanup CUDA before each job
            max_job_duration: Maximum job duration in seconds
        """
        self.queue = JobQueue(queue_file)
        self.log_dir = Path(log_dir or "logs/queue")
        self.poll_interval = poll_interval
        self.cleanup_before_job = cleanup_before_job
        self.max_job_duration = max_job_duration

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._current_job: Optional[Job] = None
        self._current_process: Optional[subprocess.Popen] = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

        # Kill current process if running
        if self._current_process:
            logger.info("Terminating current job...")
            self._current_process.terminate()
            try:
                self._current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._current_process.kill()

            # Mark job as cancelled
            if self._current_job:
                self.queue.update_job(
                    self._current_job.id,
                    status=JobStatus.CANCELLED,
                    error_message="Worker shutdown"
                )

    def _prepare_environment(self):
        """Prepare environment before running a job."""
        if self.cleanup_before_job:
            logger.info("Cleaning up before job...")
            cleanup_cuda()
            gc.collect()

    def _get_log_file(self, job: Job) -> Path:
        """Get log file path for a job."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"{job.id}_{job.type}_{timestamp}.log"

    def _run_job(self, job: Job) -> tuple[int, str]:
        """
        Run a job and return (exit_code, error_message).

        Args:
            job: Job to run

        Returns:
            (exit_code, error_message) tuple
        """
        log_file = self._get_log_file(job)

        logger.info(f"Running job {job.id}: {job.command}")
        logger.info(f"Log file: {log_file}")

        # Update job with log file and running status
        self.queue.update_job(
            job.id,
            status=JobStatus.RUNNING,
            log_file=str(log_file),
            pid=os.getpid(),
        )

        try:
            # Open log file
            with open(log_file, "w", encoding="utf-8") as f:
                # Write header
                f.write(f"{'=' * 80}\n")
                f.write(f"Job ID: {job.id}\n")
                f.write(f"Type: {job.type}\n")
                f.write(f"Name: {job.name}\n")
                f.write(f"Command: {job.command}\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Retry: {job.retry_count}/{job.max_retries}\n")
                f.write(f"{'=' * 80}\n\n")
                f.flush()

                # Run command
                self._current_process = subprocess.Popen(
                    job.command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path.cwd()),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )

                # Wait with timeout
                try:
                    exit_code = self._current_process.wait(timeout=self.max_job_duration)
                except subprocess.TimeoutExpired:
                    logger.error(f"Job {job.id} timed out after {self.max_job_duration}s")
                    self._current_process.kill()
                    exit_code = -1
                    error_message = f"Timeout after {self.max_job_duration}s"
                    return exit_code, error_message

                # Write footer
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Completed: {datetime.now().isoformat()}\n")
                f.write(f"Exit code: {exit_code}\n")
                f.write(f"{'=' * 80}\n")

            self._current_process = None

            if exit_code == 0:
                return 0, None
            else:
                # Read last lines of log for error message
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    error_lines = [l for l in lines[-50:] if "error" in l.lower() or "exception" in l.lower()]
                    error_message = "".join(error_lines[-5:]) if error_lines else f"Exit code: {exit_code}"
                return exit_code, error_message

        except Exception as e:
            logger.exception(f"Error running job {job.id}")
            return -1, str(e)

    def process_one(self) -> Optional[Job]:
        """
        Process one job from the queue.

        Returns:
            Processed job or None if queue empty
        """
        # Get next job
        job = self.queue.get_next_job()
        if not job:
            return None

        self._current_job = job
        logger.info(f"Processing job: {job.id} ({job.name})")

        # Check if this is a retry with delay
        if job.retry_count > 0:
            delay = job.get_retry_delay()
            logger.info(f"Waiting {delay}s before retry...")
            time.sleep(delay)

        # Prepare environment
        self._prepare_environment()

        # Run job
        exit_code, error_message = self._run_job(job)

        # Update status
        if exit_code == 0:
            self.queue.update_job(
                job.id,
                status=JobStatus.COMPLETED,
                exit_code=exit_code,
            )
            logger.info(f"Job {job.id} completed successfully")
        else:
            # Mark for retry or fail
            self.queue.mark_for_retry(job.id, error_message)

        self._current_job = None
        return job

    def start(self):
        """
        Start the worker loop.

        Blocks and processes jobs until stopped.
        """
        logger.info("=" * 60)
        logger.info("R-JEPA Queue Worker starting")
        logger.info(f"Queue file: {self.queue.queue_file}")
        logger.info(f"Log dir: {self.log_dir}")
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info("=" * 60)

        self._running = True

        while self._running:
            try:
                # Check for pending jobs
                job = self.queue.get_next_job()

                if job:
                    self.process_one()
                    # Small delay after job
                    time.sleep(2)
                else:
                    # No jobs, wait
                    logger.debug("No pending jobs, waiting...")
                    time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.exception(f"Worker error: {e}")
                time.sleep(self.poll_interval)

        logger.info("Worker stopped")

    def stop(self):
        """Stop the worker."""
        self._running = False


def run_worker(
    queue_file: Path = None,
    log_dir: Path = None,
    poll_interval: int = 10,
    daemon: bool = False,
):
    """
    Run the queue worker.

    Args:
        queue_file: Path to queue JSON
        log_dir: Directory for job logs
        poll_interval: Seconds between queue checks
        daemon: Run as daemon (detach from terminal)
    """
    # Ensure log directory exists
    log_dir_path = Path(log_dir or "logs/queue")
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_dir_path / "worker.log",
                encoding="utf-8",
            ),
        ],
    )

    worker = QueueWorker(
        queue_file=queue_file,
        log_dir=log_dir,
        poll_interval=poll_interval,
    )

    worker.start()


if __name__ == "__main__":
    run_worker()
