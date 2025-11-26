"""
R-JEPA Job Queue CLI

Command-line interface for managing the job queue.

Usage:
    python -m rjepa.queue add-training --config configs/rjepa/train.yaml
    python -m rjepa.queue list
    python -m rjepa.queue status
    python -m rjepa.queue worker
    python -m rjepa.queue cancel <job_id>
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .job_queue import (
    JobQueue,
    JobStatus,
    JobType,
    create_training_job,
    create_extraction_job,
)
from .worker import QueueWorker, run_worker, cleanup_cuda

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_QUEUE_FILE = Path("data/queue/jobs.json")
DEFAULT_LOG_DIR = Path("logs/queue")


def format_job(job, verbose: bool = False) -> str:
    """Format job for display."""
    status_colors = {
        "pending": "\033[33m",    # Yellow
        "running": "\033[34m",    # Blue
        "completed": "\033[32m",  # Green
        "failed": "\033[31m",     # Red
        "cancelled": "\033[90m",  # Gray
        "paused": "\033[35m",     # Magenta
    }
    reset = "\033[0m"

    color = status_colors.get(job.status, "")
    status_str = f"{color}{job.status.upper():10}{reset}"

    line = f"[{job.id}] {status_str} {job.type:15} {job.name}"

    if verbose:
        line += f"\n    Command: {job.command}"
        line += f"\n    Created: {job.created_at}"
        if job.started_at:
            line += f"\n    Started: {job.started_at}"
        if job.completed_at:
            line += f"\n    Completed: {job.completed_at}"
        if job.retry_count > 0:
            line += f"\n    Retries: {job.retry_count}/{job.max_retries}"
        if job.error_message:
            line += f"\n    Error: {job.error_message[:100]}"
        if job.log_file:
            line += f"\n    Log: {job.log_file}"

    return line


def cmd_list(args):
    """List jobs in the queue."""
    queue = JobQueue(args.queue_file)

    status_filter = None
    if args.status:
        try:
            status_filter = JobStatus(args.status)
        except ValueError:
            print(f"Invalid status: {args.status}")
            return 1

    jobs = queue.list_jobs(status=status_filter, limit=args.limit)

    if not jobs:
        print("No jobs in queue")
        return 0

    print(f"\n{'=' * 80}")
    print(f"R-JEPA Job Queue ({len(jobs)} jobs)")
    print(f"{'=' * 80}\n")

    for job in jobs:
        print(format_job(job, verbose=args.verbose))
        print()

    return 0


def cmd_status(args):
    """Show queue status."""
    queue = JobQueue(args.queue_file)
    stats = queue.get_stats()

    print(f"\n{'=' * 40}")
    print("R-JEPA Queue Status")
    print(f"{'=' * 40}")
    print(f"Total jobs:    {stats['total']}")
    print(f"  Pending:     {stats['pending']}")
    print(f"  Running:     {stats['running']}")
    print(f"  Completed:   {stats['completed']}")
    print(f"  Failed:      {stats['failed']}")
    print(f"  Cancelled:   {stats['cancelled']}")
    print(f"  Paused:      {stats['paused']}")
    print(f"{'=' * 40}\n")

    return 0


def cmd_add_training(args):
    """Add a training job."""
    job = create_training_job(
        config_path=args.config,
        output_dir=args.output,
        name=args.name,
        priority=args.priority,
        max_retries=args.retries,
        queue_file=args.queue_file,
    )

    print(f"Added training job: {job.id}")
    print(f"  Name: {job.name}")
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output}")
    print(f"  Priority: {args.priority}")
    print(f"  Max retries: {args.retries}")

    return 0


def cmd_add_extraction(args):
    """Add a latent extraction job."""
    job = create_extraction_job(
        output_dir=args.output,
        batch_size=args.batch_size,
        name=args.name,
        priority=args.priority,
        max_retries=args.retries,
        queue_file=args.queue_file,
    )

    print(f"Added extraction job: {job.id}")
    print(f"  Name: {job.name}")
    print(f"  Output: {args.output}")
    print(f"  Batch size: {args.batch_size}")

    return 0


def cmd_add_custom(args):
    """Add a custom job."""
    queue = JobQueue(args.queue_file)

    job = queue.add_job(
        job_type=JobType.CUSTOM,
        name=args.name or f"custom_{datetime.now().strftime('%H%M%S')}",
        command=args.command,
        priority=args.priority,
        max_retries=args.retries,
    )

    print(f"Added custom job: {job.id}")
    print(f"  Command: {args.command}")

    return 0


def cmd_cancel(args):
    """Cancel a job."""
    queue = JobQueue(args.queue_file)
    job = queue.cancel_job(args.job_id)

    if job:
        print(f"Cancelled job: {job.id}")
        return 0
    else:
        print(f"Job not found: {args.job_id}")
        return 1


def cmd_pause(args):
    """Pause a job."""
    queue = JobQueue(args.queue_file)
    job = queue.pause_job(args.job_id)

    if job:
        print(f"Paused job: {job.id}")
        return 0
    else:
        print(f"Job not found or not pending: {args.job_id}")
        return 1


def cmd_resume(args):
    """Resume a paused job."""
    queue = JobQueue(args.queue_file)
    job = queue.resume_job(args.job_id)

    if job:
        print(f"Resumed job: {job.id}")
        return 0
    else:
        print(f"Job not found or not paused: {args.job_id}")
        return 1


def cmd_retry(args):
    """Retry a failed job."""
    queue = JobQueue(args.queue_file)
    job = queue.get_job(args.job_id)

    if not job:
        print(f"Job not found: {args.job_id}")
        return 1

    if job.status != JobStatus.FAILED.value:
        print(f"Job is not failed: {job.status}")
        return 1

    # Reset retry count and mark as pending
    queue.update_job(
        args.job_id,
        status=JobStatus.PENDING,
        retry_count=0,
        error_message=None,
    )

    print(f"Job {args.job_id} marked for retry")
    return 0


def cmd_clear(args):
    """Clear completed jobs."""
    queue = JobQueue(args.queue_file)
    removed = queue.clear_completed(keep_last=args.keep)

    print(f"Cleared {removed} completed jobs")
    return 0


def cmd_worker(args):
    """Start the queue worker."""
    run_worker(
        queue_file=args.queue_file,
        log_dir=args.log_dir,
        poll_interval=args.poll_interval,
    )
    return 0


def cmd_cleanup(args):
    """Clean up CUDA memory."""
    print("Cleaning up CUDA memory...")
    if cleanup_cuda():
        print("CUDA memory cleaned up")
    else:
        print("CUDA not available or cleanup failed")
    return 0


def cmd_logs(args):
    """Show logs for a job."""
    queue = JobQueue(args.queue_file)
    job = queue.get_job(args.job_id)

    if not job:
        print(f"Job not found: {args.job_id}")
        return 1

    if not job.log_file:
        print(f"No log file for job: {args.job_id}")
        return 1

    log_path = Path(job.log_file)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return 1

    # Show last N lines or tail -f
    if args.follow:
        import subprocess
        subprocess.run(["tail", "-f", str(log_path)])
    else:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-args.lines:]:
                print(line, end="")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="R-JEPA Job Queue Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--queue-file", "-q",
        type=Path,
        default=DEFAULT_QUEUE_FILE,
        help="Path to queue JSON file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", aliases=["ls"], help="List jobs")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--limit", "-n", type=int, default=20, help="Max jobs to show")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    list_parser.set_defaults(func=cmd_list)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show queue status")
    status_parser.set_defaults(func=cmd_status)

    # Add training job
    add_train_parser = subparsers.add_parser("add-training", aliases=["train"], help="Add training job")
    add_train_parser.add_argument("--config", "-c", required=True, help="Config file path")
    add_train_parser.add_argument("--output", "-o", default="data/checkpoints/rjepa-queued", help="Output directory")
    add_train_parser.add_argument("--name", "-n", help="Job name")
    add_train_parser.add_argument("--priority", "-p", type=int, default=0, help="Priority (higher=first)")
    add_train_parser.add_argument("--retries", "-r", type=int, default=3, help="Max retries")
    add_train_parser.set_defaults(func=cmd_add_training)

    # Add extraction job
    add_extract_parser = subparsers.add_parser("add-extraction", aliases=["extract"], help="Add extraction job")
    add_extract_parser.add_argument("--output", "-o", required=True, help="Output directory")
    add_extract_parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    add_extract_parser.add_argument("--name", "-n", help="Job name")
    add_extract_parser.add_argument("--priority", "-p", type=int, default=0, help="Priority")
    add_extract_parser.add_argument("--retries", "-r", type=int, default=3, help="Max retries")
    add_extract_parser.set_defaults(func=cmd_add_extraction)

    # Add custom job
    add_custom_parser = subparsers.add_parser("add-custom", aliases=["add"], help="Add custom job")
    add_custom_parser.add_argument("command", help="Command to run")
    add_custom_parser.add_argument("--name", "-n", help="Job name")
    add_custom_parser.add_argument("--priority", "-p", type=int, default=0, help="Priority")
    add_custom_parser.add_argument("--retries", "-r", type=int, default=3, help="Max retries")
    add_custom_parser.set_defaults(func=cmd_add_custom)

    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", help="Job ID to cancel")
    cancel_parser.set_defaults(func=cmd_cancel)

    # Pause command
    pause_parser = subparsers.add_parser("pause", help="Pause a pending job")
    pause_parser.add_argument("job_id", help="Job ID to pause")
    pause_parser.set_defaults(func=cmd_pause)

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume a paused job")
    resume_parser.add_argument("job_id", help="Job ID to resume")
    resume_parser.set_defaults(func=cmd_resume)

    # Retry command
    retry_parser = subparsers.add_parser("retry", help="Retry a failed job")
    retry_parser.add_argument("job_id", help="Job ID to retry")
    retry_parser.set_defaults(func=cmd_retry)

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear completed jobs")
    clear_parser.add_argument("--keep", "-k", type=int, default=10, help="Keep last N completed")
    clear_parser.set_defaults(func=cmd_clear)

    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Start queue worker")
    worker_parser.add_argument("--log-dir", "-l", type=Path, default=DEFAULT_LOG_DIR, help="Log directory")
    worker_parser.add_argument("--poll-interval", "-i", type=int, default=10, help="Poll interval (seconds)")
    worker_parser.set_defaults(func=cmd_worker)

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up CUDA memory")
    cleanup_parser.set_defaults(func=cmd_cleanup)

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show job logs")
    logs_parser.add_argument("job_id", help="Job ID")
    logs_parser.add_argument("--lines", "-n", type=int, default=50, help="Number of lines")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    logs_parser.set_defaults(func=cmd_logs)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Run command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
