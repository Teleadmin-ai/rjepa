"""
R-JEPA Job Queue CLI Entry Point

Usage:
    python -m rjepa.queue <command> [args]

Examples:
    python -m rjepa.queue list
    python -m rjepa.queue add-training --config configs/rjepa/train.yaml
    python -m rjepa.queue worker
"""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
