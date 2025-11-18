#!/bin/bash
################################################################################
# Auto-Restart Wrapper for Latent Extraction
################################################################################
#
# Usage:
#   bash scripts/run_extraction_with_autorestart.sh
#
# Features:
#   - Auto-restart en cas de crash
#   - Resume automatique (--resume)
#   - Logs complets
#   - Retry max 10 fois (évite boucle infinie)
#
################################################################################

set -e

# Config
PROJECT_DIR="C:/Users/teleadmin/world-txt-model"
SCRIPT="scripts/extract_latents_optimized.py"
BATCH_SIZE=8
CHECKPOINT_EVERY=10
LOG_DIR="logs/extraction"
MAX_RETRIES=10

# Create log dir
mkdir -p "$LOG_DIR"

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/extraction_${TIMESTAMP}.log"

echo "================================================================================
AUTO-RESTART LATENT EXTRACTION WRAPPER
================================================================================

Configuration:
  - Script: $SCRIPT
  - Batch size: $BATCH_SIZE
  - Checkpoint every: $CHECKPOINT_EVERY batches
  - Max retries: $MAX_RETRIES
  - Log file: $LOG_FILE

Starting extraction...
" | tee -a "$LOG_FILE"

# Counter for retries
retry_count=0

# Main loop with auto-restart
while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "================================================================================
[ATTEMPT $(($retry_count + 1))/$MAX_RETRIES] Starting extraction at $(date)
================================================================================" | tee -a "$LOG_FILE"

    # Activate venv
    cd "$PROJECT_DIR"
    source .venv/Scripts/activate

    # Run extraction with resume (always, checkpoint handles if needed)
    if python "$SCRIPT" \
        --batch-size "$BATCH_SIZE" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --resume \
        2>&1 | tee -a "$LOG_FILE"; then

        # SUCCESS - Extraction completed
        echo "
================================================================================
[SUCCESS] Extraction completed successfully at $(date)
================================================================================" | tee -a "$LOG_FILE"
        exit 0
    else
        # FAILURE - Script crashed
        exit_code=$?
        retry_count=$(($retry_count + 1))

        echo "
================================================================================
[ERROR] Script crashed with exit code $exit_code at $(date)
[RETRY] Will restart in 30 seconds... (attempt $retry_count/$MAX_RETRIES)
================================================================================" | tee -a "$LOG_FILE"

        # Wait before retry (avoid rapid restart loop)
        sleep 30
    fi
done

# Max retries exceeded
echo "
================================================================================
[FATAL] Max retries ($MAX_RETRIES) exceeded. Extraction failed.
Please investigate the issue manually.

Log file: $LOG_FILE
================================================================================" | tee -a "$LOG_FILE"

exit 1
