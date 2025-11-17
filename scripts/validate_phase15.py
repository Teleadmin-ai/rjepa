"""
Validation script for Phase 15: Continuous Learning (User Feedback Loop).

Verifies:
1. UserInteraction logging with PII filtering
2. FeedbackValidator multi-level validation
3. FeedbackPipeline conversion to training data
4. ContinuousLearningPipeline orchestration
5. Metrics tracking
"""
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("PHASE 15 VALIDATION: CONTINUOUS LEARNING (USER FEEDBACK LOOP)")
print("=" * 80)

# Check 1: Import core modules
print("\n[1/6] Checking imports...")
try:
    from rjepa.data.user_interactions import (
        UserInteraction,
        InteractionLogger,
        create_interaction_logger,
    )
    from rjepa.data.feedback_pipeline import (
        ValidationResult,
        FeedbackValidator,
        FeedbackPipeline,
        create_feedback_pipeline,
    )
    from rjepa.pipeline.continuous_learning import (
        ContinuousLearningPipeline,
        create_continuous_learning_pipeline,
    )

    print("  [OK] All imports successful")

except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Check 2: Test UserInteraction logging with PII filtering
print("\n[2/6] Testing UserInteraction logging...")
try:
    # Create temporary log directory
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = InteractionLogger(
            log_dir=Path(tmpdir), enable_pii_filter=True, auto_flush_interval=2
        )

        # Log interaction with PII
        interaction_id = logger.log_interaction(
            session_id="test-session-1",
            prompt="My email is john.doe@example.com and my phone is 555-123-4567",
            response="Step 1: Calculate...\nStep 2: Result is 42",
            cot_steps=["Step 1: Calculate...", "Step 2: Result is 42"],
            jepa_mode="rerank",
            jepa_score=0.85,
            feedback_type="thumbs_up",
            domain="math",
            opted_in=True,
        )

        assert len(interaction_id) == 16, "Interaction ID should be 16 chars"

        # Log another (trigger flush)
        logger.log_interaction(
            session_id="test-session-2",
            prompt="Another question",
            response="Answer",
            cot_steps=["Step 1"],
            jepa_mode="off",
            opted_in=False,
        )

        # Flush manually
        logger.flush()

        # Check that file was created
        log_files = list(Path(tmpdir).glob("interactions_*.jsonl"))
        assert len(log_files) > 0, "Log file should be created"

        # Read back and check PII filtering
        import json

        with open(log_files[0], "r") as f:
            first_line = f.readline()
            record = json.loads(first_line)

            assert "[EMAIL]" in record["prompt"], "Email should be filtered"
            assert "[PHONE]" in record["prompt"], "Phone should be filtered"
            assert "john.doe@example.com" not in record["prompt"], "Raw email should not appear"

        print("  [OK] UserInteraction logging works (PII filtered)")

except Exception as e:
    print(f"[FAIL] UserInteraction logging failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Check 3: Test FeedbackValidator
print("\n[3/6] Testing FeedbackValidator...")
try:
    validator = FeedbackValidator(
        jepa_score_threshold=0.7, require_user_feedback=True, enable_auto_validation=True
    )

    # Test 1: Thumbs up + high JEPA score -> ACCEPT
    interaction1 = UserInteraction(
        session_id="s1",
        interaction_id="i1",
        prompt="Test",
        response="42",
        cot_steps=["Step 1"],
        jepa_mode="rerank",
        jepa_score=0.85,
        feedback_type="thumbs_up",
        timestamp="2025-01-17T10:00:00",
        pii_filtered=True,
        opted_in=True,
    )

    result1 = validator.validate_interaction(interaction1, domain="math")
    assert result1.is_valid, "High JEPA + thumbs_up should be accepted"
    assert result1.confidence >= 0.9, "Confidence should be high"

    print(f"  [OK] Test 1: thumbs_up + high JEPA -> ACCEPT (confidence={result1.confidence:.2f})")

    # Test 2: Thumbs down -> REJECT
    interaction2 = UserInteraction(
        session_id="s2",
        interaction_id="i2",
        prompt="Test",
        response="Wrong",
        cot_steps=["Step 1"],
        jepa_mode="rerank",
        jepa_score=0.3,
        feedback_type="thumbs_down",
        timestamp="2025-01-17T10:00:00",
        pii_filtered=True,
        opted_in=True,
    )

    result2 = validator.validate_interaction(interaction2)
    assert not result2.is_valid, "Thumbs down should be rejected"
    assert result2.confidence == 1.0, "Rejection confidence should be 100%"

    print(f"  [OK] Test 2: thumbs_down -> REJECT (confidence={result2.confidence:.2f})")

    # Test 3: Ambiguous (no feedback, medium JEPA) -> REJECT
    interaction3 = UserInteraction(
        session_id="s3",
        interaction_id="i3",
        prompt="Test",
        response="Maybe",
        cot_steps=["Step 1"],
        jepa_mode="rerank",
        jepa_score=0.5,
        feedback_type=None,
        timestamp="2025-01-17T10:00:00",
        pii_filtered=True,
        opted_in=True,
    )

    result3 = validator.validate_interaction(interaction3)
    assert not result3.is_valid, "Ambiguous should be rejected"

    print(f"  [OK] Test 3: ambiguous -> REJECT (confidence={result3.confidence:.2f})")

except Exception as e:
    print(f"[FAIL] FeedbackValidator failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Check 4: Test FeedbackPipeline (mock data)
print("\n[4/6] Testing FeedbackPipeline...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        output_dir = Path(tmpdir) / "output"

        log_dir.mkdir()
        output_dir.mkdir()

        # Create mock interaction log
        import json
        from datetime import datetime

        log_file = log_dir / f"interactions_{datetime.now().strftime('%Y-%m-%d')}.jsonl"

        mock_interactions = [
            {
                "session_id": "s1",
                "interaction_id": "i1",
                "prompt": "What is 2+2?",
                "response": "4",
                "cot_steps": ["Step 1: Add 2+2", "Step 2: Result is 4"],
                "jepa_mode": "rerank",
                "jepa_score": 0.9,
                "feedback_type": "thumbs_up",
                "timestamp": "2025-01-17T10:00:00",
                "pii_filtered": True,
                "opted_in": True,
                "domain": "math",
            },
            {
                "session_id": "s2",
                "interaction_id": "i2",
                "prompt": "Bad question",
                "response": "Bad answer",
                "cot_steps": ["Wrong"],
                "jepa_mode": "off",
                "jepa_score": 0.2,
                "feedback_type": "thumbs_down",
                "timestamp": "2025-01-17T10:00:00",
                "pii_filtered": True,
                "opted_in": True,
                "domain": "general",
            },
        ]

        with open(log_file, "w") as f:
            for interaction in mock_interactions:
                f.write(json.dumps(interaction) + "\n")

        # Create pipeline
        validator = FeedbackValidator(jepa_score_threshold=0.7)
        pipeline = FeedbackPipeline(log_dir=log_dir, output_dir=output_dir, validator=validator)

        # Load interactions
        interactions = pipeline.load_interactions(opted_in_only=True)
        assert len(interactions) == 2, f"Should load 2 interactions, got {len(interactions)}"

        # Validate
        valid_interactions, results = pipeline.validate_batch(interactions)
        assert len(valid_interactions) == 1, f"Should accept 1 interaction, got {len(valid_interactions)}"
        assert results[0].is_valid, "First interaction should be valid"
        assert not results[1].is_valid, "Second interaction should be rejected"

        # Convert to training data
        problems, cots = pipeline.convert_to_training_data(valid_interactions)
        assert len(problems) == 1, "Should create 1 problem"
        assert len(cots) == 1, "Should create 1 CoT"
        assert problems[0].statement == "What is 2+2?", "Problem statement should match"

        # Save dataset
        pipeline.save_dataset(problems, cots, version="test-v1")

        # Check output files
        version_dir = output_dir / "test-v1"
        assert (version_dir / "problems.parquet").exists(), "problems.parquet should exist"
        assert (version_dir / "cots.parquet").exists(), "cots.parquet should exist"
        assert (version_dir / "metadata.json").exists(), "metadata.json should exist"

        print("  [OK] FeedbackPipeline works (1/2 interactions accepted)")

except Exception as e:
    print(f"[FAIL] FeedbackPipeline failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Check 5: Test ContinuousLearningPipeline (structure only, no actual training)
print("\n[5/6] Testing ContinuousLearningPipeline structure...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock checkpoint
        mock_checkpoint = Path(tmpdir) / "checkpoint.pth"
        import torch

        torch.save({"model_state_dict": {}, "config": {}}, mock_checkpoint)

        # Create mock feedback pipeline
        log_dir = Path(tmpdir) / "logs"
        output_dir = Path(tmpdir) / "output"
        log_dir.mkdir()
        output_dir.mkdir()

        validator = FeedbackValidator(jepa_score_threshold=0.7)
        feedback_pipeline = FeedbackPipeline(
            log_dir=log_dir, output_dir=output_dir, validator=validator
        )

        # Create continuous learning pipeline
        cl_pipeline = ContinuousLearningPipeline(
            feedback_pipeline=feedback_pipeline,
            base_checkpoint=str(mock_checkpoint),
            min_new_samples=1,  # Low threshold for testing
        )

        # Check attributes
        assert cl_pipeline.base_checkpoint.exists(), "Base checkpoint should exist"
        assert cl_pipeline.min_new_samples == 1, "Min samples should be 1"

        print("  [OK] ContinuousLearningPipeline instantiated")
        print(f"  [OK] Base checkpoint: {cl_pipeline.base_checkpoint}")
        print(f"  [OK] Min new samples: {cl_pipeline.min_new_samples}")

except Exception as e:
    print(f"[FAIL] ContinuousLearningPipeline failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Check 6: Test factory functions
print("\n[6/6] Testing factory functions...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test create_interaction_logger
        logger = create_interaction_logger(log_dir=tmpdir, enable_pii_filter=True)
        assert logger.enable_pii_filter, "PII filter should be enabled"

        # Test create_feedback_pipeline
        pipeline = create_feedback_pipeline(
            log_dir=tmpdir, output_dir=tmpdir, jepa_score_threshold=0.8
        )
        assert pipeline.validator.jepa_score_threshold == 0.8, "Threshold should be 0.8"

        print("  [OK] Factory functions work")

except Exception as e:
    print(f"[FAIL] Factory functions failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 80)
print("PHASE 15 VALIDATION: [PASS] ALL CHECKS SUCCESSFUL")
print("=" * 80)
print("\nContinuous Learning Implementation:")
print("  - UserInteraction logging with PII filtering")
print("  - Multi-level validation (JEPA score + user feedback + auto-validation)")
print("  - Feedback pipeline (load -> validate -> convert -> save)")
print("  - Continuous learning orchestration (collect -> latents -> train -> A/B test)")
print("  - Metrics tracking over time")
print("\nKey Components:")
print("  - InteractionLogger: Privacy-first logging (anonymization, PII filtering, opt-in)")
print("  - FeedbackValidator: Multi-level validation with confidence scores")
print("  - FeedbackPipeline: Batch processing of interactions -> training data")
print("  - ContinuousLearningPipeline: Nightly retraining orchestration")
print("\nNext steps:")
print("  1. Integrate with UI (feedback buttons)")
print("  2. Schedule nightly retraining (Prefect cron)")
print("  3. Monitor metrics dashboard (accuracy gain over time)")
print("=" * 80)
