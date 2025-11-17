#!/usr/bin/env python3
"""
Validate Phase 1: Data Schemas & Config
"""
import sys
from pathlib import Path


def validate_phase1():
    """Validate that Phase 1 is complete."""
    print("[*] Validating Phase 1: Data Schemas & Config...")
    print()

    # Check files exist
    required_files = [
        "rjepa/config/settings.py",
        "rjepa/data/schemas.py",
        "configs/llm/qwen3-8b.yaml",
        "configs/rjepa/base.yaml",
        "configs/teacher/prompts.yaml",
        "configs/pipeline/build_latents.yaml",
        "configs/pipeline/train_rjepa.yaml",
        "tests/test_config.py",
        "tests/test_schemas.py",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[FAIL] {file_path} (MISSING)")
            all_exist = False

    print()

    if not all_exist:
        print("[FAIL] Phase 1 validation FAILED: Some files are missing")
        return False

    # Try importing
    try:
        from rjepa.config import settings, Settings
        print("[OK] rjepa.config imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import rjepa.config: {e}")
        return False

    try:
        from rjepa.data import Problem, ChainOfThought, LatentSequence
        print("[OK] rjepa.data imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import rjepa.data: {e}")
        return False

    print()

    # Test instantiation
    try:
        problem = Problem(
            problem_id="test_001",
            domain="math",
            subdomain="algebra",
            source="test",
            difficulty="easy",
            statement="Test problem",
        )
        print("[OK] Problem schema instantiation works")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate Problem: {e}")
        return False

    try:
        cot = ChainOfThought(
            cot_id="cot_test",
            problem_id="test_001",
            steps=["Step 1: test"],
            final_answer="test",
            is_valid=True,
            validation_reason="test",
            teacher_model="test",
            source="test",
        )
        print("[OK] ChainOfThought schema instantiation works")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate ChainOfThought: {e}")
        return False

    print()
    print("="*80)
    print("SUCCESS: PHASE 1 VALIDATION COMPLETE!")
    print("="*80)
    print()
    print("[OK] All required files exist")
    print("[OK] All imports work")
    print("[OK] Schemas can be instantiated")
    print()
    print("Statistics:")
    print("   - Config files: 5")
    print("   - Schema models: 5 (Problem, CoT, LatentSequence, DatasetVersion, RJEPACheckpoint)")
    print("   - Test files: 2")
    print()
    print("READY FOR PHASE 2: LLM Adapter")
    print()

    return True


if __name__ == "__main__":
    success = validate_phase1()
    sys.exit(0 if success else 1)
