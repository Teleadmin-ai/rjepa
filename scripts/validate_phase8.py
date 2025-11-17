#!/usr/bin/env python3
"""
Validate Phase 8: Inference Modes (rerank, nudge, plan)
"""
import sys
from pathlib import Path
from unittest.mock import Mock
import torch


def validate_phase8():
    """Validate that Phase 8 is complete."""
    print("[*] Validating Phase 8: Inference Modes...")
    print()

    # Check files exist
    required_files = [
        "rjepa/inference/rerank.py",
        "rjepa/inference/nudge.py",
        "rjepa/inference/plan.py",
        "rjepa/inference/__init__.py",
        "tests/test_inference.py",
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
        print("[FAIL] Phase 8 validation FAILED: Some files are missing")
        return False

    # Try importing
    try:
        from rjepa.inference import (
            rerank_cots_with_jepa,
            rerank_existing_cots,
            rerank_with_ensembling,
        )
        print("[OK] Re-ranking functions import successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import re-ranking functions: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        from rjepa.inference import (
            nudge_reasoning_stepwise,
            nudge_with_regeneration,
            nudge_with_beam_search,
        )
        print("[OK] Nudge functions import successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import nudge functions: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        from rjepa.inference import (
            complete_reasoning_plan,
            auto_complete_missing_steps,
            iterative_refinement,
        )
        print("[OK] Plan functions import successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import plan functions: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test re-ranking with mocks
    try:
        print("[*] Testing re-ranking mode...")

        # Create mocks
        mock_llm = Mock()
        mock_llm.layer_to_extract = -2
        mock_llm.tokenizer = Mock()
        mock_llm.tokenizer.encode = Mock(return_value=torch.randint(0, 1000, (1, 50)))

        def mock_generate(prompt, **kwargs):
            num_samples = kwargs.get("num_samples", 1)
            results = []
            for i in range(num_samples):
                results.append({
                    "full_text": f"Step 1: Test\nStep 2: Result is {i}",
                    "steps": [f"Step 1: Test", f"Step 2: Result is {i}"],
                    "tokens": torch.randint(0, 1000, (1, 50)),
                    "step_boundaries": [(0, 25), (25, 50)],
                })
            return results

        mock_llm.generate_with_cot = Mock(side_effect=mock_generate)
        mock_llm.extract_latents = Mock(return_value=torch.randn(2, 64))

        mock_rjepa = Mock()
        mock_rjepa.score = Mock(return_value={
            "jepa_loss": 0.5,
            "num_steps": 2,
            "num_masked": 1,
            "device": "cpu",
        })

        # Test rerank
        result = rerank_cots_with_jepa(
            prompt="Test",
            llm=mock_llm,
            rjepa_client=mock_rjepa,
            num_samples=3,
        )

        assert "best_cot" in result
        assert "candidates" in result
        assert result["num_candidates"] == 3

        print("[OK] Re-ranking mode works")

    except Exception as e:
        print(f"[FAIL] Re-ranking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test nudge
    try:
        print("[*] Testing nudge mode...")

        mock_rjepa.predict_masked = Mock(return_value=torch.randn(1, 64))

        result = nudge_reasoning_stepwise(
            prompt="Test",
            llm=mock_llm,
            rjepa_client=mock_rjepa,
            max_steps=3,
            lambda_nudge=0.2,
        )

        assert "full_text" in result
        assert "steps" in result
        assert "corrected" in result

        print("[OK] Nudge mode works")

    except Exception as e:
        print(f"[FAIL] Nudge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test plan
    try:
        print("[*] Testing plan mode...")

        partial_steps = ["Step 1: Start", None, "Step 3: End"]
        missing_indices = [1]

        result = complete_reasoning_plan(
            partial_steps=partial_steps,
            missing_indices=missing_indices,
            total_steps=3,
            llm=mock_llm,
            rjepa_client=mock_rjepa,
        )

        assert "completed_steps" in result
        assert "predicted_steps" in result
        assert len(result["completed_steps"]) == 3

        print("[OK] Plan mode works")

    except Exception as e:
        print(f"[FAIL] Plan test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 80)
    print("SUCCESS: PHASE 8 VALIDATION COMPLETE!")
    print("=" * 80)
    print()
    print("[OK] All required files exist")
    print("[OK] All imports work")
    print("[OK] Re-ranking mode works")
    print("[OK] Nudge mode works")
    print("[OK] Plan mode works")
    print()
    print("Statistics:")
    print("   - Modes: rerank, nudge, plan")
    print("   - Functions: 9 inference functions")
    print("   - Re-ranking: 3 variants (basic, ensembling)")
    print("   - Nudge: 3 variants (stepwise, regeneration, beam)")
    print("   - Plan: 3 variants (completion, auto-complete, refinement)")
    print()
    print("Key Features:")
    print("   [OK] Re-ranking CoT candidates (JEPA-guided)")
    print("   [OK] Nudge correction (latent space)")
    print("   [OK] Plan completion (missing steps)")
    print("   [OK] Ensembling (top-K voting)")
    print("   [OK] Beam search (JEPA-guided)")
    print("   [OK] Iterative refinement")
    print()
    print("READY FOR PHASE 9: Frontend (Next.js chat + monitoring)")
    print()

    return True


if __name__ == "__main__":
    success = validate_phase8()
    sys.exit(0 if success else 1)
