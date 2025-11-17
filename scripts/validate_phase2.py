#!/usr/bin/env python3
"""
Validate Phase 2: LLM Adapter
"""
import sys
from pathlib import Path


def validate_phase2():
    """Validate that Phase 2 is complete."""
    print("[*] Validating Phase 2: LLM Adapter...")
    print()

    # Check files exist
    required_files = [
        "rjepa/llm/adapter.py",
        "rjepa/llm/step_segmentation.py",
        "rjepa/llm/quant_utils.py",
        "rjepa/llm/server.py",
        "rjepa/llm/__init__.py",
        "docker/student-llm.Dockerfile",
        "tests/test_llm_adapter.py",
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
        print("[FAIL] Phase 2 validation FAILED: Some files are missing")
        return False

    # Try importing
    try:
        from rjepa.llm import LLMAdapter
        print("[OK] LLMAdapter imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import LLMAdapter: {e}")
        return False

    try:
        from rjepa.llm import segment_auto, check_quantization_available
        print("[OK] Utility functions import successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import utilities: {e}")
        return False

    print()

    # Check LLMAdapter instantiation (with a tiny model for speed)
    try:
        print("[*] Testing LLMAdapter instantiation (using gpt2)...")
        adapter = LLMAdapter(
            model_name="gpt2",
            quantization=None,
            layer_to_extract=-2,
        )
        print(f"[OK] LLMAdapter instantiated: {adapter}")

        # Test generation
        print("[*] Testing CoT generation...")
        results = adapter.generate_with_cot(
            prompt="What is 2+2?",
            max_new_tokens=20,
            num_samples=1,
        )
        print(f"[OK] Generated {len(results)} sample(s)")

        # Test latent extraction
        print("[*] Testing latent extraction...")
        result = results[0]
        latents = adapter.extract_latents(
            tokens=result["tokens"],
            step_boundaries=result["step_boundaries"],
        )
        print(f"[OK] Extracted latents shape: {latents.shape}")

    except Exception as e:
        print(f"[FAIL] LLMAdapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("="*80)
    print("SUCCESS: PHASE 2 VALIDATION COMPLETE!")
    print("="*80)
    print()
    print("[OK] All required files exist")
    print("[OK] All imports work")
    print("[OK] LLMAdapter can be instantiated")
    print("[OK] CoT generation works")
    print("[OK] Latent extraction works")
    print()
    print("Statistics:")
    print("   - LLM adapter classes: 1 (LLMAdapter)")
    print("   - Utility modules: 2 (step_segmentation, quant_utils)")
    print("   - API endpoints: 4 (/health, /generate, /extract_latents, /model_info)")
    print("   - Dockerfile: 1 (student-llm)")
    print("   - Tests: 1")
    print()
    print("READY FOR PHASE 3: Teacher Orchestrator")
    print()

    return True


if __name__ == "__main__":
    success = validate_phase2()
    sys.exit(0 if success else 1)
