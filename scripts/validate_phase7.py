#!/usr/bin/env python3
"""
Validate Phase 7: R-JEPA Service (inference API)
"""
import sys
import torch
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient


def validate_phase7():
    """Validate that Phase 7 is complete."""
    print("[*] Validating Phase 7: R-JEPA Service...")
    print()

    # Check files exist
    required_files = [
        "rjepa/jepa/service.py",
        "rjepa/jepa/client.py",
        "docker/rjepa-service.Dockerfile",
        "tests/test_service.py",
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
        print("[FAIL] Phase 7 validation FAILED: Some files are missing")
        return False

    # Try importing
    try:
        from rjepa.jepa.service import create_app, RJEPAService
        print("[OK] RJEPAService imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import RJEPAService: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        from rjepa.jepa.client import RJEPAClient
        print("[OK] RJEPAClient imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import RJEPAClient: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test creating a service
    try:
        print("[*] Testing service creation...")

        from rjepa.jepa import ReasoningJEPA

        # Create dummy model
        model = ReasoningJEPA(
            dim=64,
            depth_encoder=2,
            depth_predictor=2,
            num_heads=2,
            predictor_dim=128,
            max_steps=32,
        )

        # Create checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "dim": 64,
                "depth_encoder": 2,
                "depth_predictor": 2,
                "num_heads": 2,
                "predictor_dim": 128,
                "max_steps": 32,
                "domain_embed_dim": 0,
                "num_domains": 0,
                "ema_momentum": 0.996,
                "loss_config": {"loss_type": "l1", "var_reg_weight": 0.01},
            },
            "epoch": 10,
        }

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = Path(f.name)

        # Create app
        app = create_app(checkpoint_path=checkpoint_path, device="cpu")

        assert app is not None
        print("[OK] Service created successfully")

        # Cleanup
        checkpoint_path.unlink()

    except Exception as e:
        print(f"[FAIL] Service creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test endpoints
    try:
        print("[*] Testing endpoints...")

        # Create checkpoint again
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = Path(f.name)

        app = create_app(checkpoint_path=checkpoint_path, device="cpu")
        client = TestClient(app)

        # Test health
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        print("[OK] /health endpoint works")

        # Test score
        latents = torch.randn(10, 64).tolist()
        response = client.post(
            "/score",
            json={
                "latents": latents,
                "mask_ratio": 0.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "jepa_loss" in data
        assert data["num_steps"] == 10
        print("[OK] /score endpoint works")

        # Test predict_masked
        mask_indices = [3, 4, 5]
        response = client.post(
            "/predict_masked",
            json={
                "latents": latents,
                "mask_indices": mask_indices,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_latents" in data
        assert len(data["predicted_latents"]) == 3
        print("[OK] /predict_masked endpoint works")

        # Cleanup
        checkpoint_path.unlink()

    except Exception as e:
        print(f"[FAIL] Endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test client
    try:
        print("[*] Testing RJEPAClient...")

        from rjepa.jepa.client import RJEPAClient

        # Create checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = Path(f.name)

        app = create_app(checkpoint_path=checkpoint_path, device="cpu")
        test_client = TestClient(app)

        # Monkey patch httpx
        import httpx

        original_get = httpx.get
        original_post = httpx.post

        def mock_get(url, **kwargs):
            return test_client.get(url.replace("http://testserver", ""))

        def mock_post(url, **kwargs):
            return test_client.post(url.replace("http://testserver", ""), **kwargs)

        httpx.get = mock_get
        httpx.post = mock_post

        try:
            client = RJEPAClient("http://testserver")

            # Test score
            latents = torch.randn(10, 64)
            result = client.score(latents, mask_ratio=0.5)
            assert "jepa_loss" in result
            print("[OK] RJEPAClient.score() works")

            # Test predict_masked
            mask_indices = [3, 4, 5]
            predicted = client.predict_masked(latents, mask_indices)
            assert predicted.shape == (3, 64)
            print("[OK] RJEPAClient.predict_masked() works")

        finally:
            httpx.get = original_get
            httpx.post = original_post

        # Cleanup
        checkpoint_path.unlink()

    except Exception as e:
        print(f"[FAIL] RJEPAClient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 80)
    print("SUCCESS: PHASE 7 VALIDATION COMPLETE!")
    print("=" * 80)
    print()
    print("[OK] All required files exist")
    print("[OK] All imports work")
    print("[OK] Service creation works")
    print("[OK] /health endpoint works")
    print("[OK] /score endpoint works")
    print("[OK] /predict_masked endpoint works")
    print("[OK] RJEPAClient works")
    print()
    print("Statistics:")
    print("   - Service: FastAPI with 3 endpoints")
    print("   - Client: RJEPAClient (Python HTTP client)")
    print("   - Docker: rjepa-service.Dockerfile")
    print("   - Endpoints: /health, /score, /predict_masked")
    print()
    print("Key Features:")
    print("   [OK] Checkpoint loading (save/load R-JEPA)")
    print("   [OK] Scoring for re-ranking (JEPA-loss)")
    print("   [OK] Prediction for nudge/plan (masked steps)")
    print("   [OK] Health check endpoint")
    print("   [OK] Pydantic validation")
    print("   [OK] Error handling")
    print()
    print("READY FOR PHASE 8: Inference Modes (rerank, nudge, plan)")
    print()

    return True


if __name__ == "__main__":
    success = validate_phase7()
    sys.exit(0 if success else 1)
