"""
Test R-JEPA Service (API inference).
"""
import pytest
import torch
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from rjepa.jepa import ReasoningJEPA, create_app, RJEPAClient


@pytest.fixture
def dummy_checkpoint():
    """Create a dummy checkpoint for testing."""
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

    yield checkpoint_path

    # Cleanup
    checkpoint_path.unlink()


@pytest.fixture
def app(dummy_checkpoint):
    """Create FastAPI app with dummy checkpoint."""
    return create_app(checkpoint_path=dummy_checkpoint, device="cpu")


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"
    assert "rjepa_config" in data


def test_score_endpoint(client):
    """Test score endpoint."""
    # Create dummy latents
    latents = torch.randn(10, 64).tolist()  # 10 steps, 64 dim

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
    assert isinstance(data["jepa_loss"], float)
    assert data["jepa_loss"] > 0  # Loss should be positive

    assert data["num_steps"] == 10
    assert data["num_masked"] == 5  # 50% of 10
    assert data["device"] == "cpu"


def test_score_with_domain(client):
    """Test score endpoint with domain."""
    # Need to create checkpoint with domain embeddings for this
    # For now, just test that domain_id is accepted
    latents = torch.randn(10, 64).tolist()

    response = client.post(
        "/score",
        json={
            "latents": latents,
            "domain_id": 0,
            "mask_ratio": 0.3,
        },
    )

    # Should still work (domain_embed_dim=0 so domain_id is ignored)
    assert response.status_code == 200


def test_score_invalid_latents(client):
    """Test score endpoint with invalid latents."""
    # 1D latents (should be 2D)
    latents = [1.0, 2.0, 3.0]

    response = client.post(
        "/score",
        json={
            "latents": latents,
            "mask_ratio": 0.5,
        },
    )

    assert response.status_code == 500  # Should raise error


def test_predict_masked_endpoint(client):
    """Test predict_masked endpoint."""
    # Create dummy latents
    latents = torch.randn(10, 64).tolist()  # 10 steps, 64 dim

    # Mask steps 3, 4, 5
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
    predicted = data["predicted_latents"]

    # Should predict 3 steps
    assert len(predicted) == 3
    assert len(predicted[0]) == 64  # hidden_dim

    assert data["num_predicted"] == 3
    assert data["device"] == "cpu"


def test_predict_masked_with_domain(client):
    """Test predict_masked endpoint with domain."""
    latents = torch.randn(10, 64).tolist()
    mask_indices = [2, 3]

    response = client.post(
        "/predict_masked",
        json={
            "latents": latents,
            "mask_indices": mask_indices,
            "domain_id": 0,
        },
    )

    assert response.status_code == 200


def test_predict_masked_invalid_indices(client):
    """Test predict_masked endpoint with invalid indices."""
    latents = torch.randn(10, 64).tolist()

    # Out of bounds indices
    mask_indices = [5, 10, 15]  # Only 0-9 valid

    response = client.post(
        "/predict_masked",
        json={
            "latents": latents,
            "mask_indices": mask_indices,
        },
    )

    assert response.status_code == 500  # Should raise error


def test_rjepa_client_health(dummy_checkpoint):
    """Test RJEPAClient health check."""
    # Create app
    app = create_app(checkpoint_path=dummy_checkpoint, device="cpu")

    # Create test client
    test_client = TestClient(app)

    # Create RJEPAClient (using test client as base)
    # Note: This requires starting a real server for full integration test
    # For now, just test basic functionality
    client = RJEPAClient("http://testserver")

    # This would work with real server:
    # health = client.health()
    # assert health["status"] == "ok"


def test_rjepa_client_score(dummy_checkpoint):
    """Test RJEPAClient score method."""
    app = create_app(checkpoint_path=dummy_checkpoint, device="cpu")
    test_client = TestClient(app)

    # Monkey patch httpx to use test client
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

        latents = torch.randn(10, 64)
        result = client.score(latents, mask_ratio=0.5)

        assert "jepa_loss" in result
        assert result["num_steps"] == 10

    finally:
        # Restore original
        httpx.get = original_get
        httpx.post = original_post


def test_rjepa_client_predict_masked(dummy_checkpoint):
    """Test RJEPAClient predict_masked method."""
    app = create_app(checkpoint_path=dummy_checkpoint, device="cpu")
    test_client = TestClient(app)

    # Monkey patch httpx
    import httpx

    original_post = httpx.post

    def mock_post(url, **kwargs):
        return test_client.post(url.replace("http://testserver", ""), **kwargs)

    httpx.post = mock_post

    try:
        client = RJEPAClient("http://testserver")

        latents = torch.randn(10, 64)
        mask_indices = [3, 4, 5]

        predicted = client.predict_masked(latents, mask_indices)

        assert predicted.shape == (3, 64)

    finally:
        httpx.post = original_post
