"""
Test Data Pipeline components.
"""
import pytest
from pathlib import Path
import tempfile
import torch
from unittest.mock import Mock, patch

from rjepa.data.schemas import Problem, ChainOfThought
from rjepa.data.ingestion import (
    HuggingFaceDatasetIngestion,
    CustomDatasetIngestion,
    save_problems_to_parquet,
    save_cots_to_parquet,
)
from rjepa.data.sharding import DatasetSharding, LatentSharding
from rjepa.utils.io import ParquetIO, SafeTensorsIO, DuckDBIndex


def test_parquet_io():
    """Test parquet read/write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.parquet"

        # Write
        data = [
            {"id": "1", "text": "hello", "value": 42},
            {"id": "2", "text": "world", "value": 13},
        ]
        ParquetIO.write(data, output_path)

        assert output_path.exists()

        # Read
        loaded = ParquetIO.read(output_path)
        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"
        assert loaded[1]["value"] == 13


def test_safetensors_io():
    """Test safetensors read/write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.safetensors"

        # Save
        tensors = {
            "tensor_1": torch.randn(10, 128),
            "tensor_2": torch.randn(5, 64),
        }
        SafeTensorsIO.save_latents(tensors, output_path)

        assert output_path.exists()

        # Load
        loaded = SafeTensorsIO.load_latents(output_path)
        assert len(loaded) == 2
        assert loaded["tensor_1"].shape == (10, 128)
        assert loaded["tensor_2"].shape == (5, 64)


def test_duckdb_index():
    """Test DuckDB indexing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "data.parquet"

        # Create test data
        data = [
            {"id": "1", "domain": "math", "difficulty": "easy"},
            {"id": "2", "domain": "math", "difficulty": "hard"},
            {"id": "3", "domain": "code", "difficulty": "medium"},
        ]
        ParquetIO.write(data, parquet_path)

        # Create index
        db = DuckDBIndex()
        db.create_view_from_parquet("test_data", parquet_path)

        # Query
        results = db.query("SELECT * FROM test_data WHERE domain = 'math'")
        assert len(results) == 2

        results = db.query("SELECT COUNT(*) as cnt FROM test_data")
        assert results[0]["cnt"] == 3

        # Stats
        stats = db.get_stats("test_data")
        assert stats["count"] == 3
        assert len(stats["domain_distribution"]) == 2

        db.close()


def test_dataset_sharding():
    """Test dataset sharding."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test data
        records = [{"id": str(i), "value": i} for i in range(25)]

        # Save sharded
        shard_paths = DatasetSharding.save_sharded_dataset(
            records, output_dir, shard_size=10
        )

        assert len(shard_paths) == 3  # 10 + 10 + 5
        assert all(p.exists() for p in shard_paths)

        # Load sharded
        loaded = DatasetSharding.load_sharded_dataset(output_dir)
        assert len(loaded) == 25
        assert loaded[0]["id"] == "0"
        assert loaded[24]["id"] == "24"


def test_latent_sharding():
    """Test latent sharding (metadata + tensors)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test data
        metadata_records = [
            {"cot_id": f"cot_{i}", "num_steps": 3 + i, "hidden_size": 128}
            for i in range(5)
        ]

        latents_dict = {
            f"cot_{i}": torch.randn(3 + i, 128) for i in range(5)
        }

        # Save shard
        metadata_path, latents_path = LatentSharding.save_latent_shard(
            metadata_records, latents_dict, output_dir, shard_idx=0
        )

        assert metadata_path.exists()
        assert latents_path.exists()

        # Load shard
        loaded_metadata, loaded_latents = LatentSharding.load_latent_shard(
            output_dir, shard_idx=0
        )

        assert len(loaded_metadata) == 5
        assert len(loaded_latents) == 5
        assert loaded_latents["cot_0"].shape == (3, 128)
        assert loaded_latents["cot_4"].shape == (7, 128)


def test_custom_ingestion():
    """Test custom JSON ingestion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "problems.json"

        # Create test JSON
        import json

        data = [
            {
                "problem_id": "test_1",
                "domain": "math",
                "statement": "What is 2+2?",
                "answer_gold": "4",
            },
            {
                "problem_id": "test_2",
                "domain": "code",
                "statement": "Write hello world",
            },
        ]

        with open(json_path, "w") as f:
            json.dump(data, f)

        # Ingest
        problems = CustomDatasetIngestion.ingest_json_problems(json_path, "test")

        assert len(problems) == 2
        assert problems[0].problem_id == "test_1"
        assert problems[0].domain == "math"
        assert problems[1].answer_gold is None  # Not provided


def test_save_problems_to_parquet():
    """Test saving problems to parquet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "problems.parquet"

        problems = [
            Problem(
                problem_id="p1",
                domain="math",
                subdomain="algebra",
                source="test",
                difficulty="easy",
                statement="Solve x+2=5",
                answer_gold="3",
            ),
            Problem(
                problem_id="p2",
                domain="code",
                subdomain="python",
                source="test",
                difficulty="medium",
                statement="Write fibonacci",
            ),
        ]

        save_problems_to_parquet(problems, output_path)

        assert output_path.exists()

        # Load and verify
        loaded = ParquetIO.read(output_path)
        assert len(loaded) == 2
        assert loaded[0]["problem_id"] == "p1"
        assert loaded[0]["statement"] == "Solve x+2=5"


def test_save_cots_to_parquet():
    """Test saving CoTs to parquet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "cots.parquet"

        cots = [
            ChainOfThought(
                cot_id="cot1",
                problem_id="p1",
                steps=["Step 1: Subtract 2", "Step 2: x = 3"],
                final_answer="3",
                is_valid=True,
                validation_reason="Correct",
                teacher_model="test",
                source="test",
            ),
        ]

        save_cots_to_parquet(cots, output_path)

        assert output_path.exists()

        # Load and verify
        loaded = ParquetIO.read(output_path)
        assert len(loaded) == 1
        assert loaded[0]["cot_id"] == "cot1"
        assert len(loaded[0]["steps"]) == 2


@patch("rjepa.data.ingestion.load_dataset")
def test_ingest_gsm8k(mock_load_dataset):
    """Test GSM8K ingestion (mocked)."""
    # Mock dataset
    mock_dataset = [
        {
            "question": "Janet has 5 apples...",
            "answer": "Step 1\n####\n10",
        },
        {
            "question": "Bob has 3 oranges...",
            "answer": "Step 1\n####\n6",
        },
    ]

    mock_load_dataset.return_value.select.return_value = mock_dataset
    mock_load_dataset.return_value.__iter__ = lambda self: iter(mock_dataset)
    mock_load_dataset.return_value.__len__ = lambda self: len(mock_dataset)

    # Ingest
    problems = HuggingFaceDatasetIngestion.ingest_gsm8k(split="train", max_samples=2)

    assert len(problems) == 2
    assert problems[0].domain == "math"
    assert problems[0].subdomain == "arithmetic"
    assert problems[0].source == "gsm8k"
    assert "Janet" in problems[0].statement
    assert problems[0].answer_gold == "10"
