#!/usr/bin/env python3
"""
Validate Phase 4: Data Pipeline
"""
import sys
import tempfile
from pathlib import Path


def validate_phase4():
    """Validate that Phase 4 is complete."""
    print("[*] Validating Phase 4: Data Pipeline...")
    print()

    # Check files exist
    required_files = [
        "rjepa/pipeline/build_latents.py",
        "rjepa/data/ingestion.py",
        "rjepa/data/sharding.py",
        "rjepa/utils/io.py",
        "tests/test_pipeline.py",
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
        print("[FAIL] Phase 4 validation FAILED: Some files are missing")
        return False

    # Try importing
    try:
        from rjepa.utils.io import ParquetIO, SafeTensorsIO, DuckDBIndex
        print("[OK] I/O utilities import successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import I/O utils: {e}")
        return False

    try:
        from rjepa.data.sharding import DatasetSharding, LatentSharding
        print("[OK] Sharding utilities import successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import sharding: {e}")
        return False

    try:
        from rjepa.data.ingestion import (
            HuggingFaceDatasetIngestion,
            CustomDatasetIngestion,
            UserInteractionIngestion,
        )
        print("[OK] Ingestion utilities import successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import ingestion: {e}")
        return False

    try:
        from rjepa.pipeline.build_latents import build_latents_from_cots
        print("[OK] Build latents pipeline imports successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import build_latents: {e}")
        return False

    print()

    # Test instantiation
    try:
        print("[*] Testing ParquetIO...")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test.parquet"
            test_data = [{"id": "1", "value": 42}]

            ParquetIO.write(test_data, test_path)
            loaded = ParquetIO.read(test_path)

            assert len(loaded) == 1
            assert loaded[0]["value"] == 42

        print("[OK] ParquetIO works")

    except Exception as e:
        print(f"[FAIL] ParquetIO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        print("[*] Testing SafeTensorsIO...")
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test.safetensors"
            test_tensors = {"t1": torch.randn(5, 10)}

            SafeTensorsIO.save_latents(test_tensors, test_path)
            loaded = SafeTensorsIO.load_latents(test_path)

            assert "t1" in loaded
            assert loaded["t1"].shape == (5, 10)

        print("[OK] SafeTensorsIO works")

    except Exception as e:
        print(f"[FAIL] SafeTensorsIO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        print("[*] Testing DatasetSharding...")

        records = [{"id": str(i)} for i in range(25)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            shard_paths = DatasetSharding.save_sharded_dataset(
                records, output_dir, shard_size=10
            )

            assert len(shard_paths) == 3  # 10 + 10 + 5

            loaded = DatasetSharding.load_sharded_dataset(output_dir)
            assert len(loaded) == 25

        print("[OK] DatasetSharding works")

    except Exception as e:
        print(f"[FAIL] DatasetSharding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        print("[*] Testing DuckDBIndex...")

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "test.parquet"
            test_data = [
                {"id": "1", "domain": "math"},
                {"id": "2", "domain": "code"},
            ]

            ParquetIO.write(test_data, parquet_path)

            db = DuckDBIndex()
            db.create_view_from_parquet("test_view", parquet_path)

            results = db.query("SELECT COUNT(*) as cnt FROM test_view")
            assert results[0]["cnt"] == 2

            stats = db.get_stats("test_view")
            assert stats["count"] == 2

            db.close()

        print("[OK] DuckDBIndex works")

    except Exception as e:
        print(f"[FAIL] DuckDBIndex test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 80)
    print("SUCCESS: PHASE 4 VALIDATION COMPLETE!")
    print("=" * 80)
    print()
    print("[OK] All required files exist")
    print("[OK] All imports work")
    print("[OK] ParquetIO works")
    print("[OK] SafeTensorsIO works")
    print("[OK] DatasetSharding works")
    print("[OK] DuckDBIndex works")
    print()
    print("Statistics:")
    print("   - I/O utilities: 3 (ParquetIO, SafeTensorsIO, DuckDBIndex)")
    print("   - Sharding: 2 (DatasetSharding, LatentSharding)")
    print("   - Ingestion: 3 (HuggingFace, Custom, UserInteraction)")
    print("   - Pipeline: 1 (build_latents)")
    print("   - Tests: 1 (test_pipeline.py)")
    print()
    print("READY FOR PHASE 5: R-JEPA Model (encoder, predictor, EMA)")
    print()

    return True


if __name__ == "__main__":
    success = validate_phase4()
    sys.exit(0 if success else 1)
