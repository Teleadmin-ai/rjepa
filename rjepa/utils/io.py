"""
I/O utilities for R-JEPA data pipeline.

Handles:
- Parquet read/write with compression
- SafeTensors serialization for latents
- DuckDB indexing for fast queries
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import torch
from safetensors.torch import save_file, load_file
import json

logger = logging.getLogger(__name__)


class ParquetIO:
    """Parquet file I/O with compression."""

    @staticmethod
    def write(
        data: List[Dict[str, Any]],
        output_path: Path,
        compression: str = "zstd",
    ) -> None:
        """
        Write records to parquet file.

        Args:
            data: List of dicts to write
            output_path: Output parquet file path
            compression: Compression algorithm ("zstd", "snappy", "gzip")
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PyArrow table
        table = pa.Table.from_pylist(data)

        # Write parquet with compression
        pq.write_table(
            table,
            output_path,
            compression=compression,
            row_group_size=10000,
        )

        logger.info(f"Wrote {len(data)} records to {output_path}")

    @staticmethod
    def read(
        input_path: Path,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read parquet file.

        Args:
            input_path: Input parquet file path
            columns: Optional list of columns to read

        Returns:
            List of dicts
        """
        table = pq.read_table(input_path, columns=columns)

        # Use pandas conversion as fallback (more robust than to_pylist)
        try:
            return table.to_pylist()
        except (IndexError, Exception) as e:
            logger.warning(f"to_pylist() failed, using pandas fallback: {e}")
            # Fallback to pandas which is more robust
            df = table.to_pandas()
            return df.to_dict('records')

    @staticmethod
    def read_batch(
        input_path: Path,
        batch_size: int = 1000,
        columns: Optional[List[str]] = None,
    ):
        """
        Read parquet file in batches (generator).

        Args:
            input_path: Input parquet file path
            batch_size: Number of rows per batch
            columns: Optional list of columns to read

        Yields:
            Batches of records
        """
        parquet_file = pq.ParquetFile(input_path)

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=columns,
        ):
            yield batch.to_pylist()


class SafeTensorsIO:
    """SafeTensors I/O for latents."""

    @staticmethod
    def save_latents(
        latents_dict: Dict[str, torch.Tensor],
        output_path: Path,
    ) -> None:
        """
        Save latents to SafeTensors file.

        Args:
            latents_dict: Dict mapping keys to tensors
                e.g., {"cot_0": tensor([num_steps, hidden]), ...}
            output_path: Output .safetensors file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to contiguous tensors (required by safetensors)
        contiguous_dict = {
            key: tensor.contiguous()
            for key, tensor in latents_dict.items()
        }

        save_file(contiguous_dict, output_path)

        total_size = sum(t.numel() * t.element_size() for t in latents_dict.values())
        logger.info(
            f"Saved {len(latents_dict)} latent tensors to {output_path} "
            f"({total_size / 1024 / 1024:.2f} MB)"
        )

    @staticmethod
    def load_latents(
        input_path: Path,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Load latents from SafeTensors file.

        Args:
            input_path: Input .safetensors file path
            device: Device to load tensors to

        Returns:
            Dict mapping keys to tensors
        """
        latents = load_file(input_path, device=device)
        logger.info(f"Loaded {len(latents)} latent tensors from {input_path}")
        return latents


class DuckDBIndex:
    """DuckDB indexing for fast queries on parquet files."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize DuckDB connection.

        Args:
            db_path: Optional path to persistent DB file (in-memory if None)
        """
        if db_path:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = duckdb.connect(str(db_path))
        else:
            self.conn = duckdb.connect(":memory:")

        logger.info(f"Initialized DuckDB connection: {db_path or 'in-memory'}")

    def create_view_from_parquet(
        self,
        view_name: str,
        parquet_path: Union[Path, List[Path]],
    ) -> None:
        """
        Create view from parquet file(s).

        Args:
            view_name: Name for the view
            parquet_path: Path to parquet file or list of paths (glob supported)
        """
        if isinstance(parquet_path, list):
            parquet_str = ", ".join(f"'{p}'" for p in parquet_path)
            query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet([{parquet_str}])"
        else:
            query = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{parquet_path}')"

        self.conn.execute(query)
        logger.info(f"Created view '{view_name}' from parquet")

    def query(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results.

        Args:
            sql: SQL query string

        Returns:
            List of dicts (rows)
        """
        result = self.conn.execute(sql).fetchall()
        columns = [desc[0] for desc in self.conn.description]

        return [dict(zip(columns, row)) for row in result]

    def query_df(self, sql: str):
        """
        Execute SQL query and return as pandas DataFrame.

        Args:
            sql: SQL query string

        Returns:
            pandas DataFrame
        """
        return self.conn.execute(sql).df()

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics for a table/view.

        Args:
            table_name: Name of table or view

        Returns:
            Dict with statistics
        """
        stats = {}

        # Count
        count = self.conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        stats["count"] = count

        # Domains distribution
        try:
            domain_dist = self.query(
                f"SELECT domain, COUNT(*) as count FROM {table_name} GROUP BY domain ORDER BY count DESC"
            )
            stats["domain_distribution"] = domain_dist
        except Exception:
            pass

        # Difficulty distribution
        try:
            difficulty_dist = self.query(
                f"SELECT difficulty, COUNT(*) as count FROM {table_name} GROUP BY difficulty ORDER BY count DESC"
            )
            stats["difficulty_distribution"] = difficulty_dist
        except Exception:
            pass

        return stats

    def close(self):
        """Close DuckDB connection."""
        self.conn.close()


def save_metadata_json(
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save metadata as JSON.

    Args:
        metadata: Metadata dict
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metadata to {output_path}")


def load_metadata_json(input_path: Path) -> Dict[str, Any]:
    """
    Load metadata from JSON.

    Args:
        input_path: Input JSON file path

    Returns:
        Metadata dict
    """
    with open(input_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata from {input_path}")
    return metadata
