#!/usr/bin/env python3
"""
CORRECTED Latent Extraction Pipeline - Uses VALIDATED CoTs only.

CRITICAL FIX (2025-11-26):
The previous script (extract_latents_optimized.py) was GENERATING new CoTs
instead of using the validated ones from GSM8K/MATH/HumanEval datasets.

This script:
1. Loads VALIDATED CoTs from *_cots.json files (human-verified)
2. Does a FORWARD PASS only (NO generation!)
3. Extracts latents from layer -2
4. Saves in shard format for R-JEPA training

The latents now represent the LLM's internal understanding of CORRECT reasoning,
not potentially flawed generated reasoning.

USAGE:
    python scripts/extract_latents_validated.py --batch-size 8 --limit 100
    python scripts/extract_latents_validated.py --resume  # Continue from checkpoint
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
from safetensors.torch import save_file
import re


class ValidatedLatentExtractor:
    """
    Extract latents from VALIDATED CoTs only.

    Key difference from old script:
    - OLD: generate(problem) -> extract(generated_text)  # WRONG!
    - NEW: load(validated_cot) -> extract(validated_text)  # CORRECT!
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        layer_to_extract: int = -2,
        output_dir: str = "data/latents/qwen3-8b/validated",
        checkpoint_file: str = "data/latents/qwen3-8b/validated/checkpoint.json",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.layer_to_extract = layer_to_extract
        self.output_dir = Path(output_dir)
        self.checkpoint_file = Path(checkpoint_file)

        # Create output dirs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "total_time": 0.0,
            "avg_steps": [],
        }

        # Model & tokenizer (loaded later)
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer ONCE."""
        print("[LOAD] Loading Qwen3-8B model...")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")
        print(f"  Layer to extract: {self.layer_to_extract}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, self.dtype),
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        print(f"[OK] Model loaded:")
        print(f"  Hidden size: {self.model.config.hidden_size}")
        print(f"  Num layers: {self.model.config.num_hidden_layers}")
        print()

    def load_checkpoint(self) -> set:
        """Load already processed CoT IDs."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
                print(f"[RESUME] Loaded checkpoint: {len(data['processed'])} already processed")
                return set(data["processed"])
        return set()

    def save_checkpoint(self, processed: set):
        """Save checkpoint with processed IDs."""
        with open(self.checkpoint_file, "w") as f:
            json.dump({"processed": list(processed)}, f)

    def load_validated_cots(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Load all VALIDATED CoTs from JSON files.

        These are human-verified reasoning chains from:
        - GSM8K (grade school math)
        - MATH (competition math)
        - HumanEval (code)
        """
        all_cots = []

        cot_files = [
            # GSM8K
            "data/datasets/academic/math/gsm8k/train_cots.json",
            "data/datasets/academic/math/gsm8k/test_cots.json",
            # MATH (Competition)
            "data/datasets/academic/math/competition_math/train_cots.json",
            "data/datasets/academic/math/competition_math/test_cots.json",
            # HumanEval (Code)
            "data/datasets/academic/code/humaneval/test_cots.json",
        ]

        for cot_file in cot_files:
            path = Path(cot_file)
            if path.exists():
                with open(path) as f:
                    cots = json.load(f)
                    # Filter only valid CoTs
                    valid_cots = [c for c in cots if c.get("is_valid", False)]
                    all_cots.extend(valid_cots)
                    print(f"  [LOAD] {path.name}: {len(valid_cots)} valid CoTs")
            else:
                print(f"  [WARN] {path} not found")

        if limit:
            all_cots = all_cots[:limit]

        return all_cots

    def cot_to_text(self, cot: Dict[str, Any]) -> str:
        """
        Convert a validated CoT to text format.

        Input:
            {
                "steps": ["Step 1: ...", "Step 2: ..."],
                "final_answer": "72"
            }

        Output:
            "Step 1: ...\nStep 2: ...\nFinal Answer: 72"
        """
        steps_text = "\n".join(cot["steps"])

        # Add final answer if not already in last step
        final_answer = cot.get("final_answer", "")
        if final_answer and f"Final Answer" not in steps_text:
            steps_text += f"\nFinal Answer: {final_answer}"

        return steps_text

    def find_step_boundaries(self, text: str, tokens: torch.LongTensor) -> List[Tuple[int, int]]:
        """
        Find token boundaries for each step in the text.

        Returns list of (start_token_idx, end_token_idx) for each step.
        """
        # Find "Step X:" patterns
        pattern = r"Step\s+\d+:"
        matches = list(re.finditer(pattern, text))

        if not matches:
            # Fallback: entire text is one step
            return [(0, tokens.shape[0])]

        boundaries = []

        for i, match in enumerate(matches):
            start_char = match.start()
            end_char = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # Find token boundaries (approximate via encoding)
            prefix_tokens = len(self.tokenizer.encode(text[:start_char], add_special_tokens=False))
            step_tokens = len(self.tokenizer.encode(text[start_char:end_char], add_special_tokens=False))
            boundaries.append((prefix_tokens, prefix_tokens + step_tokens))

        return boundaries

    def extract_latents_from_text(self, text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]]]:
        """
        Extract latents from validated text (NO GENERATION!).

        This is the CORE function - it does a forward pass only,
        no text generation. The latents represent the LLM's internal
        understanding of the validated reasoning.

        Returns:
            latents: numpy array [num_steps, hidden_size]
            steps: list of step strings
            boundaries: list of (start, end) token indices
        """
        # Tokenize the validated text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Forward pass ONLY (no generation!)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Get hidden states at target layer
        hidden_states = outputs.hidden_states[self.layer_to_extract]  # [1, T, D]

        # Find step boundaries
        boundaries = self.find_step_boundaries(text, inputs.input_ids[0])

        # Extract steps text
        pattern = r"Step\s+\d+:"
        matches = list(re.finditer(pattern, text))
        steps = []
        for i, match in enumerate(matches):
            start_char = match.start()
            end_char = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            steps.append(text[start_char:end_char].strip())

        if not steps:
            steps = [text]

        # Average hidden states for each step
        latents = []
        seq_len = hidden_states.shape[1]

        for start, end in boundaries:
            # Clamp to valid range
            start = max(0, min(start, seq_len - 1))
            end = max(start + 1, min(end, seq_len))

            step_latent = hidden_states[0, start:end, :].mean(dim=0)  # [D]
            latents.append(step_latent.to(torch.float32).cpu().numpy())

        if not latents:
            # Fallback: use entire sequence
            latents = [hidden_states[0].mean(dim=0).to(torch.float32).cpu().numpy()]

        return np.stack(latents, axis=0), steps, boundaries

    def extract_batch(self, cots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract latents for a batch of validated CoTs.

        Note: We process one at a time for simplicity since
        validated texts have variable lengths. Batch processing
        with padding could be added later for speed.
        """
        results = []

        for cot in cots:
            try:
                # Convert CoT to text
                text = self.cot_to_text(cot)

                # Extract latents (NO GENERATION!)
                latents, steps, boundaries = self.extract_latents_from_text(text)

                results.append({
                    "status": "success",
                    "cot_id": cot["cot_id"],
                    "problem_id": cot["problem_id"],
                    "domain": cot.get("meta", {}).get("source", "unknown"),
                    "subdomain": cot.get("meta", {}).get("subdomain", ""),
                    "num_steps": len(latents),
                    "hidden_size": latents.shape[1],
                    "latents": latents,  # [num_steps, hidden_size]
                    "steps": steps,
                    "step_boundaries": boundaries,
                    "is_valid": cot.get("is_valid", True),
                    "teacher_model": cot.get("teacher_model", "unknown"),
                })

                self.stats["avg_steps"].append(len(latents))

            except Exception as e:
                results.append({
                    "status": "failed",
                    "cot_id": cot.get("cot_id", "unknown"),
                    "error": str(e),
                })

        return results

    def save_shard(self, shard_metadata: List[Dict], shard_latents: Dict[str, torch.Tensor], shard_idx: int):
        """
        Save shard to parquet + safetensors format.
        """
        shard_name = f"shard-{shard_idx:04d}"

        # Save metadata to parquet
        metadata_path = self.output_dir / f"{shard_name}.parquet"
        df = pd.DataFrame(shard_metadata)
        df.to_parquet(metadata_path, compression='snappy', index=False)

        # Save latents to safetensors
        latents_path = self.output_dir / f"{shard_name}.safetensors"
        save_file(shard_latents, str(latents_path))

        print(f"  [SHARD] Saved {shard_name}: {len(shard_metadata)} samples")

    def run(
        self,
        limit: int = None,
        batch_size: int = 8,
        resume: bool = False,
        checkpoint_every: int = 50,
        shard_size: int = 1000,
    ):
        """Main extraction pipeline."""
        print("=" * 80)
        print("CORRECTED LATENT EXTRACTION - VALIDATED CoTs ONLY")
        print("=" * 80)
        print()
        print("[INFO] This script extracts latents from VALIDATED CoTs")
        print("[INFO] NO text generation - forward pass only!")
        print("[INFO] Latents represent LLM's understanding of CORRECT reasoning")
        print()

        # Load model
        self.load_model()

        # Load checkpoint
        processed = self.load_checkpoint() if resume else set()

        # Load validated CoTs
        print(f"[LOAD] Loading validated CoTs...")
        cots = self.load_validated_cots(limit=limit)
        self.stats["total"] = len(cots)
        print(f"[OK] Loaded {len(cots)} validated CoTs")

        if processed:
            print(f"[RESUME] {len(processed)} already processed")
        print()

        # Filter already processed
        to_process = [c for c in cots if c["cot_id"] not in processed]
        self.stats["skipped"] = len(cots) - len(to_process)

        if not to_process:
            print("[DONE] All CoTs already processed!")
            return

        print(f"[CONFIG] Batch size: {batch_size}")
        print(f"[CONFIG] Shard size: {shard_size}")
        print(f"[CONFIG] CoTs to process: {len(to_process)}")
        print()

        # Process in batches, accumulate into shards
        shard_idx = 0
        shard_metadata = []
        shard_latents = {}
        start_time = time.time()

        pbar = tqdm(total=len(to_process), desc="Extracting latents (validated)", unit="cot")

        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i + batch_size]

            batch_start = time.time()
            results = self.extract_batch(batch)
            batch_time = time.time() - batch_start

            # Add successful results to current shard
            for result in results:
                if result["status"] == "success":
                    self.stats["successful"] += 1
                    processed.add(result["cot_id"])

                    cot_id = result["cot_id"]
                    latents_tensor = torch.from_numpy(result["latents"]).to(torch.float32)

                    # Add to shard
                    shard_metadata.append({
                        "cot_id": cot_id,
                        "problem_id": result["problem_id"],
                        "domain": result["domain"],
                        "subdomain": result["subdomain"],
                        "num_steps": result["num_steps"],
                        "hidden_size": result["hidden_size"],
                        "is_valid": result["is_valid"],
                        "teacher_model": result["teacher_model"],
                    })
                    shard_latents[cot_id] = latents_tensor

                else:
                    self.stats["failed"] += 1

            # Save shard if we've accumulated enough samples
            if len(shard_metadata) >= shard_size:
                self.save_shard(shard_metadata, shard_latents, shard_idx)
                shard_metadata = []
                shard_latents = {}
                shard_idx += 1

            # Checkpoint periodically
            if (i // batch_size) % checkpoint_every == 0:
                self.save_checkpoint(processed)

            # Update progress
            pbar.update(len(batch))

            elapsed_total = time.time() - start_time
            avg_time = batch_time / len(batch)
            remaining = len(to_process) - (i + len(batch))
            eta_sec = remaining * avg_time

            pbar.set_postfix({
                "success": self.stats["successful"],
                "failed": self.stats["failed"],
                "shards": shard_idx,
                "time/cot": f"{avg_time:.2f}s",
                "ETA": f"{eta_sec/3600:.1f}h",
            })

        pbar.close()

        # Save final shard
        if shard_metadata:
            self.save_shard(shard_metadata, shard_latents, shard_idx)
            shard_idx += 1

        # Final checkpoint
        self.save_checkpoint(processed)

        # Summary
        total_time = time.time() - start_time
        print()
        print("=" * 80)
        print("SUMMARY - VALIDATED LATENT EXTRACTION")
        print("=" * 80)
        print(f"Total CoTs           : {self.stats['total']}")
        print(f"Successful           : {self.stats['successful']}")
        print(f"Failed               : {self.stats['failed']}")
        print(f"Skipped (cached)     : {self.stats['skipped']}")
        print(f"Output shards        : {shard_idx}")
        if self.stats["avg_steps"]:
            print(f"Avg steps/CoT        : {np.mean(self.stats['avg_steps']):.1f}")
        if self.stats["successful"] > 0:
            avg_time = total_time / self.stats["successful"]
            print(f"Avg time/CoT         : {avg_time:.2f}s")
            print(f"Total time           : {total_time/3600:.2f}h")
        print()
        print(f"[OUTPUT] Latents saved to: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract latents from VALIDATED CoTs (no generation!)"
    )
    parser.add_argument("--limit", type=int, help="Limit number of CoTs (for testing)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Checkpoint every N batches")
    parser.add_argument("--shard-size", type=int, default=1000, help="Samples per shard")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--layer", type=int, default=-2, help="Layer to extract")
    parser.add_argument("--output-dir", default="data/latents/qwen3-8b/validated", help="Output directory")

    args = parser.parse_args()

    extractor = ValidatedLatentExtractor(
        model_name=args.model,
        layer_to_extract=args.layer,
        output_dir=args.output_dir,
        checkpoint_file=f"{args.output_dir}/checkpoint.json",
    )

    extractor.run(
        limit=args.limit,
        batch_size=args.batch_size,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
