#!/usr/bin/env python3
"""
OPTIMIZED Latent Extraction Pipeline (10-20x faster than HTTP version).

OPTIMIZATIONS:
1. Load model ONCE (no HTTP overhead)
2. Batch inference (8-16 problems in parallel on GPU)
3. Compressed storage (gzip numpy arrays)
4. Max GPU utilization (~80-90%)
5. Checkpoint/resume support

USAGE:
    python scripts/extract_latents_optimized.py --batch-size 8 --limit 1000
    python scripts/extract_latents_optimized.py --resume  # Continue from checkpoint
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
import gzip
import pickle
import pandas as pd
from safetensors.torch import save_file


class OptimizedLatentExtractor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        layer_to_extract: int = -2,
        output_dir: str = "data/latents/qwen3-8b/academic",
        checkpoint_file: str = "data/latents/checkpoint_optimized.json",
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
            "total_latents_mb": 0.0,
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
        # Set padding side to left for decoder-only models (batch generation)
        self.tokenizer.padding_side = 'left'

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, self.dtype),
            device_map="auto",  # Simple auto (worked in tests with 5-12s loading!)
            trust_remote_code=True
        )
        self.model.eval()

        print(f"[OK] Model loaded:")
        print(f"  Hidden size: {self.model.config.hidden_size}")
        print(f"  Num layers: {self.model.config.num_hidden_layers}")
        print(f"  Device: {self.model.device}")
        print()

    def load_checkpoint(self) -> set:
        """Load already processed problem IDs."""
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

    def load_problems(self, limit: int = None) -> List[Dict[str, Any]]:
        """Load all academic problems from JSON files."""
        problems = []

        # GSM8K
        gsm8k_train = Path("data/datasets/academic/math/gsm8k/train_problems.json")
        if gsm8k_train.exists():
            with open(gsm8k_train) as f:
                problems.extend(json.load(f))

        # MATH (Competition Math)
        math_train = Path("data/datasets/academic/math/competition_math/train_problems.json")
        if math_train.exists():
            with open(math_train) as f:
                problems.extend(json.load(f))

        # MATH test set aussi
        math_test = Path("data/datasets/academic/math/competition_math/test_problems.json")
        if math_test.exists():
            with open(math_test) as f:
                problems.extend(json.load(f))

        # GSM8K test set aussi
        gsm8k_test = Path("data/datasets/academic/math/gsm8k/test_problems.json")
        if gsm8k_test.exists():
            with open(gsm8k_test) as f:
                problems.extend(json.load(f))

        # HumanEval
        humaneval_test = Path("data/datasets/academic/code/humaneval/test_problems.json")
        if humaneval_test.exists():
            with open(humaneval_test) as f:
                problems.extend(json.load(f))

        if limit:
            problems = problems[:limit]

        return problems

    def segment_into_steps(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Segment text into steps and find token boundaries.

        Returns:
            steps: List of step strings
            boundaries: List of (start_token_idx, end_token_idx) tuples
        """
        import re

        # Find "Step X:" patterns
        pattern = r"Step\s+\d+:"
        matches = list(re.finditer(pattern, text))

        if not matches:
            # Fallback: entire text is one step
            tokens = self.tokenizer.encode(text)
            return [text], [(0, len(tokens))]

        steps = []
        boundaries = []

        for i, match in enumerate(matches):
            start_char = match.start()
            end_char = matches[i+1].start() if i+1 < len(matches) else len(text)

            step_text = text[start_char:end_char].strip()
            steps.append(step_text)

            # Find token boundaries (approximate)
            prefix_tokens = len(self.tokenizer.encode(text[:start_char], add_special_tokens=False))
            step_tokens = len(self.tokenizer.encode(text[start_char:end_char], add_special_tokens=False))
            boundaries.append((prefix_tokens, prefix_tokens + step_tokens))

        return steps, boundaries

    def generate_and_extract_batch(
        self,
        problems: List[Dict[str, Any]],
        max_new_tokens: int = 300,
        temperature: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Generate CoT and extract latents for a batch of problems.

        This is the CORE optimization: batch processing on GPU!
        """
        results = []

        # Prepare prompts
        system_prompt = (
            "You are a reasoning assistant. When solving problems, "
            "structure your response as explicit steps:\n"
            "Step 1: [first reasoning step]\n"
            "Step 2: [second reasoning step]\n"
            "...\n"
            "Step N: [final answer]"
        )

        prompts = [
            f"{system_prompt}\n\nProblem: {prob['statement']}\n\nSolution:"
            for prob in problems
        ]

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)

        # Generate batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Process each output
        for i, (problem, output_ids) in enumerate(zip(problems, outputs)):
            try:
                # Decode
                full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                # Segment into steps
                steps, step_boundaries = self.segment_into_steps(full_text)

                # Extract latents
                # Re-encode for latent extraction (need hidden states)
                inputs_single = self.tokenizer(
                    full_text,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    model_outputs = self.model(
                        **inputs_single,
                        output_hidden_states=True,
                        return_dict=True
                    )

                # Get hidden states at target layer
                hidden_states = model_outputs.hidden_states[self.layer_to_extract]  # [1, T, D]

                # Average over tokens for each step
                latents = []
                for start, end in step_boundaries:
                    if end <= hidden_states.shape[1]:
                        step_latent = hidden_states[0, start:end, :].mean(dim=0)  # [D]
                        # Convert to float16 before numpy (bfloat16 not supported)
                        latents.append(step_latent.to(torch.float16).cpu().numpy())

                if not latents:
                    # Fallback: use entire sequence
                    latents = [hidden_states[0].mean(dim=0).to(torch.float16).cpu().numpy()]

                latents_array = np.stack(latents, axis=0)  # [num_steps, D] float16

                # Compress latents (gzip)
                latents_bytes = gzip.compress(latents_array.tobytes())
                latents_size_mb = len(latents_bytes) / (1024 * 1024)

                results.append({
                    "status": "success",
                    "problem_id": problem["problem_id"],
                    "domain": problem["domain"],
                    "subdomain": problem["subdomain"],
                    "num_steps": len(latents),
                    "hidden_size": latents_array.shape[1],
                    "latent_shape": list(latents_array.shape),
                    "cot_text": full_text,
                    "steps": steps,
                    "step_boundaries": step_boundaries,
                    "latents_compressed": latents_bytes,
                    "latents_size_mb": latents_size_mb,
                })

                self.stats["avg_steps"].append(len(latents))
                self.stats["total_latents_mb"] += latents_size_mb

            except Exception as e:
                results.append({
                    "status": "failed",
                    "problem_id": problem["problem_id"],
                    "error": str(e),
                })

        return results

    def save_shard(self, shard_metadata: List[Dict], shard_latents: Dict[str, torch.Tensor], shard_idx: int):
        """
        Save shard to parquet + safetensors format (compatible with LatentDataset).

        Args:
            shard_metadata: List of metadata dicts (cot_id, problem_id, domain, etc.)
            shard_latents: Dict mapping cot_id -> latent tensor [num_steps, hidden_size]
            shard_idx: Shard index for naming
        """
        shard_name = f"shard-{shard_idx:04d}"

        # Save metadata to parquet
        metadata_path = self.output_dir / f"{shard_name}.parquet"
        df = pd.DataFrame(shard_metadata)
        df.to_parquet(metadata_path, compression='snappy', index=False)

        # Save latents to safetensors
        latents_path = self.output_dir / f"{shard_name}.safetensors"
        save_file(shard_latents, latents_path)

        print(f"  [SHARD] Saved {shard_name}: {len(shard_metadata)} samples")

    def save_batch_legacy(self, results: List[Dict[str, Any]], batch_id: int):
        """[DEPRECATED] Old batch format - kept for compatibility."""
        output_file = self.output_dir / f"batch_{batch_id:04d}.pkl.gz"
        with gzip.open(output_file, "wb") as f:
            pickle.dump(results, f)
        successful = sum(1 for r in results if r["status"] == "success")
        print(f"[SAVE] Saved batch {batch_id}: {successful}/{len(results)} successful -> {output_file}")

    def run(
        self,
        limit: int = None,
        batch_size: int = 8,
        resume: bool = False,
        checkpoint_every: int = 10,
        shard_size: int = 1000,
    ):
        """Main extraction pipeline."""
        print("=" * 80)
        print("PHASE 21: OPTIMIZED LATENT EXTRACTION (SHARD FORMAT)")
        print("=" * 80)
        print()

        # Load model
        self.load_model()

        # Load checkpoint
        processed = self.load_checkpoint() if resume else set()

        # Load problems
        print(f"[LOAD] Loading academic problems...")
        problems = self.load_problems(limit=limit)
        self.stats["total"] = len(problems)
        print(f"[OK] Loaded {len(problems)} problems")
        if processed:
            print(f"[RESUME] {len(processed)} already processed, {len(problems) - len(processed)} remaining")
        print()

        # Filter already processed
        to_process = [p for p in problems if p["problem_id"] not in processed]
        self.stats["skipped"] = len(problems) - len(to_process)

        if not to_process:
            print("[DONE] All problems already processed!")
            return

        print(f"[CONFIG] Batch size: {batch_size} (GPU inference)")
        print(f"[CONFIG] Shard size: {shard_size} (output format)")
        print(f"[CONFIG] Problems to process: {len(to_process)}")
        print(f"[CONFIG] Estimated shards: {len(to_process) // shard_size + 1}")
        print()

        # Process in batches, accumulate into shards
        batch_id = 0
        shard_idx = 0
        shard_metadata = []
        shard_latents = {}
        start_time = time.time()

        pbar = tqdm(total=len(to_process), desc="Extracting latents", unit="problem")

        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i + batch_size]

            batch_start = time.time()
            results = self.generate_and_extract_batch(batch)
            batch_time = time.time() - batch_start

            # Add successful results to current shard accumulators
            for result in results:
                if result["status"] == "success":
                    self.stats["successful"] += 1
                    processed.add(result["problem_id"])

                    # Generate cot_id (same as problem_id for now)
                    problem_id = result["problem_id"]
                    cot_id = problem_id

                    # Decompress latents from compressed format
                    latents_decompressed = gzip.decompress(result["latents_compressed"])
                    latents_array = np.frombuffer(latents_decompressed, dtype=np.float16)

                    # Calculate actual shape
                    hidden_size = result["hidden_size"]
                    num_elements = len(latents_array)
                    actual_num_steps = num_elements // hidden_size

                    # Convert to float32 tensor for training
                    latents_tensor = torch.from_numpy(latents_array.copy()).reshape(actual_num_steps, hidden_size).to(torch.float32)

                    # Add to shard
                    shard_metadata.append({
                        "cot_id": cot_id,
                        "problem_id": problem_id,
                        "domain": result["domain"],
                        "subdomain": result["subdomain"],
                        "num_steps": actual_num_steps,
                        "hidden_size": hidden_size,
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

            # Checkpoint
            if batch_id % checkpoint_every == 0:
                self.save_checkpoint(processed)

            # Update progress
            pbar.update(len(batch))

            elapsed_total = time.time() - start_time
            avg_time_per_batch = elapsed_total / (batch_id + 1)
            avg_time_per_problem = batch_time / len(batch)
            remaining_batches = (len(to_process) - (i + len(batch))) / batch_size
            eta_sec = remaining_batches * avg_time_per_batch

            pbar.set_postfix({
                "success": self.stats["successful"],
                "failed": self.stats["failed"],
                "shards": shard_idx,
                "time/prob": f"{avg_time_per_problem:.1f}s",
                "ETA": f"{eta_sec/3600:.1f}h",
            })

            batch_id += 1

        pbar.close()

        # Save final shard if there are remaining samples
        if shard_metadata:
            self.save_shard(shard_metadata, shard_latents, shard_idx)
            shard_idx += 1

        # Final checkpoint
        self.save_checkpoint(processed)

        # Final summary
        total_time = time.time() - start_time
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total problems       : {self.stats['total']}")
        print(f"Successful           : {self.stats['successful']}")
        print(f"Failed               : {self.stats['failed']}")
        print(f"Skipped (cached)     : {self.stats['skipped']}")
        if self.stats["avg_steps"]:
            print(f"Avg steps/problem    : {np.mean(self.stats['avg_steps']):.1f}")
        if self.stats["successful"] > 0:
            avg_time = total_time / self.stats["successful"]
            print(f"Avg time/problem     : {avg_time:.2f}s")
            print(f"Total time           : {total_time/3600:.2f}h")
            print(f"Total latents size   : {self.stats['total_latents_mb']:.1f} MB (compressed)")

            # Extrapolate
            if limit and limit < 21456:
                full_time = (21456 / self.stats["successful"]) * total_time
                print(f"\n[EXTRAPOLATION] Full 21,456 problems: {full_time/3600:.1f}h")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Optimized latent extraction (batch + compression)")
    parser.add_argument("--limit", type=int, help="Limit number of problems (for testing)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for GPU inference")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint every N batches")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--layer", type=int, default=-2, help="Layer to extract (-2 = second-to-last)")

    args = parser.parse_args()

    extractor = OptimizedLatentExtractor(
        model_name=args.model,
        layer_to_extract=args.layer,
    )

    extractor.run(
        limit=args.limit,
        batch_size=args.batch_size,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
