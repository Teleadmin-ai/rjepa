"""
LLM Adapter for R-JEPA - Generic interface for any HuggingFace LLM.

Handles:
- Model loading (with optional quantization)
- CoT generation with structured steps
- Latent extraction from hidden states
"""
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    Generic adapter for any HuggingFace LLM.

    Core functionality:
    1. Load model (with optional AWQ/GPTQ quantization)
    2. Generate structured Chain-of-Thought (CoT)
    3. Extract latents from specified layer

    Example:
        >>> adapter = LLMAdapter("Qwen/Qwen3-8B-Instruct", quantization="awq-4bit")
        >>> result = adapter.generate_with_cot("Solve: 2x + 5 = 13")
        >>> latents = adapter.extract_latents(result["tokens"], result["step_boundaries"])
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        quantization: Optional[str] = "awq-4bit",  # "awq-4bit", "gptq-4bit", None
        layer_to_extract: int = -2,  # -2 = avant-dernière couche (sweet spot)
        trust_remote_code: bool = True,
    ):
        """
        Initialize LLM Adapter.

        Args:
            model_name: HuggingFace model ID
            device: "cuda" or "cpu" (auto-detected if CUDA not available)
            dtype: "bfloat16", "float16", "float32"
            quantization: Quantization type ("awq-4bit", "gptq-4bit", None)
            layer_to_extract: Which layer to extract latents from (-2 recommended)
            trust_remote_code: Trust remote code for custom models
        """
        self.model_name = model_name
        self.device = device  # Will be overridden by device_map="auto"
        self.dtype = dtype
        self.quantization = quantization
        self.layer_to_extract = layer_to_extract

        logger.info(f"Loading LLM: {model_name}")
        logger.info(f"  Device requested: {device} (will use device_map='auto')")
        logger.info(f"  Dtype: {dtype}")
        logger.info(f"  Quantization: {quantization}")
        logger.info(f"  Layer to extract: {layer_to_extract}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional quantization
        self.model = self._load_model()
        self.model.eval()  # Always in eval mode for inference

        # Store model config
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        # Get actual device from model (device_map="auto" places it automatically)
        self.device = next(self.model.parameters()).device

        logger.info(f"Model loaded successfully!")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Actual device: {self.device}")

    def _load_model(self) -> nn.Module:
        """Load model with optional quantization."""
        if self.quantization == "awq-4bit":
            try:
                from awq import AutoAWQForCausalLM
                logger.info("Loading with AWQ 4-bit quantization...")
                model = AutoAWQForCausalLM.from_quantized(
                    self.model_name,
                    fuse_layers=True,
                    device_map="auto"
                )
                return model
            except ImportError:
                logger.warning("AWQ not installed, falling back to standard loading")

        elif self.quantization == "gptq-4bit":
            try:
                from auto_gptq import AutoGPTQForCausalLM
                logger.info("Loading with GPTQ 4-bit quantization...")
                model = AutoGPTQForCausalLM.from_quantized(
                    self.model_name,
                    device="cuda:0",
                    use_safetensors=True,
                )
                return model
            except ImportError:
                logger.warning("GPTQ not installed, falling back to standard loading")

        # Standard loading (no quantization)
        logger.info("Loading model without quantization...")
        torch_dtype = getattr(torch, self.dtype)

        # Try to detect CUDA availability for explicit device mapping
        import subprocess
        try:
            # Check nvidia-smi to verify GPU is actually available
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            has_gpu = result.returncode == 0
        except:
            has_gpu = False

        if has_gpu:
            logger.info("GPU detected via nvidia-smi, forcing CUDA placement")
            # Force placement on CUDA:0
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map={"": "cuda:0"},  # Explicit CUDA placement
                trust_remote_code=True,
            )
        else:
            logger.warning("No GPU detected, using CPU")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="cpu",
                trust_remote_code=True,
            )
        return model

    def generate_with_cot(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        step_token: str = "Step",
        num_samples: int = 1,
        force_structure: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate one or more Chain-of-Thought (CoT) with structured steps.

        IMPORTANT: We force the model to structure with "Step 1:", "Step 2:", etc.
        via a system prompt + sampling.

        Args:
            prompt: User question/problem
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            step_token: Token used for steps (default "Step")
            num_samples: Number of samples to generate
            force_structure: Force structured output with system prompt

        Returns:
            List of dicts, one per sample:
            {
              "full_text": str,               # Complete generated text
              "steps": List[str],             # ["Step 1: ...", "Step 2: ...", ...]
              "tokens": torch.LongTensor,     # [1, T] token IDs
              "step_boundaries": List[Tuple[int, int]]  # [(start, end) token indices]
            }
        """
        # System prompt to force structured output
        if force_structure:
            system_prompt = (
                "You are a reasoning assistant. When solving problems, "
                "structure your response as explicit steps:\n"
                "Step 1: [first reasoning step]\n"
                "Step 2: [second reasoning step]\n"
                "...\n"
                "Step N: [final answer]"
            )
            full_prompt = f"{system_prompt}\n\nProblem: {prompt}\n\nSolution:"
        else:
            full_prompt = f"Problem: {prompt}\n\nSolution:"

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Generate (multiple samples if requested)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results = []
        for i in range(num_samples):
            tokens = outputs[i:i+1]  # Keep batch dim
            full_text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)

            # Segment into steps
            steps, step_boundaries = self._segment_into_steps(
                full_text,
                tokens[0],
                step_token=step_token
            )

            results.append({
                "full_text": full_text,
                "steps": steps,
                "tokens": tokens,
                "step_boundaries": step_boundaries,
            })

        return results

    def _segment_into_steps(
        self,
        text: str,
        tokens: torch.LongTensor,
        step_token: str = "Step"
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Segment generated text into steps and find boundaries in tokens.

        Args:
            text: Full generated text
            tokens: Token IDs [T]
            step_token: Token marking steps (default "Step")

        Returns:
            steps: List of step strings ["Step 1: ...", ...]
            step_boundaries: List of (start_idx, end_idx) tuples in token space
        """
        import re

        # Regex to find "Step X:"
        pattern = rf"{step_token}\s+\d+:"
        matches = list(re.finditer(pattern, text))

        if not matches:
            # Fallback: entire text is one step
            return [text], [(0, len(tokens))]

        steps = []
        step_boundaries = []

        for i, match in enumerate(matches):
            start_char = match.start()
            end_char = matches[i+1].start() if i+1 < len(matches) else len(text)

            step_text = text[start_char:end_char].strip()
            steps.append(step_text)

            # Find corresponding token indices
            # (approximation: encode substring and count tokens)
            prefix_tokens = len(self.tokenizer.encode(text[:start_char]))
            current_tokens = len(self.tokenizer.encode(text[:end_char]))

            step_boundaries.append((prefix_tokens, current_tokens))

        return steps, step_boundaries

    def extract_latents(
        self,
        tokens: torch.LongTensor,
        step_boundaries: List[Tuple[int, int]],
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract latents averaged per step for a specific layer.

        CORE OF WORLD MODEL:
        - Retrieve hidden states from layer `layer_idx`
        - Average tokens within each step → one vector per step

        Args:
            tokens: [1, T] tensor of token IDs
            step_boundaries: List of (start, end) indices for each step
            layer_idx: Layer to extract (default: self.layer_to_extract = -2)

        Returns:
            H: [num_steps, hidden_size] tensor of latents
        """
        if layer_idx is None:
            layer_idx = self.layer_to_extract

        # Forward pass with hidden states extraction
        with torch.no_grad():
            outputs = self.model(
                tokens,
                output_hidden_states=True,
                return_dict=True
            )

        # outputs.hidden_states = tuple of (num_layers+1) tensors [1, T, hidden]
        # Layer 0 = embeddings, Layer 1..N = hidden layers
        # layer_idx=-2 → avant-dernière couche
        hidden_states = outputs.hidden_states[layer_idx]  # [1, T, hidden]

        # Average per step
        latents = []
        for start, end in step_boundaries:
            # Mean of tokens in this step over seq dim
            step_latent = hidden_states[0, start:end, :].mean(dim=0)  # [hidden]
            latents.append(step_latent)

        H = torch.stack(latents, dim=0)  # [num_steps, hidden]

        return H

    def __repr__(self) -> str:
        return (
            f"LLMAdapter(\n"
            f"  model={self.model_name},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  quantization={self.quantization},\n"
            f"  layer_to_extract={self.layer_to_extract}\n"
            f")"
        )
