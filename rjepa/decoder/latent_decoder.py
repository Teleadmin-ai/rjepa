"""
Latent Decoder: Transforms latent representations to text.

Architecture:
- Input: latent vector [hidden_dim] from R-JEPA layer
- Small Transformer decoder (depth=4, autoregressive)
- Output: tokens via language modeling head

Training:
- Frozen R-JEPA extracts latents from validated CoTs
- Decoder learns to generate text from latents
- Cross-entropy loss on token predictions

Usage:
- Decode predicted latents (plan mode)
- Condition generation on latent (nudge mode)
"""
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LatentDecoderConfig:
    """Configuration for LatentDecoder."""

    # Architecture
    latent_dim: int = 4096  # Input latent dimension (Qwen3-8B layer -2)
    vocab_size: int = 151936  # Qwen3 vocabulary size
    decoder_dim: int = 1024  # Decoder hidden dimension
    depth: int = 4  # Number of Transformer decoder layers
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 256  # Max tokens per step

    # Training
    tie_embeddings: bool = True  # Tie input/output embeddings


class LatentDecoder(nn.Module):
    """
    Decoder that transforms latents to text.

    Inspired by V-JEPA diffusion decoder: small model trained separately,
    R-JEPA stays frozen (world model as ground truth).

    Architecture:
    1. Latent projection: [latent_dim] -> [decoder_dim]
    2. Causal Transformer decoder
    3. Language modeling head: [decoder_dim] -> [vocab_size]
    """

    def __init__(self, config: LatentDecoderConfig):
        """
        Initialize LatentDecoder.

        Args:
            config: LatentDecoderConfig
        """
        super().__init__()

        self.config = config

        # Latent projection
        self.latent_proj = nn.Linear(config.latent_dim, config.decoder_dim)

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.decoder_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.decoder_dim) * 0.02
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.decoder_dim * config.mlp_ratio),
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (modern Transformers)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.depth,
        )

        # Language modeling head
        self.lm_head = nn.Linear(config.decoder_dim, config.vocab_size, bias=False)

        # Tie embeddings (optional, reduces parameters)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        # Layer norm before lm_head
        self.ln_f = nn.LayerNorm(config.decoder_dim)

        logger.info(
            f"LatentDecoder initialized: "
            f"latent_dim={config.latent_dim}, "
            f"decoder_dim={config.decoder_dim}, "
            f"depth={config.depth}, "
            f"vocab_size={config.vocab_size}"
        )

    def forward(
        self,
        latent: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            latent: [B, latent_dim] latent vector (from R-JEPA)
            input_ids: [B, S] token IDs (target text to generate)
            labels: [B, S] labels for language modeling (shifted input_ids)

        Returns:
            {
                "logits": [B, S, vocab_size],
                "loss": scalar (if labels provided)
            }
        """
        batch_size, seq_len = input_ids.shape

        # Project latent to decoder dimension
        latent_emb = self.latent_proj(latent)  # [B, decoder_dim]
        latent_emb = latent_emb.unsqueeze(1)  # [B, 1, decoder_dim]

        # Token embeddings + positional encoding
        token_emb = self.token_embed(input_ids)  # [B, S, decoder_dim]
        token_emb = token_emb + self.pos_encoding[:, :seq_len, :]

        # Causal mask (autoregressive)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )

        # Decoder forward
        # Memory = latent embedding (conditioning)
        # Tgt = token embeddings (autoregressive generation)
        hidden = self.decoder(
            tgt=token_emb,
            memory=latent_emb,
            tgt_mask=causal_mask,
        )  # [B, S, decoder_dim]

        # Layer norm + LM head
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)  # [B, S, vocab_size]

        outputs = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,  # Ignore padding tokens
            )
            outputs["loss"] = loss

        return outputs

    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text from latent (autoregressive).

        Args:
            latent: [1, latent_dim] latent vector
            tokenizer: Tokenizer (for BOS/EOS)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            eos_token_id: EOS token ID (stop generation)

        Returns:
            {
                "generated_ids": [1, T] tensor,
                "text": str
            }
        """
        device = latent.device

        # Start with BOS token
        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        input_ids = torch.tensor([[bos_token_id]], device=device)

        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id

        # Autoregressive generation
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(latent, input_ids)
            logits = outputs["logits"]  # [1, S, vocab_size]

            # Get logits for next token
            next_token_logits = logits[:, -1, :] / temperature  # [1, vocab_size]

            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True, dim=-1
            )
            cumulative_probs = torch.cumsum(
                nn.functional.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability > top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[:, indices_to_remove] = float('-inf')

            # Sample next token
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS token
            if next_token.item() == eos_token_id:
                break

        # Decode to text
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return {
            "generated_ids": input_ids,
            "text": text,
        }

    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_latent_decoder(
    latent_dim: int = 4096,
    vocab_size: int = 151936,
    decoder_dim: int = 1024,
    depth: int = 4,
    **kwargs
) -> LatentDecoder:
    """
    Factory function to create LatentDecoder.

    Args:
        latent_dim: Input latent dimension
        vocab_size: Vocabulary size
        decoder_dim: Decoder hidden dimension
        depth: Number of decoder layers
        **kwargs: Additional config parameters

    Returns:
        LatentDecoder instance
    """
    config = LatentDecoderConfig(
        latent_dim=latent_dim,
        vocab_size=vocab_size,
        decoder_dim=decoder_dim,
        depth=depth,
        **kwargs
    )

    model = LatentDecoder(config)

    num_params = model.get_num_params()
    logger.info(f"Created LatentDecoder with {num_params:,} parameters")

    return model
