"""
Training pipeline for Latent Decoder.

Trains the decoder to generate text from R-JEPA latents.

Philosophy:
1. R-JEPA is FROZEN (world model as ground truth)
2. Decoder learns latent→text mapping
3. Only uses validated CoTs (is_valid=True)
"""
import logging
from pathlib import Path
from typing import Optional
import yaml
import torch
from transformers import AutoTokenizer

from rjepa.decoder import LatentDecoder, LatentDecoderConfig, LatentDecoderTrainer
from rjepa.decoder.dataset import create_decoder_dataloaders

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def train_decoder_from_config(config_path: Path):
    """
    Train latent decoder from config file.

    Args:
        config_path: Path to config YAML
    """
    config = load_config(config_path)

    logger.info("="*80)
    logger.info("LATENT DECODER TRAINING")
    logger.info("="*80)

    # Load tokenizer
    tokenizer_name = config["tokenizer"]["name"]
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_decoder_dataloaders(
        latents_dir=Path(config["data"]["latents_dir"]),
        cots_train_path=Path(config["data"]["cots_train_path"]),
        cots_val_path=Path(config["data"]["cots_val_path"])
        if config["data"].get("cots_val_path")
        else None,
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        max_seq_len=config["model"]["max_seq_len"],
        llm_tag=config["data"]["llm_tag"],
        num_workers=config["data"]["num_workers"],
    )

    # Create model
    logger.info("Creating LatentDecoder model...")
    model_config = LatentDecoderConfig(
        latent_dim=config["model"]["latent_dim"],
        vocab_size=tokenizer.vocab_size,
        decoder_dim=config["model"]["decoder_dim"],
        depth=config["model"]["depth"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"],
        tie_embeddings=config["model"]["tie_embeddings"],
    )

    model = LatentDecoder(model_config)

    logger.info(
        f"Model created: {model.get_num_params():,} parameters "
        f"({model.get_num_params() / 1e6:.1f}M)"
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = LatentDecoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config["training"]["lr"],
        device=config["training"]["device"],
        use_amp=config["training"]["use_amp"],
        grad_clip=config["training"]["grad_clip"],
        log_interval=config["training"]["log_interval"],
        wandb_enabled=config["logging"]["wandb_enabled"],
    )

    # Train
    logger.info(f"Starting training for {config['training']['num_epochs']} epochs...")

    history = trainer.train(
        num_epochs=config["training"]["num_epochs"],
        save_dir=Path(config["training"]["save_dir"]),
        save_interval=config["training"]["save_interval"],
    )

    logger.info("Training completed!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final train perplexity: {history['train_perplexity'][-1]:.2f}")

    if history["val_loss"]:
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"Final val perplexity: {history['val_perplexity'][-1]:.2f}")

    # Save final model
    final_path = Path(config["training"]["save_dir"]) / "final.pth"
    trainer.save_checkpoint(final_path)
    logger.info(f"Final checkpoint saved: {final_path}")

    return history


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Latent Decoder")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/decoder/train.yaml"),
        help="Path to config file",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Train
    train_decoder_from_config(args.config)


if __name__ == "__main__":
    main()
