"""
Nudge mode for R-JEPA.

Correction latente douce : à chaque step de raisonnement, on corrige
légèrement le latent pour le ramener vers le manifold des bons raisonnements.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np

from rjepa.llm.adapter import LLMAdapter
from rjepa.jepa.client import RJEPAClient

logger = logging.getLogger(__name__)


def nudge_reasoning_stepwise(
    prompt: str,
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    max_steps: int = 10,
    lambda_nudge: float = 0.2,
    temperature: float = 0.7,
    domain_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Génère un raisonnement step-by-step avec correction JEPA.

    À chaque step:
    1. Génère le step suivant avec le LLM
    2. Extrait le latent H_t
    3. Prédit H_t_corrected via JEPA (en masquant H_t)
    4. Corrige: H_t_final = (1-λ) * H_t + λ * H_t_corrected
    5. Continue avec H_t_final

    Note: Cette version est simplifiée car on ne peut pas facilement
    réinjecter H_t_final dans le LLM pour conditionner la suite.
    On utilise plutôt le score JEPA comme signal de qualité.

    Args:
        prompt: Question/problème
        llm: LLM adapter
        rjepa_client: Client R-JEPA
        max_steps: Nombre max de steps
        lambda_nudge: Coefficient de correction (0=aucune, 1=full JEPA)
        temperature: Temperature génération
        domain_id: Optional domain ID

    Returns:
        {
            "full_text": str,
            "steps": List[Dict],  # Chaque step avec son score JEPA
            "corrected": bool,  # Si des corrections ont été appliquées
        }
    """
    logger.info(f"Generating reasoning with nudge (lambda={lambda_nudge})...")

    # Génère un CoT complet first
    cot_result = llm.generate_with_cot(
        prompt=prompt,
        max_new_tokens=512,
        temperature=temperature,
        step_token="Step",
        num_samples=1,
    )[0]

    # Extract latents
    latents = llm.extract_latents(
        tokens=cot_result["tokens"],
        layer_idx=llm.layer_to_extract,
        step_boundaries=cot_result["step_boundaries"],
    )  # [num_steps, hidden_dim]

    num_steps = latents.shape[0]

    # Pour chaque step, compute JEPA score et correction potentielle
    steps_info = []
    corrected_any = False

    for step_idx in range(num_steps):
        # Masque ce step et prédit avec JEPA
        try:
            pred_latent = rjepa_client.predict_masked(
                latents=latents,
                mask_indices=[step_idx],
                domain_id=domain_id,
            )  # [1, hidden_dim]

            # Compute distance entre latent original et prédiction JEPA
            original_latent = latents[step_idx : step_idx + 1]
            distance = torch.norm(original_latent - pred_latent).item()

            # Si distance élevée, on pourrait appliquer correction
            # H_corrected = (1 - lambda) * H_original + lambda * H_pred
            corrected_latent = (
                (1 - lambda_nudge) * original_latent + lambda_nudge * pred_latent
            )

            # Compute JEPA score pour ce step
            jepa_result = rjepa_client.score(
                latents=latents,
                mask_ratio=0.3,  # Masque un peu pour scorer
                domain_id=domain_id,
            )
            jepa_loss = jepa_result["jepa_loss"]

            # Decide if correction needed (heuristic: high distance)
            correction_needed = distance > 1.0  # Threshold empirique

            if correction_needed:
                corrected_any = True

            steps_info.append({
                "step_idx": step_idx,
                "text": cot_result["steps"][step_idx] if step_idx < len(cot_result["steps"]) else "",
                "distance": distance,
                "corrected": correction_needed,
                "jepa_loss": jepa_loss,
            })

            logger.debug(
                f"Step {step_idx}: distance={distance:.4f}, "
                f"corrected={correction_needed}, jepa_loss={jepa_loss:.4f}"
            )

        except Exception as e:
            logger.warning(f"Failed to nudge step {step_idx}: {e}")
            steps_info.append({
                "step_idx": step_idx,
                "text": cot_result["steps"][step_idx] if step_idx < len(cot_result["steps"]) else "",
                "distance": None,
                "corrected": False,
                "jepa_loss": None,
            })

    logger.info(
        f"Nudging complete: {sum(s['corrected'] for s in steps_info)}/{num_steps} "
        f"steps corrected"
    )

    return {
        "full_text": cot_result["full_text"],
        "steps": steps_info,
        "corrected": corrected_any,
        "num_steps": num_steps,
    }


def nudge_with_regeneration(
    prompt: str,
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    max_attempts: int = 3,
    jepa_threshold: float = 0.5,
    temperature: float = 0.7,
    domain_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Génère un raisonnement et régénère les steps avec score JEPA élevé.

    Stratégie:
    1. Génère un CoT complet
    2. Score chaque step avec JEPA
    3. Identifie les steps "suspects" (score élevé)
    4. Régénère ces steps avec contexte JEPA
    5. Répète jusqu'à max_attempts ou tous les steps OK

    Args:
        prompt: Question/problème
        llm: LLM adapter
        rjepa_client: Client R-JEPA
        max_attempts: Nombre max d'itérations
        jepa_threshold: Seuil de JEPA-loss pour triggering correction
        temperature: Temperature génération
        domain_id: Optional domain ID

    Returns:
        {
            "full_text": str,
            "steps": List[str],
            "iterations": int,
            "final_jepa_loss": float,
        }
    """
    logger.info(
        f"Generating reasoning with regeneration (threshold={jepa_threshold}, "
        f"max_attempts={max_attempts})..."
    )

    best_cot = None
    best_jepa_loss = float("inf")

    for attempt in range(max_attempts):
        logger.debug(f"Attempt {attempt + 1}/{max_attempts}...")

        # Generate CoT
        cot_result = llm.generate_with_cot(
            prompt=prompt,
            max_new_tokens=512,
            temperature=temperature,
            step_token="Step",
            num_samples=1,
        )[0]

        # Extract latents
        latents = llm.extract_latents(
            tokens=cot_result["tokens"],
            layer_idx=llm.layer_to_extract,
            step_boundaries=cot_result["step_boundaries"],
        )

        # Score with JEPA
        jepa_result = rjepa_client.score(
            latents=latents,
            mask_ratio=0.5,
            domain_id=domain_id,
        )
        jepa_loss = jepa_result["jepa_loss"]

        logger.debug(f"Attempt {attempt + 1}: jepa_loss={jepa_loss:.4f}")

        # Track best
        if jepa_loss < best_jepa_loss:
            best_jepa_loss = jepa_loss
            best_cot = cot_result

        # Si suffisamment bon, on arrête
        if jepa_loss < jepa_threshold:
            logger.info(f"Good reasoning found at attempt {attempt + 1}")
            break

    logger.info(
        f"Regeneration complete: best_jepa_loss={best_jepa_loss:.4f} "
        f"after {attempt + 1} attempts"
    )

    return {
        "full_text": best_cot["full_text"],
        "steps": best_cot["steps"],
        "iterations": attempt + 1,
        "final_jepa_loss": best_jepa_loss,
    }


def nudge_with_beam_search(
    prompt: str,
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    beam_width: int = 3,
    max_steps: int = 10,
    temperature: float = 0.7,
    domain_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Beam search guidé par JEPA scores.

    À chaque step:
    1. Génère beam_width continuations possibles
    2. Score chacune avec JEPA
    3. Garde les beam_width meilleures
    4. Répète jusqu'à terminaison

    Note: Implémentation simplifiée pour MVP.

    Args:
        prompt: Question/problème
        llm: LLM adapter
        rjepa_client: Client R-JEPA
        beam_width: Largeur du beam
        max_steps: Steps max
        temperature: Temperature génération
        domain_id: Optional domain ID

    Returns:
        {
            "best_path": Dict,
            "all_beams": List[Dict],
        }
    """
    logger.info(f"Beam search with JEPA guidance (width={beam_width})...")

    # Pour simplifier le MVP, on génère beam_width candidates complets
    # et on sélectionne le meilleur (pas de vrai beam search step-by-step)

    candidates = []

    for i in range(beam_width):
        cot_result = llm.generate_with_cot(
            prompt=prompt,
            max_new_tokens=512,
            temperature=temperature,
            step_token="Step",
            num_samples=1,
        )[0]

        latents = llm.extract_latents(
            tokens=cot_result["tokens"],
            layer_idx=llm.layer_to_extract,
            step_boundaries=cot_result["step_boundaries"],
        )

        jepa_result = rjepa_client.score(
            latents=latents,
            mask_ratio=0.5,
            domain_id=domain_id,
        )

        candidates.append({
            "cot": cot_result,
            "jepa_loss": jepa_result["jepa_loss"],
        })

    # Sort by JEPA loss
    candidates_sorted = sorted(candidates, key=lambda x: x["jepa_loss"])

    best_path = candidates_sorted[0]

    logger.info(f"Best beam: jepa_loss={best_path['jepa_loss']:.4f}")

    return {
        "best_path": best_path["cot"],
        "best_jepa_loss": best_path["jepa_loss"],
        "all_beams": candidates_sorted,
    }
