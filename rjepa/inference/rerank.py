"""
Re-ranking mode for R-JEPA.

Génère N chaînes de raisonnement candidates et choisit la meilleure
selon le score JEPA (cohérence avec le world model).
"""
import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from rjepa.llm.adapter import LLMAdapter
from rjepa.jepa.client import RJEPAClient

logger = logging.getLogger(__name__)


def rerank_cots_with_jepa(
    prompt: str,
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    num_samples: int = 4,
    temperature: float = 0.8,
    mask_ratio: float = 0.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    domain_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Re-rank CoT candidates using JEPA scores.

    Génère plusieurs chaînes de raisonnement, calcule leur JEPA-loss,
    et choisit la meilleure selon un score composite.

    Args:
        prompt: Question/problème à résoudre
        llm: LLM adapter (student model)
        rjepa_client: Client HTTP vers service R-JEPA
        num_samples: Nombre de candidates à générer (default: 4)
        temperature: Temperature pour génération (default: 0.8)
        mask_ratio: Ratio de masking pour scoring JEPA (default: 0.5)
        alpha: Weight pour logprob dans score final (default: 1.0)
        beta: Weight pour -JEPA-loss dans score final (default: 1.0)
        gamma: Weight pour length penalty dans score final (default: 0.0)
        domain_id: Optional domain ID pour JEPA

    Returns:
        {
            "best_cot": {
                "full_text": str,
                "steps": List[str],
                "final_answer": str,
                "score": float,
                "jepa_loss": float,
                "logprob": float,
            },
            "candidates": List[Dict],  # Tous les candidates avec scores
            "num_candidates": int,
        }
    """
    logger.info(f"Generating {num_samples} CoT candidates for prompt: {prompt[:100]}...")

    # Generate N candidates
    candidates = []

    for i in range(num_samples):
        logger.debug(f"Generating candidate {i+1}/{num_samples}...")

        # Generate CoT with student LLM
        cot_result = llm.generate_with_cot(
            prompt=prompt,
            max_new_tokens=512,
            temperature=temperature,
            step_token="Step",
            num_samples=1,
        )[0]

        # Extract latents for this CoT
        latents = llm.extract_latents(
            tokens=cot_result["tokens"],
            layer_idx=llm.layer_to_extract,
            step_boundaries=cot_result["step_boundaries"],
        )

        # Compute JEPA score
        jepa_result = rjepa_client.score(
            latents=latents,
            domain_id=domain_id,
            mask_ratio=mask_ratio,
        )

        jepa_loss = jepa_result["jepa_loss"]

        # Compute logprob (approximation)
        # Note: Pour avoir les vrais logprobs, il faudrait les récupérer du modèle
        # Pour l'instant, on utilise une approximation basée sur la longueur
        num_tokens = cot_result["tokens"].shape[1]
        logprob_approx = -0.1 * num_tokens  # Approximation simple

        # Compute composite score
        # Score = alpha * logprob + beta * (-jepa_loss) + gamma * length_penalty
        length_penalty = -0.01 * num_tokens  # Pénalité pour tokens longs
        score = alpha * logprob_approx + beta * (-jepa_loss) + gamma * length_penalty

        # Extract final answer (last step)
        final_answer = cot_result["steps"][-1] if cot_result["steps"] else ""

        candidates.append({
            "full_text": cot_result["full_text"],
            "steps": cot_result["steps"],
            "final_answer": final_answer,
            "score": score,
            "jepa_loss": jepa_loss,
            "logprob": logprob_approx,
            "num_tokens": num_tokens,
            "num_steps": len(cot_result["steps"]),
        })

        logger.debug(
            f"Candidate {i+1}: score={score:.4f}, jepa_loss={jepa_loss:.4f}, "
            f"steps={len(cot_result['steps'])}"
        )

    # Sort by score (descending)
    candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # Best candidate
    best_cot = candidates_sorted[0]

    logger.info(
        f"Best CoT selected: score={best_cot['score']:.4f}, "
        f"jepa_loss={best_cot['jepa_loss']:.4f}, "
        f"num_steps={best_cot['num_steps']}"
    )

    return {
        "best_cot": best_cot,
        "candidates": candidates_sorted,
        "num_candidates": num_samples,
    }


def rerank_existing_cots(
    cots: List[str],
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    mask_ratio: float = 0.5,
    domain_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Re-rank existing CoT candidates (déjà générés).

    Utile quand on a déjà plusieurs candidates et qu'on veut juste
    les scorer avec JEPA.

    Args:
        cots: Liste de CoT candidates (texte brut)
        llm: LLM adapter (pour extraction latents)
        rjepa_client: Client HTTP vers service R-JEPA
        mask_ratio: Ratio de masking pour scoring JEPA
        domain_id: Optional domain ID

    Returns:
        {
            "best_cot": str,
            "best_score": float,
            "candidates": List[Dict],  # Avec scores
        }
    """
    logger.info(f"Re-ranking {len(cots)} existing CoT candidates...")

    candidates = []

    for i, cot_text in enumerate(cots):
        logger.debug(f"Scoring candidate {i+1}/{len(cots)}...")

        # Tokenize and segment
        tokens = llm.tokenizer.encode(cot_text, return_tensors="pt")

        # Segment into steps (simple split on "Step X:")
        import re
        steps = re.split(r"(Step \d+:)", cot_text)
        steps = [s.strip() for s in steps if s.strip() and not s.startswith("Step")]

        # Estimate step boundaries (approximation)
        # Pour une vraie implémentation, il faudrait tokenizer chaque step
        num_steps = len(steps)
        tokens_per_step = tokens.shape[1] // max(num_steps, 1)
        step_boundaries = [
            (i * tokens_per_step, (i + 1) * tokens_per_step)
            for i in range(num_steps)
        ]

        # Extract latents
        latents = llm.extract_latents(
            tokens=tokens,
            layer_idx=llm.layer_to_extract,
            step_boundaries=step_boundaries,
        )

        # Compute JEPA score
        jepa_result = rjepa_client.score(
            latents=latents,
            domain_id=domain_id,
            mask_ratio=mask_ratio,
        )

        jepa_loss = jepa_result["jepa_loss"]

        # Score = -jepa_loss (simple, lower loss = better)
        score = -jepa_loss

        candidates.append({
            "text": cot_text,
            "steps": steps,
            "score": score,
            "jepa_loss": jepa_loss,
            "num_steps": num_steps,
        })

    # Sort by score
    candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)

    best_cot = candidates_sorted[0]

    logger.info(
        f"Best CoT selected: score={best_cot['score']:.4f}, "
        f"jepa_loss={best_cot['jepa_loss']:.4f}"
    )

    return {
        "best_cot": best_cot["text"],
        "best_score": best_cot["score"],
        "candidates": candidates_sorted,
    }


def rerank_with_ensembling(
    prompt: str,
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    num_samples: int = 8,
    temperature: float = 0.8,
    top_k: int = 3,
    domain_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Re-ranking avec ensembling des top-K candidates.

    Génère N candidates, sélectionne les top-K selon JEPA,
    et retourne une réponse ensemblée (vote majoritaire ou consensus).

    Args:
        prompt: Question/problème
        llm: LLM adapter
        rjepa_client: Client R-JEPA
        num_samples: Nombre de candidates à générer
        temperature: Temperature génération
        top_k: Nombre de top candidates à ensembler
        domain_id: Optional domain ID

    Returns:
        {
            "best_cot": Dict,  # Meilleur candidat individuel
            "ensemble_answer": str,  # Réponse consensus
            "top_k_candidates": List[Dict],
            "confidence": float,  # Confiance du consensus
        }
    """
    logger.info(f"Generating {num_samples} candidates with ensembling (top-{top_k})...")

    # Re-rank comme d'habitude
    result = rerank_cots_with_jepa(
        prompt=prompt,
        llm=llm,
        rjepa_client=rjepa_client,
        num_samples=num_samples,
        temperature=temperature,
        domain_id=domain_id,
    )

    # Extraire top-K
    top_k_candidates = result["candidates"][:top_k]

    # Extraire final answers
    final_answers = [c["final_answer"] for c in top_k_candidates]

    # Vote majoritaire (simple: réponse la plus fréquente)
    from collections import Counter
    answer_counts = Counter(final_answers)
    ensemble_answer, count = answer_counts.most_common(1)[0]
    confidence = count / top_k

    logger.info(
        f"Ensemble answer: {ensemble_answer[:100]}... "
        f"(confidence: {confidence:.2%})"
    )

    return {
        "best_cot": result["best_cot"],
        "ensemble_answer": ensemble_answer,
        "top_k_candidates": top_k_candidates,
        "confidence": confidence,
        "all_candidates": result["candidates"],
    }
