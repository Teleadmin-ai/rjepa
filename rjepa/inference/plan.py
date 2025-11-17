"""
Plan mode for R-JEPA.

Complétion de steps manquants : étant donné un raisonnement partiel,
R-JEPA prédit les latents des steps manquants et on les décode en texte.
"""
import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from rjepa.llm.adapter import LLMAdapter
from rjepa.jepa.client import RJEPAClient

logger = logging.getLogger(__name__)


def complete_reasoning_plan(
    partial_steps: List[str],
    missing_indices: List[int],
    total_steps: int,
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    domain_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Complète un plan de raisonnement avec steps manquants.

    Étant donné:
    - Step 1: "..."
    - Step 2: [MISSING]
    - Step 3: [MISSING]
    - Step 4: "..."

    R-JEPA prédit les latents des steps 2 et 3, puis on les décode en texte.

    Args:
        partial_steps: Liste des steps connus (avec None pour missing)
        missing_indices: Indices des steps manquants
        total_steps: Nombre total de steps attendus
        llm: LLM adapter
        rjepa_client: Client R-JEPA
        domain_id: Optional domain ID

    Returns:
        {
            "completed_steps": List[str],  # Tous les steps (complétés)
            "predicted_steps": Dict[int, str],  # Index -> texte prédit
            "full_text": str,
        }
    """
    logger.info(
        f"Completing reasoning plan: {len(missing_indices)} missing steps "
        f"out of {total_steps}"
    )

    # Extract latents for known steps
    known_steps_text = [s for s in partial_steps if s is not None]
    known_steps_full_text = "\n".join(known_steps_text)

    # Tokenize and extract latents
    tokens = llm.tokenizer.encode(known_steps_full_text, return_tensors="pt")

    # Segment into steps (approximation)
    num_known_steps = len(known_steps_text)
    tokens_per_step = tokens.shape[1] // max(num_known_steps, 1)
    step_boundaries = [
        (i * tokens_per_step, (i + 1) * tokens_per_step)
        for i in range(num_known_steps)
    ]

    latents_known = llm.extract_latents(
        tokens=tokens,
        layer_idx=llm.layer_to_extract,
        step_boundaries=step_boundaries,
    )  # [num_known_steps, hidden_dim]

    # Pad with zeros for missing steps to create full sequence
    hidden_dim = latents_known.shape[1]
    latents_full = torch.zeros(total_steps, hidden_dim)

    # Fill in known steps
    known_idx = 0
    for i in range(total_steps):
        if i not in missing_indices:
            if known_idx < latents_known.shape[0]:
                latents_full[i] = latents_known[known_idx]
                known_idx += 1

    # Predict missing steps with JEPA
    predicted_latents = rjepa_client.predict_masked(
        latents=latents_full,
        mask_indices=missing_indices,
        domain_id=domain_id,
    )  # [num_missing, hidden_dim]

    # Fill in predicted latents
    for idx, missing_idx in enumerate(missing_indices):
        latents_full[missing_idx] = predicted_latents[idx]

    # Decode predicted latents to text
    # Note: Ceci est la partie la plus difficile car il n'y a pas de décodeur
    # direct latent->text. On utilise une approximation via prompting.

    predicted_steps = {}

    for idx, missing_idx in enumerate(missing_indices):
        # Prompt LLM to "verbalize" this latent
        # On donne le contexte des steps avant et après
        context_before = (
            partial_steps[missing_idx - 1] if missing_idx > 0 else ""
        )
        context_after = (
            partial_steps[missing_idx + 1]
            if missing_idx + 1 < len(partial_steps)
            else ""
        )

        # Prompt pour complétion
        completion_prompt = f"""Given the reasoning context:
Before: {context_before}
After: {context_after}

What is the logical intermediate step? Provide a single reasoning step.
Step {missing_idx + 1}:"""

        # Generate completion
        completion = llm.generate_with_cot(
            prompt=completion_prompt,
            max_new_tokens=128,
            temperature=0.5,
            step_token="Step",
            num_samples=1,
        )[0]

        predicted_text = completion["full_text"].split("\n")[0].strip()
        predicted_steps[missing_idx] = predicted_text

        logger.debug(f"Predicted step {missing_idx}: {predicted_text[:100]}...")

    # Reconstruct full text
    completed_steps = []
    for i in range(total_steps):
        if i in missing_indices:
            completed_steps.append(predicted_steps[i])
        else:
            completed_steps.append(partial_steps[i])

    full_text = "\n".join(
        f"Step {i + 1}: {step}" for i, step in enumerate(completed_steps)
    )

    logger.info(f"Plan completion done: {len(predicted_steps)} steps predicted")

    return {
        "completed_steps": completed_steps,
        "predicted_steps": predicted_steps,
        "full_text": full_text,
    }


def auto_complete_missing_steps(
    prompt: str,
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    num_expected_steps: int = 5,
    domain_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Auto-complétion : génère un squelette de raisonnement puis complète.

    Stratégie:
    1. Génère un raisonnement court (outline)
    2. Identifie les gaps
    3. Utilise JEPA pour prédire les steps manquants
    4. Décode en texte

    Args:
        prompt: Question/problème
        llm: LLM adapter
        rjepa_client: Client R-JEPA
        num_expected_steps: Nombre de steps attendus
        domain_id: Optional domain ID

    Returns:
        {
            "outline": List[str],  # Squelette initial
            "completed": List[str],  # Raisonnement complet
            "full_text": str,
        }
    """
    logger.info(
        f"Auto-completing reasoning plan (expected {num_expected_steps} steps)..."
    )

    # Génère un outline court
    outline_prompt = f"""{prompt}

Provide a brief outline of the reasoning steps (step titles only):
Step 1:
Step 2:
..."""

    outline_result = llm.generate_with_cot(
        prompt=outline_prompt,
        max_new_tokens=256,
        temperature=0.5,
        step_token="Step",
        num_samples=1,
    )[0]

    outline_steps = outline_result["steps"]

    logger.info(f"Generated outline with {len(outline_steps)} steps")

    # Si moins de steps que prévu, on identifie des gaps
    if len(outline_steps) < num_expected_steps:
        # Insérer des gaps uniformément
        gap_positions = np.linspace(
            1, len(outline_steps), num_expected_steps - len(outline_steps) + 1, dtype=int
        ).tolist()[1:]

        # Create partial_steps avec None pour gaps
        partial_steps = []
        gap_idx = 0
        for i in range(num_expected_steps):
            if gap_idx < len(gap_positions) and i == gap_positions[gap_idx]:
                partial_steps.append(None)
                gap_idx += 1
            elif i - gap_idx < len(outline_steps):
                partial_steps.append(outline_steps[i - gap_idx])
            else:
                partial_steps.append(None)

        missing_indices = [i for i, s in enumerate(partial_steps) if s is None]

        # Complete plan
        completion_result = complete_reasoning_plan(
            partial_steps=partial_steps,
            missing_indices=missing_indices,
            total_steps=num_expected_steps,
            llm=llm,
            rjepa_client=rjepa_client,
            domain_id=domain_id,
        )

        return {
            "outline": outline_steps,
            "completed": completion_result["completed_steps"],
            "full_text": completion_result["full_text"],
        }

    else:
        # Déjà complet
        return {
            "outline": outline_steps,
            "completed": outline_steps,
            "full_text": outline_result["full_text"],
        }


def iterative_refinement(
    prompt: str,
    llm: LLMAdapter,
    rjepa_client: RJEPAClient,
    num_iterations: int = 3,
    domain_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Raffinement itératif : génère un raisonnement, identifie les faiblesses,
    et complète/améliore itérativement.

    Args:
        prompt: Question/problème
        llm: LLM adapter
        rjepa_client: Client R-JEPA
        num_iterations: Nombre d'itérations de raffinement
        domain_id: Optional domain ID

    Returns:
        {
            "iterations": List[Dict],  # Historique des itérations
            "final_reasoning": str,
            "final_jepa_loss": float,
        }
    """
    logger.info(f"Iterative refinement: {num_iterations} iterations...")

    iterations = []
    current_reasoning = None

    for iter_idx in range(num_iterations):
        logger.debug(f"Iteration {iter_idx + 1}/{num_iterations}...")

        # Generate or refine
        if current_reasoning is None:
            # First iteration: generate from scratch
            cot_result = llm.generate_with_cot(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.7,
                step_token="Step",
                num_samples=1,
            )[0]
        else:
            # Refine previous reasoning
            refine_prompt = f"""{prompt}

Previous reasoning:
{current_reasoning}

Identify any weak or missing steps and provide an improved reasoning:"""

            cot_result = llm.generate_with_cot(
                prompt=refine_prompt,
                max_new_tokens=512,
                temperature=0.5,
                step_token="Step",
                num_samples=1,
            )[0]

        # Extract latents and score
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

        jepa_loss = jepa_result["jepa_loss"]

        iterations.append({
            "iteration": iter_idx + 1,
            "reasoning": cot_result["full_text"],
            "jepa_loss": jepa_loss,
            "num_steps": len(cot_result["steps"]),
        })

        current_reasoning = cot_result["full_text"]

        logger.debug(f"Iteration {iter_idx + 1}: jepa_loss={jepa_loss:.4f}")

    # Best iteration
    best_iter = min(iterations, key=lambda x: x["jepa_loss"])

    logger.info(
        f"Refinement complete: best_jepa_loss={best_iter['jepa_loss']:.4f} "
        f"at iteration {best_iter['iteration']}"
    )

    return {
        "iterations": iterations,
        "final_reasoning": best_iter["reasoning"],
        "final_jepa_loss": best_iter["jepa_loss"],
    }
