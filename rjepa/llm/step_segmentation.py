"""
Step segmentation utilities for Chain-of-Thought.

Provides various strategies to segment reasoning text into steps.
"""
import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def segment_by_step_markers(
    text: str,
    step_marker: str = "Step",
    case_sensitive: bool = False
) -> List[str]:
    """
    Segment text by explicit step markers ("Step 1:", "Step 2:", etc.).

    Args:
        text: Full text to segment
        step_marker: Marker used for steps (default "Step")
        case_sensitive: Whether to match case-sensitively

    Returns:
        List of step strings
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = rf"{step_marker}\s+\d+:"
    matches = list(re.finditer(pattern, text, flags=flags))

    if not matches:
        logger.warning(f"No step markers found in text (marker='{step_marker}')")
        return [text]

    steps = []
    for i, match in enumerate(matches):
        start_char = match.start()
        end_char = matches[i+1].start() if i+1 < len(matches) else len(text)
        step_text = text[start_char:end_char].strip()
        steps.append(step_text)

    return steps


def segment_by_sentences(text: str, max_step_length: int = 200) -> List[str]:
    """
    Segment text by sentences, grouping into steps.

    Args:
        text: Full text to segment
        max_step_length: Max characters per step

    Returns:
        List of step strings
    """
    # Simple sentence splitter (can be improved with nltk/spacy)
    sentences = re.split(r'[.!?]+\s+', text)

    steps = []
    current_step = ""

    for sentence in sentences:
        if len(current_step) + len(sentence) > max_step_length and current_step:
            steps.append(current_step.strip())
            current_step = sentence
        else:
            current_step += " " + sentence if current_step else sentence

    if current_step:
        steps.append(current_step.strip())

    return steps


def segment_by_connectors(text: str) -> List[str]:
    """
    Segment text by logical connectors ("Therefore", "Thus", "Next", etc.).

    Args:
        text: Full text to segment

    Returns:
        List of step strings
    """
    connectors = [
        "Therefore", "Thus", "So", "Hence",
        "Next", "Then", "Now",
        "Finally", "In conclusion",
        "First", "Second", "Third",
    ]

    # Create pattern
    pattern = r'(' + '|'.join(connectors) + r')'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    steps = []
    current_step = ""

    for i, part in enumerate(parts):
        if part.strip() in connectors or part.strip().lower() in [c.lower() for c in connectors]:
            if current_step:
                steps.append(current_step.strip())
            current_step = part.strip()
        else:
            current_step += " " + part.strip() if current_step else part.strip()

    if current_step:
        steps.append(current_step.strip())

    return steps


def segment_auto(
    text: str,
    prefer_markers: bool = True
) -> List[str]:
    """
    Automatic segmentation with fallback strategies.

    Args:
        text: Full text to segment
        prefer_markers: Try step markers first

    Returns:
        List of step strings
    """
    if prefer_markers:
        steps = segment_by_step_markers(text)
        if len(steps) > 1:
            return steps

    # Fallback: try connectors
    steps = segment_by_connectors(text)
    if len(steps) > 1:
        return steps

    # Final fallback: sentences
    return segment_by_sentences(text)


def get_token_boundaries(
    steps_text: List[str],
    full_text: str,
    tokenizer
) -> List[Tuple[int, int]]:
    """
    Get token boundaries for each step.

    Args:
        steps_text: List of step strings
        full_text: Original full text
        tokenizer: HuggingFace tokenizer

    Returns:
        List of (start_idx, end_idx) tuples in token space
    """
    boundaries = []
    current_char = 0

    for step_text in steps_text:
        # Find step in full text
        step_start = full_text.find(step_text, current_char)
        if step_start == -1:
            logger.warning(f"Could not find step in full text: {step_text[:50]}...")
            continue

        step_end = step_start + len(step_text)

        # Convert char indices to token indices
        prefix_tokens = len(tokenizer.encode(full_text[:step_start]))
        current_tokens = len(tokenizer.encode(full_text[:step_end]))

        boundaries.append((prefix_tokens, current_tokens))
        current_char = step_end

    return boundaries
