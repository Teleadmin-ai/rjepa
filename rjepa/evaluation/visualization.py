"""
Visualization tools for R-JEPA evaluation results.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_jepa_loss_distribution(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    bins: int = 30,
):
    """
    Plot histogram of JEPA-loss distribution, separated by correctness.

    Args:
        results: List of evaluation results with jepa_loss and is_correct
        output_path: Path to save plot (if None, display only)
        bins: Number of histogram bins
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Separate by correctness
    correct_losses = [r["jepa_loss"] for r in results if r.get("is_correct") and r.get("jepa_loss") is not None]
    incorrect_losses = [r["jepa_loss"] for r in results if not r.get("is_correct") and r.get("jepa_loss") is not None]

    if not correct_losses and not incorrect_losses:
        logger.warning("No JEPA losses found in results")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    if correct_losses:
        ax.hist(correct_losses, bins=bins, alpha=0.6, label=f"Correct (n={len(correct_losses)})", color="green")
    if incorrect_losses:
        ax.hist(incorrect_losses, bins=bins, alpha=0.6, label=f"Incorrect (n={len(incorrect_losses)})", color="red")

    ax.set_xlabel("JEPA Loss")
    ax.set_ylabel("Frequency")
    ax.set_title("JEPA-Loss Distribution by Correctness")
    ax.legend()
    ax.grid(alpha=0.3)

    # Add mean lines
    if correct_losses:
        mean_correct = np.mean(correct_losses)
        ax.axvline(mean_correct, color="green", linestyle="--", linewidth=2, label=f"Mean (correct): {mean_correct:.4f}")
    if incorrect_losses:
        mean_incorrect = np.mean(incorrect_losses)
        ax.axvline(mean_incorrect, color="red", linestyle="--", linewidth=2, label=f"Mean (incorrect): {mean_incorrect:.4f}")

    ax.legend()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_correlation_scatter(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
):
    """
    Plot scatter plot of JEPA-loss vs correctness.

    IMPORTANT: Lower JEPA-loss should correlate with higher correctness.

    Args:
        results: List of evaluation results with jepa_loss and is_correct
        output_path: Path to save plot (if None, display only)
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr
    except ImportError:
        logger.error("matplotlib or scipy not installed")
        return

    # Extract data
    jepa_losses = [r["jepa_loss"] for r in results if r.get("jepa_loss") is not None]
    correctness = [1.0 if r.get("is_correct") else 0.0 for r in results if r.get("jepa_loss") is not None]

    if len(jepa_losses) < 2:
        logger.warning("Not enough data points for correlation plot")
        return

    # Compute correlation
    corr, pval = pearsonr(jepa_losses, correctness)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with jitter for visibility
    jitter = np.random.normal(0, 0.02, len(correctness))
    correctness_jittered = np.array(correctness) + jitter

    colors = ["red" if c == 0 else "green" for c in correctness]
    ax.scatter(jepa_losses, correctness_jittered, alpha=0.5, c=colors, s=50)

    ax.set_xlabel("JEPA Loss")
    ax.set_ylabel("Correctness (0 = incorrect, 1 = correct)")
    ax.set_title(f"JEPA-Loss vs Correctness\nPearson r = {corr:.4f} (p={pval:.4e})")
    ax.grid(alpha=0.3)

    # Add horizontal lines
    ax.axhline(0, color="red", linestyle="--", alpha=0.3, label="Incorrect")
    ax.axhline(1, color="green", linestyle="--", alpha=0.3, label="Correct")

    ax.legend()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_accuracy_comparison(
    metrics: Dict[str, Any],
    output_path: Optional[Path] = None,
):
    """
    Plot bar chart comparing baseline vs JEPA accuracy.

    Args:
        metrics: Dictionary with "baseline" and "jepa" accuracy metrics
        output_path: Path to save plot (if None, display only)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed")
        return

    baseline_acc = metrics["baseline"]["accuracy"]
    jepa_acc = metrics["jepa"]["accuracy"]
    delta = metrics["delta_accuracy"]
    relative_improvement = metrics["relative_improvement"]

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Baseline\n(JEPA OFF)", "With R-JEPA\n(JEPA ON)"]
    accuracies = [baseline_acc, jepa_acc]
    colors = ["#cccccc", "#4CAF50"]

    bars = ax.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor="black")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f"{acc:.2%}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Add delta annotation
    ax.annotate(
        f"\u0394 = {delta:+.2%}\n({relative_improvement:+.1f}%)",
        xy=(1, jepa_acc), xytext=(0.5, max(accuracies) * 0.95),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=12, ha="center", color="red", fontweight="bold"
    )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("R-JEPA Performance Improvement", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_mode_comparison(
    comparison_results: Dict[str, Any],
    output_path: Optional[Path] = None,
):
    """
    Plot comparison of different JEPA modes.

    Args:
        comparison_results: Results from compare_modes() function
        output_path: Path to save plot (if None, display only)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed")
        return

    summary = comparison_results["summary"]

    modes = list(summary.keys())
    accuracies = [summary[mode]["accuracy"] for mode in modes]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#cccccc", "#4CAF50", "#2196F3", "#FF9800"]
    bars = ax.bar(modes, accuracies, color=colors[:len(modes)], alpha=0.8, edgecolor="black")

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f"{acc:.2%}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Mode", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("R-JEPA Mode Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_jepa_loss_by_domain(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
):
    """
    Plot JEPA-loss statistics by domain.

    Args:
        results: List of evaluation results with domain and jepa_loss
        output_path: Path to save plot (if None, display only)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed")
        return

    # Group by domain
    domains = {}
    for r in results:
        domain = r.get("domain", "unknown")
        jepa_loss = r.get("jepa_loss")
        if jepa_loss is not None:
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(jepa_loss)

    if not domains:
        logger.warning("No domain data found")
        return

    # Compute statistics
    domain_names = list(domains.keys())
    means = [np.mean(domains[d]) for d in domain_names]
    stds = [np.std(domains[d]) for d in domain_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(domain_names))
    bars = ax.bar(x_pos, means, yerr=stds, alpha=0.8, capsize=5, edgecolor="black", color="#4CAF50")

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f"{mean:.3f}±{std:.3f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Domain", fontsize=12)
    ax.set_ylabel("JEPA Loss", fontsize=12)
    ax.set_title("JEPA-Loss by Domain", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(domain_names)
    ax.grid(axis="y", alpha=0.3)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_evaluation_report(
    results: Dict[str, Any],
    output_dir: Path,
):
    """
    Generate complete evaluation report with all visualizations.

    Args:
        results: Full evaluation results dictionary
        output_dir: Directory to save report and plots
    """
    logger.info(f"Generating evaluation report in {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    metrics = results["metrics"]
    jepa_results = results["jepa_results"]

    # Generate plots
    logger.info("Generating JEPA-loss distribution plot...")
    plot_jepa_loss_distribution(
        jepa_results,
        output_path=output_dir / "jepa_loss_distribution.png"
    )

    logger.info("Generating correlation scatter plot...")
    plot_correlation_scatter(
        jepa_results,
        output_path=output_dir / "correlation_scatter.png"
    )

    logger.info("Generating accuracy comparison plot...")
    plot_accuracy_comparison(
        metrics,
        output_path=output_dir / "accuracy_comparison.png"
    )

    if "domain" in jepa_results[0]:
        logger.info("Generating JEPA-loss by domain plot...")
        plot_jepa_loss_by_domain(
            jepa_results,
            output_path=output_dir / "jepa_loss_by_domain.png"
        )

    logger.info("Report generation completed!")
