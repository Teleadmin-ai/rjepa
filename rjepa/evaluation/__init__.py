"""
R-JEPA Evaluation Framework.

Provides tools for benchmarking and analyzing R-JEPA performance:
- Benchmark loaders (GSM8K, MATH, HumanEval)
- Metrics computation (accuracy, pass@k, correlation)
- A/B testing (JEPA on vs off)
- Visualization tools
"""
from .metrics import (
    compute_accuracy,
    compute_pass_at_k,
    compute_correlation,
    extract_answer,
    compute_metrics_summary,
)
from .benchmarks import (
    load_gsm8k,
    load_math,
    load_humaneval,
    create_mini_benchmark,
)
from .ab_testing import (
    run_ab_test,
    compare_modes,
)
from .visualization import (
    plot_jepa_loss_distribution,
    plot_correlation_scatter,
    plot_accuracy_comparison,
    plot_mode_comparison,
    plot_jepa_loss_by_domain,
    generate_evaluation_report,
)

__all__ = [
    # Metrics
    "compute_accuracy",
    "compute_pass_at_k",
    "compute_correlation",
    "extract_answer",
    "compute_metrics_summary",
    # Benchmarks
    "load_gsm8k",
    "load_math",
    "load_humaneval",
    "create_mini_benchmark",
    # A/B Testing
    "run_ab_test",
    "compare_modes",
    # Visualization
    "plot_jepa_loss_distribution",
    "plot_correlation_scatter",
    "plot_accuracy_comparison",
    "plot_mode_comparison",
    "plot_jepa_loss_by_domain",
    "generate_evaluation_report",
]
