"""
R-JEPA Teacher Package.

Provides tools for generating and validating datasets via teacher LLMs.
"""
from .client import TeacherClient, MultiSourceTeacher
from .generator import ProblemGenerator, CoTGenerator, DatasetGenerator
from .validator import Validator, MathValidator, CodeValidator, LogicValidator
from .budget_tracker import BudgetTracker

__all__ = [
    "TeacherClient",
    "MultiSourceTeacher",
    "ProblemGenerator",
    "CoTGenerator",
    "DatasetGenerator",
    "Validator",
    "MathValidator",
    "CodeValidator",
    "LogicValidator",
    "BudgetTracker",
]
