"""
Automatic validation of Chain-of-Thought solutions.

Supports:
- Math: symbolic/numerical computation (sympy)
- Code: execution in sandbox (subprocess with timeout)
- Logic: simple rule-based validation
"""
import logging
import re
import subprocess
import tempfile
from typing import Tuple, Optional, Dict, List
from pathlib import Path

from rjepa.data.schemas import Problem, ChainOfThought

logger = logging.getLogger(__name__)


class MathValidator:
    """
    Validate mathematical solutions.
    """

    def __init__(self):
        """Initialize Math Validator."""
        try:
            import sympy
            self.sympy = sympy
            logger.info("Math validator initialized (sympy available)")
        except ImportError:
            self.sympy = None
            logger.warning("sympy not installed, math validation will be limited")

    def validate(
        self,
        problem: Problem,
        cot: ChainOfThought,
    ) -> Tuple[bool, str]:
        """
        Validate a mathematical solution.

        Args:
            problem: Problem
            cot: Chain-of-Thought solution

        Returns:
            (is_valid, reason)
        """
        if not self.sympy:
            return False, "sympy not available"

        # Extract final answer from CoT
        final_answer = cot.final_answer.strip()

        # Extract expected answer from problem
        if not problem.answer_gold:
            return False, "No gold answer to compare"

        expected = problem.answer_gold.strip()

        # Try to parse and compare numerically
        try:
            # Extract numerical values
            cot_value = self._extract_number(final_answer)
            expected_value = self._extract_number(expected)

            if cot_value is None or expected_value is None:
                return False, "Could not extract numerical values"

            # Compare with tolerance
            if abs(cot_value - expected_value) < 1e-6:
                return True, "Numerical match"
            else:
                return False, f"Mismatch: {cot_value} != {expected_value}"

        except Exception as e:
            logger.error(f"Math validation failed: {e}")
            return False, f"Validation error: {e}"

    def _extract_number(self, text: str) -> Optional[float]:
        """
        Extract a number from text.

        Args:
            text: Text containing a number

        Returns:
            Extracted float or None
        """
        # Try to find numbers in text
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches:
            return float(matches[-1])  # Take last number
        return None


class CodeValidator:
    """
    Validate code solutions by execution in sandbox.
    """

    def __init__(self, timeout: int = 5):
        """
        Initialize Code Validator.

        Args:
            timeout: Execution timeout in seconds
        """
        self.timeout = timeout
        logger.info(f"Code validator initialized (timeout={timeout}s)")

    def validate(
        self,
        problem: Problem,
        cot: ChainOfThought,
    ) -> Tuple[bool, str]:
        """
        Validate a code solution by running tests.

        Args:
            problem: Problem (should contain test cases)
            cot: Chain-of-Thought solution

        Returns:
            (is_valid, reason)
        """
        # Extract code from CoT
        code = self._extract_code(cot.final_answer)
        if not code:
            return False, "No code found in solution"

        # Check if problem has test cases
        if not problem.meta_course or "tests" not in problem.meta_course:
            return False, "No test cases to run"

        tests = problem.meta_course["tests"]

        # Run tests
        try:
            results = []
            for i, test in enumerate(tests):
                passed = self._run_test(code, test)
                results.append(passed)

            if all(results):
                return True, f"All {len(results)} tests passed"
            else:
                failed = sum(1 for r in results if not r)
                return False, f"{failed}/{len(results)} tests failed"

        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return False, f"Execution error: {e}"

    def _extract_code(self, text: str) -> Optional[str]:
        """
        Extract code from text (assumes code blocks).

        Args:
            text: Text containing code

        Returns:
            Extracted code or None
        """
        # Try to find code in markdown code blocks
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        else:
            # Assume entire text is code
            return text.strip()

    def _run_test(self, code: str, test: Dict) -> bool:
        """
        Run a single test case.

        Args:
            code: Code to test
            test: Test dict with "input" and "expected" keys

        Returns:
            True if test passed
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write code + test
            f.write(code)
            f.write("\n\n")
            f.write(f"# Test case\n")
            f.write(f"result = {test['input']}\n")
            f.write(f"expected = {test['expected']}\n")
            f.write("assert result == expected, f'Expected {{expected}}, got {{result}}'\n")
            f.write("print('PASS')\n")

            temp_path = f.name

        try:
            # Run with timeout
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # Check output
            return "PASS" in result.stdout

        except subprocess.TimeoutExpired:
            logger.warning(f"Test timed out after {self.timeout}s")
            return False
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)


class LogicValidator:
    """
    Validate logic/reasoning solutions (simple rule-based).
    """

    def __init__(self):
        """Initialize Logic Validator."""
        logger.info("Logic validator initialized")

    def validate(
        self,
        problem: Problem,
        cot: ChainOfThought,
    ) -> Tuple[bool, str]:
        """
        Validate a logic solution.

        Args:
            problem: Problem
            cot: Chain-of-Thought solution

        Returns:
            (is_valid, reason)
        """
        # Extract final answer
        final_answer = cot.final_answer.strip().lower()

        # Extract expected answer
        if not problem.answer_gold:
            return False, "No gold answer to compare"

        expected = problem.answer_gold.strip().lower()

        # Simple string match
        if final_answer == expected:
            return True, "Exact match"
        elif final_answer in expected or expected in final_answer:
            return True, "Partial match"
        else:
            return False, f"Mismatch: '{final_answer}' != '{expected}'"


class Validator:
    """
    High-level validator that dispatches to domain-specific validators.
    """

    def __init__(self):
        """Initialize Validator."""
        self.math_validator = MathValidator()
        self.code_validator = CodeValidator()
        self.logic_validator = LogicValidator()

        logger.info("Validator initialized (all domains)")

    def validate(
        self,
        problem: Problem,
        cot: ChainOfThought,
    ) -> Tuple[bool, str]:
        """
        Validate a solution (dispatches to domain validator).

        Args:
            problem: Problem
            cot: Chain-of-Thought solution

        Returns:
            (is_valid, reason)
        """
        domain = problem.domain.lower()

        if domain == "math":
            return self.math_validator.validate(problem, cot)
        elif domain == "code":
            return self.code_validator.validate(problem, cot)
        elif domain == "logic":
            return self.logic_validator.validate(problem, cot)
        else:
            logger.warning(f"Unknown domain: {domain}, skipping validation")
            return False, f"Unsupported domain: {domain}"

    def validate_batch(
        self,
        problems: List[Problem],
        cots: List[ChainOfThought],
    ) -> List[Tuple[bool, str]]:
        """
        Validate a batch of solutions.

        Args:
            problems: List of problems
            cots: List of Chain-of-Thought solutions

        Returns:
            List of (is_valid, reason) tuples
        """
        results = []

        for problem, cot in zip(problems, cots):
            is_valid, reason = self.validate(problem, cot)
            results.append((is_valid, reason))

            # Update CoT validation status
            cot.is_valid = is_valid
            cot.validation_reason = reason

        valid_count = sum(1 for is_valid, _ in results if is_valid)
        logger.info(f"Validated {len(results)} solutions: {valid_count} valid ({valid_count/len(results)*100:.1f}%)")

        return results
