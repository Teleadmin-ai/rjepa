# Contributing to R-JEPA

We welcome contributions to the R-JEPA (Reasoning Joint Embedding Predictive Architecture) project!

## How to Contribute

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes: `pytest tests/`
5. Make sure your code follows our style guidelines (see below).
6. Submit your pull request!

### Issues

We use GitHub issues to track bugs and feature requests. Please ensure your description is clear and includes:

- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (OS, Python version, GPU, etc.)

### Security Issues

For security vulnerabilities, please **do not** open a public issue. Instead, email us directly at provencal.romain@teleadmin.net with details.

## Development Setup

```bash
# Clone and setup
git clone https://github.com/Teleadmin-ai/rjepa.git
cd rjepa

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# or: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[train,dev]"

# Run tests
pytest tests/
```

## Coding Style

- Python 3.11+ required
- Use `ruff` for linting and formatting
- Type hints are encouraged
- Follow PEP 8 guidelines
- Maximum line length: 100 characters

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking (optional)
mypy rjepa/
```

## Commit Messages

We follow conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example: `feat: Add support for Llama3 model family`

## License

By contributing to this repository, you agree that your contributions will be licensed under the CC BY-NC 4.0 license (see LICENSE file).

## Questions?

Feel free to open an issue or reach out to the maintainers.

---

*R-JEPA: A World Model for Text Reasoning*
