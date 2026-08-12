# Contributing to PsyCoupler

Thank you for your interest in contributing to PsyCoupler.

---

## Development Setup

```bash
git clone https://github.com/eslam-ahmed43/psycoupler.git
cd psycoupler
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

---

## Running Tests

```bash
pytest tests/ -v
```

All PRs must pass the full test suite. New features must include tests.
Test coverage for new code should be >= 90%.

---

## Submitting a Pull Request

1. Open an issue first to discuss the change
2. Fork the repo and create a branch: `git checkout -b feat/your-feature`
3. Write tests for your changes
4. Run `pytest tests/ -v` and confirm all tests pass
5. Submit a PR with a clear description of what changed and why

---

## Areas We Welcome Contributions

- Multidimensional embedding backends (v0.2)
- Granger causality module (v0.3)
- Real annotated conversation datasets for validation
- Additional sentiment extractors
- Documentation improvements
- Bug fixes

---

## Code Style

- Python 3.10+
- Type hints on all public functions
- Docstrings on all public classes and functions
- No Arabic comments in code

---

## Questions

Open a GitHub Issue or Discussion.