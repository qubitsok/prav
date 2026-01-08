# Contributing to prav-py

Thank you for your interest in contributing to prav-py!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/qubitsok/prav.git
   cd prav/prav-py
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install maturin
   pip install -e ".[dev]"
   ```

4. Build the Rust extension:
   ```bash
   maturin develop
   ```

## Running Tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=prav
```

## Code Quality

Before submitting a PR, ensure:

1. **Linting passes:**
   ```bash
   ruff check .
   ```

2. **Formatting is correct:**
   ```bash
   ruff format --check .
   ```

3. **Type checking passes:**
   ```bash
   mypy python/prav
   ```

4. **Tests pass:**
   ```bash
   pytest tests/ -v
   ```

5. **Rust code is formatted:**
   ```bash
   cargo fmt --check
   ```

6. **Rust linting passes:**
   ```bash
   cargo clippy -- -D warnings
   ```

## Pull Request Guidelines

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all quality checks pass
5. Request review

## Commit Messages

Follow conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## License

By contributing, you agree that your contributions will be licensed under the MIT OR Apache-2.0 dual license.
