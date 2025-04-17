# Contributing to CareFrame

Thank you for your interest in contributing to CareFrame! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

If you find a bug in the project:

1. Check if the bug has already been reported in the [Issues](https://github.com/yourusername/careframe/issues).
2. If not, create a new issue, providing a clear description and steps to reproduce.
3. Include relevant details like your operating system, Python version, and any error messages.

### Feature Requests

We welcome feature requests:

1. Check if the feature has already been requested or implemented.
2. Create a new issue clearly describing the feature and its potential benefits.

### Pull Requests

1. Fork the repository.
2. Create a new branch for your feature or bugfix (`git checkout -b feature/amazing-feature`).
3. Make your changes, following our coding standards.
4. Write or update tests as needed.
5. Ensure all tests pass.
6. Commit your changes with clear, descriptive commit messages.
7. Push to your branch and submit a pull request to the `main` branch.

## Development Setup

1. Clone your fork of the repository.
2. Install development dependencies:
   ```
   pip install -r requirements-dev.txt
   ```
3. Set up pre-commit hooks:
   ```
   pre-commit install
   ```

## Coding Standards

- Follow PEP 8 guidelines for Python code.
- Write clear docstrings for functions and classes.
- Maintain test coverage for new code.
- Keep dependencies minimal and well-justified.

## Testing

- Run tests before submitting a PR:
  ```
  pytest
  ```
- Add tests for new features or bug fixes.

## Documentation

- Update documentation for any user-facing changes.
- Clearly document new features, APIs, or significant changes.

## Review Process

1. All PRs require review from at least one maintainer.
2. Address review feedback promptly.
3. Once approved, maintainers will merge your PR.

Thank you for contributing to make CareFrame better! 