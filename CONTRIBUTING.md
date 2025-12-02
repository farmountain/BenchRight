# Contributing to BenchRight

Thank you for your interest in contributing to BenchRight! This document provides guidelines for contributing to the project.

## How to Contribute

### Proposing New Benchmarks

- Open an issue describing the benchmark you want to add
- Include the following in your proposal:
  - Benchmark name and purpose
  - Datasets required
  - Expected metrics and outputs
  - Reference papers or implementations

### Adding Datasets

- Document all new datasets in `docs/datasets.md`
- Include dataset source, license, and size information
- Provide download instructions or scripts
- Do not commit large data files directly to the repository

### Creating New Notebooks

- Follow the existing notebook structure and naming conventions
- Include clear markdown explanations between code cells
- Test notebooks in Google Colab before submitting
- Add any new dependencies to `requirements.txt`

### Adding New Weeks

- Follow the existing week content structure (see `week1.md`)
- Include learning objectives, theoretical content, and hands-on labs
- Update `docs/syllabus.md` with the new week's summary
- Link the new week in `README.md`

## Requirements for New Benchmark Code

Any new benchmark code must include:

- A short docstring explaining the benchmark's purpose and usage
- One minimal example in the `/examples` directory
- At least one small unit test where applicable

## Code Quality

- PRs must pass basic linting (flake8/black or ruff)
- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Keep functions focused and modular

## Submitting a Pull Request

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes following the guidelines above
4. Run linting checks locally
5. Commit your changes with clear, descriptive messages
6. Push to your fork and submit a pull request
7. Respond to any review feedback

## Questions?

If you have questions about contributing, please open an issue and we'll be happy to help.
