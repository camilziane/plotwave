# Developer Guide

## Setup

From the project root:

```bash
uv sync --all-groups
```

## Main Commands

Run the test suite:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check .
```

Run type-checking:

```bash
uv run mypy src
```

Run the example notebook:

```bash
uv run jupyter lab
```

Build the package:

```bash
uv build
```

## Typical Dev Loop

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## Project Layout

- `src/plotwave/`: library code
- `tests/`: automated tests
- `examples/getting_started.ipynb`: user-facing notebook walkthrough
- `examples/signal_helpers.py`: helper signals used by the examples

## Notes

- The package is installable with both `uv` and `pip`.
- The notebook examples are meant for users; keep them focused on the public API.
- If you change rendering or public behavior, update tests and the example notebook together.
