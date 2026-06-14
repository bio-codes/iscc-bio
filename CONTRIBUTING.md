# Contributing to iscc-bio

Thank you for your interest in improving `iscc-bio`. This project is developed in the BioCodes and ISCC ecosystem and welcomes reproducibility reports, small fixtures, documentation fixes, and code contributions.

## Reporting issues

Use the GitHub issue tracker:

https://github.com/bio-codes/iscc-bio/issues

Please include:

- the `iscc-bio` version or commit;
- operating system and Python version;
- the command or API call you ran;
- the smallest public or shareable bioimage fixture that reproduces the behavior;
- the full traceback or output, if applicable.

Do not attach private microscopy data unless you have permission to share it publicly.

## Development setup

```bash
git clone https://github.com/bio-codes/iscc-bio.git
cd iscc-bio
uv sync --python 3.11 --extra all --dev
uv run pytest
```

For the JOSS conversion experiment, Java is required for the default Bio-Formats `bfconvert` path:

```bash
uv run python experiments/joss_conversion_matching.py
```

## Pull requests

Before opening a pull request:

1. keep changes focused on one issue or feature;
2. add or update tests for behavior changes;
3. run formatting, linting, and tests:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

For new public sample fixtures, prefer small files with stable URLs, explicit byte sizes, and SHA-256 digests. Record reader/converter failures honestly rather than silently excluding difficult formats.

## Code of conduct

Be respectful and constructive. This is a scientific and open-source infrastructure project; assume good faith, document uncertainty, and avoid sharing data that cannot be redistributed.
