# LLM-powered private chatbot

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Runs fully offline.

## Usage

Ensure that you have an OpenAI-compatible inference server running at `http://localhost:1234/`. If you're using [LM Studio](https://lmstudio.ai/), a tried-and-true configuration looks like this:

<img width="800" alt="Screenshot of a tried-and-true configuration" src="https://github.com/tslmy/agent/assets/594058/6af66e62-510f-42a6-86ee-f0dcb64eea08">

Then, run the script:

```bash
chainlit run main.py -w
```

## Development

This repo uses pre-commit hooks to automate many chores. [Install](https://pre-commit.com/#install) them.

As a Python-based project, this repo registers all its dependencies in the `pyproject.toml` file. Use Poetry to [install](https://python-poetry.org/docs/basic-usage/#installing-dependencies) them.

```bash
poetry install --no-root
```

We use `--no-root` because we don't want to install the project itself as a dependency. It's an application, not a library.
