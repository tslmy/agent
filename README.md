# <img width="50" alt="logo" src="public/logo_light.png"> LLM-powered private chatbot

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Runs fully offline.

## Demo

Ask it:

> Name a type of drink that I enjoy, and then look up its country of origin. Be concise.

and it will say:

> Based on the available evidence, it appears that coffee may have originated in either Ethiopia or Yemen. However, as the legend of Kaldi suggests, Ethiopia has long been associated with coffee's history.

after looking up the user's personal notes and then consulting Wikipedia.

<img width="851" alt="Screenshot 2024-01-28 at 23 32 11" src="https://github.com/tslmy/agent/assets/594058/1ce09f03-1ff5-4e51-bed8-77d281ddad41">


## Usage

Ensure that you have an OpenAI-compatible inference server running at `http://localhost:1234/`. If you're using [LM Studio](https://lmstudio.ai/), a tried-and-true configuration looks like this:

<img width="800" alt="Screenshot of a tried-and-true configuration" src="https://github.com/tslmy/agent/assets/594058/6af66e62-510f-42a6-86ee-f0dcb64eea08">

Tips: The LLM may be biased to generate "Observations" when it is only supposed to generate up to "Action Input". To mitigate this problem, add "Observation:" as a _Stop String_ in LM Studio, so that LM Studio will stop the LLM from generating any more text after it sees "Observation:".

Then, run the script:

```bash
chainlit run main.py -w
```

## Development

This repo uses pre-commit hooks to automate many chores. [Install](https://pre-commit.com/#install) them.

As a Python-based project, this repo registers all its dependencies in the `pyproject.toml` file. Use Poetry to [install](https://python-poetry.org/docs/basic-usage/#installing-dependencies) them.

```bash
PYTHONPATH=. poetry install --no-root
```

We use `--no-root` because we don't want to install the project itself as a dependency. It's an application, not a library.

As [this article](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html) explains:
> The main use of PYTHONPATH is when we are developing some code that we want to be able to import from Python, but that we have not yet made into an installable Python package.
