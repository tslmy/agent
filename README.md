# <img width="50" alt="logo" src="public/logo_light.png"> A fully local, LLM-powered, minimal chatbot

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A good starting point for building your own personal AI.

## Features

### Local-first: nothing goes out
Unless you ask it to search Wikipedia, etc., no internet connection is required.

Why is this important? Because as long as a chatbot still sends information to the cloud (OpenAI, Azure, ...), I wouldn't trust it with sensitive info like my paystubs, health records, passwords, etc.

### Minimal: cheap to develop, easy to understand
Uses off-the-shelf components to keep the codebase small and easy to understand.

- [LM Studio](https://lmstudio.ai/) serves a LLM ([Zephyr 7B beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), as of Jan 2024) locally. No privacy concerns there -- The LLM is not fine-tuned with any private information, and the server is stateless. (Note: You can easily replace LM Studio with Ollama, etc., but I like the GUI that LM Studio provides.)
- [LlamaIndex](https://www.llamaindex.ai/) provides a natural-language querying interface. It also indexes documents into [embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture), which are stored into a [Chroma](https://www.trychroma.com/) database.
- [ChainLit](https://chainlit.io/) provides a web UI to LlamaIndex that looks like ChatGPT. (I tried building a TUI with [rich](https://github.com/Textualize/rich), but it's not very obvious how to surface the retrieved documents.)

Why is this important? Because you almost certainly have your own niche needs for your own AI chatbot, so you are most likely to be developing something solely on your own. With limited workforce, it's important to keep the codebase small and easy to understand.

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

## Structure

`main.py` is the entrypoint. It runs a [ReAct Agent](https://www.promptingguide.ai/techniques/react) ([S Yao, et, al.](https://arxiv.org/abs/2210.03629)). As an agent, the AI is capable of wielding several tools. For example,

* It can use `tool_for_my_notes.py` to look up plain-text notes you stored in a folder. For demo purposes, the folder `demo-notes/` contains some stubs that you can check out.
* It can use `tool_for_wikipedia.py` to find answers to a given question after consulting Wikipedia articles.

`chainlit.md` and `public/` are simply UI assets for the web frontend.

### Prompt engineering tricks used

To improve the precision of `tool_for_my_notes.py`, I modified the default prompt for the [sub-question query engine in LlamaIndex][sq] by asking it to generate keywords rather than complete sentences. The changes are in `sub_question_generating_prompt_in_keywords.py`.

Similarly, I also overrode the **agent-level system prompt**. Since it's quite a long prose, I put that in a separate file, `system_prompt.md`.

[sq]: https://docs.llamaindex.ai/en/latest/examples/query_engine/sub_question_query_engine.html

**One-shot examples, instead of zero-shot.** In both the `QueryEngineTool` in `tool_for_my_notes.py` and the `OnDemandLoaderTool` in `tool_for_wikipedia.py`, I added one example in the tool description. This greatly improves the quality of _Action Inputs_ generated by the LLM.

### Count of line of codes by language

Generated via `cloc --md .`:

Language|files|blank|comment|code
:-------|-------:|-------:|-------:|-------:
JSON|111|0|0|419
XML|7|0|0|411
Python|4|66|87|231
Markdown|5|62|0|103
TOML|2|26|36|43
YAML|1|0|5|41
--------|--------|--------|--------|--------
SUM:|130|154|128|1248
