# Llamacron: "When time gets right"

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

<img src="https://github.com/tslmy/agent/assets/594058/f9260527-de8a-47ac-bf73-573301fcc17e" width="200" />

_A unicorn llama, generated via MidJourney_

Interruption can be productive.

## Problem statement

What's missing?

### Humans can postpone tasks

![](https://github.com/tslmy/agent/assets/594058/4ea77ab2-77eb-4e21-a400-1e95b52dcb27)

([source](https://www.pexels.com/photo/a-small-dog-standing-in-front-of-a-window-looking-out-16479077/))

Watching a movie together in a cozy afternoon, you asked your partner to walk the dog.

They looked out the window, saying, "It's raining outside. Maybe later", and rejoined you on the couch.

When the sun came out, **without you asking again** (if you had to, reconsider your marriage), they said, "The time is right. I'll walk the dog now."

### Few AIs do that today

If you married an AI (I'm not judging), they will:
- either outright refuse to walk the dog and completely forget about it,
- or start staring at the window for 5 hours, ruining the better half of the Netflix marathon.

They lack a sense of "back burner".

![Miele KM391GBL 36" Black Gas Sealed Burner Cooktop](https://github.com/tslmy/agent/assets/594058/ef53719a-827a-4cdc-a8d1-1e3fcaa56488)

## Solution

Implement a tool that, when the last step in the chain of thought (CoT) deemed it isn't yet the right time to do something, spin off a thread to check the precondition periodically. Don't block the chat.

When the precondition is met, we resume the task. Append the task result to the chat history, as if the LLM had just said, "Hey, by the way, the sky cleared up, so I walked Fido and he's so happy."

## Demo

Ask the AI, **"Please go walk the dog."** It will say "It's raining; maybe later".

Continue the conversation by talking about something else. Perhaps **"how are you feeling right now"**. The chatbot will follow the flow.

Soon, the AI will attempt to walk the dog again, and sees that the sky has cleared up, so it will say, "I walked the dog, and he really enjoyed the park."

It's not just a UI trick. You can ask, **"Can you rephrase that?"**. The AI is aware of how the conversation has diverged.

## Future work

**The condition can become true as the conversation evolves.** (“Hey, did you just say Z is true? You know what, that actually implies that Y is true, so I’ll go ahead and do X now.“)

This means a traditional, static cron job won’t cut it. The AI has to somehow update the context of each desired action X that was held off.

**Humans know when to give up.** If the precondition turned out to be impossible to come true, remove X from its back burner.

- "Throw me a party when I marry Taylor Swift",
- "Remind me to [kill Sarah Connor](https://en.wikipedia.org/wiki/The_Terminator) when we get back to 1984",
- ...

“Dang it! Now that we realized that Y will never be the case, let’s forget about doing X for good.”

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

## Demo of its general-purpose tools

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
