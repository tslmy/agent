# A Chatbot

![logo](public/logo_light.png)

_logo generated via MidJourney_

Features:

- **Capable**. This app uses a ReAct agent.
- **Private**. This app runs completely off-line. It never sends any data to the internet.

Architecture:

- [LM Studio](https://lmstudio.ai/) serves a LLM ([Zephyr 7B beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), as of Dec 2023) locally. No privacy concerns there -- The LLM is not fine-tuned with any private information, and the server is stateless. (Note: You can easily replace LM Studio with Ollama, etc., but I like the GUI that LM Studio provides.)
- [LlamaIndex](https://www.llamaindex.ai/) provides a natural-language querying interface. It also indexes documents into [embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture), which are stored into a [Chroma](https://www.trychroma.com/) database.
- [ChainLit](https://chainlit.io/) provides a web UI to LlamaIndex that looks like ChatGPT. (I tried building a TUI with [rich](https://github.com/Textualize/rich), but it's not very obvious how to surface the retrieved documents.)
