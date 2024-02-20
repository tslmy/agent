"""
Largely based on https://docs.llamaindex.ai/en/stable/examples/tools/OnDemandLoaderTool.html.
"""

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.wikipedia import WikipediaReader

reader = WikipediaReader()


def make_tool():
    return OnDemandLoaderTool.from_defaults(
        reader,
        index_kwargs={"service_context": service_context},
        name="look_up_wikipedia",
        description="""Looks up information from Wikipedia pages.
For example, to answer "Who proposed general relativity?", you can use the following Action Input:
{"pages": ["General relativity"], "query_str": "Who proposed general relativity?"}
""",
    )


if __name__ == "__main__":
    callback_manager = CallbackManager([LlamaDebugHandler()])
    local_llm = OpenAILike(
        api_base="http://localhost:1234/v1",
        timeout=600,  # secs
        temperature=0.01,
        api_key="loremIpsum",
        # Honestly, this model name can be arbitrary.
        # I'm using this: https://huggingface.co/HuggingFaceH4/zephyr-7b-beta .
        model="zephyr beta 7B q5_k_m gguf",
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=32768,
    )
    # `ServiceContext.from_defaults` doesn't take callback manager from the LLM by default.
    # TODO: Check if this is still the case with `Settings` in 0.10.x.
    Settings.callback_manager = callback_manager
    # https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#local-embedding-models
    # HuggingFaceEmbedding requires transformers and PyTorch to be installed.
    # Run `pip install transformers torch`.
    Settings.embed_model = "local"
    tool = make_tool()
    result = tool.call(pages=["Coffee"], query_str="Which country first drink coffee?")
    print("Using just the tool itself:", result)
    agent = ReActAgent.from_tools(
        tools=[tool],
        llm=local_llm,
        verbose=True,
    )
    result = agent.query("Which country first drink coffee?")
    print("Using the tool via an agent:", result)
