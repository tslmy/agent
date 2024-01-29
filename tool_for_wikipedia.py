"""
Largely based on https://docs.llamaindex.ai/en/stable/examples/tools/OnDemandLoaderTool.html.
"""

from llama_index import ServiceContext, download_loader
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAILike
from llama_index.tools.ondemand_loader_tool import OnDemandLoaderTool

WikipediaReader = download_loader("WikipediaReader")

reader = WikipediaReader()


def make_tool(service_context: ServiceContext):
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

    service_context = ServiceContext.from_defaults(
        # https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#local-embedding-models
        # HuggingFaceEmbedding requires transformers and PyTorch to be installed.
        # Run `pip install transformers torch`.
        embed_model="local",
        # https://docs.llamaindex.ai/en/stable/examples/llm/localai.html
        # But, instead of LocalAI, I'm using "LM Studio".
        llm=local_llm,
    )
    tool = make_tool(service_context)
    result = tool.call(pages=["Coffee"], query_str="Which country first drink coffee?")
    print("Using just the tool itself:", result)
    agent = ReActAgent.from_tools(
        tools=[tool],
        llm=local_llm,
        verbose=True,
    )
    result = agent.query("Which country first drink coffee?")
    print("Using the tool via an agent:", result)
