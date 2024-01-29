"""
Largely based on https://docs.llamaindex.ai/en/stable/examples/tools/OnDemandLoaderTool.html.
"""

from llama_index import ServiceContext
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.tools.ondemand_loader_tool import OnDemandLoaderTool

reader = WikipediaReader()


def make_tool(service_context: ServiceContext):
    return OnDemandLoaderTool.from_defaults(
        reader,
        index_kwargs={"service_context": service_context},
        name="look_up_wikipedia",
        description="Looks up information from Wikipedia pages. MUST provide `pages` and `query_str`.",
    )
