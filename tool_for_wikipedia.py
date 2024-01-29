"""
Largely based on https://docs.llamaindex.ai/en/stable/examples/tools/OnDemandLoaderTool.html.
"""

from llama_index.readers.wikipedia import WikipediaReader
from llama_index.tools.ondemand_loader_tool import OnDemandLoaderTool

reader = WikipediaReader()

tool = OnDemandLoaderTool.from_defaults(
    reader,
    name="look_up_wikipedia",
    description="Looks up information from Wikipedia pages. MUST provide `pages` and `query_str`.",
)
