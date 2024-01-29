"""
Largely based on https://docs.llamaindex.ai/en/stable/examples/tools/OnDemandLoaderTool.html.
"""

from llama_index.readers.wikipedia import WikipediaReader
from llama_index.tools.ondemand_loader_tool import OnDemandLoaderTool

reader = WikipediaReader()

tool = OnDemandLoaderTool.from_defaults(
    reader,
    name="Wikipedia Tool",
    description="A tool for loading and querying articles from Wikipedia",
)