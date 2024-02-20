#!/usr/bin/env python
import logging

# TODO: Chainlit doesn't yet work with LlamaIndex 0.10.x. https://github.com/Chainlit/chainlit/issues/752
import chainlit as cl
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from rich.logging import RichHandler

# https://rich.readthedocs.io/en/latest/logging.html#handle-exceptions
logging.basicConfig(
    # level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger()

# https://rich.readthedocs.io/en/stable/traceback.html#traceback-handler
from rich.traceback import install

install(show_locals=True)

# "Phoenix can display in real time the traces automatically collected from your LlamaIndex application."
# https://docs.llamaindex.ai/en/stable/module_guides/observability/observability.html
# Or https://docs.arize.com/phoenix/integrations/llamaindex
import phoenix as px

px.launch_app()

from llama_index.core import set_global_handler

set_global_handler("arize_phoenix")


def create_callback_manager(should_use_chainlit: bool = True):
    callback_handlers = [LlamaDebugHandler()]
    if should_use_chainlit:
        callback_handlers.append(cl.LlamaIndexCallbackHandler())
    return CallbackManager(callback_handlers)


from llama_index.core.agent import ReActAgent


def create_agent(
    should_use_chainlit: bool,
    is_general_purpose: bool = True,
) -> ReActAgent:
    callback_manager = create_callback_manager(should_use_chainlit)
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama

    # https://docs.llamaindex.ai/en/stable/examples/llm/localai.html
    # But, instead of LocalAI, I'm using "LM Studio".
    Settings.llm = Ollama(
        model="zephyr:7b-beta",
        timeout=600,  # secs
        streaming=True,
        callback_manager=callback_manager,
        additional_kwargs={"stop": ["Observation:"]},
    )
    # `ServiceContext.from_defaults` doesn't take callback manager from the LLM by default.
    # TODO: Check if this is still the case with `Settings` in 0.10.x.
    Settings.callback_manager = callback_manager
    # https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#local-embedding-models
    # HuggingFaceEmbedding requires transformers and PyTorch to be installed.
    # Run `pip install transformers torch`.
    Settings.embed_model = "local"

    from tool_for_backburner import make_tools as make_tools_for_backburner
    from tool_for_my_notes import make_tool as make_tool_for_my_notes
    from tool_for_wikipedia import make_tool as make_tool_for_wikipedia

    all_tools = make_tools_for_backburner()
    if is_general_purpose:
        all_tools += [
            make_tool_for_my_notes(),
            make_tool_for_wikipedia(),
        ]
    # TODO: When we have too many tools for the Agent to comprehend in one go (In other words, the sheer amounts of two
    #  descriptions has taken most of the context window.), try `custom_obj_retriever` in
    #  https://docs.llamaindex.ai/en/latest/examples/agent/multi_document_agents-v1.html.
    #  This will allow us to retrieve the tools, instead of having to hardcode them in the code.

    from my_react_chat_formatter import MyReActChatFormatter

    chat_formatter = MyReActChatFormatter()
    return ReActAgent.from_tools(
        tools=all_tools,
        verbose=True,
        react_chat_formatter=chat_formatter,
        callback_manager=callback_manager,
    )


@cl.on_chat_start
async def factory():
    cl.user_session.set(
        "agent",
        create_agent(should_use_chainlit=True),
    )


@cl.on_message
async def main(message: cl.Message):
    """
    ChainLit provides a web GUI for this application.
    """
    agent: ReActAgent = cl.user_session.get("agent")
    response = agent.stream_chat(message.content)
    response_message = cl.Message(content="")
    for token in response.response_gen:
        await response_message.stream_token(token=token)
    if response.response:
        response_message.content = response.response
    await response_message.send()


if __name__ == "__main__":
    # If Python’s builtin readline module is previously loaded, elaborate line editing and history features will be available.

    # https://rich.readthedocs.io/en/stable/console.html#input
    from rich.console import Console

    console = Console()
    agent = create_agent(should_use_chainlit=False)
    agent.chat_repl()
