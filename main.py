#!/usr/bin/env python
import logging

import chainlit as cl
from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import OpenAILike
from rich.logging import RichHandler

from tool_for_my_notes import make_tool as make_tool_for_my_notes

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
import phoenix as px

px.launch_app()

import llama_index

llama_index.set_global_handler("arize_phoenix")


def create_callback_manager(should_use_chainlit: bool = True):
    callback_handlers = [LlamaDebugHandler()]
    if should_use_chainlit:
        callback_handlers.append(cl.LlamaIndexCallbackHandler())
    return CallbackManager(callback_handlers)


from llama_index.agent import ReActAgent


def create_agent(
    should_use_chainlit: bool, should_override_system_prompt: bool = True
) -> ReActAgent:
    callback_manager = create_callback_manager(should_use_chainlit)

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
        callback_manager=callback_manager,
    )

    service_context = ServiceContext.from_defaults(
        # https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#local-embedding-models
        # HuggingFaceEmbedding requires transformers and PyTorch to be installed.
        # Run `pip install transformers torch`.
        embed_model="local",
        # https://docs.llamaindex.ai/en/stable/examples/llm/localai.html
        # But, instead of LocalAI, I'm using "LM Studio".
        llm=local_llm,
        # `ServiceContext.from_defaults` doesn't take callback manager from the LLM by default.
        callback_manager=callback_manager,
    )

    from llama_index.memory import ChatMemoryBuffer
    from llama_index.storage.chat_store import SimpleChatStore

    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(
        llm=local_llm,
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )

    from tool_for_backburner import make_tools as make_tools_for_backburner
    from tool_for_wikipedia import make_tool as make_tool_for_wikipedia

    all_tools = make_tools_for_backburner(service_context, chat_store=chat_store) + [
        make_tool_for_my_notes(service_context),
        make_tool_for_wikipedia(service_context),
    ]
    # TODO: When we have too many tools for the Agent to comprehend in one go (In other words, the sheer amounts of two
    #  descriptions has taken most of the context window.), try `custom_obj_retriever` in
    #  https://docs.llamaindex.ai/en/latest/examples/agent/multi_document_agents-v1.html.
    #  This will allow us to retrieve the tools, instead of having to hardcode them in the code.

    from llama_index.agent.react.formatter import ReActChatFormatter

    if should_override_system_prompt:
        # Override the default system prompt for ReAct chats.
        with open("system_prompt.md") as f:
            MY_SYSTEM_PROMPT = f.read()

        class MyReActChatFormatter(ReActChatFormatter):
            system_header = MY_SYSTEM_PROMPT

        chat_formatter = MyReActChatFormatter()
    else:
        chat_formatter = ReActChatFormatter()
    return ReActAgent.from_tools(
        tools=all_tools,
        llm=local_llm,
        verbose=True,
        react_chat_formatter=chat_formatter,
        callback_manager=callback_manager,
        memory=chat_memory,
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

    See https://docs.chainlit.io/integrations/llama-index.

    Usage:

    ```shell
    chainlit run llama-index.py -w
    ```
    """
    agent: ReActAgent = cl.user_session.get("agent")
    response = await cl.make_async(agent.chat)(message.content)
    response_message = cl.Message(content="")
    response_message.content = response.response
    await response_message.send()


if __name__ == "__main__":
    # If Python’s builtin readline module is previously loaded, elaborate line editing and history features will be available.

    # https://rich.readthedocs.io/en/stable/console.html#input
    from rich.console import Console

    console = Console()
    agent = create_agent(should_use_chainlit=False)
    agent.chat_repl()
