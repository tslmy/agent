#!/usr/bin/env python

import logging
import os

import chainlit as cl
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.ollama import OllamaEmbedding
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
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


tracer_provider = register(
    project_name="agent",
    endpoint="http://localhost:6006/v1/traces",
)


LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# ruff: noqa: E402
# Keep this here to ensure imports have environment available.
env_found = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))


def create_callback_manager(should_use_chainlit: bool = False) -> CallbackManager:
    debug_logger = logging.getLogger("debug")
    debug_logger.setLevel(logging.DEBUG)
    callback_handlers = [
        LlamaDebugHandler(logger=debug_logger),
    ]
    if should_use_chainlit:
        callback_handlers.append(cl.LlamaIndexCallbackHandler())
    return CallbackManager(callback_handlers)


def set_up_llama_index(
    should_use_chainlit: bool = False,
):
    """
    One-time setup code for shared objects across all AgentRunners.
    """
    # Needed for "Retrieved the following sources" to show up on Chainlit.
    Settings.callback_manager = create_callback_manager(should_use_chainlit)
    # ============= Beginning of the code block for wiring on to models. =============
    # At least when Chainlit is involved, LLM initializations must happen upon the `@cl.on_chat_start` event,
    # not in the global scope.
    # Otherwise, it messes up with Arize Phoenix: LLM calls won't be captured as parts of an Agent Step.
    if api_key := os.environ.get("OPENAI_API_KEY", None):
        logger.info("Using OpenAI API.")
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            is_function_calling_model=True,
            is_chat_model=True,
        )
    elif api_key := os.environ.get("TOGETHER_AI_API_KEY", None):
        logger.info("Using Together AI API.")
        from llama_index.llms.openai_like import OpenAILike

        Settings.llm = OpenAILike(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            api_base="https://api.together.xyz/v1",
            api_key=api_key,
            is_function_calling_model=True,
            is_chat_model=True,
        )
    else:
        logger.info("Using Ollama's OpenAI-compatible API.")
        from llama_index.llms.openai_like import OpenAILike

        Settings.llm = OpenAILike(
            model="llama3.1",
            api_base="http://localhost:11434/v1",
            # api_base="http://10.147.20.237:11434/v1",
            api_key="ollama",
            is_function_calling_model=True,
            is_chat_model=True,
        )

    Settings.embed_model = OllamaEmbedding(
        # https://ollama.com/library/nomic-embed-text
        model_name="nomic-embed-text",
        # Uncomment the following line to use the LLM server running on my gaming PC.
        # base_url="http://10.147.20.237:11434",
    )


set_up_llama_index()


def create_agent(
    is_general_purpose: bool = True,
) -> ReActAgent:
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
    )


@cl.on_chat_start
async def factory():
    cl.user_session.set(
        "agent",
        create_agent(),
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
    agent = create_agent()
    agent.chat_repl()
