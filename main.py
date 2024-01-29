#!/usr/bin/env python
import logging
from typing import Optional

import chainlit as cl
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import OpenAILike
from llama_index.vector_stores import ChromaVectorStore
from pydantic import BaseModel

PATH_TO_NOTES = "demo-notes"
SHOULD_IGNORE_PERSISTED_INDEX = False

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
import phoenix as px

px.launch_app()

import llama_index

llama_index.set_global_handler("arize_phoenix")


def __create_index(
    input_dir: str,
    storage_context: Optional[StorageContext] = None,
    service_context: Optional[ServiceContext] = None,
) -> VectorStoreIndex:
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        # https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader.html#reading-from-subdirectories
        recursive=True,
        # https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader.html#restricting-the-files-loaded
        # Before including image files here, `mamba install pillow`.
        # Before including audio files here, `pip install openai-whisper`.
        required_exts=[".md", ".txt"],
    ).load_data()
    return VectorStoreIndex.from_documents(
        # https://docs.llamaindex.ai/en/stable/api_reference/indices/vector_store.html#llama_index.indices.vector_store.base.VectorStoreIndex.from_documents
        documents=documents,
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True,
    )


def __get_index(service_context: ServiceContext):
    logger = logging.getLogger("__get_index")
    # https://www.trychroma.com/
    import chromadb
    from chromadb.config import Settings

    db = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(
            # https://docs.trychroma.com/telemetry#opting-out
            anonymized_telemetry=False
        ),
    )
    try:
        chroma_collection = db.get_collection("notes")
    except ValueError:
        logger.info("The Chrome DB collection does not exist. Creating.")
        should_create_index = True
    else:
        logger.info("Storage exists.")
        if SHOULD_IGNORE_PERSISTED_INDEX:
            db.delete_collection("notes")
            logger.info(
                "But it's requested to ignore the persisted data. We'll delete the Chrome DB collection."
            )
            should_create_index = True
        else:
            logger.info("We'll load from the Chrome DB collection.")
            should_create_index = False

    if should_create_index:
        chroma_collection = db.create_collection("notes")
        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return __create_index(
            input_dir=PATH_TO_NOTES,
            storage_context=storage_context,
            service_context=service_context,
        )
        # If we are using file-based storage, we would have to call `persist` manually:
        # index.storage_context.persist(persist_dir=STORAGE_DIR)
        # But this doesn't apply to DBs like Chroma.
    # else, load the existing index.
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,
        storage_context=storage_context,
    )


def create_callback_manager(should_use_chainlit: bool = True):
    callback_handlers = [LlamaDebugHandler()]
    if should_use_chainlit:
        callback_handlers.append(cl.LlamaIndexCallbackHandler())
    return CallbackManager(callback_handlers)


def __create_tool_for_learning_about_me(service_context):
    """
    Creates a tool for accessing my private information, or anything about me.
    These can be my notes, my calendar, my emails, etc.
    """
    # An index is a lightweight view to the database.
    notes_index = __get_index(service_context)
    notes_query_engine = notes_index.as_query_engine(
        service_context=service_context,
        similarity_top_k=5,
        # For a query engine hidden inside an Agent, streaming really doesn't make sense.
        # https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/streaming.html#streaming
        streaming=False,
    )
    # Convert it to a tool.
    from llama_index.tools import ToolMetadata
    from llama_index.tools.query_engine import QueryEngineTool

    class NotesQueryingToolSchema(BaseModel):
        input: str

    notes_query_engine_tool = QueryEngineTool(
        query_engine=notes_query_engine,
        metadata=ToolMetadata(
            name="look_up_notes",
            description="""Search the user's notes about a particular keyword.
                Input should be the keyword that you want to search the user's notes with.
                Since the search is implemented using a vector similarity search,
                the more the input is like a piece of note itself, the better the results will be.""",
            fn_schema=NotesQueryingToolSchema,
        ),
    )
    # Sub Question Query Engine: breaks down the user's question into sub questions.
    # https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine.html
    from llama_index.query_engine import SubQuestionQueryEngine
    from llama_index.question_gen.llm_generators import LLMQuestionGenerator

    sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[notes_query_engine_tool],
        question_gen=LLMQuestionGenerator.from_defaults(
            service_context=service_context
        ),
        service_context=service_context,
        verbose=True,
    )
    # Convert it to a tool.

    class AboutTheUserToolSchema(BaseModel):
        input: str

    sub_question_query_engine_tool = QueryEngineTool(
        query_engine=sub_question_query_engine,
        metadata=ToolMetadata(
            name="about_the_user",
            description="""Provides information about the user themselves, including the user's opinions on a given topic.
            Input should be the topic about which you want to learn about the user. For example, you can ask:
            - "What does the user think about X?"
            - "Where has the user been between 12/01/2023 and 12/31/2023?"
            - "How is the user doing in terms of finances?" """,
            fn_schema=AboutTheUserToolSchema,
        ),
    )
    return sub_question_query_engine_tool


from llama_index.agent import ReActAgent


def create_agent(
    should_use_chainlit: bool, should_override_system_prompt: bool = True
) -> ReActAgent:
    callback_manager = create_callback_manager(should_use_chainlit)

    local_llm = OpenAILike(
        api_base="http://localhost:1234/v1",
        timeout=600,  # secs
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
    # `pip install llama-hub`
    from llama_hub.tools.wikipedia import WikipediaToolSpec

    tool_spec = WikipediaToolSpec()
    wikipedia_tools = tool_spec.to_tool_list()
    all_tools = [__create_tool_for_learning_about_me(service_context)] + wikipedia_tools
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
    )


@cl.on_chat_start
async def factory():
    cl.user_session.set("agent", create_agent(should_use_chainlit=True))


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
