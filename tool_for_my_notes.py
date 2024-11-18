"""
Make a tool for accessing my personal notes, stored in a directory of text files.
"""

import logging
from typing import Optional

from chromadb.errors import InvalidCollectionException
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.tools import BaseTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel

PATH_TO_NOTES = "demo-notes"
SHOULD_IGNORE_PERSISTED_INDEX = False


def __create_index(
    input_dir: str,
    storage_context: Optional[StorageContext] = None,
) -> VectorStoreIndex:
    """
    Creates an index from a directory of documents.
    """
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
        storage_context=storage_context,
        show_progress=True,
    )


def make_tool() -> BaseTool:
    """
    Creates a tool for accessing my private information, or anything about me.
    These can be my notes, my calendar, my emails, etc.
    """
    # An index is a lightweight view to the database.
    notes_index = __get_index()
    notes_query_engine = notes_index.as_query_engine(
        similarity_top_k=5,
        # For a query engine hidden inside an Agent, streaming really doesn't make sense.
        # https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/streaming.html#streaming
        streaming=False,
    )
    # Convert it to a tool.
    from llama_index.core.tools import QueryEngineTool, ToolMetadata

    class NotesQueryingToolSchema(BaseModel):
        input: str

    notes_query_engine_tool = QueryEngineTool(
        query_engine=notes_query_engine,
        metadata=ToolMetadata(
            name="look_up_notes",
            description="""Search the user's notes about a particular keyword.
                Input should be the keyword that you want to search the user's notes with.""",
            fn_schema=NotesQueryingToolSchema,
        ),
    )
    # Sub Question Query Engine: breaks down the user's question into sub questions.
    # https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine.html
    from llama_index.core.query_engine import SubQuestionQueryEngine
    from llama_index.core.question_gen import LLMQuestionGenerator

    from sub_question_generating_prompt_in_keywords import (
        SUB_QUESTION_PROMPT_TEMPLATE_WITH_KEYWORDS,
    )

    sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[notes_query_engine_tool],
        question_gen=LLMQuestionGenerator.from_defaults(
            prompt_template_str=SUB_QUESTION_PROMPT_TEMPLATE_WITH_KEYWORDS,
        ),
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
            "opinions about X", "food that I enjoy", "my financial standing", etc. """,
            fn_schema=AboutTheUserToolSchema,
        ),
    )
    return sub_question_query_engine_tool


def __get_index():
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
    except InvalidCollectionException:
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
        storage_context=storage_context,
    )
