"""
Agent-themed idea No. 1 - “when time gets right”
------------------------------------------------
**Problem statement.** Humans have “back burners”, but AI doesn’t.
You can ask someone “shall we do X?”, and they may say “hmm, potentially?? Y has to come true first before we can do X.” And you keep on your conversation. When Y becomes true in a moment, you human friend will say, “hey, now the time is right, and I think we can get our hands on X now.”

With a AI, if you ask “shall we do X?” and it thinks “Y is a prerequisite“, then it would either utterly refuse you, or enter an infinite loop of checking whether Y has become true yet. In the first case, the AI won’t raise an eyebrow even when the time gets right. In the 2nd case, you are stuck waiting for the AI to respond.

**Proposal.** Implement a meta-tool that, if it deems the time isn’t right to do something X yet, schedule a periodical task in the background. The task should check for the precondition (“Is Y true now?”). When the time has become right, perform the action X that was held off, and mention it in the conversation (“hey btw, Y has just become true, so I went ahead and did X as you asked.”)

**Here’s the tricky part:**
Y can become true as the conversation evolves. (“Hey, did you just say Z is true? You know what, that actually implies that Y is true, so I’ll go ahead and do X now.“)
This means a traditional, static cron job won’t cut it. The AI has to somehow update the context of each desired action X that was held off.
This also means the AI needs to know when to give up and remove X from its back burner. (“Dang it! Now that we realized that Y will never be the case, let’s forget about doing X for good.”)
"""

import asyncio
import logging
from enum import Enum
from threading import Thread
from typing import List

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import BaseTool, FunctionTool, ToolMetadata
from llama_index.llms.openai_like import OpenAILike
from pydantic import BaseModel
from rich.logging import RichHandler

I_WILL_GET_BACK_TO_IT = "I'll keep an eye on it. I've put this on the back burner. Consider this done for now. I can now answer without using any more tools by just saying I'll get back to it later."

# https://rich.readthedocs.io/en/latest/logging.html#handle-exceptions
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)


class ConditionStatus(Enum):
    MET = 1
    NOT_YET_MET = 2
    WILL_NEVER_BE_MET = 3


from llama_index.core.agent import ReActAgent

is_raining = True


class EmptyToolSchema(BaseModel):
    pass


def create_agent_for_evaluating_conditions() -> ReActAgent:
    # ------------------- Define tools of this agent here -------------------

    def __tool_for_checking_the_weather(*args, **kwargs):
        """
        Checks the local weather. (Takes an empty `{}` as input.)
        """
        # A stub for checking the weather.
        #
        # In case you get confused between a stub and a mock, here's a quick reminder:
        # > Stub: Like a mock class, except that it doesn't provide the ability to verify that methods have been called/not called.
        # (via https://stackoverflow.com/q/346372/1147061)
        global is_raining
        if is_raining:
            is_raining = False
            return "It is raining outside."
        return "It is sunny outside."

    all_tools = [
        FunctionTool(
            __tool_for_checking_the_weather,
            metadata=ToolMetadata(
                name="check_weather",
                description=__tool_for_checking_the_weather.__doc__,
                fn_schema=EmptyToolSchema,
            ),
        )
    ]
    # -----------------------------------------------------------------------
    from llama_index.core.agent import ReActChatFormatter

    # Override the default system prompt for ReAct chats.
    with open("system_prompt_for_condition_evaluator.md") as f:
        MY_SYSTEM_PROMPT = f.read()

    class MyReActChatFormatter(ReActChatFormatter):
        system_header: str = MY_SYSTEM_PROMPT

    chat_formatter = MyReActChatFormatter()
    return ReActAgent.from_tools(
        tools=all_tools,
        verbose=True,
        react_chat_formatter=chat_formatter,
    )


def create_agent_for_performing_actions(
    container_for_tools_for_performing_actions: List[BaseTool],
) -> ReActAgent:
    # ------------------- Define tools of this agent here -------------------

    def __tool_for_walking_the_dog(*args, **kwargs):
        """
        Walks the dog. (Takes an empty `{}` as input.)
        """
        if is_raining:
            return "The dog doesn't want to go out in the rain. You should wait till it's sunny outside."
        return "The dog enjoyed the walk outside in the sunshine."

    container_for_tools_for_performing_actions.append(
        FunctionTool(
            __tool_for_walking_the_dog,
            metadata=ToolMetadata(
                name="walk_the_dog",
                description=__tool_for_walking_the_dog.__doc__,
                fn_schema=EmptyToolSchema,
            ),
        )
    )
    # -----------------------------------------------------------------------

    from my_react_chat_formatter import MyReActChatFormatter

    chat_formatter = MyReActChatFormatter()
    return ReActAgent.from_tools(
        tools=container_for_tools_for_performing_actions,
        verbose=True,
        react_chat_formatter=chat_formatter,
    )


def make_tools() -> List[BaseTool]:

    tools_for_performing_actions: List[BaseTool] = []

    class BackburnerPuttingToolSchema(BaseModel):
        condition: str
        action: str
        action_input: str

    def put_on_backburner(
        condition: str, action: str, action_input: str, *args, **kwargs
    ):
        """
        Put a task on the back burner. The action will be performed when the condition is met.

        Arguments:
        - condition: The condition that has to be met before the action can be performed.
        - action: Name of the tool you want to use.
        - action_input: The input to the tool, in order to fulfill your goal.
          (Refer to the schema of that tool itself for the exact JSON syntax to put here.)

        Example:
            If you want to walk the dog when the sky clears up, you can use the following input:
            {"condition": "sunny weather", "action": "walk_the_dog", "action_input": "{}"}
        """
        # Start the background task of `check_backburner`.
        wait_thread = Thread(
            target=check_backburner,
            args=(
                # TODO: Can we take a snapshot/checkpoint of the CoT, instead of passing the strings?
                #  We might want this because there might should have been more steps to follow in the CoT
                #  after this current back burner step.
                condition,
                action,
                action_input,
            ),
        )
        wait_thread.start()
        return I_WILL_GET_BACK_TO_IT

    agent_for_evaluating_conditions = create_agent_for_evaluating_conditions()

    class ConditionEvaluatingToolSchema(BaseModel):
        condition: str

    def evaluate_condition(condition: str, *args, **kwargs) -> ConditionStatus:
        """
        Evaluate the condition and return its status.

        Arguments:
        - condition: The condition to be evaluated.
        Example:
            {"condition": "weather is sunny"}
        """
        logger = logging.getLogger("evaluate_condition")
        logger.info(f"Evaluating the condition: {condition}")
        # Invoke the Agent to use whatever tool it needs in order to evaluate the condition.
        response = agent_for_evaluating_conditions.query(
            f"Evaluate this condition: {condition}"
        )
        response = response.response
        logger.info(f"Condition evaluation response: {response}")
        if "will never be met" in response:
            return ConditionStatus.WILL_NEVER_BE_MET
        elif "not yet met" in response:
            return ConditionStatus.NOT_YET_MET
        else:
            return ConditionStatus.MET

    agent_for_performing_actions = create_agent_for_performing_actions(
        container_for_tools_for_performing_actions=tools_for_performing_actions,
    )

    import chainlit as cl

    async def __perform_action(
        action: str,
        action_input: str,
    ):
        """
        Perform the action.

        Arguments:
        - action: The action to be performed.
        - action_input: The input to the action.

        Example:
            {"action": "walk_the_dog", "action_input": "{}"}
        """
        logger = logging.getLogger("perform_action")
        logger.info(
            f"Performing the hanged-off action `{action}` with input `{action_input}`."
        )
        response = agent_for_performing_actions.chat(
            # TODO: We should do this in a more structured way.
            #  Can we call the lower-level method that Agent invokes to invoke a tool?
            f"Perform the action `{action}` with input `{action_input}`."
        )
        response = response.response
        logger.info(f"Action performing response: {response}")
        # Add the response as a chat history.
        # TODO: This assumes that we are using ChainLit.
        agent: ReActAgent = cl.user_session.get("agent")
        agent.chat_history.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=response)
        )
        # Display this message on the web UI.
        response_message = cl.Message(content="")
        response_message.content = response
        await response_message.send()
        return response

    def check_backburner(condition: str, action: str, action_input: str):
        """
        Check if the conditions of any action on the back burner have become true. If so, perform the action and remove it from the back burner.
        """
        from time import sleep

        logger = logging.getLogger("check_backburner")
        num_check = 0
        while True:
            sleep(10)
            num_check += 1
            logger.info(
                f"Checking the condition of `{action}` for the {num_check}th time: Has `{condition}` been met yet?"
            )
            condition_status = evaluate_condition(condition)
            if condition_status == ConditionStatus.MET:
                logger.info(f"Time seems right for `{action}`! Performing it.")
                result = asyncio.run(
                    __perform_action(
                        action,
                        action_input,
                    )
                )
                logger.info(f"Result of performing the action: {result}")
                return
            elif condition_status == ConditionStatus.WILL_NEVER_BE_MET:
                logger.warning(
                    f"Yikes! `{condition}` will never be met. Removing `{action}` from the back burner."
                )
                return

    return [
        FunctionTool(
            evaluate_condition,
            metadata=ToolMetadata(
                name="evaluate_condition",
                description=evaluate_condition.__doc__,
                fn_schema=ConditionEvaluatingToolSchema,
            ),
        ),
        FunctionTool(
            put_on_backburner,
            metadata=ToolMetadata(
                name="put_on_backburner",
                description=put_on_backburner.__doc__,
                fn_schema=BackburnerPuttingToolSchema,
            ),
        ),
    ] + tools_for_performing_actions


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
        callback_manager=callback_manager,
    )
    # `ServiceContext.from_defaults` doesn't take callback manager from the LLM by default.
    # TODO: Check if this is still the case with `Settings` in 0.10.x.
    Settings.callback_manager = callback_manager
    # https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#local-embedding-models
    # HuggingFaceEmbedding requires transformers and PyTorch to be installed.
    # Run `pip install transformers torch`.
    Settings.embed_model = "local"

    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.storage.chat_store import SimpleChatStore

    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(
        llm=local_llm,
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )

    all_tools = make_tools()

    from my_react_chat_formatter import MyReActChatFormatter

    chat_formatter = MyReActChatFormatter()
    agent = ReActAgent.from_tools(
        tools=all_tools,
        llm=local_llm,
        verbose=True,
        react_chat_formatter=chat_formatter,
        memory=chat_memory,
    )
    result = agent.query("Walk the dog.")
    print(result)
    agent.chat_repl()
