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
from dataclasses import dataclass
from enum import Enum
from threading import Thread
from typing import List

from llama_index import ServiceContext
from llama_index.core.llms.types import ChatMessage
from llama_index.llms import OpenAILike
from llama_index.tools import BaseTool, FunctionTool, ToolMetadata
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)


@dataclass
class BackburnerEntry:
    condition: str
    action: str
    action_input: str
    messages_till_last_observation: List[ChatMessage]


class ConditionStatus(Enum):
    MET = 1
    NOT_YET_MET = 2
    WILL_NEVER_BE_MET = 3


from llama_index.agent import ReActAgent

is_raining = True


def __tool_for_checking_the_weather(*args, **kwargs):
    """
    A stub for checking the weather.

    In case you get confused between a stub and a mock, here's a quick reminder:
    > Stub: Like a mock class, except that it doesn't provide the ability to verify that methods have been called/not called.
    (via https://stackoverflow.com/q/346372/1147061)
    """
    global is_raining
    if is_raining:
        is_raining = False
        return "It is raining outside."
    return "It is sunny outside."


def create_agent_for_evaluating_conditions(
    service_context: ServiceContext,
) -> ReActAgent:
    all_tools = [
        FunctionTool(
            __tool_for_checking_the_weather,
            metadata=ToolMetadata(
                name="check_weather",
                description="Checks the weather.",
            ),
        )
    ]
    from llama_index.agent.react.formatter import ReActChatFormatter

    # Override the default system prompt for ReAct chats.
    with open("system_prompt_for_condition_evaluator.md") as f:
        MY_SYSTEM_PROMPT = f.read()

    class MyReActChatFormatter(ReActChatFormatter):
        system_header = MY_SYSTEM_PROMPT

    chat_formatter = MyReActChatFormatter()
    return ReActAgent.from_tools(
        tools=all_tools,
        llm=service_context.llm,
        verbose=True,
        react_chat_formatter=chat_formatter,
        callback_manager=service_context.callback_manager,
    )


def __tool_for_walking_the_dog(*args, **kwargs):
    if is_raining:
        return "The dog doesn't want to go out in the rain. Task failed. You should hold off doing this until it's sunny outside."
    return "The dog enjoyed the walk outside in the sunshine. Well done!"


tools_for_performing_actions = [
    FunctionTool(
        __tool_for_walking_the_dog,
        metadata=ToolMetadata(
            name="walk_the_dog",
            description="Walks the dog. Requires a sunny weather.",
        ),
    )
]


def create_agent_for_performing_actions(
    service_context: ServiceContext,
) -> ReActAgent:

    from llama_index.agent.react.formatter import ReActChatFormatter

    # Override the default system prompt for ReAct chats.
    with open("system_prompt.md") as f:
        MY_SYSTEM_PROMPT = f.read()

    class MyReActChatFormatter(ReActChatFormatter):
        system_header = MY_SYSTEM_PROMPT

    chat_formatter = MyReActChatFormatter()
    return ReActAgent.from_tools(
        tools=tools_for_performing_actions,
        llm=service_context.llm,
        verbose=True,
        react_chat_formatter=chat_formatter,
        callback_manager=service_context.callback_manager,
    )


threads = []


def make_tools(service_context: ServiceContext, chat_store=None) -> List[BaseTool]:
    class BackburnerPuttingToolSchema(BaseModel):
        condition: str
        action: str
        action_input: str

    async def put_on_backburner(condition: str, action: str, action_input: str):
        """
        When the time isn't yet right to do something, put it on the back burner by using this tool.

        Args:
        - condition: The condition that has to be true before the action can be performed.
        - action: The action that has to be performed.
        - action_input: The input to the action.
        """
        logger = logging.getLogger("put_on_backburner")
        messages_so_far = chat_store.get_messages("user1")
        logger.info(
            f"Putting the action `{action}` on the back burner. Messages so far: {messages_so_far}. Keys: {chat_store.get_keys()}"
        )
        if len(messages_so_far) > 2:
            # Remove the last Observation message and the subsequent "I think the time isn't right" message.
            messages_till_last_observation = messages_so_far[-2]
        else:
            logger.warning("Something isn't right.")
            messages_till_last_observation = messages_so_far
        entry = BackburnerEntry(
            condition, action, action_input, messages_till_last_observation
        )
        # # Start the background task of `__check_backburner`.
        # asyncio.run(check_backburner(entry))
        # # Kick off the background task right away (https://stackoverflow.com/a/70719307/1147061).
        # await asyncio.sleep(0)
        wait_thread = Thread(target=check_backburner, args=(entry,))
        wait_thread.start()
        threads.append(wait_thread)
        return "I'll keep an eye on it. It's been added to the back burner. Consider this done for now."

    agent_for_evaluating_conditions = create_agent_for_evaluating_conditions(
        service_context
    )

    class ConditionEvaluatingToolSchema(BaseModel):
        condition: str

    def evaluate_condition(condition: str) -> ConditionStatus:
        """
        Evaluate the condition and return its status.

        Args:
        - condition: The condition to be evaluated.

        Returns:
        - The status of the condition.
        """
        logger = logging.getLogger("evaluate_condition")
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

    agent_for_performing_actions = create_agent_for_performing_actions(service_context)

    def __perform_action(
        action: str,
        action_input: str,
    ):
        """
        Perform the action.

        Args:
        - action: The action to be performed.
        - action_input: The input to the action.
        """
        logger = logging.getLogger("perform_action")
        logger.info(
            f"Performing the hanged-off action `{action}` with input `{action_input}`."
        )
        response = agent_for_performing_actions.chat(
            # TODO: We should do this in a more structured way.
            f"Perform the action `{action}` with input `{action_input}`."
        )
        response = response.response
        logger.info(f"Action performing response: {response}")
        # TODO: How do we expose this to ChainLit?
        return response

    def check_backburner(entry: BackburnerEntry):
        """
        Check if the conditions of any action on the back burner have become true. If so, perform the action and remove it from the back burner.
        """
        from time import sleep

        logger = logging.getLogger("check_backburner")
        num_check = 0
        while True:
            num_check += 1
            logger.info(
                f"Checking the condition of `{entry.action}` for the {num_check}th time: Has `{entry.condition}` been met yet?"
            )
            condition_status = evaluate_condition(entry.condition)
            if condition_status == ConditionStatus.MET:
                logger.info(f"Time seems right for `{entry.action}`! Performing it.")
                result = __perform_action(
                    entry.action,
                    entry.action_input,
                )
                logger.info(f"Result of performing the action: {result}")
                return
            elif condition_status == ConditionStatus.WILL_NEVER_BE_MET:
                logger.warning(
                    f"Yikes! `{entry.condition}` will never be met. Removing `{entry.action}` from the back burner."
                )
                return
            sleep(1)

    return [
        FunctionTool(
            evaluate_condition,
            metadata=ToolMetadata(
                name="evaluate_condition",
                description="""Evaluates whether a given statement ("condition") has become true or not.""",
                fn_schema=ConditionEvaluatingToolSchema,
            ),
        ),
        FunctionTool(
            put_on_backburner,
            metadata=ToolMetadata(
                name="put_on_backburner",
                description="""If you think the time isn't right to do something, use this tool to put it on the back burner. Provide the exact condition that has to be true before the action can be performed, the action that has to be performed, and the input to the action.""",
                fn_schema=BackburnerPuttingToolSchema,
            ),
        ),
    ] + tools_for_performing_actions  # We also need to expose the action-performing tools to the top-level agent, so that it can know what preconditions each tool demands, as well as just perform the actions if the precondition is met already.


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

    from llama_index.memory import ChatMemoryBuffer
    from llama_index.storage.chat_store import SimpleChatStore

    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(
        llm=local_llm,
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )
    all_tools = make_tools(service_context, chat_store=chat_store)

    # Override the default system prompt for ReAct chats.
    with open("system_prompt.md") as f:
        MY_SYSTEM_PROMPT = f.read()

    from llama_index.agent.react.formatter import ReActChatFormatter

    class MyReActChatFormatter(ReActChatFormatter):
        system_header = MY_SYSTEM_PROMPT

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
