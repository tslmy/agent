import asyncio
import logging
import uuid
from threading import Thread
from typing import Any, List, Optional, Sequence, Union

import chainlit as cl
from langchain_core.utils import print_text
from llama_index.agent import ReActAgent, ReActAgentWorker, ReActChatFormatter
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.agent.types import Task, TaskStep, TaskStepOutput
from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.chat_engine.types import AGENT_CHAT_RESPONSE_TYPE, ChatResponseMode
from llama_index.core.llms.types import ChatMessage, MessageRole
from llama_index.llms import LLM
from llama_index.memory import BaseMemory
from llama_index.objects import ObjectRetriever
from llama_index.tools import BaseTool


class DeferringTaskStepOutput(TaskStepOutput):
    """
    The same as `TaskStepOutput` but with an additional `should_defer` field.
    """

    should_defer: bool = Field(
        default=False, description="Should the next step be deferred?"
    )
    defer_till_condition: str = Field(
        default="Continue immediately",
        description="Condition that must evaluate to true before the `next_steps` can be resumed.",
    )


class DeferrableReActAgent(ReActAgent):
    """
    An agent that can defer its reasoning steps.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        memory: BaseMemory,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        context: Optional[str] = None,
    ) -> None:
        super().__init__(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            memory=memory,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
            context=context,
        )
        # The only difference between this method and the original `ReActAgent.__init__` is that it uses `DeferrableReActAgentWorker` instead of `ReActAgentWorker`.
        self.agent_worker = DeferrableReActAgentWorker.from_tools(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
        )
        # Kick off a separate thread for running deferred tasks.
        # Since we are in an async context, we keep a separate event loop spinning in this thread.
        # Credit: https://gist.github.com/dmfigol/3e7d5b84a16d076df02baa9f53271058
        self.deferral_loop = asyncio.new_event_loop()

        def __start_background_loop() -> None:
            asyncio.set_event_loop(self.deferral_loop)
            self.deferral_loop.run_forever()

        deferral_thread = Thread(target=__start_background_loop, daemon=True)
        deferral_thread.start()

    async def _achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """
        This is the same with `BaseAgentRunner._achat` but delegates the loop to a separate method.
        If LlamaIndex modifies `BaseAgentRunner._achat`, also modify this.
        TODO: Submit this split as a PR to LlamaIndex.
        """
        if chat_history is not None:
            self.memory.set(chat_history)
        task = self.create_task(message)
        return await self._achat_from_task(task, tool_choice=tool_choice, mode=mode)

    async def _achat_from_task(
        self,
        task: Task,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """
        This is the same with `BaseAgentRunner._achat` but starts with a pre-composed `Task`, instead of creating its own `Task` by calling `create_task`.
        If LlamaIndex modifies `BaseAgentRunner._achat`, also modify this.
        TODO: Submit this split as a PR to LlamaIndex.

        The `_arun_step` may emit a `DeferringTaskStepOutput` back to this method.
        If its `should_defer` field is `True`, this method will:
        1. Spin off a coroutine that periodically checks the precondition. It achieves this by:
            1. calling `create_task` again. This assigns a new `task_id` to the spun-off task.
            2. calling a utility method that waits for 10 seconds before running `_achat_from_task` with the spun-off task.
        2. Return a response along the lines of:
            > "it's not yet the right time. I've put this on my back burner and will get back to it when the time is right.
            > For the time being, can I help you with something else?"
        """
        logger = logging.getLogger("_achat_from_task")
        result_output = None
        step_id = 0
        while True:
            # pass step queue in as argument, assume step executor is stateless
            logger.info(f"Running step {step_id} for task `{task.task_id}`.")
            cur_step_output = await self._arun_step(
                task.task_id, mode=mode, tool_choice=tool_choice, **kwargs
            )
            # `cur_step_output.output` may be:
            # - "Observation: The dog doesn't want to go out in the rain. You should wait till it's sunny outside."
            # - "It's not the right time for it. I've put it on my back burner."
            if cur_step_output.should_defer:
                # This block of `if` code is unique to this custom agent.
                spun_off_task = self.create_task(
                    task.input,
                    # Call `create_task` with `extra_state=task.extra_state`,
                    # so that the reasoning chain ("current_reasoning") can be retained.
                    extra_state=task.extra_state,
                )
                # Run the spun-off task after waiting for 10 seconds.
                promise_to_run_task_after_wait = self._achat_from_task_after_wait(
                    spun_off_task
                )
                # Fulfill this promise in the separate thread we prepared. It's dedicated for running deferred tasks.
                asyncio.run_coroutine_threadsafe(
                    promise_to_run_task_after_wait, self.deferral_loop
                )
                logger.info(
                    f"Added a task to the deferral loop. Task ID: `{spun_off_task.task_id}`."
                )
                # Declare the current chain of thought as done.
                break
            if cur_step_output.is_last:
                result_output = cur_step_output
                break
            # ensure tool_choice does not cause endless loops
            tool_choice = "auto"
            step_id += 1
        return self.finalize_response(task.task_id, result_output)

    async def _achat_from_task_after_wait(self, task: Task, delay_secs: int = 4):
        logger = logging.getLogger("_achat_from_task_after_wait")
        await asyncio.sleep(delay_secs)
        logger.info(f"Running a deferred task. Task ID: `{task.task_id}`.")
        response = await self._achat_from_task(task)
        # Since 1) this method will only be called to handle deferred tasks, and 2) the original response has been
        # finalized (by saying "I've put this on my back burner and will get back to it when the time is right."), we
        # have to explicitly emit this response back to the UI (be it the TUI or the web UI).
        response = response.response
        logger.info(f"Deferred task `{task.task_id}` responded with `{response}`.")
        self.chat_history.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=response)
        )
        if cl.user_session.get("agent") is not None:
            # Display this message on the web UI.
            response_message = cl.Message(content="")
            response_message.content = response
            await response_message.send()
        else:
            # TUI.
            print_text(response)


class DeferrableReActAgentWorker(ReActAgentWorker):
    """
    A variant of `ReActAgentWorker` that can tell whether the LLM thinks that the next step should be deferred.
    """

    def _get_task_step_response(
        self, agent_response: AGENT_CHAT_RESPONSE_TYPE, step: TaskStep, is_done: bool
    ) -> DeferringTaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    # NOTE: input is unused
                    input=None,
                )
            ]
        # The only difference between this method and the original `ReActAgentWorker._get_task_step_response` is that it returns `DeferringTaskStepOutput` instead of `TaskStepOutput`, and, intuitively, includes a `should_defer`.
        should_defer = "I've put it on my back burner" in agent_response.response
        # TODO: Implement the logic for `defer_till_condition`.
        #  Without it, we are just duly repeating the same task without making use of the condition-evaluating capability.
        return DeferringTaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
            should_defer=should_defer,
        )
