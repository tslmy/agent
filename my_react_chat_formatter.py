# Override the default system prompt for ReAct chats.
import logging
from typing import List, Optional, Sequence

from llama_index.agent import ReActChatFormatter
from llama_index.agent.react.types import BaseReasoningStep, ObservationReasoningStep
from llama_index.core.llms.types import ChatMessage
from llama_index.tools import BaseTool

from tool_for_backburner import I_WILL_GET_BACK_TO_IT

with open("system_prompt.md") as f:
    MY_SYSTEM_PROMPT = f.read()

PROMPT_FOR_BACKBURNER = """
In the 2nd case, first use the `put_on_backburner` tool to put the task on your back burner, and then respond:
```
Thought: I've put the task on my back burner and should move on for now.
Answer: It's not the right time for it. I've put it on my back burner.
```

REMEMBER: You have to use the `put_on_backburner` tool before you can say `Answer: It's not the right time...`.
"""


class MyReActChatFormatter(ReActChatFormatter):
    system_header = MY_SYSTEM_PROMPT

    def format(
        self,
        tools: Sequence[BaseTool],
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        logger = logging.getLogger("MyReActChatFormatter.format")
        if current_reasoning is not None and len(current_reasoning) > 0:
            last_reasoning = current_reasoning[-1]
            if isinstance(last_reasoning, ObservationReasoningStep):
                logger.debug(
                    f"last_reasoning.observation: {last_reasoning.observation[:100]}..."
                )
                if last_reasoning.observation == I_WILL_GET_BACK_TO_IT:
                    self.system_header = MY_SYSTEM_PROMPT.replace(
                        "<x1>", PROMPT_FOR_BACKBURNER
                    )

                else:
                    self.system_header = MY_SYSTEM_PROMPT.replace("<x1>", "")
        messages = super().format(tools, chat_history, current_reasoning)
        messages[0].content = messages[0].content.replace("/*", "{").replace("*/", "}")
        return messages
