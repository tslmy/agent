from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

callback_handlers = [LlamaDebugHandler()]
callback_manager = CallbackManager(callback_handlers)
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(
    model="zephyr:7b-beta",
    timeout=600,  # secs
    streaming=True,
    callback_manager=callback_manager,
    additional_kwargs={"stop": ["Observation:"], "seed": 42},
)
Settings.callback_manager = callback_manager
Settings.embed_model = "local"

from tool_for_my_notes import make_tool as make_tool_for_my_notes
from tool_for_wikipedia import make_tool as make_tool_for_wikipedia

all_tools = [
    make_tool_for_my_notes(),
    make_tool_for_wikipedia(),
]
from my_react_chat_formatter import MyReActChatFormatter

chat_formatter = MyReActChatFormatter()
agent = ReActAgent.from_tools(
    tools=all_tools,
    verbose=True,
    react_chat_formatter=chat_formatter,
    callback_manager=callback_manager,
)
QUERY = "Name a type of drink that I enjoy, and then look up its country of origin. Be concise."
print("With stream_chat:")
agent.stream_chat(QUERY)
print("With chat:")
agent.chat(QUERY)
