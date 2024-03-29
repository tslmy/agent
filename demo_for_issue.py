from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(
    model="zephyr:7b-beta",
    timeout=600,  # secs
    streaming=True,
    additional_kwargs={"stop": ["Observation:"], "seed": 42, "temperature": 0.01},
)
Settings.embed_model = "local"

from tool_for_my_notes import make_tool as make_tool_for_my_notes
from tool_for_wikipedia import make_tool as make_tool_for_wikipedia

all_tools = [
    make_tool_for_my_notes(),
    make_tool_for_wikipedia(),
]
from my_react_chat_formatter import MyReActChatFormatter

chat_formatter = MyReActChatFormatter()

QUERY = "Name a type of drink that I enjoy, and then look up its country of origin. Be concise."

print(">>>>>>>> With stream_chat:")
agent = ReActAgent.from_tools(
    tools=all_tools,
    react_chat_formatter=chat_formatter,
)
response = agent.stream_chat(QUERY)
print(f">>>>>>>> Response: {response.response}")

agent = ReActAgent.from_tools(
    tools=all_tools,
    react_chat_formatter=chat_formatter,
)
print(">>>>>>>> With chat:")
response = agent.chat(QUERY)
print(f">>>>>>>> Response: {response.response}")
