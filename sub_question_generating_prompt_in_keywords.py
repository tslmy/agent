"""
This is a modified version of the default sub-question generating prompt in
https://github.com/run-llama/llama_index/blob/7aa9e8b7b94d054ef725f441751ede957847a89e/llama_index/question_gen/prompts.py,

which, instead of using full sentences, uses keywords to generate sub-questions.
"""

import json

from llama_index.question_gen.prompts import (
    SUFFIX,
    example_query_str,
    example_tools_str,
)
from llama_index.question_gen.types import SubQuestion

PREFIX = """\
Given a user question and a list of tools, output a few relevant sub-queries (in a Markdown code fence of JSON syntax) \
that, when composed, can help answer the given user question:

"""
example_output = [
    SubQuestion(sub_question="Uber revenue growth", tool_name="uber_10k"),
    SubQuestion(sub_question="Uber EBITDA", tool_name="uber_10k"),
    SubQuestion(sub_question="Lyft revenue growth", tool_name="lyft_10k"),
    SubQuestion(sub_question="Lyft EBITDA", tool_name="lyft_10k"),
]
example_output_str = json.dumps({"items": [x.dict() for x in example_output]}, indent=4)

EXAMPLES = f"""\
# Example 1
<Tools>
```json
{example_tools_str}
```

<User Question>
{example_query_str}


<Output>
```json
{example_output_str}
```

""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)

SUB_QUESTION_PROMPT_TEMPLATE_WITH_KEYWORDS = PREFIX + EXAMPLES + SUFFIX
