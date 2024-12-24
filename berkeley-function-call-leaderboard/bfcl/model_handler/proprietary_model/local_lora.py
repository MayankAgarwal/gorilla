import os
import re
import time
import json

from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.proprietary_model.openai import OpenAIHandler
from openai import OpenAI

from jinja2 import Template

SYSTEM_PROMPT = Template(
    """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.
You have access to the following tools:
<tools>{{ tools }}</tools>
The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>
""".strip()
)


class LocalLORA(OpenAIHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        self.is_fc_model = True

        base_url = os.getenv("BFCL_LORA_URL")
        assert base_url is not None, "'BFCL_LORA_URL' not set"

        self.client = OpenAI(base_url=base_url)
        self._output_re = re.compile(
            r"<tool_call>(.*)</tool_call>", re.IGNORECASE | re.DOTALL
        )

    #### FC methods ####

    def _query_FC(self, inference_data: dict):

        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT.render(tools=json.dumps(tools)),
        }
        message.insert(0, system_message)

        inference_data["inference_input_log"] = {"message": message, "tools": tools}

        start_time = time.time()
        api_response = self.client.chat.completions.create(
            messages=message,
            model=self.model_name,
            temperature=self.temperature,
        )
        end_time = time.time()

        return api_response, end_time - start_time

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        return super()._pre_query_processing_FC(inference_data, test_entry)

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        return super()._compile_tools(inference_data, test_entry)

    def _parse_query_response_FC(self, api_response: any) -> dict:

        tool_calls = []
        model_responses_raw: str = api_response.choices[0].message.content

        try:
            for tool_call_str in self._output_re.findall(model_responses_raw):
                tool_call_obj = json.loads(tool_call_str)
                if isinstance(tool_call_obj, list):
                    tool_calls.extend(
                        [{call["name"]: call["arguments"]} for call in tool_call_obj]
                    )
                elif isinstance(tool_call_obj, dict):
                    tool_calls.append(
                        {tool_call_obj["name"]: tool_call_obj["arguments"]}
                    )
        except Exception as err:
            try:
                for tool_call_obj in json.loads(model_responses_raw):
                    tool_calls.append(
                        {tool_call_obj["name"]: tool_call_obj["arguments"]}
                    )
            except:
                pass

        return {
            "model_responses": tool_calls,
            "model_responses_message_for_chat_history": model_responses_raw,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        return super().add_first_turn_message_FC(inference_data, first_turn_message)

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        return super()._add_next_turn_user_message_FC(inference_data, user_message)

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": model_response_data[
                    "model_responses_message_for_chat_history"
                ],
            }
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:

        for execution_result in execution_results:
            tool_message = {
                "role": "tool",
                "content": execution_result,
            }
            inference_data["message"].append(tool_message)

        return inference_data
