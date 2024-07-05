import json

from model_handler.model_style import ModelStyle
from model_handler.oss_handler import OSSHandler
from model_handler.constant import GORILLA_TO_OPENAPI
from model_handler.utils import (
    language_specific_pre_processing,
    convert_to_tool,
)


class GraniteHandler(OSSHandler):
    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)

    def _format_prompt(prompt, function, test_category):
        prompt_str = (
            "SYSTEM: You are a helpful assistant with access to the following function calls. "
            "Your task is to produce a sequence of function calls necessary to generate response to the user utterance. "
            "Use the following function calls as required."
            "\n<|function_call_library|>\n{functions_str}\n"
            'If none of the functions are relevant or the given question lacks the parameters required by the function, please output "<function_call> {"name": "no_function", "arguments": {}}".\n\n'
            "USER: {query}\nASSISTANT: "
        )

        functions = language_specific_pre_processing(functions, test_category, False)
        functions = convert_to_tool(
            functions,
            GORILLA_TO_OPENAPI,
            model_style=ModelStyle.OSSMODEL,
            test_category=test_category,
            stringify_parameters=True,
        )

        functions_str = "\n".join([json.dumps(func) for func in function])

        prompt = prompt_str.format(functions_str=functions_str, query=prompt)
        return prompt

    def __decode_output__(self, result, language="Python"):
        decoded_outputs = []
        result = [
            call.strip()
            for call in result.split("<function_call>")
            if len(call.strip()) > 0
        ]

        for res in result:
            try:
                res = json.loads(res.strip())
            except:
                decoded_outputs.append(res)
            else:
                fnname = res.get("name", "").strip()
                args = res.get("arguments", {})

                if fnname == "no_function":
                    decoded_outputs.append("No function is called")
                    continue

                if language != "Python":
                    args = {k: str(v) for k, v in args.items()}

                decoded_outputs.appedn({fnname: args})

        return decoded_outputs

    def decode_ast(self, result, language="Python"):
        return self.__decode_output__(result, language=language)

    def decode_execute(self, result):
        return self.__decode_output__(result)
