import os
import json
import subprocess
import threading
import requests
import time

import numpy as np

from transformers import AutoTokenizer


_SYSTEM_PROMPT = (
    "You are provided with a user query, "
    "a catalog of tools available to fulfill that user query, and "
    "a list of tool calls that use tools available in the catalog to fulfill user request.\n"
    "Your job is to assess whether the tool calls adequately fulfill the user request or not.\n\n"
    "You have the following tools available:\n\n{AVAILABLE_TOOLS}\n\n"
)


class RewardModelHandler:
    def __init__(self, vllm_rm_host: str, vllm_rm_port: int, dtype: str = "bfloat16"):
        self.vllm_rm_host = vllm_rm_host
        self.vllm_rm_port = vllm_rm_port
        self.rm_base_url = f"http://{self.vllm_rm_host}:{self.vllm_rm_port}"
        self.rm_model_name_or_path = None
        self.tokenizer = None

        self.dtype = dtype

    def launch_server(
        self,
        rm_gpu_ids: list[str],
        rm_model_name_or_path: str,
        gpu_memory_utilization: float,
    ):

        self.rm_model_name_or_path = rm_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.rm_model_name_or_path)

        ngpus = len(rm_gpu_ids)
        cuda_visible_devices = ",".join(rm_gpu_ids)
        pooler_config = {"normalize": False, "softmax": False}
        pooler_config = json.dumps(pooler_config)

        env_copy = os.environ.copy()
        env_copy["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        process = subprocess.Popen(
            [
                "vllm",
                "serve",
                str(self.rm_model_name_or_path),
                "--port",
                str(self.vllm_rm_port),
                "--dtype",
                self.dtype,
                "--tensor-parallel-size",
                str(ngpus),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--trust-remote-code",
                "--override-pooler-config",
                str(pooler_config),
                "--task",
                "classify",
            ],
            env=env_copy,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stop_event = threading.Event()

        def log_subprocess_output(pipe, stop_event):
            for line in iter(pipe.readline, ""):
                if stop_event.is_set():
                    break
                else:
                    print(line, end="")
            pipe.close()
            print("rm server log tracking thread stopped successfully.")

        stdout_thread = threading.Thread(
            target=log_subprocess_output, args=(process.stdout, stop_event)
        )
        stderr_thread = threading.Thread(
            target=log_subprocess_output, args=(process.stderr, stop_event)
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            # Wait for the server to be ready
            server_ready = False
            while not server_ready:
                # check if the process has terminated unexpectedly
                if process.poll() is not None:
                    # output the captured logs
                    stdout, stderr = process.communicate()
                    print(stdout)
                    print(stderr)
                    raise Exception(
                        f"RM subprocess terminated unexpectedly with code {process.returncode}"
                    )

                try:
                    response = requests.get(f"{self.rm_base_url}/v1/models")
                    if response.status_code == 200:
                        server_ready = True
                        print(f"RM server is ready!")
                except requests.exceptions.ConnectionError:
                    # If the connection is not ready, wait and try again
                    time.sleep(1)

            # signal threads to stop reading output
            stop_event.set()

        except Exception as e:
            raise e

        finally:
            # Wait for the output threads to finish
            stop_event.set()

            return process, stdout_thread, stderr_thread

    def __format_json__(self, jsondata):
        result_str = ""

        if len(jsondata) == 0:
            result_str = "```json\n[]\n```"

        elif isinstance(jsondata, str):
            result_str = "```json\n[\n\t" + jsondata + "\n]\n```"

        elif isinstance(jsondata, list):
            tool_call_str = ""
            for idx, fn in enumerate(jsondata):
                tool_call_str += "\n\t" + json.dumps(fn)
                if idx < len(jsondata) - 1:
                    tool_call_str += ","
            tool_call_str = "```json\n[" + tool_call_str + "\n]\n```"
            result_str = tool_call_str

        elif isinstance(jsondata, dict):
            tool_call_str = "```json\n[\n\t" + json.dumps(jsondata) + "\n]\n```"
            result_str = tool_call_str

        return result_str

    def __make_prompt__(
        self, functions: list[dict], conversations: list[dict], tool_calls: list[dict]
    ):

        # remove system prompt from conversations if already exists from function calling task
        sys_idx = None
        for idx, conv in enumerate(conversations):
            if conv["role"] == "system":
                sys_idx = idx
                break

        if sys_idx is not None:
            conversations.pop(sys_idx)

        system_prompt = _SYSTEM_PROMPT.replace(
            "{AVAILABLE_TOOLS}", self.__format_json__(functions)
        )

        rm_prompts = []

        for tool_call_generation in tool_calls:
            rm_conversation = (
                [{"role": "system", "content": system_prompt}]
                + conversations
                + [
                    {
                        "role": "assistant",
                        "content": self.__format_json__(tool_call_generation),
                    }
                ]
            )

            rm_prompts.append(
                self.tokenizer.apply_chat_template(rm_conversation, tokenize=False)
            )

        return rm_prompts

    def rank_generations(
        self,
        functions: list[dict],
        conversations: list[dict],
        generated_tool_calls: list[dict],
    ) -> tuple[list[int], list[float]]:

        rm_prompts = self.__make_prompt__(
            functions=functions,
            conversations=conversations,
            tool_calls=generated_tool_calls,
        )

        # TODO: Only do prediction for unique prompts

        headers = {"Content-Type": "application/json"}
        payload = {"model": self.rm_model_name_or_path, "input": rm_prompts}
        payload = json.dumps(payload)

        try:
            response = requests.request(
                "POST", f"{self.rm_base_url}/classify", headers=headers, data=payload
            )
            response.raise_for_status()

            response_json = response.json()
            # rm_scores = [data["data"][-1][0] for data in response_json["data"]]
            rm_scores = [data["probs"][0] for data in response_json["data"]]

            idxs_sorted = np.argsort(rm_scores)[::-1]
            return idxs_sorted, rm_scores

        except Exception as err:
            print(f"Error while ranking generations: {err}")
            raise err
