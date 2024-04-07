import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend

import torch


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_stdout_stderr():
    # Try to import the transformers package
    try:
        import transformers
        from transformers import pipeline
        from transformers.pipelines.conversational import (
            Conversation,
            ConversationalPipeline,
        )
    except ImportError:
        is_transformers_available = False
    else:
        is_transformers_available = True


@register_backend
class TransformersConversational(IntelligenceBackend):
    """Interface to the Transformers ConversationalPipeline."""

    stateful = False
    type_name = "transformers:conversational"

    def __init__(self, model: str, tokenizer=None, device: int | str = -1, **kwargs):
        super().__init__(model=model, device=device, **kwargs)
        self.model = model
        self.device = device

        assert is_transformers_available, "Transformers package is not installed"
        self.chatbot = pipeline(
            task="conversational", model=self.model, tokenizer=tokenizer, device=self.device
        )

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, conversation):
        conversation = self.chatbot(conversation)
        response = conversation.generated_responses[-1]
        return response

    @staticmethod
    def _msg_template(agent_name, content):
        return f"[{agent_name}]: {content}"

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        user_inputs, generated_responses = [], []
        if global_prompt:  # Prepend the global prompt if it exists
            # system_prompt = f"You are a helpful assistant.\n{global_prompt.strip()}\n{BASE_PROMPT}\n\nYour name is {agent_name}.\n\nYour role:{role_desc}"
            system_prompt = f"{global_prompt.strip()}\n\nYour name is {agent_name}.\n\nYour role:{role_desc}"
        else:
            # system_prompt = f"You are a helpful assistant. Your name is {agent_name}.\n\nYour role:{role_desc}\n\n{BASE_PROMPT}"
            system_prompt = f"Your name is {agent_name}.\n\nYour role:{role_desc}"
        # 'You are a helpful assistant.\nYou are in a university classroom and it is Natural Language Processing module. You start by introducing themselves.\nThe messages always end with the token <EOS>.\n\nYour name is Professor.\n\nYour role:You are Prof. Alpha, a knowledgeable professor in NLP. Your answer will concise and accurate. The answers should be less than 100 words.'

        all_messages = [(SYSTEM, system_prompt)]

        for msg in history_messages:
            all_messages.append((msg.agent_name, msg.content))
        if request_msg:
            all_messages.append((SYSTEM, request_msg.content))

        prev_is_user = False  # Whether the previous message is from the user
        for i, message in enumerate(all_messages):
            if i == 0:
                assert (
                    message[0] == SYSTEM
                )  # The first message should be from the system

            if message[0] != agent_name:
                if not prev_is_user:
                    user_inputs.append(self._msg_template(message[0], message[1]))
                else:
                    user_inputs[-1] += "\n" + self._msg_template(message[0], message[1])
                prev_is_user = True
            else:
                if prev_is_user:
                    generated_responses.append(message[1])
                else:
                    generated_responses[-1] += "\n" + message[1]
                prev_is_user = False

        assert len(user_inputs) == len(generated_responses) + 1
        past_user_inputs = user_inputs[:-1]
        new_user_input = user_inputs[-1]

        # Recreate a conversation object from the history messages
        conversation = Conversation(
            text=new_user_input,
            past_user_inputs=past_user_inputs,
            generated_responses=generated_responses,
        )

        # Get the response
        response = self._get_response(conversation)
        return response


# conversation = Conversation("Going to the movies tonight - any suggestions?")
#
# # Steps usually performed by the model when generating a response:
# # 1. Mark the user input as processed (moved to the history)
# conversation.mark_processed()
# # 2. Append a mode response
# conversation.append_response("The Big lebowski.")
#
# conversation.add_user_input("Is it good?")

# class LlaMaChat(TransformersConversational):
#     def __init__(self, device, **kwargs):
#         from transformers import LlamaForCausalLM, LlamaTokenizer
#         # TODO complete
#         model = LlamaForCausalLM.from_pretrained(...)
#         tokenizer = LlamaTokenizer.from_pretrained(...)
#         super().__init__(model=model, device=device, **kwargs)


class MistralChat(TransformersConversational):
    def __init__(self, device, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1',
                                                     torch_dtype=torch.float16,
                                                     # attn_implementation='flash_attention_2'
                                                     )
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
        super().__init__(model=model, tokenizer=tokenizer, device=device, **kwargs)
