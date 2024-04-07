from typing import List, Union
import re
from tenacity import RetryError
import logging
import uuid
from abc import abstractmethod
import asyncio

from .backends import IntelligenceBackend, load_backend
from .message import Message, SYSTEM_NAME
from .config import AgentConfig, Configurable, BackendConfig
from .standard_prompts import *

from datetime import datetime
import os
# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class Agent(Configurable):
    """
        An abstract base class for all the agents in the chatArena environment.
    """
    @abstractmethod
    def __init__(self, name: str, role_desc: str, global_prompt: str = None, *args, **kwargs):
        """
        Initialize the agent.

        Parameters:
            name (str): The name of the agent.
            role_desc (str): Description of the agent's role.
            global_prompt (str): A universal prompt that applies to all agents. Defaults to None.
        """
        super().__init__(name=name, role_desc=role_desc, global_prompt=global_prompt, **kwargs)
        self.name = name
        self.role_desc = role_desc
        self.global_prompt = global_prompt


class Player(Agent):
    """
    The Player class represents a player in the chatArena environment. A player can observe the environment
    and perform an action (generate a response) based on the observation.
    """

    def __init__(self, name: str, role_desc: str, backend: Union[BackendConfig, IntelligenceBackend],
                 global_prompt: str = None, **kwargs):
        """
        Initialize the player with a name, role description, backend, and a global prompt.

        Parameters:
            name (str): The name of the player.
            role_desc (str): Description of the player's role.
            backend (Union[BackendConfig, IntelligenceBackend]): The backend that will be used for decision making. It can be either a LLM backend or a Human backend.
            global_prompt (str): A universal prompt that applies to all players. Defaults to None.
        """

        if isinstance(backend, BackendConfig):
            backend_config = backend
            backend = load_backend(backend_config)
        elif isinstance(backend, IntelligenceBackend):
            backend_config = backend.to_config()
        else:
            raise ValueError(f"backend must be a BackendConfig or an IntelligenceBackend, but got {type(backend)}")

        assert name != SYSTEM_NAME, f"Player name cannot be {SYSTEM_NAME}, which is reserved for the system."

        # Register the fields in the _config
        self.scene_num = 1
        super().__init__(name=name, role_desc=role_desc, backend=backend_config,
                         global_prompt=global_prompt, **kwargs)

        self.backend = backend

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            global_prompt=self.global_prompt,
        )

    def act(self, observation: List[Message], stage) -> str:
        """
        Take an action based on the observation (Generate a response), which can later be parsed to actual actions that affect the game dyanmics.

        Parameters:
            observation (List[Message]): The messages that the player has observed from the environment.

        Returns:
            str: The action (response) of the player.
        """
        if self.name == "Designer":
            action_prompt = DESIGNER_ACTION_PROMPT.format(self.scene_num)
        elif self.name == "Global designer":
            action_prompt = GLOBAL_DESIGNER_ACTION_PROMPT
        elif self.name == 'Summarizer':
            action_prompt = SUMMARIZER_ACTION_PROMPT
        elif self.name == "Controller":
            action_prompt = CONTROLLER_ACTION_PROMPT
        elif self.name == "Reader":
            action_prompt = READER_ACTION_PROMPT
        elif stage == 'player_init':
            # role init
            action_prompt = ROLE_INIT_ACTION_PROMPT
        else:
            # role act
            action_prompt = ROLE_ACT_ACTION_PROMPT
        
        try:
            response = self.backend.query(agent_name=self.name, role_desc=self.role_desc,
                                          history_messages=observation, global_prompt=self.global_prompt,
                                          request_msg=None, action_prompt=action_prompt)
        except RetryError as e:
            err_msg = f"Agent {self.name} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
            logging.warning(err_msg)
            response = SIGNAL_END_OF_CONVERSATION + err_msg
        self.scene_num += 1
        return response

    def __call__(self, *args, **kwargs) -> str:
        return self.act(*args, **kwargs)

    async def async_act(self, observation: List[Message]) -> str:
        """
        Async version of act(). This is used when you want to generate a response asynchronously.

        Parameters:
            observation (List[Message]): The messages that the player has observed from the environment.

        Returns:
            str: The action (response) of the player.
        """
        try:
            response = self.backend.async_query(agent_name=self.name, role_desc=self.role_desc,
                                                history_messages=observation, global_prompt=self.global_prompt,
                                                request_msg=None)
        except RetryError as e:
            err_msg = f"Agent {self.name} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
            logging.warning(err_msg)
            response = SIGNAL_END_OF_CONVERSATION + err_msg

        return response

    def reset(self):
        """
        Reset the player's backend in case they are not stateless.
        This is usually called at the end of each episode.
        """
        self.backend.reset()


class Moderator(Player):
    """
    The Moderator class represents a special type of player that moderates the conversation.
    It is usually used as a component of the environment when the transition dynamics is conditioned on natural language that are not easy to parse programatically.
    """

    def __init__(self, role_desc: str, backend: Union[BackendConfig, IntelligenceBackend],
                 terminal_condition: str, global_prompt: str = None, **kwargs):
        """
        Initialize the moderator with a role description, backend, terminal condition, and a global prompt.

        Parameters:
            role_desc (str): Description of the moderator's role.
            backend (Union[BackendConfig, IntelligenceBackend]): The backend that will be used for decision making.
            terminal_condition (str): The condition that signifies the end of the conversation.
            global_prompt (str): A universal prompt that applies to the moderator. Defaults to None.
       """
        name = "Moderator"
        super().__init__(name=name, role_desc=role_desc, backend=backend, global_prompt=global_prompt, **kwargs)

        self.terminal_condition = terminal_condition

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            terminal_condition=self.terminal_condition,
            global_prompt=self.global_prompt,
        )

    def is_terminal(self, history: List[Message], *args, **kwargs) -> bool:
        """
        Check whether an episode is terminated based on the terminal condition.

        Parameters:
            history (List[Message]): The conversation history.

        Returns:
            bool: True if the conversation is over, otherwise False.
        """
        # If the last message is the signal, then the conversation is over
        if history[-1].content == SIGNAL_END_OF_CONVERSATION:
            return True

        try:
            request_msg = Message(agent_name=self.name, content=self.terminal_condition, turn=-1)
            response = self.backend.query(agent_name=self.name, role_desc=self.role_desc, history_messages=history,
                                          global_prompt=self.global_prompt, request_msg=request_msg, *args, **kwargs)
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}.")
            return True

        if re.match(r"yes|y|yea|yeah|yep|yup|sure|ok|okay|alright", response, re.IGNORECASE):
            # print(f"Decision: {response}. Conversation is ended by moderator.")
            return True
        else:
            return False


class Writer(Player):

    def __init__(self, name: str, role_desc: str, backend: Union[BackendConfig, IntelligenceBackend],
                 global_prompt: str = None, **kwargs):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scene_num = 1
        super().__init__(name, role_desc, backend, global_prompt, **kwargs)

    def act(self, observation: List[Message],
            action_prompt="Now you speak and act") -> str:
        """
        Take an action based on the observation (Generate a response), which can later be parsed to actual actions that affect the game dyanmics.

        Parameters:
            observation (List[Message]): The messages that the player has observed from the environment.

        Returns:
            str: The action (response) of the player.
        """
        
        action_prompt = f"Now please write scene {self.scene_num} of the story based on the settings and conversations."
        if self.scene_num == 1:
            action_prompt += f" Also, please make sure that this scene can be concatenated directly to the following scene."
        else:
            action_prompt += f" Also, please make sure that this scene can be concatenated directly to the previous scene and the following scene."

        try:
            response = self.backend.query(agent_name=self.name, role_desc=self.role_desc,
                                          history_messages=observation, global_prompt=self.global_prompt,
                                          request_msg=None,
                                          action_prompt=action_prompt)
        except RetryError as e:
            err_msg = f"Agent {self.name} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
            logging.warning(err_msg)
            response = SIGNAL_END_OF_CONVERSATION + err_msg
        response = f'\nScene {self.scene_num}: \n' + response
        # Save the written story to a local txt file.
        if not os.path.exists('storys'):
            os.makedirs('storys')
        with open(f"storys/story_{self.timestamp}.txt", "a+") as file:
            self.scene_num += 1
            file.write(response)
            file.write("\n")
        if not os.path.exists('storys/logs'):
            os.makedirs('storys/logs')
        with open(f"storys/logs/log_{self.timestamp}.txt", "a+") as file:
            for obs in observation:
                file.write(obs.__str__() + '\n')

        return response