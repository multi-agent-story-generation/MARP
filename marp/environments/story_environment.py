import random
from typing import List, Union, Tuple

from marp.backends import OpenAIChat
from marp.config import EnvironmentConfig
from marp.environments import TimeStep
from marp.environments.base import Environment
from marp.message import MessagePool, Message
from marp.agent import SIGNAL_END_OF_CONVERSATION, Player

PLAYER_TERMINAL = 'END'


class Story(Environment):
    type_name = "Story"

    def __init__(self, player_names: List[str],max_scene_turns, max_scenes, player_backend, player_prompt=None, summarize_act=True, **kwargs):
        super().__init__(player_names, **kwargs)
        self.global_message_pool = MessagePool()
        self.scene_message_pool = MessagePool()
        self.character_message_pools = {}
        self._current_stage = "init"
        self._next_stage = "player_init"
        self._current_turn = 0
        self._current_scene = 0
        self._max_scene_turns = max_scene_turns
        self._max_scenes = max_scenes
        self._scene_start = 0  # turn where current scene starts
        self._next_player_idx = 0
        self._arena = None
        self._current_act = Message('', '', -1)
        self._role_list = []
        self.player_backend = player_backend
        self._player_prompt = player_prompt if player_prompt else ''
        self._summarize_act = summarize_act

    def set_arena(self, arena):
        self._arena = arena

    def reset(self):
        self._current_stage = "init"
        self._next_stage = "player_init"
        self._current_turn = 0
        self.global_message_pool.reset()
        self.scene_message_pool.reset()

    def get_next_player(self) -> str:
        if self._current_stage == "init":
            return "Global designer"
        elif self._current_stage == "player_init":
            return self._role_list[self._next_player_idx]
        elif self._current_stage == "scene_init":
            return "Designer"
        elif self._current_stage == "pick":
            return "Controller"
        elif self._current_stage == "impact":
            return "Summarizer"
        elif self._current_stage == "end of scene":
            return "Writer"
        elif self._current_stage == "review":
            return "Reader"
        else:
            return self.player_names[self._next_player_idx]

    def get_observation(self, player_name=None) -> Tuple[List[Message], str]:
        if player_name is None:
            return self.scene_message_pool.get_all_messages(), self._current_stage
        elif player_name == 'Summarizer':
            temp_message_pool = self.global_message_pool.get_visible_messages(player_name, turn=self._current_turn) + \
                self.scene_message_pool.get_visible_messages(player_name, turn=self._current_turn)
            temp_message_pool.append(self._current_act)
            return temp_message_pool, self._current_stage
        elif player_name in self._role_list:
            return self.global_message_pool.get_visible_messages(player_name, turn=self._current_turn) + \
                self.scene_message_pool.get_visible_messages(player_name, turn=self._current_turn) + \
                self.character_message_pools[player_name].get_visible_messages(player_name, turn=self._current_turn), self._current_stage
        else:
            return self.global_message_pool.get_visible_messages(player_name, turn=self._current_turn) + \
                self.scene_message_pool.get_visible_messages(player_name, turn=self._current_turn), self._current_stage

    def print(self):
        self.global_message_pool.print()
        self.scene_message_pool.print()

    # def _controller_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
    #     message = Message(agent_name="Controller", content=text, turn=self._current_turn, visible_to=visible_to)
    #     self.global_message_pool.append_message(message)

    def is_terminal(self) -> bool:
        """
        check if the conversation is over
        """
        if self._current_scene == self._max_scenes:
            print(self._current_scene, self._max_scenes)
            return True
        # If the last message is the signal, then the conversation is over
        if self.scene_message_pool.last_message is None:
            return False
        if self.scene_message_pool.last_message.content.startswith(SIGNAL_END_OF_CONVERSATION):
            return True
        if self.global_message_pool.last_message is None:
            return False
        if self.global_message_pool.last_message.content.startswith(SIGNAL_END_OF_CONVERSATION):
            return True
        return False

    def _parse_global_designer_output(self, text: str):
        # global designer output format:
        # <setting description>
        # * <Player1>: <description>
        # * <Player2>: <description>
        # ...
        player_desc = text.split('* ')[1:]
        designed_players = [desc.split(':')[0] for desc in player_desc]
        descs = [str(desc.split(':')[1:]) for desc in player_desc]
        for name, desc in zip(designed_players, descs):
            player = Player(name=name, role_desc=desc + self._player_prompt, backend=self.player_backend)
            self._arena.add_player(player)
            self.character_message_pools[name] = MessagePool()
            self._role_list.append(name)
            self.player_names.append(name)
        # return all player settings, which will be added to scene message pool
        return '* ' + ''.join(player_desc)

    def _parse_designer_output(self, text: str) -> Tuple[str, List[str]]:
        try:
            setting, players = text.split('### Next up: ')
            return setting, players.split(', ')
        except ValueError:
            print('WARNING designer wrong format, using all players')
            return text, self.player_names

    def _parse_picked_player(self, text: str) -> str:
            name = text.split('Next up: ')[1]
            if name[-1] == '.':
                # remove period at the end
                name = name[:-1]
            if '<EOS' in name:
                # remove EOS
                name = name.split('<')[0]
            if name == PLAYER_TERMINAL:
                return PLAYER_TERMINAL
            for player_name in self.player_names:
                if name in player_name:
                    return player_name
            print(f'WARNING using random player, all available players are {self._role_list}')
            return random.choice(self._role_list)

    def _parse_env_manager_output(self, text: str) -> str:
        try:
            summary = text.split('### Summary:')[1]
            return summary
        except IndexError:
            print(f'WARNING env manager output format error')
            return text

    def _parse_player_output(self, name: str, text: str) -> str:
        # player output format: '### My plan:\n <explanation and plan>'
        self.character_message_pools[name].append_message(Message(agent_name=name, content=text, turn=self._current_turn))
        return text

    def step(self, player_name: str, action: str) -> TimeStep:
        terminal = False
        if self._current_stage == "init": # global designer
            player_descs = self._parse_global_designer_output(action)
            message = Message(agent_name=player_name, content=f'Players:\n {player_descs}', turn=self._current_turn)
            self.global_message_pool.append_message(message)
            self._next_stage = "player_init"
        elif self._current_stage == "player_init": # player
            self._parse_player_output(player_name, action)
            message = Message(agent_name=player_name, content=action, turn=self._current_turn)
            self.character_message_pools[player_name].append_message(message)
            if self._next_player_idx < len(self._role_list) - 1:
                self._next_stage = "player_init"
                self._next_player_idx += 1
            else:
                self._next_stage = "scene_init"
        elif self._current_stage == "scene_init": # designer
            setting, players = self._parse_designer_output(action)
            # add setting to scene message pool
            message = Message(agent_name=player_name, content=setting, turn=self._current_turn)
            self.scene_message_pool.reset()
            self.scene_message_pool.append_message(message)
            # add players of current scene
            message = Message(agent_name=player_name, content=f"Players in this scene: {', '.join(players)}", turn=self._current_turn)
            self.scene_message_pool.append_message(message)
            self._scene_start = self._current_turn
            self._next_stage = "pick"
        elif self._current_stage == "pick": # controller
            next_player = self._parse_picked_player(action)
            # controller says PLAYER_TERMINAL or max_scene_turns is reached
            if next_player == PLAYER_TERMINAL or self._current_turn - self._scene_start >= self._max_scene_turns:
                self._next_stage = "end of scene"
            else:
                self._next_player_idx = self.player_names.index(next_player)
                self._next_stage = "act"
        elif self._current_stage == "act": # player
            if self._summarize_act:
                self._current_act = Message(agent_name=player_name, content=f'The act you summarize: [{player_name}]: {action}', turn=self._current_turn)
                self._next_stage = "impact"
            else:
                message = Message(agent_name=player_name, content=action, turn=self._current_turn)
                self.scene_message_pool.append_message(message)
                self._next_stage = "pick"
        elif self._current_stage == "impact": # summarizer
            self._next_stage = "pick"
            action = self._parse_env_manager_output(action)
            message = Message(agent_name=player_name, content=action, turn=self._current_turn)
            self.scene_message_pool.append_message(message)
        elif self._current_stage == "end of scene": # writer
            message = Message(agent_name=player_name, content=action, turn=self._current_turn)
            self.global_message_pool.append_message(message)
            self._current_scene += 1
            self._next_stage = "review"
        elif self._current_stage == "review": # reader
            message = Message(agent_name=player_name, content=action, turn=self._current_turn)
            self.global_message_pool.append_message(message)
            self._next_stage = "scene_init"
        terminal = terminal or self.is_terminal()
        timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=terminal)
        self._current_turn += 1  # update current_turn every step
        self._current_stage = self._next_stage
        return timestep

    def check_action(self, action: str, player_name: str) -> bool:
        if "As an AI language model" in action:  # GPT not acting as the agent
            return False
        if "failed to generate a response" in action:
            return False
        if player_name == "Controller":
            try:
                picked_player = self._parse_picked_player(action)
            except IndexError:
                return False
            if picked_player not in self._role_list and picked_player != PLAYER_TERMINAL:
                return False
        elif player_name == "Global designer":
            try:
                player_desc = action.split('* ')[1:]
                designed_players = [desc.split(':')[0] for desc in player_desc]
                descs = [desc.split(':')[1:] for desc in player_desc]
                if len(designed_players) == 0:
                    raise IndexError
            except IndexError:
                return False
        return True

    def to_config(self) -> EnvironmentConfig:
        return super().to_config()
