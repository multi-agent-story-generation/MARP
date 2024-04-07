import json
import openai

from marp.agent import Player, Writer
from marp.backends import OpenAIChat, AutoChat
from marp.environments import Story
from marp.arena import Arena
from marp.standard_prompts import *


DEFAULT_MAX_TOKENS = 4096


BACKEND = OpenAIChat()
# BACKEND = MistralChat(device='mps')
# BACKEND = AutoChat(model='anyscale/mistralai/Mistral-7B-Instruct-v0.1')
# BACKEND = AutoChat(model='gemini-pro', temperature=0.8)

# environment_description = QUARREL_PREMISE
environment_description = IBRUSIA_PREMISE
# environment_description = ECONOMY_PREMISE

controller = Player(name="Controller", backend=BACKEND,
                        role_desc=CONTROLLER_PROMPT,
                        global_prompt=environment_description)
global_designer = Player(name="Global designer", backend=BACKEND,
                  role_desc=GLOBAL_DESIGNER_PROMPT, global_prompt=environment_description)
designer = Player(name="Designer", backend=BACKEND,
                  role_desc=DESIGNER_PROMPT, global_prompt=environment_description)
writer = Writer(name="Writer", backend=BACKEND,
                 role_desc=WRITER_PROMPT,
                 global_prompt= environment_description)
env_manager = Player(name="Summarizer", backend=BACKEND,
                     role_desc=ENV_MANAGER_PROMPT,
                     global_prompt= environment_description)
reader = Player(name="Reader", backend=BACKEND,
                role_desc=READER_PROMPT,
                global_prompt= environment_description)
players = [controller, global_designer, designer, writer, env_manager, reader]

env = Story(player_names=[p.name for p in players], max_scene_turns=10, max_scenes=5, player_backend=BACKEND, player_prompt=PLAYER_PROMPT, summarize_act=False)
# arena = Arena.from_config('story_generation.json')
arena = Arena(players=players,
              environment=env, global_prompt=environment_description)
env.set_arena(arena)

# json.dump(arena.to_config(), open('story_generation.json', 'w'))
arena.run(num_steps=2 ** 31)
