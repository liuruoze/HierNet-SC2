#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from lib import my_sc2_env as sc2_env

from absl import flags

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
import sys
import os
import lib.utils as U
import lib.config as C
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

FLAGS = flags.FLAGS
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_steps", int(1e6), "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")

flags.DEFINE_enum("agent_race", "P", sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", "T", sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", "3", sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("save_replay", True, "Whether to replays_save a replay at the end.")

flags.DEFINE_string("map", "Simple64", "Name of a map to use.")
flags.DEFINE_string("replay_dir", "fix_camera/", "dir of replay to replays_save.")


_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PROBE_TYPE_INDEX = 84

_NO_OP = sc2_actions.FUNCTIONS.no_op.id
_SELECT_POINT = sc2_actions.FUNCTIONS.select_point.id
_MOVE_CAMERA = sc2_actions.FUNCTIONS.move_camera.id
_NOT_QUEUED = [0]

camera_pos_down = [35, 47]
camera_pos_top = [25, 25]


def find_sub_mineral(obs):
    # random select a probe
    relative_type_map = obs.observation["minimap"][C._M_RELATIVE_TYPE]
    #command_center = (unit_type_map == C.TERRAN_COMMANDCENTER)
    #pos_y, pos_x = command_center.nonzero()
    #num = len(pos_y)

    imgplot = plt.imshow(relative_type_map)
    plt.show()


def main():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(
            map_name=FLAGS.map,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=FLAGS.difficulty,
            step_mul=FLAGS.step_mul,
            score_index=-1,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=False) as env:
        # env = available_actions_printer.AvailableActionsPrinter(env)

        timesteps = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])

        # random select a probe
        last_obs = None
        try:
            while True:
                timesteps = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
                obs = timesteps[0]
                for action in obs.raw_observation.actions:
                    act_fl = action.action_feature_layer
                    if act_fl.HasField("unit_command"):
                        ability_id = act_fl.unit_command.ability_id
                        print(act_fl.unit_command)

                dead_units = obs.raw_observation.observation.raw_data.event.dead_units
                if len(dead_units) > 0 and last_obs:
                    print(dead_units)
                    for tag in dead_units:
                        print(tag)
                        unit = U.find_unit_by_tag(last_obs, tag)
                        if unit:
                            find_sub_mineral(obs)
                            print("dead:")
                            print(unit)

                last_obs = obs

                if timesteps[0].last():
                    break
        except KeyboardInterrupt:
            pass

        if FLAGS.save_replay:
            env.save_replay(FLAGS.replay_dir)


if __name__ == '__main__':
    main()
