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
from pysc2.env import sc2_env

from absl import flags

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
import sys
import os

FLAGS = flags.FLAGS
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_steps", int(1e5), "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")

flags.DEFINE_enum("agent_race", "P", sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", "T", sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", "1", sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("save_replay", False, "Whether to replays_save a replay at the end.")

flags.DEFINE_string("map", "Simple64", "Name of a map to use.")
flags.DEFINE_string("replay_dir", "local/", "dir of replay to replays_save.")

flags.DEFINE_integer("game_num", 2, "The num of games to play.")


_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PROBE_TYPE_INDEX = 84

_NO_OP = sc2_actions.FUNCTIONS.no_op.id
_SELECT_POINT = sc2_actions.FUNCTIONS.select_point.id
_MOVE_CAMERA = sc2_actions.FUNCTIONS.move_camera.id
_NOT_QUEUED = [0]

camera_pos = [35, 47]


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
        done = False

        for i in range(FLAGS.game_num):
            if done:
                env.reset()

            # while loop to check start point
            while True:
                timesteps = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
                visible_map = timesteps[0].observation["minimap"][1, :, :]
                if visible_map[23, 17] == 0:
                    env.reset()
                else:
                    break

            # random select a probe
            unit_type_map = timesteps[0].observation["screen"][_UNIT_TYPE]
            pos_y, pos_x = (unit_type_map == _PROBE_TYPE_INDEX).nonzero()

            index = -5
            pos = [pos_x[index], pos_y[index]]
            env.step([sc2_actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, pos])])

            # move camera
            env.step([sc2_actions.FunctionCall(_MOVE_CAMERA, [camera_pos])])

            try:
                while True:
                    timesteps = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])

                    if timesteps[0].last():
                        if timesteps[0].reward == 1 and FLAGS.save_replay:
                            env.save_replay(FLAGS.replay_dir)
                        break
            except KeyboardInterrupt:
                pass

            done = True

        # if FLAGS.save_replay:
        #     env.save_replay(FLAGS.replay_dir)


if __name__ == '__main__':
    main()
