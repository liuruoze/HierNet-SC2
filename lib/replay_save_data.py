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
"""Run SC2 to play a game or a replay."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform
import sys
import os
import time
import numpy as np
import glob

from pysc2 import run_configs
from pysc2.lib import features


from absl import app
from absl import flags
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_bool("realtime", False, "Whether to run in realtime mode.")
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")

flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 5, "Game steps per observation.")
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_game_steps", 0, "Total game steps to run.")
flags.DEFINE_integer("max_episode_steps", 0, "Total game steps per episode.")

flags.DEFINE_bool("disable_fog", False, "Disable fog of war.")
flags.DEFINE_integer("observed_player", 1, "Which player to observe.")

flags.DEFINE_string("sc_replay_path", "D:/Blizzard App/StarCraft II/Replays/", "Dir of SC replay")
flags.DEFINE_string("replay", "new_fix/Simple64_2018-01-17-06-22-37.SC2Replay", "replay to show.")

flags.DEFINE_bool("save_data", False, "replays_save data or not")
flags.DEFINE_string("save_path", "./data/new_data/", "path to replays_save replay data")


def main(unused_argv):
    """Run SC2 to play a game or a replay."""

    if not FLAGS.replay:
        sys.exit("Must supply a replay.")

    if not FLAGS.save_path:
        sys.exit("Must supply a replays_save path.")

    if not os.path.exists(FLAGS.save_path) and FLAGS.save_data:
        os.makedirs(FLAGS.save_path)

    if FLAGS.replay and not FLAGS.replay.lower().endswith("sc2replay"):
        sys.exit("Replay must end in .SC2Replay.")

    if FLAGS.realtime and FLAGS.replay:
        # TODO(tewalds): Support realtime in replays once the game supports it.
        sys.exit("realtime isn't possible for replays yet.")

    if FLAGS.render and (FLAGS.realtime or FLAGS.full_screen):
        sys.exit("disable pygame rendering if you want realtime or full_screen.")

    if platform.system() == "Linux" and (FLAGS.realtime or FLAGS.full_screen):
        sys.exit("realtime and full_screen only make sense on Windows/MacOS.")

    if not FLAGS.render and FLAGS.render_sync:
        sys.exit("render_sync only makes sense with pygame rendering on.")

    run_config = run_configs.get()

    interface = sc_pb.InterfaceOptions()
    interface.raw = True
    interface.score = True
    interface.feature_layer.width = 24
    interface.feature_layer.resolution.x = FLAGS.screen_resolution
    interface.feature_layer.resolution.y = FLAGS.screen_resolution
    interface.feature_layer.minimap_resolution.x = FLAGS.minimap_resolution
    interface.feature_layer.minimap_resolution.y = FLAGS.minimap_resolution

    replay_data = run_config.replay_data(FLAGS.replay)
    start_replay = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=FLAGS.disable_fog,
        observed_player_id=FLAGS.observed_player)

    with run_config.start(full_screen=FLAGS.full_screen) as controller:

        info = controller.replay_info(replay_data)
        print(" Replay info ".center(60, "-"))
        print(info)
        print("-" * 60)

        controller.start_replay(start_replay)

        feature_layer = features.Features(controller.game_info())

        frame_num = info.game_duration_loops
        step_num = frame_num // FLAGS.step_mul

        path = FLAGS.save_path

        # init data
        player_data = np.zeros((step_num, 1 + 11))
        unit_data = np.zeros((step_num, 1 + 7))
        score_data = np.zeros((step_num, 1 + 13))

        frame_array = [(x+1)*FLAGS.step_mul for x in range(step_num)]
        player_data[:, 0] = unit_data[:, 0] = score_data[:, 0] = frame_array

        order_data = np.array([])
        obs = controller.observe()
        for i in range(step_num):
            # to play the game in the normal speed

            controller.step(FLAGS.step_mul)

            obs = controller.observe()
            obs_data = feature_layer.transform_obs(obs.observation)

            player_data[i, 1:] = obs_data["player"]
            unit_data[i, 1:] = obs_data["single_select"]
            score_data[i, 1:] = obs_data["score_cumulative"]

            if FLAGS.save_data:
                np.save(path + "minimap_%d.npy" % obs_data["game_loop"], obs_data["minimap"])
                np.save(path + "screen_%d.npy" % obs_data["game_loop"], obs_data["screen"])

            # [game_loop, action_type, x, y]
            # action_type: 0 : move, 1 : build_pylon, 2 : build_forge, 3: build_cannon, 4: move_camera
            action_ability_list = [16, 881, 884, 887]

            unit_set = obs.observation.raw_data.units
            for u in unit_set:
                if u.orders:
                    if u.orders[0].ability_id in action_ability_list:
                        order_temp = np.zeros(4)
                        order_temp[0] = obs.observation.game_loop
                        order_temp[1] = action_ability_list.index(u.orders[0].ability_id)
                        order_temp[2:] = [u.orders[0].target_world_space_pos.x, u.orders[0].target_world_space_pos.y]

                        order_data = np.append(order_data, order_temp)

                        for unit in unit_set:
                            if unit.build_progress < 1:
                                print(1)

        print("Score: ", obs.observation.score.score)
        print("Result: ", obs.player_result)

        if FLAGS.save_data:
            np.savetxt(path + "unit.txt", unit_data)
            np.savetxt(path + "top.txt", player_data)
            np.savetxt(path + "score.txt", score_data)
            np.savetxt(path + "order.txt", order_data.reshape(-1, 4))


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
