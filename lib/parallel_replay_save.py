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
import numpy as np

from pysc2 import run_configs
from pysc2.lib import features

from absl import app
from absl import flags
from s2clientprotocol import sc2api_pb2 as sc_pb

import threading

from lib.utils import get_building_num_array

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("realtime", False, "Whether to run in realtime mode.")
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")

flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 1, "Game steps per observation.")
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_game_steps", 0, "Total game steps to run.")
flags.DEFINE_integer("max_episode_steps", 0, "Total game steps per episode.")

flags.DEFINE_bool("disable_fog", False, "Disable fog of war.")
flags.DEFINE_integer("observed_player", 1, "Which player to observe.")

flags.DEFINE_string("replay_dir", "D:/Blizzard App/StarCraft II/Replays/new_fix/", "Dir of replays to show.")

flags.DEFINE_bool("save_data", True, "replays_save data or not")
flags.DEFINE_string("save_path", "./data/new_data/", "path to replays_save replay data")

flags.DEFINE_integer("parallel", 1, "How many threads to run in parallel.")

LOCK = threading.Lock()
COUNT = 0


def replays_save(files):
    """Run SC2 to play a game or a replay."""
    global COUNT
    while COUNT > 0:
        with LOCK:
            COUNT -= 1
            count = COUNT

        run_config = run_configs.get()

        interface = sc_pb.InterfaceOptions()
        interface.raw = True
        interface.score = True
        interface.feature_layer.width = 24
        interface.feature_layer.resolution.x = FLAGS.screen_resolution
        interface.feature_layer.resolution.y = FLAGS.screen_resolution
        interface.feature_layer.minimap_resolution.x = FLAGS.minimap_resolution
        interface.feature_layer.minimap_resolution.y = FLAGS.minimap_resolution

        replay = FLAGS.replay_dir + files[count]
        print(replay)
        replay_data = run_config.replay_data(replay)

        with run_config.start(full_screen=FLAGS.full_screen) as controller:

            info = controller.replay_info(replay_data)
            print(" Replay info ".center(60, "-"))
            print(info)
            print("-" * 60)

            if info.base_build != 60321:
                continue

            start_replay = sc_pb.RequestStartReplay(
                replay_data=replay_data,
                options=interface,
                disable_fog=FLAGS.disable_fog,
                observed_player_id=FLAGS.observed_player)

            controller.start_replay(start_replay)

            if not os.path.exists(FLAGS.save_path + str(count)) and FLAGS.save_data:
                os.makedirs(FLAGS.save_path + str(count))
            save_path = FLAGS.save_path + str(count) + "/"

            feature_layer = features.Features(controller.game_info())

            frame_num = info.game_duration_loops
            step_num = frame_num // FLAGS.step_mul

            # init data
            player_data = np.zeros((step_num, 1 + 11))
            unit_data = np.zeros((step_num, 1 + 7))
            score_data = np.zeros((step_num, 1 + 13))

            frame_array = [(x + 1) * FLAGS.step_mul for x in range(step_num)]
            player_data[:, 0] = unit_data[:, 0] = score_data[:, 0] = frame_array

            pos_x = pos_y = 0

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
                    np.save(save_path + "minimap_%d.npy" % obs_data["game_loop"], obs_data["minimap"])
                    np.save(save_path + "screen_%d.npy" % obs_data["game_loop"], obs_data["screen"])

                # [game_loop, action_type, x, y]
                # action_type: 0 : move, 1 : build_pylon, 2 : build_forge, 3: build_cannon, 4: move_camera
                action_ability_list_feature = [1, 881, 884, 887]
                action_ability_list_order = [16, 881, 884, 887]

                for action in obs.actions:
                    act_fl = action.action_feature_layer
                    if act_fl.HasField("unit_command"):
                        if act_fl.unit_command.ability_id in action_ability_list_feature:
                            pos_x = act_fl.unit_command.target_screen_coord.x
                            pos_y = act_fl.unit_command.target_screen_coord.y

                # get the num of building
                building_num_array = get_building_num_array(obs)

                # get the unit order
                unit_set = obs.observation.raw_data.units
                for u in unit_set:
                    if u.orders:
                        if u.orders[0].ability_id in action_ability_list_order:
                            building_len = building_num_array.shape[0]
                            order_temp = np.zeros(4 + building_len)

                            order_temp[0] = obs.observation.game_loop
                            order_temp[1] = action_ability_list_order.index(u.orders[0].ability_id)
                            order_temp[[2, 3]] = [pos_x, pos_y]
                            order_temp[4:] = building_num_array

                            order_data = np.append(order_data, order_temp)
                            break

            print("Score: ", obs.observation.score.score)
            print("Result: ", obs.player_result)

            if FLAGS.save_data:
                np.savetxt(save_path + "unit.txt", unit_data)
                np.savetxt(save_path + "top.txt", player_data)
                np.savetxt(save_path + "score.txt", score_data)
                np.savetxt(save_path + "order.txt", order_data.reshape(-1, 4+building_len))


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


def main(unuse):

    if not FLAGS.save_path:
        sys.exit("Must supply a replays_save path.")

    if not os.path.exists(FLAGS.save_path) and FLAGS.save_data:
        os.makedirs(FLAGS.save_path)

    if FLAGS.render and (FLAGS.realtime or FLAGS.full_screen):
        sys.exit("disable pygame rendering if you want realtime or full_screen.")

    if platform.system() == "Linux":
        sys.exit("This version is only support for Windows")

    if not FLAGS.render and FLAGS.render_sync:
        sys.exit("render_sync only makes sense with pygame rendering on.")

    if not os.path.exists(FLAGS.replay_dir):
        sys.exit("replay path not exists")

    files = os.listdir(FLAGS.replay_dir)
    global COUNT
    COUNT = len(files)

    threads = []
    for i in range(FLAGS.parallel-1):
        t = threading.Thread(target=replays_save, args=(files,))
        threads.append(t)
        t.daemon = True
        t.start()

    replays_save(files)

    for t in threads:
        t.join()


if __name__ == "__main__":
    app.run(main)
