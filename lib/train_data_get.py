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

import csv
import os
import platform
import sys

import numpy as np
from absl import app
from absl import flags
from pysc2 import run_configs
from pysc2.lib import features
from s2clientprotocol import sc2api_pb2 as sc_pb

import lib.utils as U
from agent_network import HierNetwork
from lib import environment, config as C

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
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

flags.DEFINE_string("sc_replay_path", "D:/Blizzard App/StarCraft II/Replays/", "Dir of SC replay")
flags.DEFINE_bool("save_data", False, "replays_save data or not")
flags.DEFINE_string("save_path", "./data/new_data/", "path to replays_save replay data")

flags.DEFINE_string("csv_path", "data/simple64.csv", "path to sub-goal save file")
flags.DEFINE_string("replay_version", "4.2.1", "versions replay played")
flags.DEFINE_integer("subgoal_intervel", 30, "second between subgoal changed")
flags.DEFINE_integer("subgoal_count", 3, "how many subgoals")
flags.DEFINE_integer("obs_array_count", 10, "store the nearest obs of frames ")
flags.DEFINE_bool("judge_subgoal", False, "whether to auto judge subgoal")

flags.DEFINE_string("replay", "Simple64_2018-04-10-14-59-49.SC2Replay", "replay to show.")
flags.DEFINE_bool("run_single", False, "whether to run only one replay, not batch")


class SubGoalType:
    Economy, Army, Attack = range(3)


def readSubGoal():
    """Read sub-goal data from csv file"""
    if not FLAGS.csv_path:
        sys.exit("Must supply a csv_path.")

    with open(FLAGS.csv_path, 'r') as f:
        reader = csv.reader(f)
        content = list(reader)

    sub_goal_dict = {}
    ver_dict = {}
    info_dict = {}
    for replay_info in content:
        index = replay_info[0]
        if int(index) < 36:
            continue

        subgoal_list = []
        index = replay_info[0]
        version = replay_info[1]
        difficulty = replay_info[2]
        # 0 player, 1 agent+player, 2 all agent
        play_type = replay_info[3]
        replay_name = replay_info[4] + '.SC2Replay'
        # find all subgoals
        for subgoal in replay_info[5:]:
            [timespan, subgoal_type] = subgoal.split('=')
            [start_time, end_time] = timespan.split('-')
            [start_min, start_sec] = start_time.split(':')
            start_tick = int(start_min) * 60 + int(start_sec)
            subgoal_list.append([start_tick, end_time, int(subgoal_type)])
        sub_goal_dict[replay_name] = subgoal_list
        info_dict[replay_name] = difficulty
        ver_dict[replay_name] = version

    return sub_goal_dict, ver_dict, info_dict


def getSubGoalFrame(frame_num, replay, fps):
    # build the frames start and end for each subgoal
    sub_goal_frames = []
    sub_goal_dict, _, _ = readSubGoal()
    replay_sub_goals = sub_goal_dict[os.path.basename(replay)]
    for [start_tick, end_time, subgoal_type] in replay_sub_goals:
        start_frame = start_tick * fps
        if end_time == "end":
            end_frame = frame_num - 1
        else:
            [end_min, end_sec] = end_time.split(':')
            end_tick = int(end_min) * 60 + int(end_sec)
            end_frame = end_tick * fps
        sub_goal_frames.append([int(start_frame), int(end_frame), subgoal_type])
    return sub_goal_frames


def get_tech_data_array(timestep_array, high_goal, ability_id, command=None):
    tech_array = np.array([])
    for timestep in timestep_array:
        if timestep:
            high_input, tech_cost, _ = U.get_input(timestep)
            # tech_label = 0  # no_op
            if ability_id == C._A_BUILD_PYLON_S:
                tech_label = 0
            elif ability_id == C._A_BUILD_ASSIMILATOR_S:
                tech_label = 1
            elif ability_id == C._A_BUILD_GATEWAY_S:
                tech_label = 2
            elif ability_id == C._A_BUILD_CYBER_S:
                tech_label = 3
            elif ability_id == -1:
                tech_label = 4
            tech_input = np.concatenate([high_input, high_goal, tech_cost], axis=0)
            tech_record = np.concatenate([tech_input, [tech_label]], axis=0)
            if command:
                if command.target_screen_coord:
                    pos = command.target_screen_coord
                    tech_record = np.concatenate([tech_record, [pos.x, pos.y]], axis=0)
            tech_array = np.append(tech_array, tech_record)
    return tech_array


def get_pop_data_array(timestep_array, high_goal, ability_id, command=None):
    pop_array = np.array([])
    for timestep in timestep_array:
        if timestep:
            high_input, _, pop_num = U.get_input(timestep)
            pop_label = -1
            if ability_id == C._A_TRAIN_PROBE:
                pop_label = 0
            elif ability_id == C._A_SMART_SCREEN:
                if command.target_screen_coord:
                    pos = command.target_screen_coord
                    if abs(pos.x - C.gas1_pos[0]) <= 1 and abs(pos.y - C.gas1_pos[1]) <= 1:
                        pop_label = 1
                        # print('gas_1')
                    if abs(pos.x - C.gas2_pos[0]) <= 1 and abs(pos.y - C.gas2_pos[1]) <= 1:
                        pop_label = 2
                        # print('gas_2')
            elif ability_id == C._A_TRAIN_ZEALOT:
                pop_label = 3
            elif ability_id == C._A_TRAIN_STALKER:
                pop_label = 4
            pop_input = np.concatenate([high_input, high_goal, pop_num], axis=0)
            pop_record = np.concatenate([pop_input, [pop_label]], axis=0)
            if pop_label >= 0:
                pop_array = np.append(pop_array, pop_record)
    return pop_array


def showRawObs(obs):
    if obs.observation:
        player_data = obs.observation.player_common
        print(player_data)
    if len(obs.chat) > 0:
        for ch in obs.chat:
            print(ch.message)
            if ch.message == 'gg':
                print('find gg!')


def findCommandGas(command):
    if command.ability_id == C._A_SMART_SCREEN:
        if command.target_screen_coord:
            pos = command.target_screen_coord
            if abs(pos.x - C.gas1_pos[0]) <= 1 and abs(pos.y - C.gas1_pos[1]) <= 1:
                print('gas_1')
            if abs(pos.x - C.gas2_pos[0]) <= 1 and abs(pos.y - C.gas2_pos[1]) <= 1:
                print('gas_2')


def run(replay_name, replay_version, difficulty, run_config, interface, net):
    replay_path = 'D:/sc2/multi_agent/init/data/replays/' + replay_name
    print(replay_path)

    tech_ability_list = [C._A_BUILD_PYLON_S, C._A_BUILD_ASSIMILATOR_S, C._A_BUILD_GATEWAY_S, C._A_BUILD_CYBER_S]
    pop_ability_list = [C._A_SMART_SCREEN, C._A_TRAIN_PROBE, C._A_TRAIN_ZEALOT, C._A_TRAIN_STALKER]
    attack_ability_list = [C._A_ATTACK_ATTACK_MINIMAP_S, C._A_ATTACK_MINIMAP_S]
    all_ability_list = tech_ability_list + pop_ability_list + attack_ability_list

    replay_data = run_config.replay_data(replay_path)
    start_replay = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=FLAGS.disable_fog,
        observed_player_id=FLAGS.observed_player)

    with run_config.start(full_screen=FLAGS.full_screen, game_version=replay_version) as controller:
        info = controller.replay_info(replay_data)
        print(" Replay info ".center(60, "-"))
        print(info)
        print("-" * 60)
        print(" Replay difficulty: ", difficulty)
        C.difficulty = difficulty

        frame_num = info.game_duration_loops
        step_num = frame_num // FLAGS.step_mul
        sub_goal_frames = getSubGoalFrame(frame_num, replay=replay_path, fps=FLAGS.fps)

        obs_array_count = FLAGS.obs_array_count
        obs_array = [None] * obs_array_count

        controller.start_replay(start_replay)
        feature_layer = features.Features(controller.game_info())
        path = FLAGS.save_path

        high_data = np.array([])
        tech_data = np.array([])
        pop_data = np.array([])
        begin_attack = False
        for i in range(step_num):
            # to play the game in the normal speed
            controller.step(FLAGS.step_mul)
            obs = controller.observe()
            timestep = environment.TimeStep(step_type=None,
                                            reward=None,
                                            discount=None,
                                            observation=None, raw_observation=obs)
            high_goal = net.predict_high(timestep)
            # print('high_goal:', high_goal)

            obs_data = feature_layer.transform_obs(obs.observation)
            frame_idx = obs_data["game_loop"][0]
            subgoals = [1 if start <= frame_idx <= end else 0 for [start, end, subgoal] in sub_goal_frames]
            obs_array[int(i / 2) % obs_array_count] = timestep

            use_rule = True
            if use_rule:
                gateway_count = U.get_unit_num(timestep, C._GATEWAY_TYPE_INDEX)
                cyber_count = U.get_unit_num(timestep, C._CYBER_TYPE_INDEX)
                pylon_count = U.get_unit_num(timestep, C._PYLON_TYPE_INDEX)

                player_common = timestep.raw_observation.observation.player_common
                subgoals[0] = 0 if player_common.food_workers >= 22 else 1
                subgoals[1] = 1 if 1 <= gateway_count else 0
                subgoals[2] = 1 if player_common.army_count >= 10 else 0

                use_no_op = False
                if use_no_op:
                    build_wait = False
                    if gateway_count >= 4 and cyber_count >= 1 and pylon_count >= 8:
                        build_wait = True
                    if gateway_count >= 6 and pylon_count >= 10:
                        build_wait = True
                    if build_wait:
                        tech_record = get_tech_data_array(obs_array, np.array(subgoals), -1)
                        tech_data = np.append(tech_data, tech_record)

            for action in obs.actions:
                act_fl = action.action_feature_layer
                if act_fl.HasField("unit_command"):
                    high_input, tech_cost, pop_num = U.get_input(timestep, difficulty)
                    ability_id = act_fl.unit_command.ability_id
                    if ability_id in tech_ability_list:
                        # [showRawObs(timestep.raw_observation) for timestep in obs_array]
                        tech_record = get_tech_data_array(obs_array, np.array(subgoals), ability_id)
                        tech_data = np.append(tech_data, tech_record)
                    if ability_id in pop_ability_list:
                        pop_record = get_pop_data_array(obs_array, np.array(subgoals), ability_id, act_fl.unit_command)
                        pop_data = np.append(pop_data, pop_record)
                        # print('len of pop_data:', pop_data.shape)
                    if act_fl.unit_command.ability_id in attack_ability_list:
                        begin_attack = True

            print('subgoals:', subgoals)

            if FLAGS.save_data:
                record = np.zeros(C._SIZE_HIGH_NET_INPUT + C._SIZE_HIGH_NET_OUT)
                high_input, tech_cost, pop_num = U.get_input(timestep, difficulty)
                record[0:C._SIZE_HIGH_NET_INPUT] = high_input
                record[C._SIZE_HIGH_NET_INPUT:] = np.array(subgoals)
                high_data = np.append(high_data, record)

        if FLAGS.save_data:
            with open(path + "high.txt", 'ab') as f:
                np.savetxt(f, high_data.reshape(-1, C._SIZE_HIGH_NET_INPUT + C._SIZE_HIGH_NET_OUT))
            with open(path + "tech.txt", 'ab') as f:
                np.savetxt(f, tech_data.reshape(-1, 26 + 1))
            with open(path + "pop.txt", 'ab') as f:
                np.savetxt(f, pop_data.reshape(-1, 30 + 1))


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

    sub_goal_dict, ver_dict, info_dict = readSubGoal()
    run_single = FLAGS.run_single

    net = HierNetwork()
    net.initialize()
    net.restore_high()

    if run_single:
        replay_name = FLAGS.replay
        run(replay_name, ver_dict[replay_name], info_dict[replay_name], run_config, interface, net)
    else:
        [run(replay_name, ver_dict[replay_name], info_dict[replay_name], run_config, interface, net) for replay_name in
         sub_goal_dict.keys()]


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
