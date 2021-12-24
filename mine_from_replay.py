#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Transform replay data: "

import shutil
import csv
import os
import sys
import traceback
import random
import pickle
import enum
import copy

from absl import flags
from absl import app
from tqdm import tqdm

import matplotlib.pyplot as plt

from pysc2.lib import point
from pysc2.lib import features as Feat
from pysc2.lib import actions as A
from pysc2.lib.actions import FUNCTIONS as F
from pysc2 import run_configs

from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as com_pb

import lib.utils as U
import lib.config as C

from prefixspan import PrefixSpan

__author__ = "Ruo-Ze Liu"

debug = False

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

flags.DEFINE_integer("max_steps_of_replay", int(22.4 * 60 * 60), "Max game steps of a replay, max for 1 hour of game.")
flags.DEFINE_integer("small_max_steps_of_replay", 256, "Max game steps of a replay when debug.")


flags.DEFINE_bool("disable_fog", False, "Whether tp disable fog of war.")
flags.DEFINE_integer("observed_player", 1, "Which player to observe. For 2 player game, this can be 1 or 2.")

flags.DEFINE_integer("save_type", 1, "0 is torch_tensor, 1 is python_pickle, 2 is numpy_array")
flags.DEFINE_string("replay_version", "3.16.1", "the replays released by blizzard are all 3.16.1 version")

# note, replay path should be absoulte path
flags.DEFINE_string("no_server_replay_path", "D:/work/Experiment/TODO/new_three_layer_server/data/replay/", "path of replay data")

flags.DEFINE_bool("save_data", False, "replays_save data or not")
flags.DEFINE_string("save_path", "./data/replay_data/", "path to replays_save replay data")
FLAGS(sys.argv)


RACE = ['Terran', 'Zerg', 'Protoss', 'Random']
RESULT = ['Victory', 'Defeat', 'Tie']

RAW = False

DATA_FROM = 0
DATA_NUM = 30
STEPS = 20 * 60 * 22.4

STATISTIC_ACTIONS_INTERVAL = 5 * 22.4

SELECT_SEPARATE_NUMBER = 2000
SMART_TARGET_SEPARATE = 4500


class SaveType(enum.IntEnum):
    torch_tensor = 0
    python_pickle = 1
    numpy_array = 2


if FLAGS.save_type == 0:
    SAVE_TYPE = SaveType.torch_tensor
elif FLAGS.save_type == 1:
    SAVE_TYPE = SaveType.python_pickle
else:
    SAVE_TYPE = SaveType.numpy_array


def getFuncCall(o, feat):
    func_call = None

    #func_call = feat.reverse_action(o.actions[0])

    print('len(o.actions): ', len(o.actions)) if 1 else None

    action_list = []
    for action in o.actions:
        func_call = feat.reverse_action(action)

        if func_call.function == 0:
            # no op
            pass
        elif func_call.function == 1:
            # camera move
            pass
        elif func_call.function == 3:
            # we do not consider the smart rect
            pass        
        else:
            print('func_call: ', func_call) if 1 else None
            action_list.append(func_call)

    if len(action_list) > 0:
        return action_list[0]
    else:
        return None


def transoform_replays(on_server=False):

    if on_server:
        REPLAY_PATH = P.replay_path  
        max_steps_of_replay = FLAGS.max_steps_of_replay
    else:
        REPLAY_PATH = FLAGS.no_server_replay_path
        max_steps_of_replay = STEPS  # 2000

    run_config = run_configs.get()  # 
    print('REPLAY_PATH:', REPLAY_PATH)
    replay_files = os.listdir(REPLAY_PATH)
    print('length of replay_files:', len(replay_files))
    replay_files.sort()

    screen_resolution = point.Point(FLAGS.screen_resolution, FLAGS.screen_resolution)
    minimap_resolution = point.Point(FLAGS.minimap_resolution, FLAGS.minimap_resolution)
    camera_width = 24

    interface = sc_pb.InterfaceOptions(

    )

    screen_resolution.assign_to(interface.feature_layer.resolution)
    minimap_resolution.assign_to(interface.feature_layer.minimap_resolution)

    replay_length_list = []
    noop_length_list = []

    all_func_call_list = []

    from_index = DATA_FROM
    end_index = DATA_FROM + DATA_NUM

    with run_config.start(game_version=FLAGS.replay_version, full_screen=False) as controller:

        for i, replay_file in enumerate(tqdm(replay_files)):
            try:
                replay_path = REPLAY_PATH + replay_file
                print('replay_path:', replay_path)

                do_write = False
                if i >= from_index:
                    if end_index is None:
                        do_write = True
                    elif end_index is not None and i < end_index:
                        do_write = True

                if not do_write:
                    continue 

                replay_data = run_config.replay_data(replay_path)
                replay_info = controller.replay_info(replay_data)

                print('replay_info', replay_info) if debug else None
                print('type(replay_info)', type(replay_info)) if debug else None

                print('replay_info.player_info：', replay_info.player_info) if debug else None
                infos = replay_info.player_info

                observe_id_list = []
                observe_result_list = []
                for info in infos:
                    print('info：', info) if debug else None
                    player_info = info.player_info
                    result = info.player_result.result
                    print('player_info', player_info) if debug else None
                    if player_info.race_actual == com_pb.Protoss:
                        observe_id_list.append(player_info.player_id)
                        observe_result_list.append(result)

                print('observe_id_list', observe_id_list) if debug else None
                print('observe_result_list', observe_result_list) if debug else None

                win_observe_id = 0

                for i, result in enumerate(observe_result_list):
                    if result == sc_pb.Victory:
                        win_observe_id = observe_id_list[i]
                        break

                # we observe the winning one
                print('win_observe_id', win_observe_id)

                if win_observe_id == 0:
                    print('no win_observe_id found! continue')
                    continue

                start_replay = sc_pb.RequestStartReplay(
                    replay_data=replay_data,
                    options=interface,
                    disable_fog=False,  # FLAGS.disable_fog
                    observed_player_id=win_observe_id
                )

                print(" Replay info ".center(60, "-")) if debug else None
                print(replay_info) if debug else None
                print("-" * 60) if debug else None
                controller.start_replay(start_replay)

                feat = Feat.Features(controller.game_info()) 

                print("feat obs spec:", feat.observation_spec()) if debug else None
                print("feat action spec:", feat.action_spec()) if debug else None

                prev_obs = None
                i = 0
                record_i = 0
                save_steps = 0
                noop_count = 0

                obs_list, func_call_list, z_list, delay_list = [], [], [], [] 
                feature_list, label_list = [], []
                step_dict = {}

                # initial build order
                player_bo = []
                player_ucb = []

                no_op_window = []
                show = False

                unit_type = None

                while True:
                    o = controller.observe()
                    try:
                        try:
                            func_call = None
                            no_op = False

                            if i % STATISTIC_ACTIONS_INTERVAL == 0:
                                all_func_call_list.append(func_call_list)
                                func_call_list = []

                            if o.actions:
                                func_call = getFuncCall(o, feat)
                                print('func_call', func_call)

                                # we didn't consider move_camera (1)
                                # in macro actions;
                                if func_call is not None:
                                    if func_call.function == F.move_camera.id:
                                        func_call = None
                            else:
                                no_op = True

                            if func_call is not None:
                                save_steps += 1

                                int_id = func_call.function
                                print('func_call.function ', func_call.function)

                                if int_id == F.select_point.id:  # 2: # select_point                               
                                    arguments = func_call.arguments
                                    print('arguments', arguments)

                                    [x, y] = arguments[1]

                                    obs = feat.transform_obs(o.observation)
                                    unit_type_map = obs["screen"][U._UNIT_TYPE]

                                    unit_type = unit_type_map[y, x]

                                    print('unit_type', unit_type)

                                    if show:
                                        imgplot = plt.imshow(unit_type_map)
                                        plt.show()

                                    if unit_type is not None:
                                        int_id = SELECT_SEPARATE_NUMBER + unit_type

                                elif int_id == F.Smart_screen.id:  # 451:  # Smart_screen
                                    arguments = func_call.arguments
                                    print('arguments', arguments)
                                    [x, y] = arguments[1]

                                    obs = feat.transform_obs(o.observation)
                                    unit_type_map = obs["screen"][U._UNIT_TYPE]

                                    unit_type = unit_type_map[y, x]

                                    print('unit_type', unit_type)

                                    if unit_type is not None:
                                        int_id = SMART_TARGET_SEPARATE + unit_type

                                func_call_list.append(int_id)

                        except Exception as e:
                            traceback.print_exc()

                        if i >= max_steps_of_replay:  # test the first n frames
                            print("max frames test, break out!")
                            break

                        if o.player_result:  # end of game
                            print(o.player_result)
                            break

                    except Exception as inst:
                        traceback.print_exc() 

                    controller.step()
                    i += 1

                if SAVE_TYPE == SaveType.torch_tensor:
                    pass

                elif SAVE_TYPE == SaveType.python_pickle:
                    all_func_call_list.append(func_call_list)

                elif SAVE_TYPE == SaveType.numpy_array:
                    pass

                replay_length_list.append(save_steps)
                noop_length_list.append(noop_count)
                # We only test the first one replay            
            except Exception as inst:
                traceback.print_exc() 

        print('begin save!')

        if SAVE_TYPE == SaveType.torch_tensor:
            pass

        elif SAVE_TYPE == SaveType.python_pickle:
            save_path = FLAGS.save_path + 'actions.dat'

            print("all_func_call_list", all_func_call_list)

            with open(save_path, 'wb') as fp:
                pickle.dump(all_func_call_list, fp)

        elif SAVE_TYPE == SaveType.numpy_array:
            pass   

        print('end save!')

    print("end")
    print("replay_length_list:", replay_length_list)
    print("noop_length_list:", noop_length_list)


def analyse_data(on_server=False):
    read_path = FLAGS.save_path + 'actions.dat'
    print('read_path', read_path)

    with open(read_path, 'rb') as fp:
        all_func_call_list = pickle.load(fp)
        print('all_func_call_list', all_func_call_list) if 0 else None

        ps = PrefixSpan(all_func_call_list)

        # make sure the first actions should be select action, like select point (2), select group (4),
        # and select army (7).
        # meanwhile, the macro actions should not have the same actions, like len(set(patt)) equals to len(patt) 
        result = ps.topk(75, filter=lambda patt, matches: len(set(patt)) == len(patt) and len(patt) >= 2 and
                         (patt[0] in [F.select_army.id] or patt[0] > SELECT_SEPARATE_NUMBER))

        all_seq_output = []
        for i in result:
            print('frq', i[0]) if 0 else None
            seq = i[1]
            seq_output = []
            for j in seq:
                unit_type = None
                if j > SELECT_SEPARATE_NUMBER and j < SMART_TARGET_SEPARATE:  # select point
                    unit_type = j - SELECT_SEPARATE_NUMBER
                    j = F.select_point.id

                elif j >= SMART_TARGET_SEPARATE:  # smart screen
                    unit_type = j - SMART_TARGET_SEPARATE

                    # Function.ability(451, "Smart_screen", cmd_screen, 1),
                    j = F.Smart_screen.id

                    # specfila for smart screen on resource
                    resource_minerals_list = [C.UNIT_MAP_INV['MineralField'], C.UNIT_MAP_INV['MineralField750'], C.UNIT_MAP_INV['MineralField450']]
                    resource_gas_list = [C.UNIT_MAP_INV['Assimilator'], C.UNIT_MAP_INV['Extractor'], C.UNIT_MAP_INV['Refinery']]

                    if unit_type in resource_minerals_list or unit_type in resource_gas_list:

                        # Function.ability(264, "Harvest_Gather_screen", cmd_screen, 3666),
                        j = F.Harvest_Gather_screen.id
                        if unit_type in resource_minerals_list:

                            # Unified into a single resource type
                            unit_type = C.UNIT_MAP_INV['MineralField']

                if unit_type is not None: 
                    print(F[j].name, C.UNIT_MAP.get(unit_type, "None"), unit_type) if 0 else None
                else:
                    print(F[j].name) if 0 else None

                seq_output.append([j, unit_type])
            all_seq_output.append([i[0], seq_output])

        # filter
        filter_seq_output = []
        for i in all_seq_output:
            frq = i[0]
            seq = i[1]

            select_point_size = 0
            first_is_select = True
            smart_screen_to_none = False
            first_select_type = None
            wrong_action_mapping = False
            for jdx, j in enumerate(seq):
                action = j[0]
                unit_type = j[1]

                # if the first action is not any select action, skip it
                if jdx == 0 and not (action in [F.select_point.id, F.select_control_group.id, F.select_army.id]):
                    first_is_select = False
                    break

                if jdx == 0 and action in [F.select_point.id]:
                    first_select_type = unit_type

                # if the seq contains more than one select action, skip it
                if action in [F.select_point.id, F.select_control_group.id, F.select_army.id]:
                    select_point_size += 1

                if jdx > 0 and first_select_type == C.UNIT_MAP_INV['Nexus']:

                    # Function.ability(485, "Train_Probe_quick", cmd_quick, 1006),
                    if action not in [F.Train_Probe_quick.id]:
                        wrong_action_mapping = True
                        break

                # if the smart_screen has a 0 target unit_type, skip it
                if action == F.Smart_screen.id and unit_type == 0:
                    smart_screen_to_none = True
                    break

            # if the first action is not any select action, skip it
            if not first_is_select:
                continue

            # if the seq contains more than one select action, skip it
            if select_point_size > 1:
                continue

            # if the smart_screen has a 0 target unit_type, skip it
            if smart_screen_to_none:
                continue

            # if the selected unit types can't match the subsequent actions, skit it
            if wrong_action_mapping:
                continue

            filter_seq_output.append([frq, seq])

        # transform to some easily reading format
        macros_list = []
        for i in filter_seq_output:
            frq = i[0]
            seq = i[1]
            out_string = 'frq:' + str(frq) + " "
            string_list = [] 
            for jdx, j in enumerate(seq):
                action = j[0]
                unit_type = j[1]
                if unit_type is not None:
                    string = F[action].name + "(" + C.UNIT_MAP.get(unit_type, "None") + ")"
                else:
                    string = F[action].name
                string_list.append(string)
            output_func = ' -> '.join(string_list)
            print(out_string + output_func) if 0 else None
            macros_list.append([out_string, output_func])

        macros_list_copy = copy.deepcopy(macros_list)

        # remove the repeated one
        filter_macros_list = []
        for i, macro in enumerate(macros_list):
            in_others = False
            print('macro', macro) if 0 else None
            for j, others in enumerate(macros_list_copy):
                if i < j and macro[1] in others[1]:
                    in_others = True
                    break
            print('in_others', in_others) if 0 else None
            if not in_others:
                filter_macros_list.append(macro)

        for i in filter_macros_list:
            print(i)

        output_file = FLAGS.save_path + "generated_marco_actions.txt"
        with open(output_file, 'w') as file:      
            for i in filter_macros_list:
                file.write(i[0] + " " + i[1] + "\n")


def run(analyse):
    if not analyse:
        transoform_replays()
    else:
        analyse_data()


if __name__ == '__main__':
    run(analyse=0)
    run(analyse=1)
