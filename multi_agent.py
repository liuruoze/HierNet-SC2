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
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

import lib.utils as U
from new_network import HierNetwork
from lib import config as C
from lib import transform_pos as T
from lib import human_expert_statis as HES
from lib import option as M
from lib import environment
from lib import my_sc2_env as sc2_env
from lib.replay_buffer import Buffer, Cnn_Buffer

from logging import warning as logging

import param as P


class MultiAgent(base_agent.BaseAgent):
    """My first agent for starcraft."""

    def __init__(self, index=0, rl_training=False, restore_model=False, global_buffer=None, net=None):
        super(MultiAgent, self).__init__()
        self.net = net
        self.index = index
        self.global_buffer = global_buffer
        self.restore_model = restore_model

        # count num
        self.step = 0

        self.controller_wait_secs = 8
        self.base_wait_secs = 1
        self.battle_wait_secs = 1

        self.controller_flag = False
        self.base_flag = True
        self.battle_flag = True

        self.env = None
        self.obs = None

        self.use_fight = False
        self.restore_fight = False

        # buffer
        self.controller_buffer = Buffer()
        self.base_buffer = Buffer()
        self.tech_buffer = Buffer()
        self.pop_buffer = Buffer()
        self.battle_buffer = Buffer()

        self.fight_buffer = Cnn_Buffer()
        self.fight_temp = None

        self.num_players = 2
        self.on_select = None
        self._result = None
        self.is_end = False

        self.rl_training = rl_training
        self.update_counter = 0

        self.use_triple_reward = False
        self.use_only_result_reward = True

        self.use_battle_net = True

        self.use_trick_1 = False
        self.use_trick_2 = False
        self.use_trick_3 = False

    def reset(self):
        super(MultiAgent, self).reset()
        self.step = 0
        self.obs = None
        self._result = None
        self.is_end = False

        self.controller_flag = False
        self.base_flag = True
        self.battle_flag = True

        self.controller_buffer.reset()
        self.base_buffer.reset()
        self.tech_buffer.reset()
        self.pop_buffer.reset()
        self.battle_buffer.reset()

        self.fight_buffer.reset()

        self.fight_temp = None

    def set_env(self, env):
        self.env = env

    def init_network(self):
        self.net.initialize()
        if self.restore_model:
            self.net.restore_controller()
            self.net.restore_base()
            self.net.restore_tech()
            self.net.restore_pop()
            self.net.restore_battle()

            if self.restore_fight:
                self.net.restore_fight()

    def reset_old_network(self):
        self.net.reset_old_network()

    def save_model(self):
        self.net.save_controller()
        self.net.save_base()
        self.net.save_tech()
        self.net.save_pop()
        self.net.save_battle()

        if self.use_fight:
            self.net.save_fight()

    def update_network(self, result_list):
        self.update_counter += 1

        if self.use_trick_1:
            pass

        if P.use_alternative_update:
            logging('use_alternative_update, update_counter: %d', int(self.update_counter))
            if self.update_counter % 20 < 10:
                logging('only update controller')
                self.net.Update_controller(self.global_buffer.controller_buffer)
            else:
                logging('update all')
                self.net.Update_controller(self.global_buffer.controller_buffer)
                self.net.Update_base_net(self.global_buffer.base_buffer)
                self.net.Update_tech_net(self.global_buffer.tech_buffer)
                self.net.Update_pop_net(self.global_buffer.pop_buffer)
                self.net.Update_battle_net(self.global_buffer.battle_buffer)
                if self.use_fight:
                    self.net.Update_fight(self.global_buffer.fight_buffer)
        else:

            self.net.Update_controller(self.global_buffer.controller_buffer)
            self.net.Update_base_net(self.global_buffer.base_buffer)
            self.net.Update_tech_net(self.global_buffer.tech_buffer)
            self.net.Update_pop_net(self.global_buffer.pop_buffer)
            self.net.Update_battle_net(self.global_buffer.battle_buffer)

            if self.use_fight:
                self.net.Update_fight(self.global_buffer.fight_buffer)

        self.net.Update_result(result_list)

    def update_summary(self, counter):
        return self.net.Update_summary(counter)

    def get_controller_input(self):
        controller_input, _, _ = U.get_input(self.obs)
        return controller_input

    def get_base_input(self):
        high_input, tech_cost, pop_num = U.get_input(self.obs)
        base_input = np.concatenate([high_input, tech_cost, pop_num], axis=0)
        return base_input

    def get_tech_input(self):
        high_input, tech_cost, _ = U.get_input(self.obs)
        tech_obs = np.concatenate([high_input, tech_cost], axis=0)
        return tech_obs

    def get_pop_input(self):
        high_input, _, pop_num = U.get_input(self.obs)
        pop_obs = np.concatenate([high_input, pop_num], axis=0)
        return pop_obs

    def get_battle_input(self):
        high_input, tech_cost, pop_num = U.get_input(self.obs)
        battle_input = np.concatenate([high_input, tech_cost, pop_num], axis=0)
        return battle_input

    def tech_step(self, tech_action):
        # to execute a tech_action
        # [pylon, gas1, gas2, gateway, cyber]

        if tech_action == 0:  # pylon
            no_unit_index = U.get_unit_mask_screen(self.obs, size=2)
            pos = U.get_pos(no_unit_index)
            M.build_by_idle_worker(self, C._BUILD_PYLON_S, pos)

        elif tech_action == 1 and not U.find_gas(self.obs, 1):  # gas_1
            gas_1 = U.find_gas_pos(self.obs, 1)
            gas_1_pos = T.world_to_screen_pos(self.env.game_info, gas_1.pos, self.obs)
            M.build_by_idle_worker(self, C._BUILD_ASSIMILATOR_S, gas_1_pos)

        elif tech_action == 1 and not U.find_gas(self.obs, 2):  # gas_2
            gas_2 = U.find_gas_pos(self.obs, 2)
            gas_2_pos = T.world_to_screen_pos(self.env.game_info, gas_2.pos, self.obs)
            M.build_by_idle_worker(self, C._BUILD_ASSIMILATOR_S, gas_2_pos)

        elif tech_action == 2:  # gateway
            power_index = U.get_power_mask_screen(self.obs, size=5)
            pos = U.get_pos(power_index)
            M.build_by_idle_worker(self, C._BUILD_GATEWAY_S, pos)

        elif tech_action == 3:  # cyber
            power_index = U.get_power_mask_screen(self.obs, size=3)
            pos = U.get_pos(power_index)
            M.build_by_idle_worker(self, C._BUILD_CYBER_S, pos)

        else:
            self.safe_action(C._NO_OP, 0, [])

    def pop_step(self, pop_action):
        # to execute a pop_action
        # [ mineral_probe, zealot, stalker]

        if pop_action == 0:  # mineral_probe
            M.mineral_worker(self)

        elif pop_action == 1:  # zealot
            M.train_army(self, C._TRAIN_ZEALOT)

        elif pop_action == 2:  # stalker
            M.train_army(self, C._TRAIN_STALKER)

        else:
            self.safe_action(C._NO_OP, 0, [])

    def fight_step(self, fight_action, fight_pos):
        army, army_pos = U.get_best_army(self.env.game_info, self.obs)
        if army:
            self.safe_action(C._MOVE_CAMERA, 0, [army_pos])

            if fight_action == 1:  # attack
                M.attack_step(self, fight_pos)

            elif fight_action == 2:  # retreat
                M.retreat_step(self, fight_pos)

        else:
            self.safe_action(C._NO_OP, 0, [])

    def play(self):
        # M.set_source(self)

        self.safe_action(C._NO_OP, 0, [])
        while True:
            controller_input = self.get_controller_input()
            _, controller_pred, controller_v_pred = self.net.controller.act(controller_input, verbose=False)
            controller_reward = 0
            # enemy_num = controller_input[18]
            # army_count = controller_input[10]
            self.controller_flag = False

            if controller_pred == 0:  # base net
                self.safe_action(C._MOVE_CAMERA, 0, [C.base_camera_pos])
                while (not self.controller_flag) and (not self.is_end):
                    self.safe_action(C._NO_OP, 0, [])

                    if self.base_flag:
                        base_input = self.get_base_input()
                        _, base_pred, base_v_pred = self.net.base_net.act(base_input, verbose=False)
                        base_reward = 0
                        self.base_flag = False

                        if base_pred == 0:
                            base_reward += self.tech_action()
                        elif base_pred == 1:
                            base_reward += self.pop_action()
                        else:
                            self.safe_action(C._NO_OP, 0, [])

                        now_base_input = self.get_base_input()
                        base_v_pred_next = self.net.base_net.get_values(now_base_input)
                        base_v_pred_next = self.get_values(base_v_pred_next)

                        if self.use_only_result_reward:
                            base_reward = 0

                        self.base_buffer.append(base_input, base_pred, base_reward, base_v_pred, base_v_pred_next)
                        controller_reward += base_reward

            else:               # battle net
                while (not self.controller_flag) and (not self.is_end):
                    self.safe_action(C._NO_OP, 0, [])

                    if self.battle_flag:

                        if self.use_battle_net:        
                            battle_input = self.get_battle_input()
                            _, battle_pred, battle_v_pred = self.net.battle_net.act(battle_input, verbose=False)
                            battle_reward = 0
                            self.battle_flag = False

                            if battle_pred == 0:
                                battle_reward += self.pop_action() 
                            elif battle_pred == 1:
                                battle_reward += self.fight_action()
                            else:
                                self.safe_action(C._NO_OP, 0, [])

                            now_battle_input = self.get_battle_input()
                            battle_v_pred_next = self.net.battle_net.get_values(now_battle_input)
                            battle_v_pred_next = self.get_values(battle_v_pred_next)

                            if self.use_only_result_reward:
                                battle_reward = 0

                            self.battle_buffer.append(battle_input, battle_pred, battle_reward, battle_v_pred, battle_v_pred_next)
                            controller_reward += battle_reward
                        else:
                            battle_reward = 0
                            self.battle_flag = False
                            battle_reward += self.fight_action()

                            if self.use_trick_2:
                                self.pop_action()  # TODO: remove

                            controller_reward += battle_reward

            now_controller_input = self.get_controller_input()
            controller_v_pred_next = self.net.controller.get_values(now_controller_input)
            controller_v_pred_next = self.get_values(controller_v_pred_next)

            if self.use_only_result_reward:
                controller_reward = 0

            self.controller_buffer.append(controller_input, controller_pred, controller_reward, controller_v_pred, controller_v_pred_next)

            if self.is_end:
                logging('self.is_end')

                if self.rl_training:
                    logging('self.rl_training')
                    if not self.use_only_result_reward:
                        logging('self.use_only_result_reward %d', int(self.use_only_result_reward))
                        self.controller_buffer.rewards[-1] -= 10 * self.obs.raw_observation.observation.game_loop // (22.4 * 60)
                        self.controller_buffer.rewards[-1] += 50 * self.result['win']

                        self.base_buffer.rewards[-1] -= 10 * self.obs.raw_observation.observation.game_loop // (22.4 * 60)
                        self.base_buffer.rewards[-1] += 50 * self.result['win']

                        self.battle_buffer.rewards[-1] -= 10 * self.obs.raw_observation.observation.game_loop // (22.4 * 60)
                        self.battle_buffer.rewards[-1] += 50 * self.result['win']

                        self.tech_buffer.rewards[-1] += 50 * self.result['win']
                        self.pop_buffer.rewards[-1] += 50 * self.result['win']
                    else:
                        logging('self.use_only_result_reward %d', int(self.use_only_result_reward))
                        self.controller_buffer.rewards[-1] += P.reward_weight[0] * self.result['reward']

                        logging('self.controller_buffer.rewards' + str(self.controller_buffer.rewards)[1:-1])

                        self.base_buffer.rewards[-1] += P.reward_weight[1] * self.result['reward']
                        self.battle_buffer.rewards[-1] += P.reward_weight[4] * self.result['reward']

                        self.tech_buffer.rewards[-1] += P.reward_weight[2] * self.result['reward']
                        self.pop_buffer.rewards[-1] += P.reward_weight[3] * self.result['reward']

                    self.global_buffer.append(self.controller_buffer, self.base_buffer, self.tech_buffer, self.pop_buffer, self.battle_buffer)

                break

    def tech_action(self):
        # act
        tech_obs = self.get_tech_input()
        act_array, act, v_pred = self.net.tech_net.act(tech_obs, verbose=False)
        self.tech_step(act)

        # TODO: next state should be the action call next once
        # get next state and next values
        now_tech_obs = self.get_tech_input()
        v_pred_next = self.net.tech_net.get_values(now_tech_obs)
        v_pred_next = self.get_values(v_pred_next)

        reward = 0
        if self.rl_training and self.obs.raw_observation.observation.game_loop < 10000:
            # get reward
            pylon_num_idx = 14
            gas_num_idx = 15
            gateway_num_idx = 16
            cyber_num_idx = 17

            last_building_count = tech_obs[[pylon_num_idx, gas_num_idx, gateway_num_idx, cyber_num_idx]]
            last_building_count += tech_obs[-4:]

            now_building_count = now_tech_obs[[pylon_num_idx, gas_num_idx, gateway_num_idx, cyber_num_idx]]
            now_building_count += now_tech_obs[-4:]

            if now_building_count[0] <= HES.pylon:
                reward += now_building_count[0] - last_building_count[0]
            else:
                reward -= now_building_count[0] - last_building_count[0]

            if now_building_count[1] <= HES.gas:
                reward += now_building_count[1] - last_building_count[1]
            else:
                reward -= now_building_count[1] - last_building_count[1]

            if now_building_count[2] <= HES.gateway:
                reward += now_building_count[2] - last_building_count[2]
            else:
                reward -= now_building_count[2] - last_building_count[2]

            if now_building_count[3] <= HES.cyber:
                reward += (now_building_count[3] - last_building_count[3]) * 1
            else:
                reward -= (now_building_count[3] - last_building_count[3]) * 1

        if self.use_only_result_reward:
            reward = 0

        self.tech_buffer.append(tech_obs, act, reward, v_pred, v_pred_next)
        return reward

    def pop_action(self):
        # act
        pop_obs = self.get_pop_input()
        act_array, act, v_pred = self.net.pop_net.act(pop_obs, verbose=False)
        self.pop_step(act)

        # get next state and next values
        now_pop_obs = self.get_pop_input()
        v_pred_next = self.net.pop_net.get_values(now_pop_obs)
        v_pred_next = self.get_values(v_pred_next)

        reward = 0
        if self.rl_training:
            worker_index = C._SIZE_HIGH_NET_INPUT
            army_index = 10

            last_worker_count = pop_obs[worker_index] + pop_obs[worker_index + 9]
            now_worker_count = now_pop_obs[worker_index] + now_pop_obs[worker_index + 9]
            ideal_worker_count = now_pop_obs[worker_index + 1]
            last_army_count = pop_obs[army_index] + pop_obs[worker_index + 10] + pop_obs[worker_index + 11]
            now_army_count = now_pop_obs[army_index] + now_pop_obs[worker_index + 10] + now_pop_obs[worker_index + 11]

            if now_worker_count <= ideal_worker_count:
                reward += now_worker_count - last_worker_count
            elif act == 0:
                reward -= now_worker_count - last_worker_count

            if now_army_count <= HES.army:
                reward += now_army_count - last_army_count

        if self.use_only_result_reward:
            reward = 0

        self.pop_buffer.append(pop_obs, act, reward, v_pred, v_pred_next)

        return reward

    def fight_action(self):
        reward = 0

        # move camera
        army, army_pos = U.get_best_army(self.env.game_info, self.obs)
        if army:
            self.safe_action(C._MOVE_CAMERA, 0, [army_pos])
            self.select(C._SELECT_ARMY, C._ARMY_INDEX, [[0]])

        if not self.use_fight:
            M.attack_step(self)

            if self.use_trick_3:
                if self.obs.raw_observation.observation.player_common.army_count >= 8:
                    # M.control_step(self)
                    # M.attack_step(self)
                    reward = 1

        return reward

    def set_flag(self):
        if self.step % C.time_wait(self.controller_wait_secs) == 1:
            self.controller_flag = True

        if self.step % C.time_wait(self.base_wait_secs) == 1:
            self.base_flag = True

        if self.step % C.time_wait(self.battle_wait_secs) == 1:
            self.battle_flag = True

    def safe_action(self, action, unit_type, args):
        if M.check_params(self, action, unit_type, args, 1):
            obs = self.env.step([sc2_actions.FunctionCall(action, args)])[0]
            self.obs = obs
            self.step += 1
            self.update_result()
            self.set_flag()

    def select(self, action, unit_type, args):
        # safe select
        if M.check_params(self, action, unit_type, args, 0):
            self.obs = self.env.step([sc2_actions.FunctionCall(action, args)])[0]
            self.on_select = unit_type
            self.update_result()
            self.step += 1
            self.set_flag()

        # else:
        # print('Unavailable_actions id:', action, ' and type:', unit_type, ' and args:', args)

    @property
    def result(self):
        return self._result

    def update_result(self):
        if self.obs is None:
            return
        if self.obs.last() or self.env.state == environment.StepType.LAST:
            self.is_end = True
            outcome = 0
            o = self.obs.raw_observation
            player_id = o.observation.player_common.player_id
            for r in o.player_result:
                if r.player_id == player_id:
                    outcome = sc2_env._possible_results.get(r.result, 0)
            frames = o.observation.game_loop
            result = dict()
            result['outcome'] = outcome
            result['reward'] = self.obs.reward
            result['frames'] = frames

            if not self.use_triple_reward:
                result['win'] = -1   # TODO, change it
                if result['reward'] == 1:
                    result['win'] = 1
            else:
                result['win'] = self.obs.reward

            self._result = result
            print('play end, total return', self.obs.reward)
            self.step = 0

    def get_values(self, values):
        # check if the game is end
        if self.is_end:
            return 0
        else:
            return values
