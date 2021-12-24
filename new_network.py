import platform

import numpy as np
import tensorflow as tf

import lib.config as C
import lib.utils as U
from algo.ppo import Policy_net, PPOTrain
from algo.ppo_cnn import Policy_net as Policy_net_cnn
from algo.ppo_cnn import PPOTrain as PPOTrain_cnn
import param as P


class HierNetwork(object):

    def __init__(self, sess=None, summary_writer=tf.summary.FileWriter("logs/"), rl_training=False, reuse=False,
                 cluster=None, index=0, device='/gpu:0'):
        self.system = platform.system()

        self.controller_model_path_load = C._LOAD_MODEL_PATH + "controller/probe"
        self.base_model_path_load = C._LOAD_MODEL_PATH + "base/probe"
        self.tech_model_path_load = C._LOAD_MODEL_PATH + "tech/probe"
        self.pop_model_path_load = C._LOAD_MODEL_PATH + "pop/probe"
        self.battle_model_path_load = C._LOAD_MODEL_PATH + "battle/probe"
        self.fight_model_path_load = C._LOAD_MODEL_PATH + "fight/probe"

        self.controller_model_path_save = C._SAVE_MODEL_PATH + "controller/probe"
        self.base_model_path_save = C._SAVE_MODEL_PATH + "base/probe"
        self.tech_model_path_save = C._SAVE_MODEL_PATH + "tech/probe"
        self.pop_model_path_save = C._SAVE_MODEL_PATH + "pop/probe"
        self.battle_model_path_save = C._SAVE_MODEL_PATH + "battle/probe"
        self.fight_model_path_save = C._SAVE_MODEL_PATH + "fight/probe"

        self.rl_training = rl_training

        self.reuse = reuse
        self.sess = sess
        self.cluster = cluster
        self.index = index
        self.device = device

        self.use_fight_net = False

        self._create_graph()

        self.rl_saver = tf.train.Saver()
        self.summary_writer = summary_writer

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def reset_old_network(self):
        self.controller_ppo.assign_policy_parameters()
        self.base_ppo.assign_policy_parameters()
        self.tech_ppo.assign_policy_parameters()
        self.pop_ppo.assign_policy_parameters()
        self.battle_ppo.assign_policy_parameters()

        self.controller_ppo.reset_mean_returns()
        self.base_ppo.reset_mean_returns()
        self.tech_ppo.reset_mean_returns()
        self.pop_ppo.reset_mean_returns()
        self.battle_ppo.reset_mean_returns()

        self.sess.run(self.results_sum.assign(0))
        self.sess.run(self.game_num.assign(0))

    def _create_graph(self):
        if self.reuse:
            tf.get_variable_scope().reuse_variables()
            assert tf.get_variable_scope().reuse

        # with tf.device("/job:ps/task:0"):
        worker_device = "/job:worker/task:%d" % self.index + self.device
        print("worker_device:", worker_device)
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device, cluster=self.cluster)):
            self.results_sum = tf.get_variable(name="results_sum", shape=[], initializer=tf.zeros_initializer)
            self.game_num = tf.get_variable(name="game_num", shape=[], initializer=tf.zeros_initializer)

            self.global_steps = tf.get_variable(name="iter_steps", shape=[], dtype=tf.int32, 
                                                initializer=tf.zeros_initializer, trainable=False)

            self.mean_win_rate = tf.summary.scalar('mean_win_rate_dis', self.results_sum / self.game_num)
            self.merged = tf.summary.merge([self.mean_win_rate])

            scope = "Controller"
            with tf.variable_scope(scope):
                ob_space = C._SIZE_HIGH_NET_INPUT
                act_space = C._SIZE_CONTROLLER_OUT
                self.controller = Policy_net('policy', self.sess, ob_space, act_space)
                self.controller_old = Policy_net('old_policy', self.sess, ob_space, act_space)
                self.controller_ppo = PPOTrain('PPO', self.sess, self.controller, self.controller_old, epoch_num=P.update_num[0], lr=P.lr_list[0])
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.controller_saver = tf.train.Saver(var_list=var_list)

            scope = "Base_net"
            with tf.variable_scope(scope):
                ob_space = C._SIZE_HIGH_NET_INPUT + C._SIZE_TECH_NET_INPUT + C._SIZE_POP_NET_INPUT
                act_space = C._SIZE_BASE_NET_OUT
                self.base_net = Policy_net('policy', self.sess, ob_space, act_space)
                self.base_net_old = Policy_net('old_policy', self.sess, ob_space, act_space)
                self.base_ppo = PPOTrain('PPO', self.sess, self.base_net, self.base_net_old, epoch_num=P.update_num[1], lr=P.lr_list[1])
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.base_saver = tf.train.Saver(var_list=var_list)

            scope = "Tech_net"
            with tf.variable_scope(scope):
                ob_space = C._SIZE_HIGH_NET_INPUT + C._SIZE_TECH_NET_INPUT
                act_space = C._SIZE_TECH_NET_OUT
                self.tech_net = Policy_net('policy', self.sess, ob_space, act_space)
                self.tech_net_old = Policy_net('old_policy', self.sess, ob_space, act_space)
                self.tech_ppo = PPOTrain('PPO', self.sess, self.tech_net, self.tech_net_old, epoch_num=P.update_num[2], lr=P.lr_list[2])
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.tech_saver = tf.train.Saver(var_list=var_list)

            scope = "Pop_net"
            with tf.variable_scope(scope):
                ob_space = C._SIZE_HIGH_NET_INPUT + C._SIZE_POP_NET_INPUT
                act_space = C._SIZE_POP_NET_OUT
                self.pop_net = Policy_net('policy', self.sess, ob_space, act_space)
                self.pop_net_old = Policy_net('old_policy', self.sess, ob_space, act_space)
                self.pop_ppo = PPOTrain('PPO', self.sess, self.pop_net, self.pop_net_old, epoch_num=P.update_num[3], lr=P.lr_list[3])
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.pop_saver = tf.train.Saver(var_list=var_list)

            scope = "Battle_net"
            with tf.variable_scope(scope):
                ob_space = C._SIZE_HIGH_NET_INPUT + C._SIZE_TECH_NET_INPUT + C._SIZE_POP_NET_INPUT
                act_space = C._SIZE_BATTLE_NET_OUT
                self.battle_net = Policy_net('policy', self.sess, ob_space, act_space)
                self.battle_net_old = Policy_net('old_policy', self.sess, ob_space, act_space)
                self.battle_ppo = PPOTrain('PPO', self.sess, self.battle_net, self.battle_net_old, epoch_num=P.update_num[4], lr=P.lr_list[4])
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.battle_saver = tf.train.Saver(var_list=var_list)

            scope = "Fight_net"
            if self.use_fight_net:
                with tf.variable_scope(scope):
                    ob_space = C._SIZE_HIGH_NET_INPUT + C._SIZE_TECH_NET_INPUT + C._SIZE_POP_NET_INPUT
                    act_space_array = [C._SIZE_FIGHT_NET_OUT, 8]
                    self.fight_net = Policy_net_cnn('policy', self.sess, ob_space, act_space_array)
                    self.fight_net_old = Policy_net_cnn('old_policy', self.sess, ob_space, act_space_array)
                    self.fight_ppo = PPOTrain_cnn('PPO', self.sess, self.fight_net, self.fight_net_old)
                fight_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
                self.fight_saver = tf.train.Saver(var_list=fight_var_list)

    def Update_result(self, result_list):
        self.sess.run(self.results_sum.assign_add(result_list.count(1)))
        self.sess.run(self.game_num.assign_add(len(result_list)))

    def Update_summary(self, counter):
        print("Update summary........")

        controller_summary = self.controller_ppo.get_summary_dis()
        self.summary_writer.add_summary(controller_summary, counter)

        base_summary = self.base_ppo.get_summary_dis()
        self.summary_writer.add_summary(base_summary, counter)

        tech_summary = self.tech_ppo.get_summary_dis()
        self.summary_writer.add_summary(tech_summary, counter)

        pop_summary = self.pop_ppo.get_summary_dis()
        self.summary_writer.add_summary(pop_summary, counter)

        battle_summary = self.battle_ppo.get_summary_dis()
        self.summary_writer.add_summary(battle_summary, counter)

        summary = self.sess.run(self.merged)
        self.summary_writer.add_summary(summary, counter)

        print("Update summary finished!")

        self.sess.run(self.global_steps.assign(counter))
        steps = int(self.sess.run(self.global_steps))
        win_game = int(self.sess.run(self.results_sum))
        all_game = int(self.sess.run(self.game_num))
        #print('all_game:', all_game)
        win_rate = win_game / float(all_game) if all_game != 0 else 0.

        return steps, win_rate

    def Update_controller(self, buffer):
        print("Update controller...............")
        print(len(buffer.observations))
        self.controller_ppo.ppo_train_dis(buffer.observations, buffer.actions, buffer.rewards,
                                          buffer.values, buffer.values_next, buffer.gaes, buffer.returns, buffer.return_values)

    def Update_base_net(self, buffer):
        print("Update base net...............")
        print(len(buffer.observations))
        self.base_ppo.ppo_train_dis(buffer.observations, buffer.actions, buffer.rewards, buffer.values,
                                    buffer.values_next, buffer.gaes, buffer.returns, buffer.return_values)

    def Update_tech_net(self, buffer):
        print("Update tech net...............")
        print(len(buffer.observations))
        self.tech_ppo.ppo_train_dis(buffer.observations, buffer.actions, buffer.rewards, buffer.values,
                                    buffer.values_next, buffer.gaes, buffer.returns, buffer.return_values)

    def Update_pop_net(self, buffer):
        print("Update pop net...............")
        print(len(buffer.observations))
        self.pop_ppo.ppo_train_dis(buffer.observations, buffer.actions, buffer.rewards, buffer.values,
                                   buffer.values_next, buffer.gaes, buffer.returns, buffer.return_values)

    def Update_battle_net(self, buffer):
        print("Update battle net...............")
        print(len(buffer.observations))
        self.battle_ppo.ppo_train_dis(buffer.observations, buffer.actions, buffer.rewards, buffer.values,
                                      buffer.values_next, buffer.gaes, buffer.returns, buffer.return_values)

    def Update_fight_net(self, buffer):
        print("Update fight net...............")
        print('rewards:', buffer.rewards)
        self.fight_ppo.ppo_train_dis(buffer.observations, buffer.map_data, buffer.battle_actions, buffer.battle_pos,
                                     buffer.rewards, buffer.values, buffer.values_next, buffer.gaes, buffer.returns, buffer.return_values)

    def save_controller(self):
        self.controller_saver.save(self.sess, self.controller_model_path_save)
        print("controller has been saved in", self.controller_model_path_save)

    def save_base(self):
        self.base_saver.save(self.sess, self.base_model_path_save)
        print("base_net has been saved in", self.base_model_path_save)

    def save_tech(self):
        self.tech_saver.save(self.sess, self.tech_model_path_save)
        print("tech_net has been saved in", self.tech_model_path_save)

    def save_pop(self):
        self.pop_saver.save(self.sess, self.pop_model_path_save)
        print("pop_net has been saved in", self.pop_model_path_save)

    def save_battle(self):
        self.battle_saver.save(self.sess, self.battle_model_path_save)
        print("Battle_net has been saved in", self.battle_model_path_save)

    def save_fight(self):
        self.fight_saver.save(self.sess, self.fight_model_path_save)
        print("Fight_net has been saved in", self.fight_model_path_save)

    def restore_controller(self):
        self.controller_saver.restore(self.sess, self.controller_model_path_load)
        print("Restore controller from", self.controller_model_path_load)

    def restore_base(self):
        self.base_saver.restore(self.sess, self.base_model_path_load)
        print("Restore base from", self.base_model_path_load)

    def restore_tech(self):
        self.tech_saver.restore(self.sess, self.tech_model_path_load)
        print("Restore tech from", self.tech_model_path_load)

    def restore_pop(self):
        self.pop_saver.restore(self.sess, self.pop_model_path_load)
        print("Restore pop from", self.pop_model_path_load)

    def restore_battle(self):
        self.battle_saver.restore(self.sess, self.battle_model_path_load)
        print("Battle_net has been restored from", self.battle_model_path_load)

    def restore_fight(self):
        self.fight_saver.restore(self.sess, self.fight_model_path_load)
        print("Fight_net has been saved in", self.fight_model_path_load)
