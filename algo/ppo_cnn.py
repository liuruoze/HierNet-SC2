import tensorflow as tf
import numpy as np
import copy
import lib.layer as layer
import lib.config as C
import param as P


class Policy_net:

    def __init__(self, name: str, sess, ob_space, act_space_array, activation=tf.nn.relu):
        """
        :param name: string
        """
        self.sess = sess
        self.map_width = 64
        self.map_channels = C.MAP_CHANNELS

        self.use_norm = False
        self.sl_training = False

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, ob_space], name='obs')
            self.map_data = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.map_channels, self.map_width, self.map_width],
                                           name="map_data")

            with tf.variable_scope('policy_net'):
                with tf.variable_scope('controller'):
                    layer_1 = layer.dense_layer(self.obs, 256, "DenseLayer1", func=activation)
                    layer_2 = layer.dense_layer(layer_1, 256, "DenseLayer2", func=activation)
                    self.controller_info = layer.dense_layer(layer_2, 64, "Info", func=None)

                with tf.variable_scope('battle'):
                    self.minimap_info = self.cnn_map(self.map_data)
                    self.battle_info = tf.concat([self.controller_info, self.minimap_info], axis=1)

                    layer_5 = layer.dense_layer(self.battle_info, 256, "DenseLayer1", func=activation)
                    self.battle_probs = layer.dense_layer(layer_5, act_space_array[0], "battle_output",
                                                          func=tf.nn.softmax)
                    self.battle_act = tf.multinomial(tf.log(self.battle_probs), num_samples=1)
                    self.battle_act = tf.reshape(self.battle_act, shape=[-1])

                    layer_6 = layer.dense_layer(self.battle_info, 512, "PosLayer1", func=activation)
                    layer_7 = layer.dense_layer(layer_6, 256, "PosLayer2", func=activation)
                    self.battle_pos_probs = layer.dense_layer(layer_7, act_space_array[1], "battle_pos",
                                                              func=tf.nn.softmax)
                    self.battle_pos = tf.multinomial(tf.log(self.battle_pos_probs), num_samples=1)
                    self.battle_pos = tf.reshape(self.battle_pos, shape=[-1])

            with tf.variable_scope('value_net'):
                layer_1 = layer.dense_layer(self.obs, 256, "DenseLayer1", func=activation)

                minimap_info = self.cnn_map(self.map_data)
                layer_1 = tf.concat([layer_1, minimap_info], axis=1)

                layer_2 = layer.dense_layer(layer_1, 128, "DenseLayer2", func=activation)
                layer_3 = layer.dense_layer(layer_2, 128, "DenseLayer3", func=activation)
                self.v_preds = layer.dense_layer(layer_3, 1, "DenseLayer4", func=None)

            self.scope = tf.get_variable_scope().name

    def get_action(self, obs, map, verbose=True):
        battle_act_probs, battle_act, battle_pos_probs, battle_pos, v_preds \
            = self.sess.run([self.battle_probs, self.battle_act, self.battle_pos_probs, self.battle_pos, self.v_preds],
                            feed_dict={self.obs: obs.reshape([1, -1]), self.map_data: map.reshape(
                                [1, self.map_channels, self.map_width, self.map_width])})
        if verbose:
            print("Battle:", 'act_probs:', battle_act_probs, 'act:', battle_act)
            print("Battle:", 'pos_probs:', battle_pos_probs, 'pos:', battle_pos)
            print("Value:", v_preds)
            print("\n")

        return battle_act[0], battle_pos[0], v_preds[0]

    def get_values(self, obs, map):
        v_preds = self.sess.run(self.v_preds, feed_dict={self.obs: obs.reshape([1, -1]),
                                                         self.map_data: map.reshape(
                                                             [1, self.map_channels, self.map_width, self.map_width])})
        v_preds = np.asscalar(v_preds)
        return v_preds

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def cnn_map(self, map_data, trainable=True):

        map_data = tf.transpose(map_data, [0, 2, 3, 1])
        with tf.variable_scope("cnn"):
            if self.use_norm:
                map_data = layer.batch_norm(map_data, self.sl_training, "BN", trainable=trainable)
            c1 = layer.conv2d_layer(map_data, 3, 32, "Conv1", trainable=trainable)
            c1 = layer.max_pool(c1)
            if self.use_norm:
                c1 = layer.batch_norm(c1, self.sl_training, "Norm1", trainable=trainable)

            c2 = layer.conv2d_layer(c1, 3, 64, "Conv2", trainable=trainable)
            c2 = layer.max_pool(c2)

            if self.use_norm:
                c2 = layer.batch_norm(c2, self.sl_training, "Norm2", trainable=trainable)

            c3 = layer.conv2d_layer(c2, 3, 64, "Conv3", trainable=trainable)
            c3 = layer.max_pool(c3)

            if self.use_norm:
                c3 = layer.batch_norm(c3, self.sl_training, "Norm3", trainable=trainable)

            c4 = layer.conv2d_layer(c3, 3, 3, "Conv4", trainable=trainable)
            if self.use_norm:
                c4 = layer.batch_norm(c4, self.sl_training, "Norm4", trainable=trainable)

            c4 = tf.reshape(c4, [-1, self.map_width * 3])

        return c4


class PPOTrain:

    def __init__(self, name, sess, Policy, Old_Policy, gamma=0.995, clip_value=0.2, c_1=0.01, c_2=1e-6, epoch_num=20):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        :param epoch_num: num for update
        """
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.sess = sess
        self.epoch_num = epoch_num

        self.gamma = P.gamma
        self.lamda = P.lamda
        self.batch_size = P.batch_size
        self.clip_value = P.clip_value
        self.c_1 = P.c_1
        self.c_2 = P.c_2
        self.adam_lr = P.lr
        self.restore_model = P.restore_model

        self.adam_epsilon = 1e-5
        self.update_count = 0

        with tf.variable_scope(name):
            pi_trainable = self.Policy.get_trainable_variables()
            old_pi_trainable = self.Old_Policy.get_trainable_variables()

            # assign_operations for policy parameter values to old policy parameters
            with tf.variable_scope('assign_op'):
                self.assign_ops = []
                for v_old, v in zip(old_pi_trainable, pi_trainable):
                    self.assign_ops.append(tf.assign(v_old, v))

            # inputs for train_op
            with tf.variable_scope('train_inp'):
                self.battle_actions = tf.placeholder(dtype=tf.int32, shape=[None], name='battle_actions')
                self.battle_pos = tf.placeholder(dtype=tf.int32, shape=[None], name='battle_pos')

                self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
                self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
                self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
                self.returns = tf.placeholder(dtype=tf.float32, shape=[None], name='returns')

                # define distribute variable
                self.returns_sum = tf.get_variable(name="returns_sum", shape=[], initializer=tf.zeros_initializer)
                self.loss_p_sum = tf.get_variable(name="loss_p_sum", shape=[], initializer=tf.zeros_initializer)
                self.loss_v_sum = tf.get_variable(name="loss_v_sum", shape=[], initializer=tf.zeros_initializer)
                self.loss_e_sum = tf.get_variable(name="loss_e_sum", shape=[], initializer=tf.zeros_initializer)
                self.loss_all_sum = tf.get_variable(name="loss_all_sum", shape=[], initializer=tf.zeros_initializer)

                self.proc_num = tf.get_variable(name="proc_num", shape=[], initializer=tf.zeros_initializer)

            battle_act_probs = self.Policy.battle_probs
            battle_act_probs_old = self.Old_Policy.battle_probs
            battle_pos_probs = self.Policy.battle_pos_probs
            battle_pos_probs_old = self.Old_Policy.battle_pos_probs

            # probabilities of actions which agent took with policy
            battle_act_probs = battle_act_probs * tf.one_hot(indices=self.battle_actions,
                                                             depth=battle_act_probs.shape[1])
            battle_act_probs = tf.reduce_sum(battle_act_probs, axis=1)
            battle_pos_probs = battle_pos_probs * tf.one_hot(indices=self.battle_pos, depth=battle_pos_probs.shape[1])
            battle_pos_probs = tf.reduce_sum(battle_pos_probs, axis=1)

            act_probs = battle_act_probs * battle_pos_probs

            # probabilities of actions which agent took with old policy
            battle_act_probs_old = battle_act_probs_old * tf.one_hot(indices=self.battle_actions,
                                                                     depth=battle_act_probs_old.shape[1])
            battle_act_probs_old = tf.reduce_sum(battle_act_probs_old, axis=1)
            battle_pos_probs_old = battle_pos_probs_old * tf.one_hot(indices=self.battle_pos,
                                                                     depth=battle_pos_probs_old.shape[1])
            battle_pos_probs_old = tf.reduce_sum(battle_pos_probs_old, axis=1)

            act_probs_old = battle_act_probs_old * battle_pos_probs_old

            with tf.variable_scope('loss'):
                # construct computation graph for loss_clip
                # ratios = tf.divide(act_probs, act_probs_old)
                ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0)) - tf.log(
                    tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value,
                                                  clip_value_max=1 + self.clip_value)
                loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                self.loss_clip = -tf.reduce_mean(loss_clip)

                # construct computation graph for loss of entropy bonus
                battle_act_entropy = -tf.reduce_sum(
                    self.Policy.battle_probs * tf.log(tf.clip_by_value(self.Policy.battle_probs, 1e-10, 1.0)), axis=1)
                battle_pos_entropy = -tf.reduce_sum(
                    self.Policy.battle_pos_probs * tf.log(tf.clip_by_value(self.Policy.battle_pos_probs, 1e-10, 1.0)),
                    axis=1)
                entropy = battle_act_entropy + battle_pos_entropy
                self.entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)

                # construct computation graph for loss of value function
                v_preds = self.Policy.v_preds
                loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
                self.loss_vf = tf.reduce_mean(loss_vf)

                # construct computation graph for loss
                self.total_loss = self.loss_clip + self.c_1 * self.loss_vf - self.c_2 * self.entropy

                self.sum_mean_returns = tf.summary.scalar('mean_return_dis',
                                                          self.returns_sum / (self.proc_num + 0.0001))
                self.sum_p_loss = tf.summary.scalar('policy_loss_dis', self.loss_p_sum / (self.proc_num + 0.0001))
                self.sum_v_loss = tf.summary.scalar('value_loss_dis', self.loss_v_sum / (self.proc_num + 0.0001))
                self.sum_e_loss = tf.summary.scalar('entropy_loss_dis', self.loss_e_sum / (self.proc_num + 0.0001))
                self.sum_total_loss = tf.summary.scalar('total_loss_dis', self.loss_all_sum / (self.proc_num + 0.0001))

            self.merged_dis = tf.summary.merge(
                [self.sum_mean_returns, self.sum_p_loss, self.sum_v_loss, self.sum_e_loss,
                 self.sum_total_loss])

            optimizer = tf.train.AdamOptimizer(learning_rate=self.adam_lr, epsilon=self.adam_epsilon)

            self.gradients = optimizer.compute_gradients(self.total_loss, var_list=pi_trainable)

            self.train_op = optimizer.minimize(self.total_loss, var_list=pi_trainable)
            self.train_value_op = optimizer.minimize(self.loss_vf, var_list=pi_trainable)

    def get_loss(self, obs, map_data, battle_actions, battle_pos, gaes, rewards, v_preds_next):
        loss_p, loss_v, loss_e, loss_all = self.sess.run([self.loss_clip, self.loss_vf, self.entropy, self.total_loss],
                                                         feed_dict={self.Policy.obs: obs,
                                                                    self.Policy.map_data: map_data,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.Old_Policy.map_data: map_data,
                                                                    self.battle_actions: battle_actions,
                                                                    self.battle_pos: battle_pos,
                                                                    self.rewards: rewards,
                                                                    self.v_preds_next: v_preds_next,
                                                                    self.gaes: gaes}
                                                         )
        return loss_p, loss_v, loss_e, loss_all

    def train(self, obs, map_data, battle_actions, battle_pos, gaes, rewards, v_preds_next):
        _, total_loss = self.sess.run([self.train_op, self.total_loss], feed_dict={self.Policy.obs: obs,
                                                                                   self.Policy.map_data: map_data,
                                                                                   self.Old_Policy.obs: obs,
                                                                                   self.Old_Policy.map_data: map_data,
                                                                                   self.battle_actions: battle_actions,
                                                                                   self.battle_pos: battle_pos,
                                                                                   self.rewards: rewards,
                                                                                   self.v_preds_next: v_preds_next,
                                                                                   self.gaes: gaes})
        return total_loss

    def train_value(self, obs, map_data, gaes, rewards, v_preds_next):
        _, value_loss = self.sess.run([self.train_value_op, self.loss_vf], feed_dict={self.Policy.obs: obs,
                                                                                      self.Policy.map_data: map_data,
                                                                                      self.Old_Policy.obs: obs,
                                                                                      self.Old_Policy.map_data: map_data,
                                                                                      self.rewards: rewards,
                                                                                      self.v_preds_next: v_preds_next,
                                                                                      self.gaes: gaes})
        return value_loss

    def get_summary_dis(self):
        return self.sess.run(self.merged_dis)

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return self.sess.run(self.assign_ops)

    def reset_mean_returns(self):
        self.sess.run(self.returns_sum.assign(0))
        self.sess.run(self.loss_p_sum.assign(0))
        self.sess.run(self.loss_v_sum.assign(0))
        self.sess.run(self.loss_e_sum.assign(0))
        self.sess.run(self.loss_all_sum.assign(0))

        self.sess.run(self.proc_num.assign(0))

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * self.lamda * gaes[t + 1]
        return gaes

    def ppo_train_dis(self, observations, map_data, battle_actions, battle_pos, rewards, v_preds, v_preds_next,
                      gaes, returns, verbose=False):
        if verbose:
            print('PPO train now..........')

        # convert list to numpy array for feeding tf.placeholder
        # ob_space = C._SIZE_HIGH_NET_INPUT + C._SIZE_HIGH_NET_OUT + C._SIZE_POP_NET_INPUT
        # observations = np.reshape(observations, newshape=[-1] + list(ob_space))
        observations = np.array(observations).astype(dtype=np.float32)
        map_data = np.array(map_data).astype(dtype=np.float32)

        battle_actions = np.array(battle_actions).astype(dtype=np.int32)
        battle_pos = np.array(battle_pos).astype(dtype=np.int32)

        gaes = np.array(gaes).astype(dtype=np.float32).reshape(-1)
        gaes = (gaes - gaes.mean()) / gaes.std()
        rewards = np.array(rewards).astype(dtype=np.float32).reshape(-1)
        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32).reshape(-1)
        inp = [observations, map_data, battle_actions, battle_pos, gaes, rewards, v_preds_next]

        if observations.shape[0] <= 0:
            return

        # self.assign_policy_parameters()
        # train
        # batch_size = max(observations.shape[0] // 10, self.batch_size)
        batch_size = self.batch_size
        # print('batch_size is:', batch_size)
        for epoch in range(self.epoch_num + 10):
            # sample indices from [low, high)
            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=batch_size)
            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data

            if self.restore_model and self.update_count < -1:
                value_loss = self.train_value(obs=sampled_inp[0],
                                              map_data=sampled_inp[1],
                                              gaes=sampled_inp[4],
                                              rewards=sampled_inp[5],
                                              v_preds_next=sampled_inp[6])
            else:
                total_loss = self.train(obs=sampled_inp[0],
                                        map_data=sampled_inp[1],
                                        battle_actions=sampled_inp[2],
                                        battle_pos=sampled_inp[3],
                                        gaes=sampled_inp[4],
                                        rewards=sampled_inp[5],
                                        v_preds_next=sampled_inp[6])
                if verbose:
                    print('total_loss:', total_loss)

        self.update_count += 1

        sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)
        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
        loss_p, loss_v, loss_e, loss_all = self.get_loss(obs=sampled_inp[0],
                                                         map_data=sampled_inp[1],
                                                         battle_actions=sampled_inp[2],
                                                         battle_pos=sampled_inp[3],
                                                         gaes=sampled_inp[4],
                                                         rewards=sampled_inp[5],
                                                         v_preds_next=sampled_inp[6])

        self.sess.run(self.loss_p_sum.assign_add(loss_p))
        self.sess.run(self.loss_v_sum.assign_add(loss_v))
        self.sess.run(self.loss_e_sum.assign_add(loss_e))
        self.sess.run(self.loss_all_sum.assign_add(loss_all))
        print('returns:', returns)
        print('np.mean(returns):', np.mean(returns))
        self.sess.run(self.returns_sum.assign_add(np.mean(returns)))
        self.sess.run(self.proc_num.assign_add(1))

        if verbose:
            print('PPO train end..........')
        return
