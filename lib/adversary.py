# https://github.com/andrewliao11/gail-tf
# import lib.tf_util as U

import tensorflow as tf
import numpy as np
import lib.layer as layer
# ================================================================
# Flat vectors
# ================================================================


def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return intprod(var_shape(x))


def intprod(x):
    return int(np.prod(x))


def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])

# ================================================================
# logit_bernoulli_entropy
# ================================================================


def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent

# ================================================================
# Discriminator
# ================================================================


class TransitionClassifier(object):

    def __init__(self, sess, hidden_size, input_size, output_size,
                 use_norm, pop_batch_norm, entcoeff=0.001, lr_rate=1e-3, scope="adversary"):
        self.scope = scope
        self.observation_shape = (input_size,)  # env.observation_space.shape
        self.actions_shape = (output_size,)  # env.action_space.shape
        self.hidden_size = hidden_size
        self.lr = lr_rate
        self.sess = sess
        self.use_norm = use_norm
        self.pop_batch_norm = pop_batch_norm
        self.build_ph()
        # Build grpah
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        #var_list = self.get_trainable_variables()
        # self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
        #                              self.losses + [flatgrad(self.total_loss, var_list)])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert_actions_ph")

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            data = tf.concat([obs_ph, acs_ph], axis=1)  # concatenate the two input -> form a transition

            '''p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.relu)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)'''

            if self.use_norm:
                data = layer.batch_norm(data, self.pop_batch_norm, 'BN')
            d1 = layer.dense_layer(data, 128, "DenseLayer1", is_training=self.pop_batch_norm, trainable=True,
                                   norm=self.use_norm)
            d2 = layer.dense_layer(d1, 32, "DenseLayer2", is_training=self.pop_batch_norm, trainable=True,
                                   norm=self.use_norm)
            dout = layer.dense_layer(d2, 1, "DenseLayer3", func=None,
                                     is_training=self.pop_batch_norm, trainable=True, norm=None)
        return dout

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def train(self, ob_batch, ac_batch, ob_expert, ac_expert):
        feed_dict = {self.generator_obs_ph: ob_batch,
                     self.generator_acs_ph: ac_batch,
                     self.expert_obs_ph: ob_expert,
                     self.expert_acs_ph: ac_expert,
                     self.pop_batch_norm: True,
                     }
        train_op = [self.train_step] + self.losses
        result = self.sess.run(train_op, feed_dict=feed_dict)
        print('train result:', result)

    def get_reward(self, obs, acs):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obs_ph: obs,
                     self.generator_acs_ph: acs,
                     self.pop_batch_norm: False,
                     }
        reward = self.sess.run(self.reward_op, feed_dict)
        return reward
