from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

import tensorflow as tf
from absl import app
from absl import flags
from pysc2 import maps
from pysc2.lib import stopwatch

import lib.config as C
import param as P
import multi_agent as multi_agent
# from pysc2.env import sc2_env
from lib import my_sc2_env as sc2_env
from lib.replay_buffer import Global_Buffer
from new_network import HierNetwork

from datetime import datetime
import multiprocessing as mp
import numpy as np
from logging import warning as logging

import server_param as SP

#USED_DEVICES = "0,1,2,3"
USED_DEVICES = "4,5,6,7"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES

FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_integer("num_for_update", 1000, "Number of episodes for each train.")
flags.DEFINE_string("log_path", "./logs/", "Path for log.")
flags.DEFINE_string("device", "0,1,2,3", "Device for training.")

flags.DEFINE_string("map", "Simple64", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")

flags.DEFINE_enum("agent_race", "P", sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", "T", sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", "A", sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("max_agent_steps", 18000, "Total agent steps.")
flags.DEFINE_integer("max_iters", 100, "the rl agent max run iters")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("save_replay", False, "Whether to replays_save a replay at the end.")
flags.DEFINE_string("replay_dir", "multi-agent/", "dir of replay to replays_save.")

flags.DEFINE_string("restore_model_path", "./model/20211130-131356/", "path for restore model")
flags.DEFINE_bool("restore_model", False, "Whether to restore old model")

flags.DEFINE_integer("parallel", 10, "How many processes to run in parallel.")
flags.DEFINE_integer("thread_num", 5, "How many thread to run in the process.")
flags.DEFINE_integer("port_num", 6370, "the start port to create distribute tf")
#flags.DEFINE_integer("port_num", 6470, "the start port to create distribute tf")


flags.DEFINE_string("game_version", "3.16.1", "game version of SC2")

FLAGS(sys.argv)
if FLAGS.training:
    PARALLEL = FLAGS.parallel
    THREAD_NUM = FLAGS.thread_num
    MAX_AGENT_STEPS = FLAGS.max_agent_steps
    DEVICE = ['/gpu:' + dev for dev in FLAGS.device.split(',')]
    #DEVICE = ['/cpu:0']
else:
    PARALLEL = 1
    THREAD_NUM = 1
    MAX_AGENT_STEPS = 1e5
    DEVICE = ['/cpu:0']

TRAIN_ITERS = FLAGS.max_iters
MAX_AGENT_STEPS = FLAGS.max_agent_steps

LOG = FLAGS.log_path
if not os.path.exists(LOG):
    os.makedirs(LOG)

SERVER_DICT = {"worker": [], "ps": []}

# define some global variable
UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
Counter = 0
Waiting_Counter = 0
Update_Counter = 0
Result_List = []


''' 
ps -ef |grep liuruoze | awk '{print $2}' | xargs kill -9
ps -ef |grep liuruoze | grep main.py | awk '{print $2}' | xargs kill -9
ps -ef |grep liuruoze | grep SC2 | awk '{print $2}' | xargs kill -9
'''

''' 
ps -ef |grep liuruoze | grep 'SC2_x64' | awk '{print $2}' | xargs kill -9
kill -9 `ps -ef |grep liuruoze | grep main | awk '{print $2}' `

'''


def run_thread(agent, game_num, Synchronizer, difficulty):
    global UPDATE_EVENT, ROLLING_EVENT, Counter, Waiting_Counter, Update_Counter, Result_List

    num = 0
    proc_name = mp.current_process().name

    C._FPS = 2.8
    step_mul = 8

    if difficulty == 'A':
        C.difficulty = 10
    else:
        C.difficulty = difficulty

    env = sc2_env.SC2Env(
        map_name=FLAGS.map,
        agent_race=FLAGS.agent_race,
        bot_race=FLAGS.bot_race,
        difficulty=difficulty,
        step_mul=step_mul,
        score_index=-1,
        game_steps_per_episode=FLAGS.max_agent_steps,
        screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
        minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
        visualize=False,
        game_version=FLAGS.game_version)
    # env = available_actions_printer.AvailableActionsPrinter(env)
    agent.set_env(env)

    while True:
        try:
            agent.play()
        except Exception as e:
            print("Catch exception: ", e, ", agent will be reset.")
            env.close()
            agent.reset()
            env = sc2_env.SC2Env(
                map_name=FLAGS.map,
                agent_race=FLAGS.agent_race,
                bot_race=FLAGS.bot_race,
                difficulty=difficulty,
                step_mul=step_mul,
                score_index=-1,
                game_steps_per_episode=FLAGS.max_agent_steps,
                screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
                minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
                visualize=False,
                game_version=FLAGS.game_version)
            agent.set_env(env)
            print("Environment has restarted.")
            agent.play()

        if FLAGS.training:
            # check if the num of episodes is enough to update
            num += 1
            Counter += 1
            reward = agent.result['reward']
            Result_List.append(reward)
            logging("(diff: %d) %d epoch: %s get %d/%d episodes! return: %d!" %
                    (int(C.difficulty), Update_Counter, proc_name, len(Result_List), game_num * THREAD_NUM, reward))

            # time for update
            if num == game_num:
                num = 0
                ROLLING_EVENT.clear()
                # worker stops rolling, wait for update
                if agent.index != 0 and THREAD_NUM > 1:
                    Waiting_Counter += 1
                    if Waiting_Counter == THREAD_NUM - 1:  # wait for all the workers stop
                        UPDATE_EVENT.set()
                    ROLLING_EVENT.wait()

                # update!
                else:
                    if THREAD_NUM > 1:
                        UPDATE_EVENT.wait()

                    Synchronizer.wait()  # wait for other processes to go here

                    agent.update_network(Result_List)
                    Result_List.clear()
                    agent.global_buffer.reset()

                    Synchronizer.wait()

                    Update_Counter += 1

                    # finish update
                    UPDATE_EVENT.clear()
                    Waiting_Counter = 0
                    ROLLING_EVENT.set()

        if FLAGS.save_replay:
            env.save_replay(FLAGS.replay_dir)

        agent.reset()


def Worker(index, update_game_num, Synchronizer, cluster):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    worker = tf.train.Server(cluster, job_name="worker", task_index=index, config=config)

    sess = tf.Session(target=worker.target, config=config)
    Net = HierNetwork(sess=sess, summary_writer=None, rl_training=FLAGS.training,
                      cluster=cluster, index=index, device=DEVICE[index % len(DEVICE)])

    global_buffer = Global_Buffer()
    agents = []
    for i in range(THREAD_NUM):
        agent = multi_agent.MultiAgent(index=i, global_buffer=global_buffer, net=Net,
                                       restore_model=FLAGS.restore_model, rl_training=FLAGS.training)
        agents.append(agent)

    logging("Worker %d: waiting for cluster connection..." % index)
    sess.run(tf.report_uninitialized_variables())
    logging("Worker %d: cluster ready!" % index)

    while len(sess.run(tf.report_uninitialized_variables())):
        logging("Worker %d: waiting for variable initialization..." % index)
        time.sleep(1)
    logging("Worker %d: variables initialized" % index)

    game_num = np.ceil(update_game_num // THREAD_NUM)

    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()

    # Run threads
    threads = []
    for i in range(THREAD_NUM - 1):
        t = threading.Thread(target=run_thread, args=(agents[i], game_num, Synchronizer, FLAGS.difficulty))
        threads.append(t)
        t.daemon = True
        t.start()
        time.sleep(3)

    run_thread(agents[-1], game_num, Synchronizer, FLAGS.difficulty)

    for t in threads:
        t.join()


def Parameter_Server(Synchronizer, cluster, log_path):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name="ps", task_index=0, config=config)

    sess = tf.Session(target=server.target, config=config)
    summary_writer = tf.summary.FileWriter(log_path)
    Net = HierNetwork(sess=sess, summary_writer=summary_writer, rl_training=FLAGS.training,
                      cluster=cluster, index=0, device=DEVICE[0 % len(DEVICE)])
    agent = multi_agent.MultiAgent(index=-1, net=Net, restore_model=FLAGS.restore_model, rl_training=FLAGS.training)

    logging("Parameter server: waiting for cluster connection...")
    sess.run(tf.report_uninitialized_variables())
    logging("Parameter server: cluster ready!")

    logging("Parameter server: initializing variables...")
    agent.init_network()
    logging("Parameter server: variables initialized")

    update_counter = 0
    max_win_rate = 0.
    latest_win_rate = 0.

    while update_counter < TRAIN_ITERS:
        agent.reset_old_network()

        # wait for update
        Synchronizer.wait()
        logging("Update Network!")
        # TODO count the time , compare cpu and gpu
        time.sleep(1)

        # update finish
        Synchronizer.wait()
        logging("Update Network finished!")

        steps, win_rate = agent.update_summary(update_counter)
        logging("Steps: %d, win rate: %f" % (steps, win_rate))

        update_counter += 1
        if win_rate >= max_win_rate:
            agent.save_model()
            max_win_rate = win_rate

        latest_win_rate = win_rate
        # agent.net.save_latest_policy()

    return max_win_rate, latest_win_rate


def _main(unused_argv):
    """Run agents"""
    maps.get(FLAGS.map)  # Assert the map exists.

    # create distribute tf cluster
    start_port = FLAGS.port_num
    SERVER_DICT["ps"].append("localhost:%d" % start_port)
    for i in range(PARALLEL):
        SERVER_DICT["worker"].append("localhost:%d" % (start_port + 1 + i))

    Cluster = tf.train.ClusterSpec(SERVER_DICT)

    global LOG
    if FLAGS.training:
        now = datetime.now()
        LOG = "./logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

        model_path = "./model/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            os.makedirs(model_path + "controller")
            os.makedirs(model_path + "base")
            os.makedirs(model_path + "tech")
            os.makedirs(model_path + "pop")
            os.makedirs(model_path + "battle")

        C._SAVE_MODEL_PATH = model_path

    if FLAGS.restore_model:
        C._LOAD_MODEL_PATH = FLAGS.restore_model_path
        P.restore_model = FLAGS.restore_model

    UPDATE_GAME_NUM = FLAGS.num_for_update
    per_update_num = np.ceil(UPDATE_GAME_NUM / PARALLEL)

    Synchronizer = mp.Barrier(PARALLEL + 1)
    # Run parallel process
    procs = []
    for index in range(PARALLEL):
        p = mp.Process(name="Worker_%d" % index, target=Worker, args=(index, per_update_num, Synchronizer, Cluster))
        procs.append(p)
        p.daemon = True
        p.start()
        time.sleep(1)

    max_win_rate, latest_win_rate = Parameter_Server(Synchronizer, Cluster, LOG)

    print('#######################')
    print('Best Win_rate:', max_win_rate)
    print('Latest Win_rate:', latest_win_rate)
    print('#######################')

    for p in procs:
        p.join()

    if FLAGS.profile:
        print(stopwatch.sw)


if __name__ == "__main__":
    app.run(_main)
