#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import os
import uuid
import argparse

import cv2
import tensorflow as tf
import six
from six.moves import queue


from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu


import gym
from simulator import SimulatorProcess, SimulatorMaster, TransitionExperience
from common import Evaluator, eval_model_multithread, play_n_episodes
from atari_wrapper import MapState, FrameStack, FireResetEnv, LimitLength

if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 6000
EVAL_EPISODE = 50
BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 15     # batch for efficient forward
SIMULATOR_PROC = 50
PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None

NUM_ACTIONS = None
ENV_NAME = None


def get_player(train=False, dumpdir=None):
    env = gym.make(ENV_NAME)
    if dumpdir:
        env = gym.wrappers.Monitor(env, dumpdir)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    env = FrameStack(env, 4)
    if train:
        env = LimitLength(env, 60000)
    return env


class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return get_player(train=True)


class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputDesc(tf.uint8, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward1'),
                InputDesc(tf.float32, (None,), 'futurereward2'),
                InputDesc(tf.float32, (None,), 'updateweight1'),
                InputDesc(tf.float32, (None,), 'updateweight2'),
                InputDesc(tf.float32, (None,), 'action_prob'),
                ]

    def _get_NN_prediction(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            h = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            h = MaxPooling('pool0', h, 2)
            h = Conv2D('conv1', h, out_channel=32, kernel_shape=5)
            h = MaxPooling('pool1', h, 2)
            h = Conv2D('conv2', h, out_channel=64, kernel_shape=4)
            h = MaxPooling('pool2', h, 2)
            h = Conv2D('conv3', h, out_channel=64, kernel_shape=3)

        h = FullyConnected('fc0', h, 512, nl=tf.identity)
        h = PReLU('prelu', h)
        logits = FullyConnected('fc-pi', h, out_dim=NUM_ACTIONS, nl=tf.identity)    # unnormalized policy
        value1 = FullyConnected('fc-v_1', h, 1, nl=tf.identity) 
        value2 = FullyConnected('fc-v_2', h, 1, nl=tf.identity)
        return logits, value1, value2

    def _build_graph(self, inputs):
        state, action, futurereward1, futurereward2, updateweight1, updateweight2, action_prob = inputs
        logits, value1, value2 = self._get_NN_prediction(state)
        value1 = tf.squeeze(value1, [1], name='pred_value_1')  # (B,)
        value2 = tf.squeeze(value2, [1], name='pred_value_2')  # (B,)
        policy = tf.nn.softmax(logits, name='policy')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        advantage1 = tf.subtract(tf.stop_gradient(value1), futurereward1, name='advantage_1')
        advantage2 = tf.subtract(tf.stop_gradient(value2), futurereward2, name='advantage_2')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, NUM_ACTIONS), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss1 = tf.reduce_sum(log_pi_a_given_s * advantage1 * importance * updateweight1, name='policy_loss_1')
        policy_loss2 = tf.reduce_sum(log_pi_a_given_s * advantage2 * importance * updateweight2, name='policy_loss_2')
        policy_loss = tf.add(policy_loss1, policy_loss2, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')
        value_loss1 = tf.nn.l2_loss((value1 - futurereward1) * tf.sqrt(updateweight1), name='value_loss_1')
        value_loss2 = tf.nn.l2_loss((value2 - futurereward2) * tf.sqrt(updateweight2), name='value_loss_2')
        value_loss = tf.add(value_loss1, value_loss2, name='value_loss')

        pred_reward1 = tf.reduce_mean(value1, name='predict_reward_1')
        pred_reward2 = tf.reduce_mean(value2, name='predict_reward_2')
        pred_reward_avg = tf.add(pred_reward1 * 0.5, pred_reward2 * 0.5, name='predict_reward_avg')
        advantage1 = symbf.rms(advantage1, name='rms_advantage_1')
        advantage2 = symbf.rms(advantage2, name='rms_advantage_2')
        advantage_avg = symbf.rms(advantage1 * 0.5 + advantage2 * 0.5, name='rms_advantage_avg')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        self.cost = tf.add_n([policy_loss1, policy_loss2, xentropy_loss * entropy_beta, value_loss1, value_loss2])
        self.cost = tf.truediv(self.cost,
                               tf.cast(tf.shape(futurereward1)[0], tf.float32),
                               name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss, value_loss,
                                   pred_reward1, pred_reward2, pred_reward_avg,
                                   advantage1, advantage2, advantage_avg,
                                   self.cost, tf.reduce_mean(importance, name='importance'))

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, model, gpus):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self._gpus = gpus

    def _setup_graph(self):
        # create predictors on the available predictor GPUs.
        nr_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['state'], ['policy', 'pred_value_1', 'pred_value_2'],
            self._gpus[k % nr_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, state, ident):
        def cb(outputs):
            try:
                distrib, value1, value2 = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            rand_num = np.random.rand()
            if rand_num < 0.5:
                updateweight1, updateweight2 = 1.0, 0.0
            else:
                updateweight2, updateweight1 = 1.0, 0.0
            client = self.clients[ident]
            client.memory.append(TransitionExperience(
                state, action, reward=None, value1=value1, value2=value2,
                updateweight1=updateweight1, updateweight2=updateweight2, prob=distrib[action]))
            self.send_queue.put([ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _on_episode_over(self, ident):
        self._parse_memory(0, 0, ident, True)

    def _on_datapoint(self, ident):
        client = self.clients[ident]
        if len(client.memory) == LOCAL_TIME_MAX + 1:
            R1 = client.memory[-1].value2
            R2 = client.memory[-1].value1
            self._parse_memory(R1, R2, ident, False)

    def _parse_memory(self, init_r1, init_r2, ident, isOver):
        client = self.clients[ident]
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R1 = float(init_r1)
        R2 = float(init_r2)
        for idx, k in enumerate(mem):
            R1 = np.clip(k.reward, -1, 1) + GAMMA * R1
            R2 = np.clip(k.reward, -1, 1) + GAMMA * R2
            self.queue.put([k.state, k.action, R1, R2, k.updateweight1, k.updateweight2, k.prob])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []


def get_config():
    nr_gpu = get_nr_gpu()
    global PREDICTOR_THREAD
    if nr_gpu > 0:
        if nr_gpu > 1:
            # use half gpus for inference
            predict_tower = list(range(nr_gpu))[-nr_gpu // 2:]
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        PREDICTOR_THREAD = 1
        predict_tower, train_tower = [0], [0]

    # setup simulator processes
    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    M = Model()
    master = MySimulatorMaster(namec2s, names2c, M, predict_tower)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    return TrainConfig(
        model=M,
        dataflow=dataflow,
        callbacks=[
            ModelSaver(max_to_keep=2),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            HumanHyperParamSetter('learning_rate'),
            HumanHyperParamSetter('entropy_beta'),
            master,
            StartProcOrThread(master),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                every_k_epochs=1),
        ],
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
        tower=train_tower
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--env', help='env', required=True)
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train', 'gen_submit'], default='train')
    parser.add_argument('--output', help='output directory for submission', default='output_dir')
    parser.add_argument('--episode', help='number of episode to eval', default=100, type=int)
    args = parser.parse_args()

    ENV_NAME = args.env
    logger.info("Environment Name: {}".format(ENV_NAME))
    NUM_ACTIONS = get_player().action_space.n
    logger.info("Number of actions: {}".format(NUM_ACTIONS))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['policy']))
        if args.task == 'play':
            play_n_episodes(get_player(train=False), pred,
                            args.episode, render=True)
        elif args.task == 'eval':
            eval_model_multithread(pred, args.episode, get_player)
        elif args.task == 'gen_submit':
            play_n_episodes(
                get_player(train=False, dumpdir=args.output),
                pred, args.episode)
            # gym.upload(args.output, api_key='xxx')
    else:
        dirname = os.path.join('train_log', 'train-atari-{}'.format(ENV_NAME))
        logger.set_logger_dir(dirname)

        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SimpleTrainer() if config.nr_tower == 1 else AsyncMultiGPUTrainer(config.tower)
        launch_train_with_config(config, trainer)
