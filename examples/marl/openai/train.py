"""
This example is modified from the original OpenAI MADDPG training script.
Original script can be found below:
https://github.com/openai/maddpg/blob/master/experiments/train.py

To run this example, please make sure you have the OpenAI MADDPG installed,
see README under examples/marl/openai for details.

"""

import argparse
import json
import logging
import os
import random
import uuid

from datetime import datetime

import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from gridworld.log import logger
from gridworld.multiagent_env import MultiAgentEnv
from gridworld.multiagent_list_interface_env import MultiAgentListInterfaceEnv
from gridworld.scenarios.buildings import make_env_config

logger.setLevel(logging.ERROR)


class CoordinatedMultiBuildingControlEnv(MultiAgentEnv):
    """ Extend the original multiagent environment to include coordination.
    In addition to the original agent-level reward, grid-level reward/penalty
    is considered, if agents fail to coordinate to satisfy the system-level
    constraint(s).
    In this example, we consider the voltage constraints: agents need to
    coordinate so the common bus voltage is within the ANSI C.84.1 limit.
    Otherwise, the voltage violation penalty will be shared by all agents.
    """

    VOLTAGE_LIMITS = [0.95, 1.05]
    VV_UNIT_PENALTY = 1e4

    # Overwriting the default transform behavior.
    def reward_transform(self, rew_dict) -> dict:
        """ Adding system wide penalty and slip it evenly on all agents.
        """

        voltage_violation = self.get_voltage_violation()
        sys_penalty = voltage_violation * self.VV_UNIT_PENALTY

        # split the penalty equally among agents.
        agent_num = len(rew_dict)
        for key in rew_dict.keys():
            rew_dict[key] -= (sys_penalty / agent_num)

        return rew_dict

    def meta_transform(self, meta) -> dict:
        """ Augment meta info for logging purpose. """
        sys_info = {'voltage_violation': self.get_voltage_violation()}
        meta.update(sys_info)
        return meta

    def get_voltage_violation(self):
        """ Obtain voltage of the bus where all buildings connect to.
        """

        assert len(set(
            self.agent_name_bus_map.values())) == 1, \
            "In this example, all buildings should be on the same bus."

        bus_id = list(set(self.agent_name_bus_map.values()))[0]

        common_bus_voltage = self.pf_solver.get_bus_voltage_by_name(bus_id)
        voltage_violation = max([
            0.0,
            self.VOLTAGE_LIMITS[0] - common_bus_voltage,
            common_bus_voltage - self.VOLTAGE_LIMITS[1]
        ])

        return voltage_violation


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for"
                                     " multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple",
                        help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25,
                        help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=350000,
                        help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0,
                        help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg",
                        help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg",
                        help="policy of adversaries")
    parser.add_argument("--sys-load", type=float, default=1.2,
                        help="system load fraction")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='default_experiment',
                        help="name of the experiment")
    parser.add_argument("--save-dir", type=str,
                        default="./trained_policy_gridworld/",
                        help="directory in which training state and model"
                             " should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes"
                             " are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model"
                             " are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000,
                        help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str,
                        default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64,
              layer_num=3, activation=tf.nn.tanh, activate_final_layer=False):
    # This model takes as input an observation and returns values of all
    # actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        for idx in range(layer_num):
            out = layers.fully_connected(out, num_outputs=num_units,
                                         activation_fn=activation)

        final_layer_activation = activation if activate_final_layer else None
        out = layers.fully_connected(out, num_outputs=num_outputs,
                                     activation_fn=final_layer_activation)
        return out


def make_env(arglist):
    """ Make the example coordinated multi-building environment.
    """

    env_config = make_env_config(
        building_config={},
        pv_config={
            "profile_csv": "pv_profile.csv",
            "scaling_factor": 40.
        },
        storage_config={
            "max_power": 15.,
            "storage_range": (3., 50.)
        },
        system_load_rescale_factor=arglist.sys_load,
        num_buildings=3
    )

    env = MultiAgentListInterfaceEnv(
        CoordinatedMultiBuildingControlEnv,
        env_config
    )

    return env


def set_up_save_dir(arglist):
    """ Make the directory to save configuration log and learning results.
    """

    date_str = str(datetime.now())[:19].replace(' ', '-').replace(':', '-')
    uuid_str = uuid.uuid4().hex
    folder_name = arglist.exp_name + '-' + date_str + '-' + uuid_str
    save_dir = arglist.save_dir + '/' + folder_name + '/'

    os.makedirs(os.path.join(save_dir, 'policy_model'))
    os.makedirs(os.path.join(save_dir, 'learning_curves'))

    settings = {'learning_rate': arglist.lr,
                'batch_size': arglist.batch_size,
                'num_episodes': arglist.num_episodes,
                'random_seed': arglist.seed,
                'sys_load': arglist.sys_load,
                }
    settings_file_path = os.path.join(save_dir, 'settings.json')
    with open(settings_file_path, 'w') as f:
        json.dump(settings, f)

    return save_dir


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    """ The main MADDPG training script from OpenAI.
    We made some changes below regarding the random seed and results saving.
    """

    with U.single_threaded_session():

        if arglist.seed is None:
            arglist.seed = np.random.randint(1000)

        np.random.seed(arglist.seed)
        tf.set_random_seed(arglist.seed)
        random.seed(arglist.seed)

        # Create environment
        env = make_env(arglist)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        exp_save_dir = set_up_save_dir(arglist)

        episode_rewards = [0.0]  # sum of rewards for all agents
        # individual agent reward
        agent_rewards = [[0.0] for _ in range(env.n)]
        mean_p_loss = []
        mean_q_loss = []
        final_p_loss = []
        final_q_loss = []
        voltage_violation_hist = [0.0]
        final_voltage_violation = []
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i],
                                 new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            voltage_violation_hist[-1] += info_n['voltage_violation']

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                voltage_violation_hist.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir \
                                + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                if loss is not None:
                    mean_q_loss.append(loss[0])
                    mean_p_loss.append(loss[1])

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(exp_save_dir + 'policy_model/model', saver=saver)
                # print statement depends on whether or not there are
                # adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {},"
                          " v_violation: {}, mean p loss: {}, mean q loss: {},"
                          " time: {}".format(train_step, len(episode_rewards),
                        np.mean(episode_rewards[
                                -arglist.save_rate:]),
                        np.mean(voltage_violation_hist[
                                -arglist.save_rate:]),
                        np.mean(mean_p_loss),
                        np.mean(mean_q_loss),
                        round(time.time() - t_start, 3)))
                else:
                    print(
                        "steps: {}, episodes: {}, mean episode reward: {},"
                        " agent episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards),
                            np.mean(episode_rewards[-arglist.save_rate:]),
                            [np.mean(rew[-arglist.save_rate:]) for rew in
                             agent_rewards], round(time.time() - t_start, 3)))

                final_p_loss.append(np.mean(mean_p_loss))
                final_q_loss.append(np.mean(mean_q_loss))
                mean_p_loss = []
                mean_q_loss = []
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[
                                                -arglist.save_rate:]))
                final_voltage_violation.append(np.mean(voltage_violation_hist[
                                                       -arglist.save_rate:]),)
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[
                                                       -arglist.save_rate:]))

            # saves episode reward periodically for plotting training curves.
            if len(episode_rewards) % 500 == 0:

                curve_dir = exp_save_dir + 'learning_curves/'

                rew_file_name = curve_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)

                agrew_file_name = curve_dir \
                                  + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)

                ploss_file_name = curve_dir + arglist.exp_name + '_ploss.pkl'
                with open(ploss_file_name, 'wb') as fp:
                    pickle.dump(final_p_loss, fp)

                qloss_file_name = curve_dir + arglist.exp_name + '_qloss.pkl'
                with open(qloss_file_name, 'wb') as fp:
                    pickle.dump(final_q_loss, fp)

                vv_file_name = curve_dir + arglist.exp_name + '_vvio.pkl'
                with open(vv_file_name, 'wb') as fp:
                    pickle.dump(final_voltage_violation, fp)

                if len(episode_rewards) >= arglist.num_episodes:
                    print('...Finished total of {} episodes.'.format(
                        len(episode_rewards)))
                    break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
