#!/usr/bin/env python3
import sys

import os
import ray
import time
import glob
import torch
import wandb
import shutil
import random
import argparse
import numpy as np
import torch.nn as nn

import pytorchrl as prl
from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.algorithms import RND_PPO
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.atari import atari_train_env_factory
from pytorchrl.agent.storages.on_policy.ppod2_buffer import PPOD2Buffer
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor, get_memory_network


def main():
    print("GPU available: {}".format(torch.cuda.is_available()))

    # Get and log config
    args = get_args()

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=args.experiment_name, name=args.agent_name, config=args, mode=mode):

        # Sanity check, make sure that logging matches execution
        args = wandb.config

        info_keywords = []
        if args.episodic_life:
            info_keywords += ['EpisodicReward', 'Lives']
        if args.clip_rewards:
            info_keywords += ['UnclippedReward']
        if args.env_id == "MontezumaRevengeNoFrameskip-v4":
            info_keywords += ['VisitedRooms']

        # Define Train Vector of Envs
        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            env_fn=atari_train_env_factory,
            env_kwargs={
                "env_id": args.env_id,
                "frame_stack": args.frame_stack,
                "episodic_life": args.episodic_life,
                "clip_rewards": args.clip_rewards,
                "sticky_actions": args.sticky_actions,
                "use_domain_knowledge": True,
                "domain_knowledge_embedding": "room"
            },
            vec_env_size=args.num_env_processes, log_dir=args.log_dir,
            info_keywords=tuple(info_keywords))

        # Define RL training algorithm
        algo_factory, algo_name = RND_PPO.create_factory(
            lr=args.lr, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
            entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
            max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
            use_clipped_value_loss=False, gamma_intrinsic=args.gamma_intrinsic,
            ext_adv_coeff=args.ext_adv_coeff, int_adv_coeff=args.int_adv_coeff,
            predictor_proportion=args.predictor_proportion, gamma=args.gamma,
            pre_normalization_steps=args.pre_normalization_steps,
            pre_normalization_length=args.num_steps,
            intrinsic_rewards_network=get_feature_extractor("CNN"),
            intrinsic_rewards_target_network_kwargs={
                "output_sizes": [512],
                "activation": nn.LeakyReLU,
                "final_activation": False,
                "rgb_norm": False,
            },
            intrinsic_rewards_predictor_network_kwargs={
                "output_sizes": [512, 512, 512],
                "activation": nn.LeakyReLU,
                "final_activation": False,
                "rgb_norm": False,
            },
        )

        # Define RL Policy
        actor_factory = OnPolicyActor.create_factory(
            obs_space, action_space, algo_name,
            feature_extractor_network=get_feature_extractor(args.feature_extractor_net),
            restart_model=args.restart_model, recurrent_net=get_memory_network(args.recurrent_net))

        # Define rollouts storage
        supp_demos_dir = args.log_dir + "/supplementary_demos/"
        os.makedirs(supp_demos_dir, exist_ok=True)
        storage_factory = PPOD2Buffer.create_factory(
            size=args.num_steps, rho=args.rho, phi=args.phi,
            total_buffer_demo_capacity=args.buffer_capacity,
            gae_lambda=args.gae_lambda, initial_reward_threshold=1.0,
            demo_dtypes={prl.OBS: np.uint8, prl.ACT: np.int8, prl.REW: np.float16},
            supplementary_demos_dir=supp_demos_dir,
        )

        # Define scheme
        params = {}

        # add core modules
        params.update({
            "algo_factory": algo_factory,
            "actor_factory": actor_factory,
            "storage_factory": storage_factory,
            "train_envs_factory": train_envs_factory,
        })

        scheme = Scheme(**params)
        wandb.config.update(scheme.get_agent_components())  # Log agent components

        # Define learner
        training_steps = args.target_env_steps
        learner = Learner(scheme, target_steps=training_steps, log_dir=args.log_dir)

        # Define train loop
        iterations = 0
        start_time = time.time()
        while not learner.done():

            learner.step()

            if iterations % args.log_interval == 0:
                log_data = learner.get_metrics(add_episodes_metrics=True)
                log_data = {k.split("/")[-1]: v for k, v in log_data.items()}
                wandb.log(log_data, step=learner.num_samples_collected)
                learner.print_info()

            if iterations % args.save_interval == 0:
                # Save current model version
                save_name = learner.save_model()

            if args.max_time != -1 and (time.time() - start_time) > args.max_time:
                break

            iterations += 1

    print("Finished!")
    sys.exit()


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # Configuration file, keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile)

    # Wandb
    parser.add_argument(
        '--experiment_name', default=None, help='Name of the wandb experiment the agent belongs to')
    parser.add_argument(
        '--agent_name', default=None, help='Name of the wandb run')
    parser.add_argument(
        '--wandb_key', default=None, help='Init key from wandb account')

    # Environment specs
    parser.add_argument(
        '--env-id', type=str, default=None,
        help='Gym environment id (default None)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Env seed (default 0)')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=1,
        help='Number of frame to stack in observation (default no stack)')
    parser.add_argument(
        '--clip-rewards', action='store_true', default=False,
        help='Clip env rewards between -1 and 1')
    parser.add_argument(
        '--episodic-life', action='store_true', default=False,
        help='Treat end-of-life as end-of-episode')
    parser.add_argument(
        '--sticky-actions', action='store_true', default=False,
        help='Use sticky actions')

    # RND PPOD specs
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--rho', type=float, default=0.3,
        help='PPO+D rho parameter (default: 0.3)')
    parser.add_argument(
        '--phi', type=float, default=0.0,
        help='PPO+D phi parameter (default: 0.0)')
    parser.add_argument(
        '--gae-lambda', type=float, default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--num-steps', type=int, default=20000,
        help='number of forward steps in PPO (default: 20000)')
    parser.add_argument(
        '--ppo-epoch', type=int, default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch', type=int, default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param', type=float, default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--gamma-intrinsic', type=float, default=0.99,
        help='rnd ppo intrinsic gamma parameter (default: 0.99)')
    parser.add_argument(
        '--ext-adv-coeff', type=float, default=2.0,
        help='rnd ppo external advantage coefficient parameter (default: 2.0)')
    parser.add_argument(
        '--int-adv-coeff', type=float, default=1.0,
        help='rnd ppo internal advantage coefficient parameter (default: 1.0)')
    parser.add_argument(
        '--predictor-proportion', type=float, default=1.0,
        help='rnd ppo proportion of batch samples to use to update predictor net (default: 1.0)')
    parser.add_argument(
        '--pre-normalization-steps', type=int, default=50,
        help='rnd ppo number of pre-normalization steps parameter (default: 50)')
    parser.add_argument(
        '--buffer-capacity', type=int, default=50,
        help='Max number of demos allowed int he buffer (default: 50)')
    parser.add_argument(
        '--demos-dir', default='/tmp/atari_ppod',
        help='target directory to store and retrieve demos.')

    # Feature extractor model specs
    parser.add_argument(
        '--feature-extractor-net', default='MLP', help='Type of nn. Options include MLP, CNN, Fixup')
    parser.add_argument(
        '--restart-model', default=None,
        help='Restart training using the model given')
    parser.add_argument(
        '--restart-reference-model', default=None,
        help='Path to reward predictor model')
    parser.add_argument(
        '--recurrent-net', default=None, help='Recurrent neural networks to use')

    # Scheme specs
    parser.add_argument(
        '--num-env-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-grad-workers', type=int, default=1,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--com-grad-workers', default='synchronised',
        help='communication patters grad workers (default: synchronised)')
    parser.add_argument(
        '--num-col-workers', type=int, default=1,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--com-col-workers', default='synchronised',
        help='communication patters col workers (default: synchronised)')
    parser.add_argument(
        '--cluster', action='store_true', default=False,
        help='script is running in a cluster')

    # General training specs
    parser.add_argument(
        '--target-env-steps', type=int, default=10e7,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--max-time', type=int, default=-1,
        help='stop script after this amount of time in seconds (default: no limit)')
    parser.add_argument(
        '--log-interval', type=int, default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval', type=int, default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--log-dir', default='/tmp/pybullet_ppo',
        help='directory to save agent logs (default: /tmp/pybullet_ppo)')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
