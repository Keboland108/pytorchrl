import os
import obstacle_tower_env
from obstacle_tower_env import ObstacleTowerEnv
from pytorchrl.envs.common import FrameStack, FrameSkip
from pytorchrl.envs.obstacle_tower_unity3d_challenge.wrappers import (
ReducedActionEnv, BasicObstacleEnv, RewardShapeObstacleEnv)
from pytorchrl.agent.algos import PPO
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.storages import GAEBuffer
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from pytorchrl.envs import obstacle_train_env_factory




def obstacle_train_env_factory(
        index_worker=0, rank=0, frame_skip=0, frame_stack=1, min_floor=0,
        max_floor=50, reduced_actions=True, reward_shape=True):
    """
    Create train Obstacle Tower Unity3D environment.
    Useful info_keywords 'floor', 'start', 'seed'.
    Parameters
    ----------
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    min_floor : int
        Minimum floor the agent can be spawned in.
    max_floor : int
        Maximum floor the agent can be spawned in.
    reduced_actions : bool
        Whether or not to use the action wrapper to reduce the number of available actions.
    reward_shape : bool
        Whether or not to use the reward shape wrapper.
    Returns
    -------
    env : gym.Env
        Train environment.
    """
    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    exe = os.path.join(os.path.dirname(
        obstacle_tower_env.__file__), 'ObstacleTower/obstacletower')

    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=index_worker + rank,
        greyscale=False, docker_training=False, realtime_mode=False)

    if reduced_actions:
        env = ReducedActionEnv(env)

    env = BasicObstacleEnv(env, max_floor=max_floor, min_floor=min_floor)

    if reward_shape:
        env = RewardShapeObstacleEnv(env)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env

# Define Train Vector of Envs
envs_factory, action_space, obs_space = VecEnv.create_factory(
    env_fn=obstacle_train_env_factory,
    env_kwargs={"frame_skip": 2, "frame_stack": 4},
    vec_env_size=8, log_dir='/tmp/obstacle_tower_agent',
    info_keywords=('floor', 'start', 'seed'))

# Define RL training algorithm
algo_factory = PPO.create_factory(
    lr=2.5e-5, num_epochs=2, clip_param=0.15, entropy_coef=0.01,
     value_loss_coef=0.2, max_grad_norm=0.5, num_mini_batch=8,
    use_clipped_value_loss=True, gamma=0.99)

# Define RL Policy
actor_factory = OnPolicyActor.create_factory(
    obs_space, action_space,
    feature_extractor_network=get_feature_extractor("Fixup"),
    recurrent_policy=True)

# Define rollouts storage
storage_factory = GAEBuffer.create_factory(size=800, gae_lambda=0.95)