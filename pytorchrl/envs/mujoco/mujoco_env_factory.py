import gym
from pytorchrl.envs.common import FrameStack, FrameSkip, DelayedReward, CartPoleActionWrapper


def mujoco_train_env_factory(env_id, index_col_worker, index_grad_worker, index_env=0, seed=0, frame_skip=0, frame_stack=1, reward_delay=1):
    """
    Create train MuJoCo environment.

    Parameters
    ----------
    env_id : str
        Environment name.
    index_col_worker : int
        Index of the collection worker running this environment.
    index_grad_worker : int
        Index of the gradient worker running the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Environment random seed.
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    reward_delay : int
        Only return accumulated reward every `reward_delay` steps to simulate sparse reward environment.

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    env = gym.make(env_id)
    env.seed(index_grad_worker * 1000 + 100 * index_col_worker + index_env + seed)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    if reward_delay > 1:
        env = DelayedReward(env, delay=reward_delay)
    
    if env_id == "CartPole-v0" or env_id == "CartPole-v1":
        env = CartPoleActionWrapper(env)

    return env


def mujoco_test_env_factory(env_id, index_col_worker, index_grad_worker, index_env=0, seed=0, frame_skip=0, frame_stack=1, reward_delay=1):
    """
    Create test MuJoCo environment.

    Parameters
    ----------
    env_id : str
        Environment name.
    index_col_worker : int
        Index of the collection worker running this environment.
    index_grad_worker : int
        Index of the gradient worker running the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Environment random seed.
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    reward_delay : int
        Only return accumulated reward every `reward_delay` steps to simulate sparse reward environment.

    Returns
    -------
    env : gym.Env
        Test environment.
    """
    env = gym.make(env_id)
    env.seed(index_grad_worker * 1000 + 100 * index_col_worker + index_env + seed)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    if reward_delay > 1:
        env = DelayedReward(env, delay=reward_delay)

    if env_id == "CartPole-v0" or env_id == "CartPole-v1":
        env = CartPoleActionWrapper(env)

    return env
