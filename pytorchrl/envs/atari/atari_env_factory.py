from pytorchrl.envs.atari.wrappers import wrap_deepmind, make_atari, MontezumaVisitedRoomEnv
from pytorchrl.envs.common import DelayedReward


def atari_train_env_factory(
        env_id, index_col_worker, index_grad_worker, index_env=0, seed=0, frame_stack=1, reward_delay=1,
        episodic_life=True, clip_rewards=False, max_episode_steps=4500):
    """
    Create train Atari environment.

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
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    reward_delay : int
        Only return accumulated reward every `reward_delay` steps to simulate sparse reward environment.
    episodic_life : bool
        Whether or not simulate end of episode when losing a life.
    clip_rewards : bool
        Whether or not to clip rewards between -1 and 1.
    max_episode_steps : int
        Maximum number of steps per episode.

    Returns
    -------
    env : gym.Env
        Train environment.
    """
    env = make_atari(env_id, max_episode_steps=max_episode_steps)
    env.seed(index_grad_worker * 1000 + 100 * index_col_worker + index_env + seed)
    env = wrap_deepmind(
        env, episode_life=episodic_life,
        clip_rewards=clip_rewards,
        scale=False,
        frame_stack=frame_stack)

    if env_id == "MontezumaRevengeNoFrameskip-v4":
        env = MontezumaVisitedRoomEnv(env, 3)

    if reward_delay > 1:
        env = DelayedReward(env, delay=reward_delay)

    return env


def atari_test_env_factory(env_id, index_col_worker, index_grad_worker, index_env=0, seed=0, frame_stack=1, reward_delay=1):
    """
    Create test Atari environment.

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
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    reward_delay : int
        Only return accumulated reward every `reward_delay` steps to simulate sparse reward environment.

    Returns
    -------
    env : gym.Env
        Test environment.
    """
    env = make_atari(env_id)
    env.seed(index_grad_worker * 1000 + 100 * index_col_worker + index_env + seed)
    env = wrap_deepmind(
        env, episode_life=False,
        clip_rewards=False,
        scale=False,
        frame_stack=frame_stack)

    if reward_delay > 1:
        env = DelayedReward(env, delay=reward_delay)

    return env