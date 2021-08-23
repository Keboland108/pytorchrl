import os
import animalai
from animalai.envs.gym.environment import AnimalAIEnv
from pytorchrl.envs.common import FrameStack, FrameSkip, DelayedReward
from pytorchrl.envs.animal_olympics.wrappers import RetroEnv, FilterActionEnv, LabAnimal, RewardShaping


def animal_train_env_factory(
        index_col_worker, index_grad_worker, index_env=0, frame_skip=0, frame_stack=1,
        arenas_dir=None, reduced_actions=True, reward_shape=True, exe_path=None, reward_delay=1):
    """
    Create train Animal Olympics Unity3D environment.

    Parameters
    ----------
     index_col_worker : int
        Index of the collection worker running this environment.
    index_grad_worker : int
        Index of the gradient worker running the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    arenas_dir:
        path to dir containing arenas .yaml files.
    reduced_actions : bool
        Whether or not to use the action wrapper to reduce the number of available actions.
    reward_shape : bool
        Whether or not to use the reward shape wrapper.
    exe_path : str
        Path to obstacle environment executable.
    reward_delay : int
        Only return accumulated reward every `reward_delay` steps to simulate sparse reward environment.

    Returns
    -------
    env : gym.Env
        Train environment.
    """
    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    if exe_path:
        exe = exe_path
    else:
        exe = os.path.join(os.path.dirname(animalai.__file__), '../../env/AnimalAI')

    id = 1 + index_grad_worker * 10000 + 1000 * index_col_worker + index_env * 100

    print("\n index_col_worker is {}, index_grad_worker is {}, index_env is {}, ID IS {} \n".format(
        index_col_worker, index_grad_worker, index_env, id))

    env = AnimalAIEnv(exe, retro=False, worker_id=id,
                      seed=id, n_arenas=1, arenas_configurations=None,
                      greyscale=False, inference=False, resolution=None)

    env = RetroEnv(env)

    if reduced_actions:
        env = FilterActionEnv(env)

    env = LabAnimal(env, arenas_dir)

    if reward_shape:
        env = RewardShaping(env)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    if reward_delay > 1:
        env = DelayedReward(env, delay=reward_delay)

    return env


def animal_test_env_factory(
        index_col_worker, index_grad_worker, index_env=0, frame_skip=0, frame_stack=1,
        arenas_dir=None, reduced_actions=True, reward_shape=True, exe_path=None, reward_delay=1):
    """
    Create train Animal Olympics Unity3D environment.

    Parameters
    ----------
     index_col_worker : int
        Index of the collection worker running this environment.
    index_grad_worker : int
        Index of the gradient worker running the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    arenas_dir:
        path to dir containing arenas .yaml files.
    reduced_actions : bool
        Whether or not to use the action wrapper to reduce the number of available actions.
    reward_shape : bool
        Whether or not to use the reward shape wrapper.
    exe_path : str
        Path to obstacle environment executable.
    reward_delay : int
        Only return accumulated reward every `reward_delay` steps to simulate sparse reward environment.

    Returns
    -------
    env : gym.Env
        Train environment.
    """
    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    if exe_path:
        exe = exe_path
    else:
        exe = os.path.join(os.path.dirname(animalai.__file__), '../../env/AnimalAI')

    id = index_grad_worker * 1000 + 100 * index_col_worker + index_env
    env = AnimalAIEnv(exe, retro=False, worker_id=id,
                      seed=id, n_arenas=1, arenas_configurations=None,
                      greyscale=False, inference=False, resolution=None)

    env = RetroEnv(env)

    if reduced_actions:
        env = FilterActionEnv(env)

    env = LabAnimal(env, arenas_dir)

    if reward_shape:
        env = RewardShaping(env)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    if reward_delay > 1:
        env = DelayedReward(env, delay=reward_delay)

    return env