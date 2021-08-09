import gym
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import pytorchrl as prl
from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.utils import Scale, Unscale, init
from pytorchrl.agent.actors.memory_networks import GruNet
from pytorchrl.agent.actors.feature_extractors import MLP, default_feature_extractor


class OffPolicyActor(nn.Module):
    """
    Actor critic class for Off-Policy algorithms.

    It contains a policy network (actor) to predict next actions and one or two
    Q networks.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    noise : str
        Type of exploration noise that will be added to the deterministic actions.
    deterministic : bool
        Whether using DDPG, TD3 or any other deterministic off-policy actor.
    obs_feature_extractor : nn.Module
        PyTorch nn.Module to extract features from observation in all networks.
    obs_feature_extractor_kwargs : dict
        Keyword arguments for the obs extractor network.
    act_feature_extractor : nn.Module
        PyTorch nn.Module to extract features from actions in all networks.
    act_feature_extractor_kwargs : dict
        Keyword arguments for the act extractor network.
    common_feature_extractor : nn.Module
        PyTorch nn.Module to extract joint features from the concatenation of
        action and observation features.
    common_feature_extractor_kwargs : dict
        Keyword arguments for the common extractor network.
    recurrent_nets : bool
        Whether to use a RNNs as feature extractors.
    sequence_overlap : float
        From 0.0 to 1.0, how much consecutive rollout sequences will overlap.
    recurrent_nets_kwargs : dict
        Keyword arguments for the memory network.
    create_double_q_critic : bool
        Whether to instantiate a second Q network or not.

    Examples
    --------
    """
    def __init__(self,
                 input_space,
                 action_space,
                 noise=None,
                 deterministic=False,
                 sequence_overlap=0.5,
                 recurrent_nets_kwargs={},
                 recurrent_nets=False,
                 obs_feature_extractor=None,
                 obs_feature_extractor_kwargs={},
                 act_feature_extractor=None,
                 act_feature_extractor_kwargs={},
                 common_feature_extractor=MLP,
                 common_feature_extractor_kwargs={},
                 number_of_critics=2):

        super(OffPolicyActor, self).__init__()

        self.noise = noise
        self.input_space = input_space
        self.action_space = action_space
        self.deterministic = deterministic
        self.act_feature_extractor = act_feature_extractor
        self.act_feature_extractor_kwargs = act_feature_extractor_kwargs
        self.obs_feature_extractor = obs_feature_extractor
        self.obs_feature_extractor_kwargs = obs_feature_extractor_kwargs
        self.common_feature_extractor = common_feature_extractor
        self.common_feature_extractor_kwargs = common_feature_extractor_kwargs
        self.recurrent_nets = recurrent_nets
        self.recurrent_nets_kwargs = recurrent_nets_kwargs
        self.sequence_overlap = np.clip(sequence_overlap, 0.0, 1.0)
        self.number_of_critics = number_of_critics

        #######################################################################
        #                           POLICY NETWORK                            #
        #######################################################################

        self.create_policy("policy_net")

        #######################################################################
        #                             Q-NETWORKS                              #
        #######################################################################

        for i in range(number_of_critics):
            self.create_critic("q{}".format(i + 1))

    @classmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            noise=None,
            deterministic=False,
            restart_model=None,
            sequence_overlap=0.5,
            recurrent_nets_kwargs={},
            recurrent_nets=False,
            obs_feature_extractor=None,
            obs_feature_extractor_kwargs={},
            act_feature_extractor=None,
            act_feature_extractor_kwargs={},
            common_feature_extractor=MLP,
            common_feature_extractor_kwargs={},
            number_of_critics=2

    ):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.
        noise : str
            Type of exploration noise that will be added to the deterministic actions.
        deterministic : bool
            Whether using DDPG, TD3 or any other deterministic off-policy actor.
        obs_feature_extractor : nn.Module
            PyTorch nn.Module to extract features from observation in all networks.
        obs_feature_extractor_kwargs : dict
            Keyword arguments for the obs extractor network.
        act_feature_extractor : nn.Module
            PyTorch nn.Module to extract features from actions in all networks.
        act_feature_extractor_kwargs : dict
            Keyword arguments for the act extractor network.
        common_feature_extractor : nn.Module
            PyTorch nn.Module to extract joint features from the concatenation of
            action and observation features.
        common_feature_extractor_kwargs : dict
            Keyword arguments for the common extractor network.
        recurrent_nets : bool
            Whether to use a RNNs as feature extractors.
        sequence_overlap : float
            From 0.0 to 1.0, how much consecutive rollout sequences will overlap.
        recurrent_nets_kwargs : dict
            Keyword arguments for the memory network.
        create_double_q_critic : bool
            Whether to instantiate a second Q network or not.

        Returns
        -------
        create_actor_critic_instance : func
            creates a new OffPolicyActor class instance.
        """

        def create_actor_critic_instance(device):
            """Create and return an actor critic instance."""
            policy = cls(input_space=input_space,
                         action_space=action_space,
                         noise=noise,
                         deterministic=deterministic,
                         sequence_overlap=sequence_overlap,
                         recurrent_nets_kwargs=recurrent_nets_kwargs,
                         recurrent_nets=recurrent_nets,
                         obs_feature_extractor=obs_feature_extractor,
                         obs_feature_extractor_kwargs=obs_feature_extractor_kwargs,
                         act_feature_extractor=act_feature_extractor,
                         act_feature_extractor_kwargs=act_feature_extractor_kwargs,
                         common_feature_extractor=common_feature_extractor,
                         common_feature_extractor_kwargs=common_feature_extractor_kwargs,
                         number_of_critics=number_of_critics)

            if restart_model:
                policy.load_state_dict(
                    torch.load(restart_model, map_location=device))
            policy.to(device)

            return policy

        return create_actor_critic_instance

    @property
    def is_recurrent(self):
        """Returns True if the actor network are recurrent."""
        return self.recurrent_nets

    @property
    def recurrent_hidden_state_size(self):
        """Size of policy recurrent hidden state"""
        return self.recurrent_size

    def actor_initial_states(self, obs):
        """
        Returns all actor inputs required to predict initial action.

        Parameters
        ----------
        obs : torch.tensor
            Initial environment observation.

        Returns
        -------
        obs : torch.tensor
            Initial environment observation.
        rhs : dict
            Initial recurrent hidden state (will contain zeroes).
        done : torch.tensor
            Initial done tensor, indicating the environment is not done.
        """

        if isinstance(obs, dict):
            num_proc = list(obs.values())[0].size(0)
            dev = list(obs.values())[0].device
        else:
            num_proc = obs.size(0)
            dev = obs.device

        done = torch.zeros(num_proc, 1).to(dev)
        rhs_act = torch.zeros(num_proc, self.recurrent_size).to(dev)

        rhs = {"rhs_act": rhs_act}
        rhs.update({"rhs_q{}".format(i + 1): rhs_act.clone() for i in range(self.number_of_critics)})

        return obs, rhs, done

    def burn_in_recurrent_states(self, data_batch):
        """
        Applies a recurrent burn-in phase to data_batch as described in
        (https://openreview.net/pdf?id=r1lyTjAqYX). Initial B steps are used
        to compute on-policy recurrent hidden states. data_batch is then
        updated, discarding B first steps in all tensors.

        Parameters
        ----------
        data_batch : dict
            data batch containing all required tensors to compute Algorithm loss.

        Returns
        -------
        data_batch : dict
            Updated data batch after burn-in phase.
        """

        # (T, N, -1) tensors that have been flatten to (T * N, -1)
        N = data_batch[prl.RHS]["rhs_act"].shape[0]  # number of sequences
        T = int(data_batch[prl.DONE].shape[0] / N)  # sequence lengths
        B = int(self.sequence_overlap * T)  # sequence burn-in length

        if B == 0:
            return data_batch

        # Split tensors into burn-in and no-burn-in
        chunk_sizes = [B, T - B] * N
        burn_in_data = {k: {} for k in data_batch}
        non_burn_in_data = {k: {} for k in data_batch}
        for k, v in data_batch.items():

            if k in (prl.RHS, prl.RHS2):
                burn_in_data[k] = v
                continue
            if not isinstance(v, (torch.Tensor, dict)):
                non_burn_in_data[k] = v
                continue
            if isinstance(v, dict):
                for x, y in v.items():
                    if not isinstance(y, torch.Tensor):
                        non_burn_in_data[k][x] = v
                        continue
                    sequence_slices = torch.split(y, chunk_sizes)
                    burn_in_data[k][x] = torch.cat(sequence_slices[0::2])
                    non_burn_in_data[k][x] = torch.cat(sequence_slices[1::2])
            else:
                sequence_slices = torch.split(v, chunk_sizes)
                burn_in_data[k] = torch.cat(sequence_slices[0::2])
                non_burn_in_data[k] = torch.cat(sequence_slices[1::2])

        # Do burn-in
        with torch.no_grad():

            act, _, _, rhs, _ = self.get_action(
                burn_in_data[prl.OBS], burn_in_data[prl.RHS], burn_in_data[prl.DONE])
            act2, _, _, rhs2, _ = self.get_action(
                burn_in_data[prl.OBS2], burn_in_data[prl.RHS2], burn_in_data[prl.DONE2])

            rhs = self.get_q_scores(
                burn_in_data[prl.OBS], rhs, burn_in_data[prl.DONE], act).get("rhs")
            rhs2 = self.get_q_scores(
                burn_in_data[prl.OBS2], rhs2, burn_in_data[prl.DONE2], act2).get("rhs2")

            for k in rhs:
                rhs[k] = rhs[k].detach()
            for k in rhs2:
                rhs2[k] = rhs2[k].detach()

            non_burn_in_data[prl.RHS] = rhs
            non_burn_in_data[prl.RHS2] = rhs2

        return non_burn_in_data

    def get_action(self, obs, rhs, done, deterministic=False):
        """
        Predict and return next action, along with other information.

        Parameters
        ----------
        obs : torch.tensor
            Current environment observation.
        rhs : dict
            Current recurrent hidden states.
        done : torch.tensor
            Current done tensor, indicating if episode has finished.
        deterministic : bool
            Whether to randomly sample action from predicted distribution or take the mode.

        Returns
        -------
        action : torch.tensor
            Next action sampled.
        clipped_action : torch.tensor
            Next action sampled, but clipped to be within the env action space.
        logp_action : torch.tensor
            Log probability of `action` within the predicted action distribution.
        rhs : dict
            Updated recurrent hidden states.
        entropy_dist : torch.tensor
            Entropy of the predicted action distribution.
        """

        x = self.policy_net.common_feature_extractor(self.policy_net.obs_feature_extractor(obs))

        if self.recurrent_nets:
            x, rhs["rhs_act"] = self.policy_net.memory_net(x, rhs["rhs_act"], done)

        (action, clipped_action, logp_action, entropy_dist, dist) = self.policy_net.dist(
            x, deterministic=deterministic)

        if self.unscale:
            action = self.unscale(action)
            clipped_action = self.unscale(clipped_action)

        return action, clipped_action, logp_action, rhs, entropy_dist, dist

    def evaluate_actions(self, obs, rhs, done, action):
        """
        Evaluate log likelihood of action given obs and the current
        policy network. Returns also entropy distribution.

        Parameters
        ----------
        obs : torch.tensor
            Environment observation.
        rhs : dict
            Recurrent hidden states.
        done : torch.tensor
            Done tensor, indicating if episode has finished.
        action : torch.tensor
            Evaluated action.

        Returns
        -------
        logp_action : torch.tensor
            Log probability of `action` according to the action distribution
            predicted with current version of the policy_net.
        entropy_dist : torch.tensor
            Entropy of the action distribution predicted with current version
            of the policy_net.
        rhs : dict
            Updated recurrent hidden states.
        """

        if self.scale:
            action = self.scale(action)

        x = self.policy_common_feature_extractor(self.policy_obs_feature_extractor(obs))

        if self.recurrent_nets:
            x, rhs["rhs_act"] = self.policy_memory_net(x, rhs["rhs_act"], done)

        logp_action, entropy_dist, dist = self.dist.evaluate_pred(x, action)

        return logp_action, entropy_dist, dist

    def get_q_scores(self, obs, rhs, done, actions=None):
        """
        Return Q scores of the given observations and actions.

        Parameters
        ----------
        obs : torch.tensor
            Environment observation.
        rhs : dict
            Current recurrent hidden states.
        done : torch.tensor
            Current done tensor, indicating if episode has finished.
        actions : torch.tensor
             Evaluated actions.

        Returns
        -------
        q1 : torch.tensor
            Q score according to current q1 network version.
        q2 : torch.tensor
            Q score according to current q2 network version.
        rhs : dict
            Updated recurrent hidden states.
        """

        outputs = {}
        for i in range(self.number_of_critics):
            q = getattr(self, "q{}".format(i + 1))
            features = q.obs_feature_extractor(obs)

            if actions is not None:
                act_features = q.act_feature_extractor(actions)
                features = torch.cat([features, act_features], -1)
            features = q.common_feature_extractor(features)

            if self.recurrent_nets:
                features, rhs["rhs_q{}".format(1 + 1)] = self.q1_memory_net(
                    features, rhs["rhs_q{}".format(i + 1)], done)

            q_scores = q.predictor(features)
            outputs["q{}".format(i + 1)] = q_scores

        outputs["rhs"] = rhs
        return outputs

    def create_critic(self, name):
        """

        This actor defines defines q networks as:
        -----------------------------------------

            obs_feature_extractor
        q =                       + common_feature_extractor +
            act_feature_extractor

            + memory_net + q_prediction_layer


        Parameters
        ----------
        name

        Returns
        -------

        """

        # ---- 1. Define action feature extractor -----------------------------

        act_extractor = self.act_feature_extractor or nn.Identity
        q_act_feature_extractor = act_extractor(
            self.action_space, **self.act_feature_extractor_kwargs)

        # ---- 2. Define obs feature extractor -----------------------------

        obs_extractor = self.obs_feature_extractor or nn.Identity
        q_obs_feature_extractor = obs_extractor(
            self.input_space, **self.obs_feature_extractor_kwargs)
        obs_feature_size = int(np.prod(q_obs_feature_extractor(
            torch.randn(1, *self.input_space.shape)).shape))

        # ---- 3. Define shared feature extractor -----------------------------

        if isinstance(self.action_space, gym.spaces.Discrete):
            act_feature_size = 0
            q_outputs = self.action_space.n

        elif isinstance(self.action_space, gym.spaces.Box):
            act_feature_size = int(np.prod(q_act_feature_extractor(
                torch.randn(1, *self.action_space.shape)).shape)) if self.act_feature_extractor \
                else np.prod(self.action_space.shape)
            q_outputs = 1

        else:
            raise NotImplementedError

        feature_size = obs_feature_size + act_feature_size
        q_common_feature_extractor = self.common_feature_extractor(
            feature_size, **self.common_feature_extractor_kwargs)

        # ---- 4. Define memory network ---------------------------------------

        feature_size = int(np.prod(q_common_feature_extractor(
            torch.randn(1, feature_size)).shape))
        q_memory_net = GruNet(feature_size, **self.recurrent_nets_kwargs) if\
            self.recurrent_nets else nn.Identity()
        feature_size = self.q1_memory_net.num_outputs if self.recurrent_nets\
            else feature_size

        # ---- 5. Define prediction layer -------------------------------------

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        q_predictor = init_(nn.Linear(feature_size, q_outputs))

        # ---- 6. Concatenate all q1 net modules ------------------------------

        q_net = nn.Sequential(OrderedDict([
            ('obs_feature_extractor', q_obs_feature_extractor),
            ('act_feature_extractor', q_act_feature_extractor),
            ('common_feature_extractor', q_common_feature_extractor),
            ('memory_net', q_memory_net),
            ("predictor", q_predictor),
        ]))

        setattr(self, name, q_net)

    def create_policy(self, name):
        """

        This actor defines policy network as:
        -------------------------------------

        policy = obs_feature_extractor + common_feature_extractor +

                + memory_net + action distribution


        Parameters
        ----------
        name

        Returns
        -------

        """

        # ---- 1. Define Obs feature extractor --------------------------------

        if len(self.input_space.shape) == 3:  # If inputs are images, CNN required
            self.obs_feature_extractor = default_feature_extractor(self.input_space)
        obs_extractor = self.obs_feature_extractor or nn.Identity
        policy_obs_feature_extractor = obs_extractor(
            self.input_space, **self.obs_feature_extractor_kwargs)

        # ---- 2. Define Common feature extractor -----------------------------

        feature_size = int(np.prod(policy_obs_feature_extractor(
            torch.randn(1, *self.input_space.shape)).shape))

        policy_common_feature_extractor = self.common_feature_extractor(
            feature_size, **self.common_feature_extractor_kwargs)

        # ---- 3. Define memory network  --------------------------------------

        feature_size = int(np.prod(policy_common_feature_extractor(
            torch.randn(1, feature_size)).shape))
        self.recurrent_size = feature_size
        if self.recurrent_nets:
            policy_memory_net = GruNet(feature_size, **self.recurrent_nets_kwargs)
            feature_size = self.policy_memory_net.num_outputs
        else:
            policy_memory_net = nn.Identity()

        # ---- 4. Define action distribution ----------------------------------

        if isinstance(self.action_space, gym.spaces.Discrete):
            dist = get_dist("Categorical")(feature_size, self.action_space.n)
            self.scale = None
            self.unscale = None

        elif isinstance(self.action_space, gym.spaces.Box) and not self.deterministic:
            dist = get_dist("SquashedGaussian")(feature_size, self.action_space.shape[0])
            self.scale = Scale(self.action_space)
            self.unscale = Unscale(self.action_space)

        elif isinstance(self.action_space, gym.spaces.Box) and self.deterministic:
            dist = get_dist("Deterministic")(feature_size,
                self.action_space.shape[0], noise=self.noise)
            self.scale = Scale(self.action_space)
            self.unscale = Unscale(self.action_space)
        else:
            raise NotImplementedError

        # ---- 5. Concatenate all policy modules ------------------------------

        policy_net = nn.Sequential(OrderedDict([
            ('obs_feature_extractor', policy_obs_feature_extractor),
            ('common_feature_extractor', policy_common_feature_extractor),
            ('memory_net', policy_memory_net), ('dist', dist),
        ]))

        setattr(self, name, policy_net)
