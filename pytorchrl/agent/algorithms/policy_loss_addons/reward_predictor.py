import torch
import numpy as np

import pytorchrl as prl
from pytorchrl.utils import RunningMeanStd
from pytorchrl.agent.algorithms.policy_loss_addons import PolicyLossAddOn


class RewardPredictor(PolicyLossAddOn):

    def __init__(self, predictor_net_factory, predictor_net_kwargs, masked_sparse_obs_ratio=0.0):
        """
        Class to train a reward predictor in parallel with the RL actor.

        Parameters
        ----------
        predictor_net_factory : func
            Method to create the network that predict the reward (output has shape (1,)).
        predictor_net_factory : dict
            Keyword argument to be used in predictor_net_factory.
        masked_sparse_obs_ratio : func
            Ratio, between 0.0 and 1.0, of states with reward 0.0 to be masked in the loss function.
        """
        self.predictor_net_kwargs = predictor_net_kwargs
        self.predictor_net_factory = predictor_net_factory
        self.masked_sparse_obs_ratio = np.clip(masked_sparse_obs_ratio, 0.0, 1.0)

    def setup(self, actor, device):
        """Setup addon module by creating and instance of the network."""

        self.actor = actor
        self.device = device

        # Cast behavior weights to torch tensors
        self.actor.reward_predictor = self.predictor_net_factory(**self.predictor_net_kwargs).to(device)
        self.max_pred_errors_rms = RunningMeanStd(shape=(1,), device=device)
        self.mean_pred_errors_rms = RunningMeanStd(shape=(1,), device=device)
        self.min_pred_errors_rms = RunningMeanStd(shape=(1,), device=device)
        self.actor.error_threshold = torch.nn.parameter.Parameter(
            data=torch.tensor(1000000, dtype=torch.float32), requires_grad=False)

    def compute_loss_term(self, data, actor_dist, info):
        """
        Calculate and add KL Attraction loss term.

        Parameters
        ----------
        actor_dist : torch.distributions.Distribution
            Actor action distribution for actions in data[prl.OBS]
        data : dict
            data batch containing all required tensors to compute loss term.
        info : dict
            Dictionary to store log information.

        Returns
        -------
        attraction_kl_loss_term : torch.tensor
            KL loss term.
        info : dict
            Updated info dict.
        """

        o, rhs, r = data[prl.OBS], data[prl.RHS], data[prl.REW]
        pred_r = self.actor.reward_predictor(o)
        error = torch.abs(r - pred_r)
        loss = 0.5 * error.pow(2)
        mask = torch.rand(loss.size(), device=self.device)
        mask = (mask >= self.masked_sparse_obs_ratio).float()
        mask[r != 0.0] = 1.0
        loss = (mask * loss).sum() / mask.sum()

        if len(error[r != 0.0]) > 0:
            self.max_pred_errors_rms.update(error[r != 0.0].max().reshape(-1, 1))
            self.mean_pred_errors_rms.update(error[r != 0.0].mean().reshape(-1, 1))
            self.min_pred_errors_rms.update(error[r != 0.0].min().reshape(-1, 1))
        self.actor.error_threshold.data = self.pred_errors_rms.mean.float()

        info.update({
            "reward_predictor_loss": loss.item(),
            "reward_predictor_error": error.mean().item(),
            "max_reward_pred_error_rms": self.max_pred_errors_rms.mean.float().item(),
            "mean_reward_pred_error_rms": self.mean_pred_errors_rms.mean.float().item(),
            "min_reward_pred_error_rms": self.min_pred_errors_rms.mean.float().item(),
        })

        return loss, info
