import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
from pytorchrl.agent.actors.feature_extractors.ensemble_layer import EnsembleFC

from pytorchrl.agent.actors.utils import init

LOG_STD_MAX = 2  # Maximum std allowed value. Used for clipping.
LOG_STD_MIN = -20  # Minimum std allowed value. Used for clipping.


class DiagGaussian(nn.Module):
    """
    Isotropic gaussian probability distribution.

    Parameters
    ----------
    num_inputs : int
        Size of input feature maps.
    num_outputs : int
        Number of dims in output space.
    predict_log_std : bool
        Whether to use a nn.linear layer to predict the output std.
    """

    def __init__(self, num_inputs, num_outputs, predict_log_std=False):
        super(DiagGaussian, self).__init__()

        self.predict_log_std = predict_log_std

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.mean = init_(nn.Linear(num_inputs, num_outputs))

        if predict_log_std:
            self.log_std = init_(nn.Linear(num_inputs, num_outputs))
        else:
            self.log_std = nn.Parameter(torch.zeros(num_outputs).unsqueeze(0))

    def forward(self, x, deterministic=False):
        """
        Predict distribution parameters from x (obs features) and return
        predicted values (sampled and clipped), sampled log
        probability and distribution entropy.

        Parameters
        ----------
        x : torch.tensor
            Feature maps extracted from environment observations.
        deterministic : bool
            Whether to randomly sample from predicted distribution or take the mode.

        Returns
        -------
        pred: torch.tensor
            Predicted value.
        clipped_pred: torch.tensor
            Predicted value (clipped to be within [-1, 1] range).
        logp : torch.tensor
            Log probability of `pred` according to the predicted distribution.
        entropy_dist : torch.tensor
            Entropy of the predicted distribution.
        dist : torch.Distribution
            Action probability distribution.
        """
        # Predict distribution parameters
        mean = self.mean(x)
        logstd = self.log_std(x) if self.predict_log_std else torch.zeros(
            mean.size()).to(x.device) + self.log_std

        # logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)

        # Create distribution and sample
        dist = Normal(mean, logstd.exp())

        if deterministic:
            pred = mean
        else:
            pred = dist.sample()

        # Apply clipping to avoid being outside output space
        clipped_pred = torch.clamp(pred, -1, 1)

        # Action log probability
        logp = dist.log_prob(pred).sum(-1, keepdim=True)

        # Distribution entropy
        entropy_dist = dist.entropy().sum(-1).mean()

        return pred, clipped_pred, logp, entropy_dist, dist

    def evaluate_pred(self, x, pred):
        """
        Return log prob of `pred` under the distribution generated from
        x (obs features). Also return entropy of the generated distribution.

        Parameters
        ----------
        x : torch.tensor
            obs feature map obtained from a policy_net.
        pred : torch.tensor
            Prediction to evaluate.

        Returns
        -------
        logp : torch.tensor
            Log probability of `pred` according to the predicted distribution.
        entropy_dist : torch.tensor
            Entropy of the predicted distribution.
        dist : torch.Distribution
            Action probability distribution.
        """

        # Predict distribution parameters
        mean = self.mean(x)
        logstd = self.log_std(x) if self.predict_log_std else torch.zeros(
            mean.size()).to(x.device) + self.log_std

        # logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)

        # Create distribution
        dist = Normal(mean, logstd.exp())

        # Evaluate log prob under dist
        logp = dist.log_prob(pred).sum(-1, keepdim=True)

        # Distribution entropy
        entropy_dist = dist.entropy().sum(-1).mean()

        return logp, entropy_dist, dist

class DiagGaussianEnsemble(nn.Module):
    """Ensemble gaussian output layer 
    
        Parameters
        ----------
        num_inputs: int
            Size of input feature maps.
        num_outputs: int
            Output size of the gaussian layer.
        ensemble_size: int
            Ensemble size in the output layer.
    """
    def __init__(self, num_inputs: int, num_outputs: int, ensemble_size: int)-> None:
        super(DiagGaussianEnsemble, self).__init__()

        self.num_outputs = num_outputs
        self.ensemble_size = ensemble_size
        self.mu = EnsembleFC(in_features=num_inputs, out_features=num_outputs, ensemble_size=ensemble_size)

    def forward(self, x: torch.Tensor):
        """Forward pass"""
        return self.mu(x)
