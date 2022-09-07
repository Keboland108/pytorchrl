import gym
import numpy as np
import torch
import torch.nn as nn
from pytorchrl.agent.actors.utils import init
from transformers import OpenAIGPTConfig, OpenAIGPTModel


class GPT(nn.Module):
    """
    Wrapper to be able to use GPT model from transformers.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    transformers_config : list
        Hidden layers sizes.
    """
    def __init__(self, input_space, transformers_config):
        super(GPT, self).__init__()

        # Define feature extractor
        self.feature_extractor = OpenAIGPTModel(transformers_config)

    def forward(self, inputs):
        """
        Forward pass Neural Network

        Parameters
        ----------
        inputs : torch.tensor
            Input data.

        Returns
        -------
        out3 : torch.tensor
            Output feature map.
        """

        batch_size = inputs.size(0)
        inputs1 = inputs.view(batch_size, -1)

        # masks
        mask_attn = (inputs < 0.0)
        has_masked_tokens = (mask_attn == True).any()

        # forward pass
        inputs[mask_attn] = 0.0
        out1 = self.feature_extractor(
            input_ids=inputs1.long(),  # Shape (batch_size, sequence_length)
            attention_mask=1 - mask_attn.long(),  # Shape (batch_size, sequence_length)
        ).last_hidden_state

        if has_masked_tokens:  # Data collection
            out2 = []
            mask_dim0, mask_dim1 = torch.where(mask_attn == False)
            for i in range(batch_size):
                idx = max(mask_dim1[mask_dim0 == i])
                out2.append(out1[i, idx])
            out3 = torch.stack(out2)
        else:  # Gradient computation
            out3 = out1
        inputs[mask_attn] = - 1.0  # TODO: Should go back to whichever value had before!
        return out3
