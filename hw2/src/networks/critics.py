import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        ###########################################################################
        return self.network(obs).squeeze(-1)    # shape: (B,1) -> (B,)
                                                # q_values (B,)와 맞춰줌
        ###########################################################################

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # TODO: compute the loss using the observations and q_values
        ###########################################################################
        # target은 state-value V(s)가 실제 Q(q_values)에 가까워지도록
        values = self.forward(obs)
        loss = F.mse_loss(values, q_values)    # MSE Loss
        ###########################################################################

        # TODO: perform an optimizer step
        ###########################################################################
        self.optimizer.zero_grad()  # gradient 초기화
        loss.backward()             # gradient 계산
        self.optimizer.step()       # parameter update
        ###########################################################################

        return {
            "Baseline Loss": loss.item(),
        }