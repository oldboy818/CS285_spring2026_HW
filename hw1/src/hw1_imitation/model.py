"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
    ############################################################################################
        sizes = [state_dim, *hidden_dims, chunk_size * action_dim] #[입력, *, 출력]
        items = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            items.append(nn.Linear(a, b))
            items.append(nn.ReLU())
        self.net = nn.Sequential(*items[:-1])  # 마지막 ReLU 제거
    ############################################################################################

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # raise NotImplementedError
    ############################################################################################
        action_chunk_pred = self.net(state).reshape(-1, self.chunk_size, self.action_dim)       # (batch, chunk_size, action_dim)
        loss = nn.MSELoss(reduction='none')(action_chunk_pred, action_chunk).sum(dim=(1, 2)).mean()   # HW1 식(1).
                                                                            # reduction='none': 평균내지 않고 원소별 제곱 오차
                                                                            # (batch, chunk_size, action_dim) -> (batch,)
        return loss
    ############################################################################################

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
    ############################################################################################
        with torch.no_grad():
            action_chunk_pred = self.net(state).reshape(-1, self.chunk_size, self.action_dim)   # (batch, chunk_size, action_dim)
        return action_chunk_pred
    ############################################################################################

class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
    ############################################################################################
        # 입력: [state, action_chunk_interpolation, tau]
        # 출력: vector field v
        in_dim = state_dim + chunk_size * action_dim + 1        # (batch, state_dim + chunk_size * action_dim + 1)
        sizes = [in_dim, *hidden_dims, chunk_size * action_dim] # [입력, *, 출력]
        items = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            items.append(nn.Linear(a, b))
            items.append(nn.ReLU())
        self.net = nn.Sequential(*items[:-1])  # 마지막 ReLU 제거
    ############################################################################################

    def compute_loss(
        self,
        state: torch.Tensor,        # (batch, state_dim)
        action_chunk: torch.Tensor, # A_t, 정답. (batch, chunk_size, action_dim)
    ) -> torch.Tensor:
        # raise NotImplementedError
    ############################################################################################
        # define interpolation. A_tau = tau * A_t + (1 - tau) * A_0
        action_chunk_noise = torch.randn_like(action_chunk)         # A_0 / (batch, chunk_size, action_dim)
        tau = torch.rand(state.shape[0], 1, 1, device=state.device) # (batch, 1, 1)
        action_chunk_tau = tau * action_chunk + (1 - tau) * action_chunk_noise    # A_tau = tau * A_t + (1 - tau) * A_0 / (batch, chunk_size, action_dim)
        
        # define vector field. v(state, A_tau, tau)
        x = torch.cat([state, action_chunk_tau.flatten(1), tau.flatten(1)], dim=1)      # (batch, state_dim + chunk flatten + 1)
                                                                                        # 차원 수 맞추기 위해 flatten. 신경망은 하나의 벡터 (batch, ~)를 입력으로
        vector_field_pred = self.net(x).reshape(-1, self.chunk_size, self.action_dim)   # (batch, chunk_size, action_dim)

        # loss
        target = action_chunk - action_chunk_noise  # (batch, chunk_size, action_dim)   # A_1 - A_0
        loss = nn.MSELoss(reduction='none')(vector_field_pred, target).sum(dim=(1, 2)).mean()

        return loss
    ############################################################################################

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # raise NotImplementedError
    ############################################################################################
        # Euler integration. A_{τ + 1/n} = A_τ + (1/n)·v_θ(o, A_τ, τ)
        with torch.no_grad():
            action_chunk = torch.randn(state.shape[0], self.chunk_size, self.action_dim, device=state.device)
                                                                    # (batch, chunk_size, action_dim)
            # denoising steps
            n = 1.0 / num_steps

            for step in range(num_steps):
                tau = torch.full((state.shape[0], 1, 1), step * n, device=state.device)
                                            # denoising step에 따라 tau를 0에서 1로 변하는 값으로
                                            # shape (batch, 1, 1)
                x = torch.cat([state, action_chunk.flatten(1), tau.flatten(1)], dim=1)
                action_chunk = action_chunk + n * self.net(x).reshape(-1, self.chunk_size, self.action_dim)

        return action_chunk
    ############################################################################################


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
