"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 학습 데이터셋 다운로드 및 로드
    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    # 배치를 꺼내는 DataLoader.
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    ### TODO: PUT YOUR MAIN TRAINING LOOP HERE ###
    # Note that, for grading purposes, you must call the evaluate_policy function periodically in your
    # training loop, and call logger.dump_for_grading() at the end of training.
    ####################################################################################
    from hw1_imitation.evaluation import evaluate_policy

    # Optimizer 정의.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    model.optimizer = optimizer
    
    step = 0
    
    for epoch in range(config.num_epochs):
        
        for state, action_chunk in loader:   # epoch 내에서 배치 단위로 데이터를 가져옴. 평가 주기와 로그 주기를 고려하여 step(배치) 단위로 진행
            state = state.to(device)
            action_chunk = action_chunk.to(device)

            loss = model.compute_loss(state, action_chunk)  # 

            # Backpropagation
            model.optimizer.zero_grad() # gradient 0으로 초기화
            loss.backward()             # backpropagation으로 gradient 계산
            model.optimizer.step()      # gradient 업데이트

            step += 1

            # 로그(log training metrics) 주기마다 loss, epoch, step을 wandb에 기록
            if step % config.log_interval == 0: 
                logger.log({"loss": loss.item(), "epoch": epoch}, step) # loss.item()은 loss tensor에서 scalar 값만 추출
                                                                                # tensor를 그대로 logger 안의 리스트에 쌓이면 메모리 문제 발생 가능성

            # 평가 주기마다 정책 평가
            if step % config.eval_interval == 0: 
                evaluate_policy(
                    model,
                    normalizer,
                    device,
                    chunk_size=config.chunk_size,
                    video_size=config.video_size,
                    num_video_episodes=config.num_video_episodes,
                    flow_num_steps=config.flow_num_steps,
                    step=step,
                    logger=logger,
                )
                model.train()  # 평가 후 다시 학습 모드로 전환
    ####################################################################################

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()