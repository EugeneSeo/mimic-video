# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

import torch
import wandb

from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _to_scalar(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().float().item()
    return None


class WandbLogger(Callback):
    """Initialize W&B from environment variables and log scalar training metrics."""

    def __init__(self, log_every_n: int | None = None):
        self.log_every_n = log_every_n
        self._enabled = False

    @distributed.rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model
        mode = os.environ.get("WANDB_MODE", "online")
        self._enabled = _env_flag("WANDB_ENABLED", default=False) and mode != "disabled"
        if not self._enabled:
            return

        if self.log_every_n is None:
            self.log_every_n = int(os.environ.get("WANDB_LOG_EVERY_N", "1"))

        init_kwargs = {
            "project": os.environ.get("WANDB_PROJECT"),
            "entity": os.environ.get("WANDB_ENTITY"),
            "name": os.environ.get("WANDB_NAME") or self.config.job.name,
            "group": os.environ.get("WANDB_GROUP") or self.config.job.group,
            "dir": os.environ.get("WANDB_DIR"),
            "mode": mode,
        }
        run_id = os.environ.get("WANDB_RUN_ID")
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"

        wandb.init(**{key: value for key, value in init_kwargs.items() if value})
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("*", step_metric="trainer/global_step")
        log.info(
            f"W&B logging initialized: project={init_kwargs['project']} "
            f"name={init_kwargs['name']} mode={mode}"
        )

        if iteration:
            wandb.log({"trainer/resumed_from_iteration": iteration}, step=iteration)

    @distributed.rank0_only
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, data_batch
        if not self._enabled or wandb.run is None:
            return
        assert self.log_every_n is not None
        if iteration % self.log_every_n != 0:
            return

        metrics: dict[str, int | float] = {
            "trainer/global_step": iteration,
            "train/loss": loss.detach().float().item(),
        }
        for key, value in output_batch.items():
            scalar = _to_scalar(value)
            if scalar is not None:
                metrics[f"train/{key}"] = scalar

        wandb.log(metrics, step=iteration)

    @distributed.rank0_only
    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model
        if self._enabled and wandb.run is not None:
            wandb.log({"trainer/final_iteration": iteration}, step=iteration)
            wandb.finish()
