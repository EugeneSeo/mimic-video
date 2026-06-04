# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

import numpy as np
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from omegaconf import MISSING

from cosmos_predict2.callbacks.video_eval import VideoEvalCallback
from cosmos_predict2.callbacks.so101_two_camera_video_eval import SO101TwoCameraVideoEvalCallback
from cosmos_predict2.configs.defaults.data_video import train_datasets
from imaginaire.lazy_config import LazyCall as L

BASE: dict = dict(
    defaults=[
        {"override /model": "predict2_video2world_ddp_2b_480p_10fps"},
        {"override /video_dataset_train": MISSING},
        {"override /video_dataset_val": MISSING},
        {"override /dataloader_val": "vanilla"},
        {"override /dataloader_train": "vanilla"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "constant"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        cpu_offloading_activations=False,
        cpu_offloading_weights=False,
    ),
    trainer=dict(
        distributed_parallelism="ddp",
        grad_accum_iter=1,
        max_iter=1_000_000,
        logging_iter=1_000,
        run_validation=False,
        callbacks=dict(
            video_eval=L(VideoEvalCallback)(fuse_lora=MISSING),
        ),
    ),
    optimizer=dict(
        lr=MISSING,
    ),
)

lrs = np.logspace(-5, -3, 9)[[5]]
bszs = [32]
ranks = [256]


def get_local_batch_size(global_bsz: int) -> int:
    res = global_bsz / parallel_state.get_data_parallel_world_size()

    if not res.is_integer():
        msg = "That batch size doesn't work with the number of gpus you have."
        raise ValueError(msg)

    return int(res)


cs = ConfigStore.instance()

for rank in ranks:
    for dataset in train_datasets:
        for lr in lrs:
            for bsz in bszs:
                train_type = f"lora_rank{rank}" if rank is not None else "fullft"
                name = f"v2w_{dataset}_{train_type}_lr{lr:.3e}_bsz{bsz}"

                cfg = copy.deepcopy(BASE)

                cfg["defaults"][1]["override /video_dataset_train"] = dataset
                cfg["defaults"][2]["override /video_dataset_val"] = dataset

                cfg["optimizer"]["lr"] = lr.item()
                cfg["job"]["name"] = name
                cfg["dataloader_train"] = {"batch_size": L(get_local_batch_size)(global_bsz=bsz)}

                if rank is not None:
                    cfg["model"]["config"].update(
                        dict(
                            train_architecture="lora",
                            lora_rank=rank,
                            lora_alpha=32,
                            init_lora_weights=True,
                            lora_target_modules="q_proj,k_proj,v_proj,output_proj,x_embedder.proj.1,linear_1,linear_2,mlp.layer1,mlp.layer2",
                        )
                    )
                cfg["trainer"]["callbacks"]["video_eval"]["fuse_lora"] = rank is not None

                cs.store(
                    group="experiment",
                    package="_global_",
                    name=name,
                    node=cfg,
                )

def register_so101_two_camera_video_experiment(*, rank: int, lr: float, bsz: int) -> None:
    cfg = copy.deepcopy(BASE)
    name = f"v2w_so101_two_camera_lora_rank{rank}_lr{lr:.3e}_bsz{bsz}"
    cfg["defaults"][0]["override /model"] = "predict2_video2world_ddp_2b_480p_10fps"
    cfg["defaults"][1]["override /video_dataset_train"] = "so101_two_camera"
    cfg["defaults"][2]["override /video_dataset_val"] = "so101_two_camera"
    cfg["optimizer"]["lr"] = lr
    cfg["job"]["group"] = "video2world_so101_two_camera"
    cfg["job"]["name"] = name
    cfg["dataloader_train"] = {"batch_size": L(get_local_batch_size)(global_bsz=bsz)}
    cfg["trainer"]["grad_accum_iter"] = max(1, 64 // bsz)
    cfg["trainer"]["max_iter"] = 30_000
    cfg["trainer"]["logging_iter"] = 100
    cfg["trainer"]["validation_iter"] = 1_000
    cfg["checkpoint"] = {"save_iter": 1_000}
    cfg["model"]["config"].update(
        dict(
            train_architecture="lora",
            lora_rank=rank,
            lora_alpha=32,
            init_lora_weights=True,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,x_embedder.proj.1,linear_1,linear_2,mlp.layer1,mlp.layer2",
            train_adaln_modulation=False,
            save_trainable_only=True,
            model_manager_config=dict(
                dit_path="${oc.env:SO101_TWO_CAMERA_INIT_DIT_PATH,/cluster/scratch/eugseo/mimic_video_checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt}",
                text_encoder_path="",
            ),
        )
    )
    cfg["checkpoint"] = {"save_iter": 100, "strict_resume": False, "keep_latest_only": True}
    cfg["trainer"]["callbacks"]["video_eval"]["fuse_lora"] = False
    cfg["trainer"]["callbacks"]["so101_video_preview"] = L(SO101TwoCameraVideoEvalCallback)(
        dataset_dir="${oc.env:SO101_TWO_CAMERA_VIDEO_DIR,/tmp/nonexistent_so101_two_camera_video}",
        iteration_list="${oc.env:SO101_V2W_WANDB_VIDEO_ITERS,none}",
        every_n="${oc.env:SO101_V2W_WANDB_VIDEO_EVERY_N,50}",
        start_after="${oc.env:SO101_V2W_WANDB_VIDEO_START_AFTER,300}",
        num_samples="${oc.env:SO101_V2W_WANDB_VIDEO_NUM_SAMPLES,2}",
        num_sampling_steps="${oc.env:SO101_V2W_WANDB_VIDEO_STEPS,20}",
    )
    cs.store(group="experiment", package="_global_", name=name, node=cfg)


for so101_two_camera_bsz in [4, 8]:
    register_so101_two_camera_video_experiment(rank=256, lr=float(np.logspace(-5, -3, 9)[5]), bsz=so101_two_camera_bsz)
