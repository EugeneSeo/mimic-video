import copy
import itertools as it
import pathlib

import numpy as np
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from omegaconf import MISSING

from cosmos_predict2.configs.defaults.data_action import DATA_CONFIGS
from cosmos_predict2.configs.defaults.world2action_model import VIDEO_MODEL_CKPT_NAMES
from cosmos_predict2.configs.defaults.world2action_pipe import ACTION_DECODER_NETS
from imaginaire.lazy_config import LazyCall as L

BASE: dict = dict(
    defaults=[
        {"override /model": MISSING},
        {"override /world2action_pipe": MISSING},
        {"override /data_config": MISSING},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mimic"},
        {"override /dataloader_train": "mimic"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            train_architecture="base",
            # video_sigma_mode="logitnormal",
            pipe_config=dict(xattn_layer_idx=MISSING),
            video_pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=MISSING,
    ),
    scheduler=dict(
        f_max=[1],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[500_000],
    ),
    job=dict(
        project="vam",
        group=MISSING,
        name=MISSING,
    ),
    model_parallel=dict(
        cpu_offloading_activations=False,
        cpu_offloading_weights=False,
    ),
    checkpoint=dict(save_iter=1_000),
    trainer=dict(
        distributed_parallelism="ddp",
        grad_accum_iter=1,
        max_iter=500_000,
        logging_iter=1_000,
        validation_iter=1_000,
        run_validation=True,
    ),
)

cs = ConfigStore.instance()
cs.store(name="config", node=BASE)

world2action_pipes = ACTION_DECODER_NETS.keys()
xattn_layer_idxs = [20]
lrs = np.logspace(-5, -3, 9)[[4]]
bszs = [1, 128, 256]


def get_local_batch_size(global_bsz: int) -> int:
    res = global_bsz / parallel_state.get_data_parallel_world_size()

    if not res.is_integer():
        msg = "That batch size doesn't work with the number of gpus you have."
        raise ValueError(msg)

    return int(res)


for video_ckpt, data_config, xattn_layer_idx, lr, bsz in it.product(
    VIDEO_MODEL_CKPT_NAMES, DATA_CONFIGS.keys(), xattn_layer_idxs, lrs, bszs
):
    if data_config.startswith("so101"):
        continue

    pipes = [pipe for pipe in world2action_pipes if data_config.startswith(pipe)]
    if not pipes:
        continue
    if len(pipes) > 1:
        raise AssertionError("data_config to pipe should be n-to-1")
    pipe = pipes[0]

    exp_name = f"w2a_{data_config}_{video_ckpt}_lr{lr:.3e}_layer{xattn_layer_idx}_bsz{bsz}"

    cfg = copy.deepcopy(BASE)
    cfg["defaults"][0]["override /model"] = video_ckpt
    cfg["defaults"][1]["override /world2action_pipe"] = pipe
    cfg["defaults"][2]["override /data_config"] = data_config
    cfg["model"]["config"]["pipe_config"]["xattn_layer_idx"] = xattn_layer_idx
    cfg["optimizer"]["lr"] = lr.item()
    cfg["job"]["group"] = pipe
    cfg["job"]["name"] = exp_name
    cfg["dataloader_train"] = {"batch_size": L(get_local_batch_size)(global_bsz=bsz)}

    if "libero" in data_config:
        cfg["checkpoint"]["save_iter"] = 99999999
        cfg["trainer"]["run_validation"] = False

    cs.store(
        group="experiment",
        package="_global_",
        name=exp_name,
        node=cfg,
    )


SO101_VIDEO_CKPTS = [
    "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused",
    "v2w_pretrained_cosmos",
]

SO101_ACTION_DECODER_PARTIAL_INITS = {
    "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused": (
        pathlib.Path(__file__).parents[3]
        / "checkpoints"
        / "action_decoder"
        / (
            "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_"
            "lr1.000e-04_layer20_bsz256_iter_000014112.pt"
        )
    ),
    "v2w_pretrained_cosmos": (
        pathlib.Path(__file__).parents[3]
        / "checkpoints"
        / "action_decoder"
        / "w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
    ),
}


def register_so101_experiment(
    *,
    init_name: str,
    video_ckpt: str,
    lr: float,
    bsz: int,
    data_config: str = "so101",
    action_dit_path: str = "",
) -> None:
    cfg = copy.deepcopy(BASE)
    cfg["defaults"][0]["override /model"] = video_ckpt
    cfg["defaults"][1]["override /world2action_pipe"] = "so101"
    cfg["defaults"][2]["override /data_config"] = data_config
    cfg["model"]["config"]["pipe_config"]["xattn_layer_idx"] = 20
    if data_config in {"so101_delta_30hz", "so101_delta_30hz_gripper_absolute"}:
        cfg["model"]["config"]["pipe_config"]["net"] = {"max_horizon": 91}
    cfg["model"]["config"]["action_dit_path"] = action_dit_path
    cfg["model"]["config"]["allow_partial_action_dit_load"] = bool(action_dit_path)
    cfg["optimizer"]["lr"] = lr
    cfg["job"]["group"] = data_config
    cfg["job"]["name"] = f"w2a_{data_config}_{init_name}_{video_ckpt}_lr{lr:.3e}_layer20_bsz{bsz}"
    cfg["dataloader_train"] = {"batch_size": L(get_local_batch_size)(global_bsz=bsz)}
    cfg["trainer"]["grad_accum_iter"] = max(1, 64 // bsz)
    cfg["trainer"]["max_iter"] = 30_000
    cfg["trainer"]["validation_iter"] = 1_000
    cfg["trainer"]["logging_iter"] = 100
    cfg["checkpoint"]["save_iter"] = 1_000

    cs.store(
        group="experiment",
        package="_global_",
        name=cfg["job"]["name"],
        node=cfg,
    )


for so101_data_config, so101_video_ckpt, so101_lr, so101_bsz in it.product(
    ["so101", "so101_relative", "so101_delta", "so101_delta_30hz", "so101_delta_30hz_gripper_absolute"],
    SO101_VIDEO_CKPTS,
    [1e-4, 3e-4],
    [4, 8, 16, 32, 64],
):
    register_so101_experiment(
        init_name="random",
        video_ckpt=so101_video_ckpt,
        lr=so101_lr,
        bsz=so101_bsz,
        data_config=so101_data_config,
    )
    register_so101_experiment(
        init_name="partial_bridge_init",
        video_ckpt=so101_video_ckpt,
        lr=so101_lr,
        bsz=so101_bsz,
        data_config=so101_data_config,
        action_dit_path=str(SO101_ACTION_DECODER_PARTIAL_INITS[so101_video_ckpt].resolve()),
    )

# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_FUSED_ATTN=0 torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=...
