---
license: apache-2.0
library_name: pytorch
tags:
  - robotics
  - video-prediction
  - lora
  - so101
  - mimic-video
  - multi-object
---

# Mimic Video SO-101 Multi-Object 2-Camera HStack 5fps V2W LoRA

This repository contains a LoRA adapter for the Mimic Video / Cosmos Predict2
video-to-world backbone, fine-tuned on `dreamdifferent/so101_multi_object_new`.

## Checkpoint

- Adapter: `iter_000001000.pt`
- Training data layout: front and wrist cameras horizontally stacked
- Video rate: 5 fps
- Input/output resolution: 480x640
- Conditional frames: 5
- Prediction horizon: 56 future frames
- Base checkpoint expected by the adapter:
  `v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt`

This checkpoint is an unfused LoRA adapter. It should be loaded on top of the
base video backbone rather than merged into a standalone full checkpoint.

This repository uses the Mimic Video checkpoint format, not the standard Hugging
Face PEFT adapter format. The adapter weights are stored in a single PyTorch
state-dict file (`iter_000001000.pt`) whose tensors are all LoRA parameters,
and the LoRA hyperparameters / target modules are recorded in the training
config. It is intended to be loaded with the Mimic Video / Cosmos Predict2 code,
not directly with `AutoPeftModel.from_pretrained(...)`.

## Previews

`previews/` may contain ground-truth versus predicted SO-101 videos rendered
from the 1000-step adapter checkpoint.

## Notes

The dataset was converted from the original SO-101 videos by actual frame
subsampling to 5 fps. The fps value was not changed only at the container level.

This is the heterogeneous / multi-object counterpart of the SO-101 bottle-only
2-camera hstack 5 fps Video2World LoRA.
