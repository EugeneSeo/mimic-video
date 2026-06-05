# SO-101 Policy Server

This directory contains the SO-101 online prediction server used by the
experiment-aware wrappers in `scripts/so101/`.

```text
policy_server.py       TCP/websocket prediction server
run_server.sh          low-level direct wrapper around policy_server.py
```

Prefer the root-level SO-101 workflow:

```bash
bash scripts/so101/setup_env.sh so101-bottle-delta30
bash scripts/so101/check_assets.sh so101-bottle-delta30
bash scripts/so101/run_policy_server.sh so101-bottle-delta30
```

The wrapper loads `scripts/so101/experiments/<experiment>.env` and optional
`scripts/so101/artifacts/<experiment>.env`, then passes explicit checkpoint and
stats paths into `policy_server.py`.

## Experiments

```text
so101-bottle-absolute5       bottle dataset, absolute 5Hz action decoder
so101-bottle-delta30         bottle dataset, 30Hz incremental-delta action decoder
so101-multi-object-delta30   multi-object dataset, 30Hz incremental-delta action decoder
```

## Request Schema

```python
{
    "observation/images/front": hstack_image,  # uint8 RGB, HWC or CHW
    "observation/state": joint_state,          # shape (6,) or (1, 6)
    "prompt": "pick up the bottle",
    "observation/prompt_embedding": embedding, # optional, shape (512, 1024)
    "reset_history": False,
    "seed": 0,
}
```

The image is the 480x640 two-camera hstack frame:

```text
left half:  front camera
right half: wrist camera
```

If the server is not started with `--load-text-encoder`, the client must send
`observation/prompt_embedding`.

## Response Schema

```python
{
    "actions": np.ndarray,  # shape (action_horizon, 6)
    "server_timing": {...},
    "metadata": {
        "action_horizon": 90,
        "action_dim": 6,
        "action_target_frequency": 30,
        "action_delta_mode": "incremental_delta",
    },
}
```

For `DATA_CONFIG=so101`, actions are absolute joint targets. For
`DATA_CONFIG=so101_delta` and `so101_delta_30hz`, actions are incremental deltas
and deployment should cumulatively apply them at `action_target_frequency`.
