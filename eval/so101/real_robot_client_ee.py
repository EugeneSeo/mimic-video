"""SO-101 EE-space real-robot client for the mimic-video policy server.

The EE policy served by ``eval/so101/policy_server.py`` predicts 5 Hz chunks
with rows laid out as::

    [relative_ee_xyz(3), relative_ee_rot6d(6), absolute_gripper(1)]

The relative EE pose is expressed with respect to the end-effector pose at the
time the request observation was captured.  This client therefore latches the
request-time FK pose, reconstructs each target pose as ``T_world = T_req @
T_rel``, solves IK into SO-101 joint targets, and sends those targets to the
follower robot.  It is dry-run by default; pass ``--execute`` to command motors.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import pickle
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Iterable
from typing import Any

import numpy as np
from PIL import Image

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import websocket_protocol as framed_protocol

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PORTS_JSON = REPO_ROOT / "config" / "so101_ports.json"
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "config" / "calibration" / "robots" / "so_follower"

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_NAME = "gripper"
JOINT_NAMES = (*ARM_JOINT_NAMES, GRIPPER_NAME)
ACTION_DIM = 10

LOGGER = logging.getLogger("so101_real_robot_client_ee")


def _to_pickle_safe(obj: Any) -> Any:
    """Convert NumPy values to plain containers for cross-NumPy pickle fallback."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _to_pickle_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_pickle_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_pickle_safe(value) for value in obj)
    return obj


class FramedConnection:
    """Length-prefixed TCP connection with payload protocol auto-detection."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.payload_protocol: str | None = None

    def recv_auto(self) -> Any:
        payload = self._recv_payload()
        try:
            obj = framed_protocol.unpackb(payload)
        except Exception as msgpack_exc:
            try:
                obj = pickle.loads(payload)
            except Exception:
                raise msgpack_exc
            self.payload_protocol = "pickle-numpy"
            return obj
        self.payload_protocol = framed_protocol.PAYLOAD_PROTOCOL
        return obj

    def send(self, obj: Any) -> None:
        payload = self._pack(obj)
        self.sock.sendall(framed_protocol._HEADER_STRUCT.pack(len(payload)) + payload)

    def recv(self) -> Any:
        payload = self._recv_payload()
        if self.payload_protocol == "pickle-numpy":
            return pickle.loads(payload)
        return framed_protocol.unpackb(payload)

    def _recv_payload(self) -> bytes:
        header = framed_protocol.recv_exact(self.sock, framed_protocol._HEADER_STRUCT.size)
        (payload_size,) = framed_protocol._HEADER_STRUCT.unpack(header)
        if payload_size <= 0:
            raise ValueError(f"Invalid framed payload size: {payload_size}")
        return framed_protocol.recv_exact(self.sock, payload_size)

    def _pack(self, obj: Any) -> bytes:
        if self.payload_protocol == "pickle-numpy":
            return pickle.dumps(_to_pickle_safe(obj), protocol=pickle.HIGHEST_PROTOCOL)
        return framed_protocol.packb(obj)


@dataclasses.dataclass(frozen=True)
class ClientArgs:
    host: str
    port: int
    ports_json: pathlib.Path
    robot_id: str
    calibration_dir: pathlib.Path
    urdf_path: pathlib.Path
    target_frame_name: str
    front_camera_index_or_path: str
    wrist_camera_index_or_path: str
    camera_width: int
    camera_height: int
    camera_fps: int
    image_width: int
    image_height: int
    task: str
    prompt_embedding_path: pathlib.Path | None
    prompt_embedding_key: str | None
    control_fps: float
    actions_per_chunk: int | None
    max_relative_target: float | None
    max_ee_step_m: float | None
    max_rotation_step_deg: float | None
    max_joint_step_deg: float | None
    gripper_min: float
    gripper_max: float
    ik_position_weight: float
    ik_orientation_weight: float
    duration_s: float | None
    max_steps: int | None
    seed: int
    execute: bool
    log_every_n: int


@dataclasses.dataclass(frozen=True)
class PolicyObservation:
    request: dict[str, Any]
    ee_pose_world: np.ndarray
    joint_state: np.ndarray


@dataclasses.dataclass(frozen=True)
class JointTarget:
    raw_action: np.ndarray
    target_pose_world: np.ndarray
    joint_target: np.ndarray
    clipped: bool


class SharedTargetQueue:
    """Thread-safe joint-target queue with explicit after-chunk replanning."""

    def __init__(self) -> None:
        self.targets: deque[JointTarget] = deque()
        self.condition = threading.Condition()
        self.stop = False
        self.error: BaseException | None = None
        self.request_pending = False
        self.request_in_flight = False
        self.num_chunks = 0
        self.last_server_timing: dict[str, Any] = {}

    def should_request_locked(self) -> bool:
        return self.request_pending and not self.request_in_flight

    def begin_request_locked(self) -> None:
        self.request_pending = False
        self.request_in_flight = True

    def request_next_chunk(self) -> None:
        with self.condition:
            if not self.stop and self.error is None and not self.request_in_flight:
                self.request_pending = True
                self.condition.notify_all()

    def replace(self, targets: list[JointTarget], server_timing: dict[str, Any]) -> None:
        with self.condition:
            self.targets.clear()
            self.targets.extend(targets)
            self.request_pending = False
            self.request_in_flight = False
            self.num_chunks += 1
            self.last_server_timing = server_timing
            self.condition.notify_all()

    def pop(self) -> JointTarget | None:
        with self.condition:
            if not self.targets:
                return None
            target = self.targets.popleft()
            self.condition.notify_all()
            return target

    def wait_for_first_target(self, timeout_s: float | None = None) -> None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self.condition:
            while not self.targets and self.error is None and not self.stop:
                if deadline is None:
                    self.condition.wait(timeout=0.1)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for first policy target chunk.")
                self.condition.wait(timeout=min(0.1, remaining))
            if self.error is not None:
                raise RuntimeError("Policy request thread failed.") from self.error

    def mark_error(self, error: BaseException) -> None:
        with self.condition:
            self.error = error
            self.stop = True
            self.request_pending = False
            self.request_in_flight = False
            self.condition.notify_all()

    def request_stop(self) -> None:
        with self.condition:
            self.stop = True
            self.condition.notify_all()

    def snapshot(self) -> tuple[int, int, dict[str, Any]]:
        with self.condition:
            return len(self.targets), self.num_chunks, dict(self.last_server_timing)


def parse_args() -> ClientArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ports-json", type=pathlib.Path, default=DEFAULT_PORTS_JSON)
    parser.add_argument("--robot-id", default="follower")
    parser.add_argument("--calibration-dir", type=pathlib.Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument(
        "--urdf-path",
        type=pathlib.Path,
        default=os.environ.get("SO101_URDF_PATH"),
        help="SO-101 URDF path. Can also be provided via SO101_URDF_PATH.",
    )
    parser.add_argument("--target-frame-name", default="gripper_frame_link")
    parser.add_argument("--front-camera-index-or-path", required=True)
    parser.add_argument("--wrist-camera-index-or-path", required=True)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--task",
        default="pick up the yellow bottle and place it into the white container",
        help="Task prompt. Must exactly match a server-side prompt embedding when the server uses prompt cache.",
    )
    parser.add_argument(
        "--prompt-embedding-path",
        type=pathlib.Path,
        default=None,
        help="Optional .npy/.npz prompt embedding if the server cannot encode prompt text.",
    )
    parser.add_argument("--prompt-embedding-key", default=None, help="Array key for --prompt-embedding-path .npz files.")
    parser.add_argument("--control-fps", type=float, default=5.0, help="Motor command FPS; EE policy targets are 5 Hz.")
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=5,
        help="Execute only the first N actions from each 15-action chunk. Use 0 or negative to keep all actions.",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="LeRobot safety cap for target-present joint gap. Use a negative value to disable.",
    )
    parser.add_argument("--max-ee-step-m", type=float, default=0.05, help="Clip per-row EE translation step. Negative disables.")
    parser.add_argument(
        "--max-rotation-step-deg",
        type=float,
        default=30.0,
        help="Clip per-row EE orientation step in degrees. Negative disables.",
    )
    parser.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=10.0,
        help="Clip per-row arm joint target step in degrees. Negative disables.",
    )
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--ik-position-weight", type=float, default=1.0)
    parser.add_argument(
        "--ik-orientation-weight",
        type=float,
        default=0.01,
        help="Weak orientation tracking by default, matching LeRobot RobotKinematics default.",
    )
    parser.add_argument("--duration-s", type=float, default=None, help="Optional rollout duration. Default: run until Ctrl+C.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional maximum number of control steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Actually send targets to the robot. Default is dry-run logging.")
    parser.add_argument("--log-every-n", type=int, default=1)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    ns = parser.parse_args()

    logging.basicConfig(level=getattr(logging, ns.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if ns.urdf_path is None:
        parser.error("--urdf-path is required, or set SO101_URDF_PATH")
    ns.urdf_path = pathlib.Path(ns.urdf_path)
    if not ns.urdf_path.exists():
        parser.error(f"--urdf-path does not exist: {ns.urdf_path}")
    if ns.control_fps <= 0:
        parser.error("--control-fps must be positive")
    if ns.image_width % 2 != 0:
        parser.error("--image-width must be even for front/wrist hstack")
    if ns.actions_per_chunk is not None and ns.actions_per_chunk < 0:
        ns.actions_per_chunk = 0
    if ns.max_steps is not None and ns.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if ns.duration_s is not None and ns.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if ns.log_every_n <= 0:
        parser.error("--log-every-n must be positive")
    if ns.gripper_min > ns.gripper_max:
        parser.error("--gripper-min must be <= --gripper-max")

    max_relative_target = None if ns.max_relative_target is not None and ns.max_relative_target < 0 else ns.max_relative_target
    max_ee_step_m = None if ns.max_ee_step_m is not None and ns.max_ee_step_m < 0 else ns.max_ee_step_m
    max_rotation_step_deg = (
        None if ns.max_rotation_step_deg is not None and ns.max_rotation_step_deg < 0 else ns.max_rotation_step_deg
    )
    max_joint_step_deg = None if ns.max_joint_step_deg is not None and ns.max_joint_step_deg < 0 else ns.max_joint_step_deg
    actions_per_chunk = None if ns.actions_per_chunk is None or ns.actions_per_chunk <= 0 else ns.actions_per_chunk
    return ClientArgs(
        host=ns.host,
        port=ns.port,
        ports_json=ns.ports_json,
        robot_id=ns.robot_id,
        calibration_dir=ns.calibration_dir,
        urdf_path=ns.urdf_path,
        target_frame_name=ns.target_frame_name,
        front_camera_index_or_path=ns.front_camera_index_or_path,
        wrist_camera_index_or_path=ns.wrist_camera_index_or_path,
        camera_width=ns.camera_width,
        camera_height=ns.camera_height,
        camera_fps=ns.camera_fps,
        image_width=ns.image_width,
        image_height=ns.image_height,
        task=ns.task,
        prompt_embedding_path=ns.prompt_embedding_path,
        prompt_embedding_key=ns.prompt_embedding_key,
        control_fps=ns.control_fps,
        actions_per_chunk=actions_per_chunk,
        max_relative_target=max_relative_target,
        max_ee_step_m=max_ee_step_m,
        max_rotation_step_deg=max_rotation_step_deg,
        max_joint_step_deg=max_joint_step_deg,
        gripper_min=ns.gripper_min,
        gripper_max=ns.gripper_max,
        ik_position_weight=ns.ik_position_weight,
        ik_orientation_weight=ns.ik_orientation_weight,
        duration_s=ns.duration_s,
        max_steps=ns.max_steps,
        seed=ns.seed,
        execute=ns.execute,
        log_every_n=ns.log_every_n,
    )


def parse_camera_index_or_path(value: str) -> int | pathlib.Path:
    try:
        return int(value)
    except ValueError:
        return pathlib.Path(value)


def load_follower_port(path: pathlib.Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return str(payload["follower"])
    except KeyError as exc:
        raise KeyError(f"Missing 'follower' entry in ports file: {path}") from exc


def load_prompt_embedding(path: pathlib.Path, key: str | None) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as payload:
            if key is not None:
                if key not in payload:
                    raise KeyError(f"Embedding key {key!r} not found in {path}; available={list(payload.keys())}")
                embedding = payload[key]
            else:
                keys = list(payload.keys())
                if len(keys) != 1:
                    raise ValueError(f"{path} contains multiple arrays {keys}; pass --prompt-embedding-key")
                embedding = payload[keys[0]]
    else:
        embedding = np.load(path)

    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape == (1, 512, 1024):
        embedding = embedding[0]
    if embedding.shape != (512, 1024):
        raise ValueError(f"Expected prompt embedding shape (512, 1024) or (1, 512, 1024), got {embedding.shape}")
    return np.ascontiguousarray(embedding, dtype=np.float32)


def resize_pad(image: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
    image = to_uint8_rgb(image)
    h, w = image.shape[:2]
    scale = min(out_w / w, out_h / h)
    resized_w = max(int(round(w * scale)), 1)
    resized_h = max(int(round(h * scale)), 1)
    resized = Image.fromarray(image, "RGB").resize((resized_w, resized_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), 0)
    canvas.paste(resized, ((out_w - resized_w) // 2, (out_h - resized_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def compose_two_view_frame(front: np.ndarray, wrist: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if width % 2 != 0:
        raise ValueError(f"Two-camera horizontal canvas requires even width, got {width}")
    half_w = width // 2
    front_half = resize_pad(front, out_h=height, out_w=half_w)
    wrist_half = resize_pad(wrist, out_h=height, out_w=half_w)
    return np.ascontiguousarray(np.concatenate([front_half, wrist_half], axis=1), dtype=np.uint8)


def to_uint8_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmin(array)) < -0.05:
            array = (array + 1.0) / 2.0
        if float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {array.shape}")
    return np.ascontiguousarray(array)


def build_robot(args: ClientArgs):
    try:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise ImportError(
            "This client expects LeRobot v0.5.x in the active environment. "
            "Run with: conda run -n lerobot python eval/so101/real_robot_client_ee.py ..."
        ) from exc

    follower_port = load_follower_port(args.ports_json)
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=parse_camera_index_or_path(args.front_camera_index_or_path),
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=parse_camera_index_or_path(args.wrist_camera_index_or_path),
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        ),
    }
    config = SO101FollowerConfig(
        port=follower_port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir,
        cameras=cameras,
        max_relative_target=args.max_relative_target,
        use_degrees=True,
    )
    return SO101Follower(config)


def build_kinematics(args: ClientArgs):
    try:
        from lerobot.model.kinematics import RobotKinematics
    except ImportError as exc:
        raise ImportError("Could not import lerobot.model.kinematics.RobotKinematics.") from exc
    try:
        return RobotKinematics(
            urdf_path=str(args.urdf_path),
            target_frame_name=args.target_frame_name,
            joint_names=list(ARM_JOINT_NAMES),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize LeRobot RobotKinematics. The SO-101 EE client requires `placo` "
            "in the active LeRobot environment. Install/activate an environment with lerobot, placo, "
            f"and the SO-101 URDF available. urdf_path={args.urdf_path}"
        ) from exc


def joint_state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    values = []
    for joint_name in JOINT_NAMES:
        key = f"{joint_name}.pos"
        if key not in observation:
            raise KeyError(f"Robot observation missing joint key {key!r}; keys={sorted(observation)}")
        values.append(float(observation[key]))
    return np.asarray(values, dtype=np.float64)


def action_dict_from_target(target: Iterable[float]) -> dict[str, float]:
    return {f"{joint_name}.pos": float(value) for joint_name, value in zip(JOINT_NAMES, target, strict=True)}


def rotmat_to_6d(rot_mat: np.ndarray) -> np.ndarray:
    return np.asarray(rot_mat, dtype=np.float64)[:3, :2].reshape(6).astype(np.float32)


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    a = np.asarray(rot6d, dtype=np.float64).reshape(3, 2)
    x = a[:, 0]
    y = a[:, 1]
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-8:
        raise ValueError(f"Invalid rot6d first column norm: {rot6d}")
    x = x / x_norm
    y = y - x * np.dot(x, y)
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-8:
        raise ValueError(f"Invalid rot6d second column after orthogonalization: {rot6d}")
    y = y / y_norm
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def action_to_relative_pose(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"Expected one EE action row shape ({ACTION_DIM},), got {action.shape}")
    rel_t = np.eye(4, dtype=np.float64)
    rel_t[:3, 3] = action[:3]
    rel_t[:3, :3] = rot6d_to_rotmat(action[3:9])
    return rel_t


def rotation_angle_rad(rot_mat: np.ndarray) -> float:
    cos_angle = (np.trace(rot_mat) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def rotmat_to_rotvec(rot_mat: np.ndarray) -> np.ndarray:
    angle = rotation_angle_rad(rot_mat)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    skew = np.array(
        [
            rot_mat[2, 1] - rot_mat[1, 2],
            rot_mat[0, 2] - rot_mat[2, 0],
            rot_mat[1, 0] - rot_mat[0, 1],
        ],
        dtype=np.float64,
    )
    if abs(np.pi - angle) < 1e-5:
        eigvals, eigvecs = np.linalg.eig(rot_mat)
        axis = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
        axis /= max(np.linalg.norm(axis), 1e-8)
        return axis * angle
    return skew * (angle / (2.0 * np.sin(angle)))


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(rotvec, dtype=np.float64) / theta
    kx, ky, kz = axis
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def clip_pose_step(prev_pose: np.ndarray, target_pose: np.ndarray, args: ClientArgs) -> tuple[np.ndarray, bool]:
    clipped = False
    result = np.array(target_pose, dtype=np.float64, copy=True)
    if args.max_ee_step_m is not None:
        delta_pos = result[:3, 3] - prev_pose[:3, 3]
        delta_norm = float(np.linalg.norm(delta_pos))
        if delta_norm > args.max_ee_step_m and delta_norm > 0.0:
            result[:3, 3] = prev_pose[:3, 3] + delta_pos * (args.max_ee_step_m / delta_norm)
            clipped = True
    if args.max_rotation_step_deg is not None:
        rel_rot = prev_pose[:3, :3].T @ result[:3, :3]
        angle = rotation_angle_rad(rel_rot)
        max_angle = np.deg2rad(args.max_rotation_step_deg)
        if angle > max_angle and angle > 0.0:
            clipped_rel_rot = rotvec_to_rotmat(rotmat_to_rotvec(rel_rot) * (max_angle / angle))
            result[:3, :3] = prev_pose[:3, :3] @ clipped_rel_rot
            clipped = True
    return result, clipped


def clip_joint_step(prev_joint: np.ndarray, target_joint: np.ndarray, max_step_deg: float | None) -> tuple[np.ndarray, bool]:
    if max_step_deg is None:
        return target_joint, False
    result = np.array(target_joint, dtype=np.float64, copy=True)
    delta = result[: len(ARM_JOINT_NAMES)] - prev_joint[: len(ARM_JOINT_NAMES)]
    clipped_delta = np.clip(delta, -max_step_deg, max_step_deg)
    clipped = not np.allclose(delta, clipped_delta)
    result[: len(ARM_JOINT_NAMES)] = prev_joint[: len(ARM_JOINT_NAMES)] + clipped_delta
    return result, clipped


def capture_policy_observation(
    robot: Any,
    robot_lock: threading.Lock,
    kinematics: Any,
    args: ClientArgs,
    *,
    reset_history: bool,
    request_seed: int,
    prompt_embedding: np.ndarray | None,
) -> PolicyObservation:
    with robot_lock:
        robot_observation = robot.get_observation()
    joint_state = joint_state_from_observation(robot_observation)
    ee_pose_world = np.asarray(kinematics.forward_kinematics(joint_state[: len(ARM_JOINT_NAMES)]), dtype=np.float64)
    ee_state = np.concatenate(
        [
            ee_pose_world[:3, 3].astype(np.float32),
            rotmat_to_6d(ee_pose_world[:3, :3]),
            np.asarray([joint_state[-1]], dtype=np.float32),
        ],
        axis=0,
    )
    hstack = compose_two_view_frame(
        robot_observation["front"],
        robot_observation["wrist"],
        height=args.image_height,
        width=args.image_width,
    )
    request: dict[str, Any] = {
        "observation/images/hstack": hstack,
        "observation/state": np.ascontiguousarray(ee_state, dtype=np.float32),
        "prompt": args.task,
        "reset_history": reset_history,
        "seed": request_seed,
    }
    if prompt_embedding is not None:
        request["observation/prompt_embedding"] = prompt_embedding
    return PolicyObservation(request=request, ee_pose_world=ee_pose_world, joint_state=joint_state)


def validate_metadata(metadata: Any, prompt_embedding: np.ndarray | None) -> tuple[int, int]:
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected server metadata dict, got {type(metadata)!r}: {metadata!r}")
    action_horizon = int(metadata.get("action_horizon", 15))
    action_dim = int(metadata.get("action_dim", ACTION_DIM))
    action_delta_mode = metadata.get("action_delta_mode")
    if action_dim != ACTION_DIM:
        raise ValueError(f"Expected EE action_dim={ACTION_DIM}, got {action_dim} from server metadata={metadata}")
    if action_horizon <= 0:
        raise ValueError(f"Invalid action_horizon={action_horizon} from server metadata={metadata}")
    if action_delta_mode not in (None, "relative_ee"):
        raise ValueError(f"Expected action_delta_mode='relative_ee', got {action_delta_mode!r}")
    if metadata.get("load_text_encoder") is False and prompt_embedding is None:
        LOGGER.warning(
            "Server metadata reports load_text_encoder=false and no --prompt-embedding-path was provided. "
            "This is OK only if the server has a prompt embedding manifest for the task prompt."
        )
    LOGGER.info("Connected to EE policy server metadata=%s", metadata)
    return action_horizon, action_dim


def validate_actions(response: Any, expected_shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(response, str):
        raise RuntimeError(f"Server returned error:\n{response}")
    if not isinstance(response, dict):
        raise TypeError(f"Expected response dict, got {type(response)!r}: {response!r}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.shape != expected_shape:
        raise ValueError(f"Expected actions shape {expected_shape}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Server returned non-finite actions")
    return actions, dict(response.get("server_timing", {}))


def build_joint_targets_from_actions(
    actions: np.ndarray,
    policy_obs: PolicyObservation,
    kinematics: Any,
    args: ClientArgs,
) -> list[JointTarget]:
    targets: list[JointTarget] = []
    prev_pose = policy_obs.ee_pose_world.copy()
    prev_joint = policy_obs.joint_state.copy()
    ik_seed = policy_obs.joint_state.copy()

    for row in np.asarray(actions, dtype=np.float32):
        target_pose = policy_obs.ee_pose_world @ action_to_relative_pose(row)
        target_pose, pose_clipped = clip_pose_step(prev_pose, target_pose, args)

        ik_joint = np.asarray(
            kinematics.inverse_kinematics(
                ik_seed,
                target_pose,
                position_weight=args.ik_position_weight,
                orientation_weight=args.ik_orientation_weight,
            ),
            dtype=np.float64,
        )
        full_target = np.array(prev_joint, dtype=np.float64, copy=True)
        full_target[: len(ARM_JOINT_NAMES)] = ik_joint[: len(ARM_JOINT_NAMES)]
        full_target[-1] = np.clip(float(row[9]), args.gripper_min, args.gripper_max)
        full_target, joint_clipped = clip_joint_step(prev_joint, full_target, args.max_joint_step_deg)

        targets.append(
            JointTarget(
                raw_action=np.asarray(row, dtype=np.float32),
                target_pose_world=target_pose.astype(np.float64),
                joint_target=full_target.astype(np.float64),
                clipped=pose_clipped or joint_clipped,
            )
        )
        prev_pose = (
            np.asarray(kinematics.forward_kinematics(full_target[: len(ARM_JOINT_NAMES)]), dtype=np.float64)
            if joint_clipped
            else target_pose
        )
        prev_joint = full_target
        ik_seed = full_target
    return targets


def policy_request_loop(
    conn: FramedConnection,
    robot: Any,
    robot_lock: threading.Lock,
    kinematics: Any,
    args: ClientArgs,
    queue: SharedTargetQueue,
    expected_shape: tuple[int, int],
    prompt_embedding: np.ndarray | None,
) -> None:
    request_idx = 0
    try:
        while True:
            with queue.condition:
                while not queue.stop and not queue.should_request_locked():
                    queue.condition.wait(timeout=0.05)
                if queue.stop:
                    return
                queue.begin_request_locked()

            policy_obs = capture_policy_observation(
                robot,
                robot_lock,
                kinematics,
                args,
                reset_history=request_idx == 0,
                request_seed=args.seed + request_idx,
                prompt_embedding=prompt_embedding,
            )
            started_at = time.perf_counter()
            conn.send(policy_obs.request)
            response = conn.recv()
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            actions, server_timing = validate_actions(response, expected_shape)
            received_shape = actions.shape
            if args.actions_per_chunk is not None:
                actions = actions[: args.actions_per_chunk]
            if len(actions) == 0:
                raise ValueError("No actions left after applying --actions-per-chunk")
            joint_targets = build_joint_targets_from_actions(actions, policy_obs, kinematics, args)
            server_timing.setdefault("client_roundtrip_ms", elapsed_ms)
            queue.replace(joint_targets, server_timing)
            LOGGER.info(
                "received_chunk=%d received_actions=%s queued_targets=%d request_ee_xyz=%s server_timing=%s",
                request_idx,
                received_shape,
                len(joint_targets),
                np.array2string(policy_obs.ee_pose_world[:3, 3], precision=4),
                server_timing,
            )
            request_idx += 1
    except BaseException as exc:  # noqa: BLE001 - propagate to main thread through queue state.
        queue.mark_error(exc)


def run_control_loop(robot: Any, robot_lock: threading.Lock, args: ClientArgs, queue: SharedTargetQueue) -> None:
    period_s = 1.0 / args.control_fps
    start = time.monotonic()
    motor_step = 0
    policy_step = 0
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    LOGGER.info(
        "Starting %s EE control loop at %.3f Hz; actions_per_chunk=%s",
        mode,
        args.control_fps,
        "all" if args.actions_per_chunk is None else args.actions_per_chunk,
    )

    while True:
        now = time.monotonic()
        if args.duration_s is not None and now - start >= args.duration_s:
            LOGGER.info("Stopping after duration %.2fs", args.duration_s)
            return
        if args.max_steps is not None and motor_step >= args.max_steps:
            LOGGER.info("Stopping after max_steps=%d", args.max_steps)
            return
        with queue.condition:
            if queue.error is not None:
                raise RuntimeError("Policy request thread failed.") from queue.error

        tick_started = time.perf_counter()
        target = queue.pop()
        queue_size, num_chunks, server_timing = queue.snapshot()
        if target is None:
            queue.request_next_chunk()
            LOGGER.warning("motor_step=%d policy_step=%d queue_underrun: holding current position", motor_step, policy_step)
            sleep_s = period_s - (time.perf_counter() - tick_started)
            if sleep_s > 0:
                time.sleep(sleep_s)
            motor_step += 1
            continue

        with robot_lock:
            observation = robot.get_observation()
            current_joint = joint_state_from_observation(observation)
            target_action = action_dict_from_target(target.joint_target)
            sent_action = robot.send_action(target_action) if args.execute else target_action

        if motor_step % args.log_every_n == 0:
            LOGGER.info(
                "motor_step=%d policy_step=%d mode=%s queue=%d chunks=%d clipped=%s current_joint=%s "
                "raw_ee_action=%s target_xyz=%s target_joint=%s sent=%s timing=%s",
                motor_step,
                policy_step,
                mode,
                queue_size,
                num_chunks,
                target.clipped,
                np.array2string(current_joint, precision=3),
                np.array2string(target.raw_action, precision=4),
                np.array2string(target.target_pose_world[:3, 3], precision=4),
                np.array2string(target.joint_target, precision=3),
                sent_action,
                server_timing,
            )

        motor_step += 1
        policy_step += 1
        remaining_targets, _, _ = queue.snapshot()
        if remaining_targets == 0:
            LOGGER.info(
                "completed queued EE target chunk at policy_step=%d; capturing fresh observation for next server request",
                policy_step,
            )
            queue.request_next_chunk()

        sleep_s = period_s - (time.perf_counter() - tick_started)
        if sleep_s > 0:
            time.sleep(sleep_s)


def main(args: ClientArgs) -> None:
    prompt_embedding = None
    if args.prompt_embedding_path is not None:
        prompt_embedding = load_prompt_embedding(args.prompt_embedding_path, args.prompt_embedding_key)
        LOGGER.info("Loaded prompt embedding from %s", args.prompt_embedding_path)

    kinematics = build_kinematics(args)
    with socket.create_connection((args.host, args.port)) as sock:
        conn = FramedConnection(sock)
        metadata = conn.recv_auto()
        LOGGER.info("Detected server payload protocol: %s", conn.payload_protocol)
        action_horizon, action_dim = validate_metadata(metadata, prompt_embedding)
        expected_shape = (action_horizon, action_dim)
        queued_horizon = action_horizon if args.actions_per_chunk is None else min(args.actions_per_chunk, action_horizon)
        LOGGER.info(
            "Using queued_horizon=%d from server_horizon=%d. Next server request is triggered only after queued targets finish.",
            queued_horizon,
            action_horizon,
        )

        robot = build_robot(args)
        robot_lock = threading.Lock()
        queue = SharedTargetQueue()
        request_thread: threading.Thread | None = None
        try:
            LOGGER.info(
                "Connecting SO-101 follower id=%s calibration_dir=%s execute=%s urdf=%s target_frame=%s",
                args.robot_id,
                args.calibration_dir,
                args.execute,
                args.urdf_path,
                args.target_frame_name,
            )
            robot.connect(calibrate=True)
            LOGGER.info("Robot connected. observation_features=%s action_features=%s", robot.observation_features, robot.action_features)

            request_thread = threading.Thread(
                target=policy_request_loop,
                name="policy-request-loop",
                args=(conn, robot, robot_lock, kinematics, args, queue, expected_shape, prompt_embedding),
                daemon=True,
            )
            request_thread.start()
            queue.request_next_chunk()
            queue.wait_for_first_target(timeout_s=120.0)
            run_control_loop(robot, robot_lock, args, queue)
        finally:
            queue.request_stop()
            if request_thread is not None:
                request_thread.join(timeout=2.0)
            try:
                if robot.is_connected:
                    robot.disconnect()
                    LOGGER.info("Robot disconnected.")
            except Exception:  # noqa: BLE001 - cleanup best effort.
                LOGGER.exception("Failed during robot disconnect cleanup")


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        sys.exit(130)
