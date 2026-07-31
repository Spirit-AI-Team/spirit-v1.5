"""FSDP2-safe model and resumable training checkpoint handling."""

import copy
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor


_PACKED_MLP_SUFFIX = ".mlp.gate_up_proj.weight"
_TRAINING_STATE_FORMAT = "spirit-fsdp2-training-state-v1"


def _export_reference_text_mlp_keys(state_dict):
    """Restore stock Qwen gate/up keys in a full dense exported state dict."""
    packed_keys = [
        name
        for name in state_dict
        if name.endswith(_PACKED_MLP_SUFFIX)
    ]
    for packed_name in packed_keys:
        packed_weight = state_dict.pop(packed_name)
        if packed_weight.ndim != 2 or packed_weight.shape[0] % 2:
            raise RuntimeError(
                f"Invalid packed Text MLP weight {packed_name}: "
                f"shape={tuple(packed_weight.shape)}"
            )
        prefix = packed_name[: -len("gate_up_proj.weight")]
        gate_name = prefix + "gate_proj.weight"
        up_name = prefix + "up_proj.weight"
        if gate_name in state_dict or up_name in state_dict:
            raise RuntimeError(
                f"Text MLP checkpoint key collision under prefix {prefix}"
            )
        gate_weight, up_weight = packed_weight.chunk(2, dim=0)
        state_dict[gate_name] = gate_weight
        state_dict[up_name] = up_weight
    return len(packed_keys)


def _packed_text_mlp_parameter_shapes(model):
    shapes = {}
    for name, parameter in model.named_parameters():
        if not name.endswith(_PACKED_MLP_SUFFIX):
            continue
        shape = tuple(parameter.shape)
        if len(shape) != 2 or shape[0] % 2:
            raise RuntimeError(
                f"Invalid packed Text MLP parameter {name}: shape={shape}"
            )
        shapes[name] = shape
    return shapes


def _separate_mlp_names(packed_name):
    if not packed_name.endswith(_PACKED_MLP_SUFFIX):
        raise ValueError(f"Not a packed Text MLP parameter: {packed_name}")
    prefix = packed_name[: -len("gate_up_proj.weight")]
    return prefix + "gate_proj.weight", prefix + "up_proj.weight"


def _clone_checkpoint_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _replace_group_parameter(group, old_name, replacement_names):
    parameters = group.get("params")
    if not isinstance(parameters, list):
        raise RuntimeError("Optimizer param group must contain a params list")
    indices = [index for index, name in enumerate(parameters) if name == old_name]
    if len(indices) != 1:
        raise RuntimeError(
            f"Expected optimizer parameter {old_name} exactly once, "
            f"found {len(indices)}"
        )
    index = indices[0]
    parameters[index : index + 1] = list(replacement_names)


def export_reference_text_mlp_optimizer_state(
    optimizer_state_dict,
    packed_parameter_shapes,
):
    """Convert canonical packed optimizer FQNs/states to stock Qwen FQNs.

    Full-size per-parameter tensors such as Adam's ``exp_avg`` and
    ``exp_avg_sq`` are split on dimension zero. Scalar state such as ``step``
    and non-tensor metadata are duplicated. Param-group ordering and
    hyperparameters are preserved.
    """
    state = optimizer_state_dict.get("state")
    groups = optimizer_state_dict.get("param_groups")
    if not isinstance(state, dict) or not isinstance(groups, list):
        raise RuntimeError("Invalid canonical optimizer state_dict structure")

    converted = 0
    for packed_name, packed_shape in packed_parameter_shapes.items():
        group_matches = [
            group for group in groups if packed_name in group.get("params", [])
        ]
        has_state = packed_name in state
        if not group_matches and not has_state:
            continue
        if len(group_matches) != 1:
            raise RuntimeError(
                f"Packed optimizer parameter {packed_name} must occur in one "
                f"param group, found {len(group_matches)}"
            )
        gate_name, up_name = _separate_mlp_names(packed_name)
        if gate_name in state or up_name in state:
            raise RuntimeError(
                f"Optimizer state key collision for {gate_name}/{up_name}"
            )
        if any(
            gate_name in group.get("params", [])
            or up_name in group.get("params", [])
            for group in groups
        ):
            raise RuntimeError(
                f"Optimizer param-group collision for {gate_name}/{up_name}"
            )

        if has_state:
            packed_state = state.pop(packed_name)
            if not isinstance(packed_state, dict):
                raise RuntimeError(
                    f"Optimizer state for {packed_name} must be a dictionary"
                )
            gate_state = {}
            up_state = {}
            for state_name, value in packed_state.items():
                if (
                    isinstance(value, torch.Tensor)
                    and tuple(value.shape) == tuple(packed_shape)
                ):
                    gate_value, up_value = value.chunk(2, dim=0)
                    # Keep storage-sharing views while serializing. Cloning
                    # both Adam moments here would transiently duplicate the
                    # complete optimizer state on rank 0.
                    gate_state[state_name] = gate_value.detach()
                    up_state[state_name] = up_value.detach()
                else:
                    gate_state[state_name] = _clone_checkpoint_value(value)
                    up_state[state_name] = _clone_checkpoint_value(value)
            state[gate_name] = gate_state
            state[up_name] = up_state

        _replace_group_parameter(
            group_matches[0], packed_name, (gate_name, up_name)
        )
        converted += 1
    return converted


def _equal_checkpoint_values(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    return type(left) is type(right) and left == right


def import_reference_text_mlp_optimizer_state(
    optimizer_state_dict,
    packed_parameter_shapes,
):
    """Merge stock Qwen gate/up optimizer states into packed training state."""
    state = optimizer_state_dict.get("state")
    groups = optimizer_state_dict.get("param_groups")
    if not isinstance(state, dict) or not isinstance(groups, list):
        raise RuntimeError("Invalid canonical optimizer state_dict structure")

    converted = 0
    for packed_name, packed_shape in packed_parameter_shapes.items():
        gate_name, up_name = _separate_mlp_names(packed_name)
        gate_groups = [
            group for group in groups if gate_name in group.get("params", [])
        ]
        up_groups = [
            group for group in groups if up_name in group.get("params", [])
        ]
        gate_has_state = gate_name in state
        up_has_state = up_name in state
        present = bool(gate_groups or up_groups or gate_has_state or up_has_state)
        if not present:
            continue
        if len(gate_groups) != 1 or len(up_groups) != 1:
            raise RuntimeError(
                f"Separate optimizer parameters {gate_name}/{up_name} must "
                "each occur in exactly one param group"
            )
        if gate_groups[0] is not up_groups[0]:
            raise RuntimeError(
                f"Cannot pack {gate_name}/{up_name}: they belong to different "
                "optimizer param groups"
            )
        if packed_name in state or any(
            packed_name in group.get("params", []) for group in groups
        ):
            raise RuntimeError(
                f"Packed optimizer key already exists: {packed_name}"
            )
        if gate_has_state != up_has_state:
            raise RuntimeError(
                f"Incomplete optimizer state for {gate_name}/{up_name}"
            )

        if gate_has_state:
            gate_state = state.pop(gate_name)
            up_state = state.pop(up_name)
            if not isinstance(gate_state, dict) or not isinstance(up_state, dict):
                raise RuntimeError("Separate optimizer states must be dictionaries")
            if set(gate_state) != set(up_state):
                raise RuntimeError(
                    f"Optimizer state fields differ for {gate_name}/{up_name}"
                )
            gate_shape = (packed_shape[0] // 2, *packed_shape[1:])
            packed_state = {}
            for state_name in gate_state:
                gate_value = gate_state[state_name]
                up_value = up_state[state_name]
                if (
                    isinstance(gate_value, torch.Tensor)
                    and isinstance(up_value, torch.Tensor)
                    and tuple(gate_value.shape) == tuple(gate_shape)
                    and tuple(up_value.shape) == tuple(gate_shape)
                ):
                    if gate_value.dtype != up_value.dtype:
                        raise RuntimeError(
                            f"Optimizer tensor dtype differs for field {state_name}"
                        )
                    packed_state[state_name] = torch.cat(
                        (gate_value, up_value), dim=0
                    )
                else:
                    if not _equal_checkpoint_values(gate_value, up_value):
                        raise RuntimeError(
                            f"Non-parameter optimizer state {state_name} differs "
                            f"for {gate_name}/{up_name}"
                        )
                    packed_state[state_name] = _clone_checkpoint_value(gate_value)
            state[packed_name] = packed_state

        group = gate_groups[0]
        parameters = group["params"]
        gate_index = parameters.index(gate_name)
        up_index = parameters.index(up_name)
        insertion_index = min(gate_index, up_index)
        group["params"] = [
            name for name in parameters if name not in {gate_name, up_name}
        ]
        group["params"].insert(insertion_index, packed_name)
        converted += 1
    return converted


def _full_model_state_dict(model):
    if not dist.is_initialized():
        return model.state_dict()
    # This call is collective. With cpu_offload=True, rank 0 receives the full
    # CPU state dict and all other ranks receive an empty dict.
    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def _full_optimizer_state_dict(model, optimizer):
    return get_optimizer_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def _replace_latest_file(source_path, latest_path):
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    try:
        latest_path.hardlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, latest_path)


def _cpu_checkpoint_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_checkpoint_value(item) for item in value)
    return copy.deepcopy(value)


def save_model(
    model,
    step: int,
    output_dir: str,
    rank: int,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
):
    """Save stock-compatible model weights and optional resumable state.

    The optimizer is exported using canonical parameter FQNs and the stock
    separate gate/up layout. Therefore the file can be loaded by either a
    stock Qwen model or this runtime's packed Text MLP after migration.
    """
    if optimizer is not None:
        active_gradients = sum(
            parameter.grad is not None
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if active_gradients:
            raise RuntimeError(
                "Refusing to save optimizer state in the middle of gradient "
                f"accumulation: {active_gradients} gradients are still active"
            )
    state_dict = _full_model_state_dict(model)
    if rank == 0:
        exported_text_mlps = _export_reference_text_mlp_keys(state_dict)
        invalid = [
            (name, type(value).__name__)
            for name, value in state_dict.items()
            if not isinstance(value, torch.Tensor) or isinstance(value, DTensor)
        ]
        if invalid:
            raise RuntimeError(
                "FSDP2 full state dict contains non-dense tensors: "
                f"{invalid[:20]}"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Cloning independently deliberately breaks the Qwen tied-weight
        # storage alias, which safetensors cannot encode. Loading copies both
        # equal values into the model's already-tied Parameter.
        tensors = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in state_dict.items()
        }
        save_path = output_path / f"model_step_{step}.safetensors"
        latest_path = output_path / "model.safetensors"
        save_file(tensors, save_path)
        _replace_latest_file(save_path, latest_path)
        print(
            f"[FSDP2_CHECKPOINT] saved={save_path} "
            f"exported_packed_text_mlps={exported_text_mlps}",
            flush=True,
        )
    del state_dict

    if optimizer is not None:
        optimizer_state = _full_optimizer_state_dict(model, optimizer)
        if rank == 0:
            packed_shapes = _packed_text_mlp_parameter_shapes(model)
            exported_optimizer_mlps = export_reference_text_mlp_optimizer_state(
                optimizer_state,
                packed_shapes,
            )
            training_state = {
                "format": _TRAINING_STATE_FORMAT,
                "step": int(step),
                "text_mlp_optimizer_layout": "separate_gate_up",
                # get_optimizer_state_dict(..., cpu_offload=True) already
                # returned dense CPU tensors. Avoid another full optimizer
                # clone, which is material for multi-billion-parameter jobs.
                "optimizer": optimizer_state,
                "scheduler": (
                    _cpu_checkpoint_value(scheduler.state_dict())
                    if scheduler is not None
                    else None
                ),
                "scaler": (
                    _cpu_checkpoint_value(scaler.state_dict())
                    if scaler is not None
                    else None
                ),
            }
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            training_path = output_path / f"training_state_step_{step}.pt"
            latest_training_path = output_path / "training_state.pt"
            torch.save(training_state, training_path)
            _replace_latest_file(training_path, latest_training_path)
            print(
                f"[FSDP2_TRAINING_CHECKPOINT] saved={training_path} "
                f"layout=separate_gate_up "
                f"exported_packed_text_mlps={exported_optimizer_mlps}",
                flush=True,
            )
        del optimizer_state

    if dist.is_initialized():
        dist.barrier()


save_model._supports_training_state = True


def _resolve_training_state_path(checkpoint_path):
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "training_state.pt"
    return path


def read_training_checkpoint_step(checkpoint_path):
    """Read only the trusted local training-state metadata needed at startup."""
    path = _resolve_training_state_path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != _TRAINING_STATE_FORMAT:
        raise RuntimeError(
            f"Unsupported training checkpoint format in {path}: "
            f"{payload.get('format')!r}"
        )
    return int(payload["step"])


def load_training_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    *,
    scheduler=None,
    scaler=None,
):
    """Restore optimizer/scheduler/scaler state into a packed FSDP2 model.

    Model weights remain loaded through the existing pretrained/model
    checkpoint path. This function restores the training state and returns the
    saved absolute micro-step so the runtime can continue its step counter.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    path = _resolve_training_state_path(checkpoint_path)
    if rank == 0:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != _TRAINING_STATE_FORMAT:
            raise RuntimeError(
                f"Unsupported training checkpoint format in {path}: "
                f"{payload.get('format')!r}"
            )
        if payload.get("text_mlp_optimizer_layout") != "separate_gate_up":
            raise RuntimeError(
                "Training checkpoint must use separate_gate_up optimizer layout"
            )
        optimizer_state = payload["optimizer"]
        imported_mlps = import_reference_text_mlp_optimizer_state(
            optimizer_state,
            _packed_text_mlp_parameter_shapes(model),
        )
        metadata = {
            "step": int(payload["step"]),
            "scheduler": payload.get("scheduler"),
            "scaler": payload.get("scaler"),
            "imported_mlps": imported_mlps,
        }
    else:
        optimizer_state = {}
        metadata = None

    if dist.is_initialized():
        objects = [metadata]
        dist.broadcast_object_list(objects, src=0)
        metadata = objects[0]
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        broadcast_from_rank0=dist.is_initialized(),
    )
    set_optimizer_state_dict(
        model,
        optimizer,
        optimizer_state,
        options=options,
    )
    if scheduler is not None:
        if metadata["scheduler"] is None:
            raise RuntimeError("Checkpoint does not contain scheduler state")
        scheduler.load_state_dict(metadata["scheduler"])
    if scaler is not None:
        if metadata["scaler"] is None:
            raise RuntimeError("Checkpoint does not contain scaler state")
        scaler.load_state_dict(metadata["scaler"])
    if rank == 0:
        print(
            f"[FSDP2_TRAINING_CHECKPOINT] loaded={path} "
            f"step={metadata['step']} "
            f"imported_separate_text_mlps={metadata['imported_mlps']}",
            flush=True,
        )
    return metadata["step"]
