"""Small inference-only replacements for the original VITS hparams/checkpoint helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class HParams:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = HParams(**value)
            setattr(self, key, value)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __repr__(self) -> str:
        return repr(self.__dict__)


def load_hparams(config_path: Path) -> HParams:
    with config_path.open("r", encoding="utf-8") as handle:
        return HParams(**json.load(handle))


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, int, float]:
    """Load the trusted local VITS checkpoint without loading an optimizer."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=True)
    return model, int(checkpoint.get("iteration", 0)), float(checkpoint.get("learning_rate", 0.0))
