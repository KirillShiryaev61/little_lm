from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

class PreparedData(NamedTuple):
    x_train: torch.Tensor
    y_train: torch.Tensor

class ResultForwardNet(NamedTuple):
    x_one_hot: torch.Tensor
    hidden_states: torch.Tensor
    probs: torch.Tensor

@dataclass
class Weights:
    w1: torch.Tensor
    w2: torch.Tensor