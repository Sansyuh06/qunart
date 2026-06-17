import torch
import numpy as np
from typing import Dict


def compute_weight_importance(model) -> Dict[str, np.ndarray]:
    """L2 norm of each output neuron's weights. Useful for per-neuron pruning."""
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            scores[name] = (
                module.weight.norm(dim=1, p=2).detach().cpu().to(torch.float32).numpy()
            )
    return scores


def compute_mlp_neuron_importance(gate_weight, up_weight, down_weight) -> np.ndarray:
    """
    Importance of each MLP intermediate neuron for a Llama-style gated MLP.
    Combines the outgoing gate/up rows and the incoming down column.
    """
    gate = gate_weight.detach().cpu().to(torch.float32)
    up = up_weight.detach().cpu().to(torch.float32)
    down = down_weight.detach().cpu().to(torch.float32)
    out_importance = gate.norm(dim=1).numpy() + up.norm(dim=1).numpy()
    in_importance = down.norm(dim=0).numpy()
    return out_importance + in_importance

