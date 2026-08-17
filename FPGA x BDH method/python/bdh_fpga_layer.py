import math
from typing import Optional, Tuple

import numpy as np
import torch

from bdh_fpga_driver import BDHFPGAEngine


def symmetric_int8_quantize(values: np.ndarray) -> Tuple[np.ndarray, float]:
    values_f32 = np.asarray(values, dtype=np.float32)
    max_abs = float(np.max(np.abs(values_f32))) if values_f32.size else 0.0
    if max_abs == 0.0:
        return np.zeros_like(values_f32, dtype=np.int8), 1.0
    scale = 127.0 / max_abs
    quantized = np.clip(np.rint(values_f32 * scale), -128, 127).astype(np.int8)
    return quantized, scale


class BDHFPGALinearLayer(torch.nn.Module):
    weight: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        fpga_driver: Optional[BDHFPGAEngine] = None,
        use_fpga: bool = True,
        sparse_mode: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(
                "weight must be a 2D tensor shaped "
                "[out_features, in_features]"
            )
        self.register_buffer(
            "weight", weight.detach().to(dtype=torch.float32).contiguous()
        )
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
                raise ValueError(
                    "bias must be a 1D tensor matching out_features"
                )
            self.register_buffer(
                "bias", bias.detach().to(dtype=torch.float32).contiguous()
            )
        else:
            self.bias = None
        self.fpga_driver = fpga_driver
        self.use_fpga = use_fpga
        self.sparse_mode = sparse_mode
        self.verbose = verbose

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        fpga_driver: BDHFPGAEngine,
        use_fpga: bool = True,
        sparse_mode: bool = True,
        verbose: bool = False,
    ) -> "BDHFPGALinearLayer":
        return cls(
            linear.weight.detach(),
            linear.bias.detach() if linear.bias is not None else None,
            fpga_driver=fpga_driver,
            use_fpga=use_fpga,
            sparse_mode=sparse_mode,
            verbose=verbose,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_fpga:
            raise RuntimeError(
                "CPU fallback is disabled; set use_fpga=True with "
                "a hardware driver"
            )
        if self.fpga_driver is None:
            raise RuntimeError(
                "BDHFPGALinearLayer requires a BDHFPGAEngine instance"
            )
        if x.shape[-1] != self.weight.shape[1]:
            raise ValueError(
                "input feature dimension does not match layer weight"
            )

        original_shape = x.shape[:-1]
        out_features = int(self.weight.shape[0])
        flat_x_np = (
            x.detach()
            .to(dtype=torch.float32, device="cpu")
            .reshape(-1, x.shape[-1])
            .numpy()
        )
        weights_np = self.weight.detach().cpu().numpy()

        weights_i8, weight_scale = symmetric_int8_quantize(weights_np)
        x_i8, x_scale = symmetric_int8_quantize(flat_x_np)
        dequant_scale = weight_scale * x_scale
        if dequant_scale == 0.0:
            raise ZeroDivisionError("invalid INT8 dequantization scale")

        acc = self.fpga_driver.compute_layer_projection(
            weights_i8,
            x_i8,
            sparse_mode=self.sparse_mode,
        )

        out_fp32 = acc.astype(np.float32) / dequant_scale
        if self.bias is not None:
            out_fp32 = out_fp32 + self.bias.detach().cpu().numpy()

        y = torch.from_numpy(out_fp32).reshape(*original_shape, out_features)
        return y.to(device=x.device, dtype=torch.float32)

    def _forward_vector(
        self, x: torch.Tensor, token_index: int = 0
    ) -> torch.Tensor:
        assert self.fpga_driver is not None, (
            "BDHFPGALinearLayer requires a BDHFPGAEngine instance"
        )
        weights_np = self.weight.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()

        weights_i8, weight_scale = symmetric_int8_quantize(weights_np)
        x_i8, x_scale = symmetric_int8_quantize(x_np)
        dequant_scale = weight_scale * x_scale
        if dequant_scale == 0.0:
            raise ZeroDivisionError("invalid INT8 dequantization scale")

        acc = self.fpga_driver.compute_layer_projection(
            weights_i8,
            x_i8,
            sparse_mode=self.sparse_mode,
        )
        out_fp32 = acc.astype(np.float32) / dequant_scale
        if self.bias is not None:
            out_fp32 = out_fp32 + self.bias.detach().cpu().numpy()

        return torch.from_numpy(out_fp32).to(dtype=torch.float32)


class BDHHebbianAttention(torch.nn.Module):
    q_proj: torch.nn.Module
    k_proj: torch.nn.Module
    v_proj: torch.nn.Module
    out_proj: torch.nn.Module

    def __init__(
        self,
        embed_dim: int,
        fpga_driver: Optional[BDHFPGAEngine] = None,
        use_fpga: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.fpga_driver = fpga_driver
        self.use_fpga = use_fpga
        self.verbose = verbose

    def attach_fpga(self, fpga_driver: BDHFPGAEngine) -> None:
        self.fpga_driver = fpga_driver
        if isinstance(self.q_proj, torch.nn.Linear):
            self.q_proj = BDHFPGALinearLayer.from_linear(
                self.q_proj, fpga_driver, sparse_mode=True, verbose=self.verbose
            )
        if isinstance(self.k_proj, torch.nn.Linear):
            self.k_proj = BDHFPGALinearLayer.from_linear(
                self.k_proj, fpga_driver, sparse_mode=True, verbose=self.verbose
            )
        if isinstance(self.v_proj, torch.nn.Linear):
            self.v_proj = BDHFPGALinearLayer.from_linear(
                self.v_proj, fpga_driver, sparse_mode=False, verbose=self.verbose
            )
        if isinstance(self.out_proj, torch.nn.Linear):
            self.out_proj = BDHFPGALinearLayer.from_linear(
                self.out_proj, fpga_driver, sparse_mode=False, verbose=self.verbose
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fpga and self.fpga_driver is None:
            raise RuntimeError(
                "BDHHebbianAttention requires physical FPGA execution"
            )
        q = torch.relu(self.q_proj(x))
        k = torch.relu(self.k_proj(x))
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.embed_dim
        )
        y = torch.matmul(scores, v)
        return self.out_proj(y)
