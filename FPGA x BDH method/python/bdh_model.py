from typing import Optional

import torch

from bdh_fpga_driver import BDHFPGAEngine
from bdh_fpga_layer import BDHFPGALinearLayer, BDHHebbianAttention


class BDHBlock(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        fpga_driver: Optional[BDHFPGAEngine] = None,
        use_fpga: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.attn = BDHHebbianAttention(
            embed_dim, fpga_driver=fpga_driver, use_fpga=use_fpga
        )
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, ff_dim)
        self.fc2 = torch.nn.Linear(ff_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        ff = torch.relu(self.fc1(self.norm2(x)))
        return x + self.fc2(ff)


class BabyDragonHatchling(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        ff_dim: int = 256,
        num_layers: int = 2,
        fpga_driver: Optional[BDHFPGAEngine] = None,
        use_fpga: bool = False,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                BDHBlock(
                    embed_dim=embed_dim,
                    ff_dim=ff_dim,
                    fpga_driver=fpga_driver,
                    use_fpga=use_fpga,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


def replace_bdh_with_fpga(
    model: torch.nn.Module,
    fpga_engine: BDHFPGAEngine,
    verbose: bool = False,
) -> torch.nn.Module:
    for _name, module in model.named_modules():
        if isinstance(module, BDHHebbianAttention):
            module.attach_fpga(fpga_engine)
            module.use_fpga = True

    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear):
            sparse_mode = child.out_features >= child.in_features
            setattr(
                model,
                name,
                BDHFPGALinearLayer.from_linear(
                    child,
                    fpga_engine,
                    use_fpga=True,
                    sparse_mode=sparse_mode,
                    verbose=verbose,
                ),
            )
        else:
            replace_bdh_with_fpga(child, fpga_engine, verbose=verbose)
    return model
