import argparse
from pathlib import Path

import torch

from bdh_fpga_driver import BDHFPGAEngine
from bdh_model import BabyDragonHatchling, replace_bdh_with_fpga


def build_token_sequence(
    batch: int = 1, seq_len: int = 8, embed_dim: int = 64
) -> torch.Tensor:
    positions = torch.arange(seq_len * embed_dim, dtype=torch.float32)
    signal = (
        0.55 * torch.sin(positions / 11.0)
        + 0.30 * torch.cos(positions / 5.0)
        + torch.linspace(-0.25, 0.25, seq_len * embed_dim)
    )
    return signal.reshape(batch, seq_len, embed_dim)


def build_reference_model() -> BabyDragonHatchling:
    torch.manual_seed(20260812)
    model = BabyDragonHatchling(embed_dim=64, ff_dim=128, num_layers=2)
    model.eval()
    return model


def run_hardware_check(
    port: str,
    baudrate: int,
    timeout: float,
    verbose: bool,
) -> None:
    cpu_model = build_reference_model()
    x = build_token_sequence()
    with torch.no_grad():
        y_cpu = cpu_model(x)

    print(
        f"Opening Tang Nano 9K BDH UART: port={port} baudrate={baudrate}",
        flush=True,
    )
    with BDHFPGAEngine(
        port=port, baudrate=baudrate, timeout=timeout, verbose=verbose
    ) as fpga:
        if not fpga.validate_bitstream():
            raise RuntimeError(
                "BDH MAC bitstream NOT detected on Tang Nano 9K! "
                "The board is running factory firmware (BL702 menu). "
                "Flash the BDH bitstream with: "
                "openFPGALoader -b tangnano9k rtl/impl/pnr/project.fs"
            )
        print("Bitstream validation PASSED", flush=True)

        fpga_model = build_reference_model()
        fpga_model.load_state_dict(cpu_model.state_dict())
        replace_bdh_with_fpga(fpga_model, fpga, verbose=verbose)
        fpga_model.eval()
        with torch.no_grad():
            y_fpga = fpga_model(x)

    cosine = torch.nn.functional.cosine_similarity(
        y_fpga.reshape(-1), y_cpu.reshape(-1), dim=0
    ).item()
    mae = torch.mean(torch.abs(y_fpga - y_cpu)).item()
    print(f"Cosine Similarity: {cosine:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    assert cosine > 0.98, f"FPGA cosine similarity too low: {cosine:.6f}"
    assert mae < 0.05, f"FPGA mean absolute error too high: {mae:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict physical Tang Nano 9K BDH hardware verification."
    )
    parser.add_argument(
        "--port",
        default="COM3" if "\\" in str(Path.cwd()) else "/dev/ttyUSB0",
        help="Serial device connected to the Tang Nano 9K BL702 USB-UART.",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")
    args, _unknown = parser.parse_known_args()
    return args


def test_bdh_projection_on_physical_fpga() -> None:
    args = parse_args()
    run_hardware_check(
        args.port,
        args.baudrate,
        args.timeout,
        args.verbose,
    )


if __name__ == "__main__":
    args = parse_args()
    run_hardware_check(
        args.port,
        args.baudrate,
        args.timeout,
        args.verbose,
    )
