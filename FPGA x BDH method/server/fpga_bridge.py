import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple

import numpy as np
import serial.tools.list_ports

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from bdh_fpga_driver import BDHFPGAEngine, HardwareError  # noqa: E402


@dataclass(frozen=True)
class FPGAStatus:
    connected: bool
    port: Optional[str]
    baudrate: int
    zero_skip: bool
    description: Optional[str] = None

    def as_payload(self) -> Dict[str, object]:
        reason = self.description or ("DEVICE CONNECTED" if self.connected else "NO TANG NANO 9K DETECTED")
        payload: Dict[str, object] = {
            "type": "status",
            "connected": self.connected,
            "baudrate": self.baudrate,
            "zero_skip": self.zero_skip,
            "local_reason": reason,
        }
        if self.port is not None:
            payload["port"] = self.port
        if self.description is not None:
            payload["description"] = self.description
        return payload


class FPGABridge:
    def __init__(
        self,
        baudrate: int = 115200,
        timeout: float = 0.35,
        zero_skip: bool = True,
    ) -> None:
        self.baudrate = baudrate
        self.timeout = timeout
        self.zero_skip = zero_skip
        self.engine: Optional[BDHFPGAEngine] = None
        self.status = FPGAStatus(False, None, baudrate, zero_skip, "SCANNING USB PORTS...")

    def detected_ports(self) -> Dict[str, str]:
        return {
            port.device: port.description
            for port in serial.tools.list_ports.comports()
        }

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self.status = FPGAStatus(False, None, self.baudrate, self.zero_skip)

    def candidate_ports(self) -> List[Tuple[str, str]]:
        ports = [
            (port.device, port.description)
            for port in serial.tools.list_ports.comports()
        ]
        if sys.platform.startswith("win"):
            ports.extend(
                (f"COM{i}", "Windows COM probe") for i in range(1, 33)
            )
        else:
            usb_ports = [
                (f"/dev/ttyUSB{i}", "Linux USB serial probe")
                for i in range(8)
            ]
            acm_ports = [
                (f"/dev/ttyACM{i}", "Linux ACM serial probe")
                for i in range(8)
            ]
            ports.extend(usb_ports + acm_ports)
        seen: set[str] = set()
        unique_ports = []
        for port, description in ports:
            if port not in seen:
                unique_ports.append((port, description))
                seen.add(port)
        return unique_ports

    def scan_once(self) -> FPGAStatus:
        if self.engine is not None:
            detected = self.detected_ports()
            if self.status.port in detected and self.engine.serial.is_open:
                self.status = FPGAStatus(
                    True,
                    self.status.port,
                    self.baudrate,
                    self.zero_skip,
                    detected[self.status.port],
                )
                return self.status
            self.close()

        detected = self.detected_ports()
        baudrates_to_try = [115200, 921600]
        found_port = None
        for port, description in self.candidate_ports():
            if description.endswith("probe") and port not in detected:
                continue
            found_port = port
            for br in baudrates_to_try:
                try:
                    engine = BDHFPGAEngine(
                        port=port,
                        baudrate=br,
                        timeout=self.timeout,
                        verbose=False,
                    )
                    engine.zero_skip = self.zero_skip
                    if not engine.validate_bitstream():
                        engine.close()
                        self.status = FPGAStatus(
                            False,
                            port,
                            br,
                            self.zero_skip,
                            f"{port} DETECTED (FACTORY BL702 FIRMWARE - FLASH BDH BITSTREAM)",
                        )
                        return self.status
                    engine.reset_accumulator()
                    self.engine = engine
                    self.baudrate = br
                    self.status = FPGAStatus(
                        True, port, br, self.zero_skip, f"{port} CONNECTED @ {br} (BDH BITSTREAM VALIDATED)"
                    )
                    return self.status
                except HardwareError:
                    continue

        if found_port:
            self.status = FPGAStatus(
                False,
                found_port,
                self.baudrate,
                self.zero_skip,
                f"{found_port} DETECTED (NEEDS BDH BITSTREAM FLASHED)",
            )
        else:
            self.status = FPGAStatus(
                False, None, self.baudrate, self.zero_skip, "NO TANG NANO 9K USB PORT DETECTED"
            )
        return self.status

    def compute_dot_product(
        self,
        weights: np.ndarray,
        activations: np.ndarray,
        sparse_mode: bool = True,
    ) -> int:
        if self.engine is None:
            raise HardwareError("Tang Nano 9K not detected on serial port!")
        try:
            return self.engine.compute_dot_product(
                weights, activations, sparse_mode=sparse_mode
            )
        except HardwareError:
            self.close()
            raise

    def compute_layer_projection(
        self,
        weights: np.ndarray,
        activations: np.ndarray,
        sparse_mode: bool = True,
    ) -> np.ndarray:
        if self.engine is None:
            raise HardwareError("Tang Nano 9K not detected on serial port!")
        try:
            return self.engine.compute_layer_projection(
                weights, activations, sparse_mode=sparse_mode
            )
        except HardwareError:
            self.close()
            raise

    async def status_stream(self) -> AsyncIterator[FPGAStatus]:
        while True:
            yield await asyncio.to_thread(self.scan_once)
            await asyncio.sleep(1.0)
