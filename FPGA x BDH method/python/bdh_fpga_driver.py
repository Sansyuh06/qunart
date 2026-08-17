import importlib
import time
from typing import Any, Optional, Tuple

import numpy as np

try:
    serial: Any = importlib.import_module("serial")
except ImportError:
    serial = None

SerialException: Any = (
    getattr(serial, "SerialException", OSError)
    if serial is not None
    else OSError
)

CMD_RESET_ACC = 0x00
CMD_MAC_PAIR = 0x01
CMD_MAC_STREAM = 0x02
CMD_READ_ACC = 0x03
CMD_MAC_SPARSE = 0x04
MAX_STREAM_PAIRS = 65535


class HardwareError(RuntimeError):
    pass


class BDHFPGAEngine:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 2.0,
        write_timeout: float = 2.0,
        verbose: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.verbose = verbose
        self.zero_skip = True

        if serial is None:
            raise HardwareError(
                "pyserial is required for Tang Nano 9K UART access. "
                "Install dependencies with: python -m pip install -r "
                "requirements.txt"
            )

        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
                write_timeout=write_timeout,
            )
        except (SerialException, OSError) as exc:
            msg = "Tang Nano 9K not detected on serial port!"
            raise HardwareError(msg) from exc

        if not self.serial.is_open:
            raise HardwareError("Tang Nano 9K not detected on serial port!")

        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

        # Drain any unsolicited text (e.g. factory BL702 menu)
        import time as _time
        _time.sleep(0.1)
        stale = self.serial.read(self.serial.in_waiting or 0)
        if stale and self.verbose:
            print(
                f"Drained {len(stale)} stale bytes from {port}",
                flush=True,
            )

    def validate_bitstream(self) -> bool:
        """Send a known dot product to verify the BDH MAC bitstream.

        Returns True if the FPGA correctly computes [1,1,1,1]·[1,1,1,1]=4.
        Returns False if the board is running factory firmware or a wrong
        bitstream.
        """
        test_w = np.array([1, 1, 1, 1], dtype=np.int8)
        test_a = np.array([1, 1, 1, 1], dtype=np.int8)
        try:
            result = self._stream_dot_product(test_w, test_a, False)
        except (HardwareError, OSError):
            return False
        if self.verbose:
            print(
                f"Bitstream validation: [1,1,1,1]·[1,1,1,1] = {result} "
                f"(expected 4)",
                flush=True,
            )
        return result == 4


    def close(self) -> None:
        if getattr(self, "serial", None) is not None and self.serial.is_open:
            self.serial.close()

    def __enter__(self) -> "BDHFPGAEngine":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def reset_accumulator(self) -> None:
        self._write_packet(bytes([CMD_RESET_ACC]))

    def _stream_dot_product(
        self,
        weights_i8: np.ndarray,
        activations_i8: np.ndarray,
        sparse_mode: bool,
    ) -> int:
        length = int(weights_i8.shape[0])
        self.reset_accumulator()

        offset = 0
        while offset < length:
            chunk_len = min(length - offset, MAX_STREAM_PAIRS)
            w_chunk = weights_i8[offset:offset + chunk_len]
            a_chunk = activations_i8[offset:offset + chunk_len]
            payload = np.empty(chunk_len * 2, dtype=np.int8)
            payload[0::2] = w_chunk
            payload[1::2] = a_chunk
            command = CMD_MAC_SPARSE if sparse_mode else CMD_MAC_STREAM
            packet = (
                bytes([command])
                + chunk_len.to_bytes(2, byteorder="little", signed=False)
                + payload.tobytes()
            )
            self._write_packet(packet)
            offset += chunk_len

        self._write_packet(bytes([CMD_READ_ACC]))
        data = self._read_exactly(4)
        return int.from_bytes(data, byteorder="little", signed=True)

    def compute_dot_product(
        self,
        weights: np.ndarray,
        activations: np.ndarray,
        sparse_mode: bool = True,
    ) -> int:
        weights_i8 = self._validate_int8_vector(weights, "weights")
        activations_i8 = self._validate_int8_vector(activations, "activations")
        if weights_i8.shape[0] != activations_i8.shape[0]:
            raise ValueError(
                "weights and activations must have identical length"
            )

        length = int(weights_i8.shape[0])
        zero_count = int(np.count_nonzero(activations_i8 == 0))
        result = self._stream_dot_product(
            weights_i8, activations_i8, sparse_mode=sparse_mode
        )
        if self.verbose:
            sparsity = (zero_count / length) if length else 0.0
            useful = max(length - zero_count, 1)
            speedup = (length / useful) if sparse_mode and length else 1.0
            print(
                f"bdh_fpga length={length} sparse={sparse_mode} "
                f"zero_sparsity={sparsity:.2%} "
                f"estimated_zero_skip_speedup={speedup:.2f}x",
                flush=True,
            )
        return result

    def compute_layer_projection(
        self,
        weights: np.ndarray,
        activations: np.ndarray,
        sparse_mode: bool = True,
    ) -> np.ndarray:
        weights_i8 = self._validate_int8_matrix(weights, "weights")

        is_1d = False
        orig_batch_shape: Optional[Tuple[int, ...]] = None

        if not isinstance(activations, np.ndarray):
            raise TypeError("activations must be a numpy.ndarray")
        if activations.dtype != np.int8:
            raise TypeError("activations must have dtype np.int8")

        if activations.ndim == 1:
            is_1d = True
            act_2d = activations.reshape(1, -1)
        elif activations.ndim == 2:
            act_2d = activations
        elif activations.ndim > 2:
            orig_batch_shape = activations.shape[:-1]
            act_2d = activations.reshape(-1, activations.shape[-1])
        else:
            raise ValueError("activations must be at least 1D")

        act_2d = np.ascontiguousarray(act_2d, dtype=np.int8)

        m_out, n_in = weights_i8.shape
        b_batch, n_act = act_2d.shape
        if n_in != n_act:
            raise ValueError(
                "weights and activations inner dimension must match"
            )

        results = np.zeros((b_batch, m_out), dtype=np.int32)
        total_zeros = 0
        total_elements = b_batch * m_out * n_in

        for b in range(b_batch):
            a_vec = act_2d[b]
            act_zeros = int(np.count_nonzero(a_vec == 0))
            for m in range(m_out):
                w_vec = weights_i8[m]
                acc = self._stream_dot_product(
                    w_vec, a_vec, sparse_mode=sparse_mode
                )
                results[b, m] = acc
            total_zeros += act_zeros * m_out

        if self.verbose:
            sparsity = (
                (total_zeros / total_elements) if total_elements else 0.0
            )
            useful = max(total_elements - total_zeros, 1)
            speedup = (
                (total_elements / useful)
                if sparse_mode and total_elements
                else 1.0
            )
            print(
                f"bdh_fpga layer shape=({b_batch}, {m_out}, {n_in}) "
                f"sparse={sparse_mode} zero_sparsity={sparsity:.2%} "
                f"estimated_zero_skip_speedup={speedup:.2f}x",
                flush=True,
            )

        if is_1d:

            return results[0]
        if orig_batch_shape is not None:
            return results.reshape(*orig_batch_shape, m_out)
        return results

    def _write_packet(self, packet: bytes) -> None:
        try:
            written = self.serial.write(packet)
            self.serial.flush()
        except (SerialException, OSError) as exc:
            raise HardwareError("Tang Nano 9K UART write failed!") from exc
        if written != len(packet):
            raise HardwareError("Tang Nano 9K UART write was incomplete!")

    def _read_exactly(self, size: int) -> bytes:
        deadline = time.monotonic() + self.timeout
        chunks = bytearray()
        while len(chunks) < size:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise HardwareError("Tang Nano 9K UART response timed out!")
            try:
                chunk = self.serial.read(size - len(chunks))
            except (SerialException, OSError) as exc:
                raise HardwareError("Tang Nano 9K UART read failed!") from exc
            if chunk:
                chunks.extend(chunk)
        return bytes(chunks)

    @staticmethod
    def _validate_int8_vector(values: np.ndarray, name: str) -> np.ndarray:
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if values.dtype != np.int8:
            raise TypeError(f"{name} must have dtype np.int8")
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1D vector")
        return np.ascontiguousarray(values, dtype=np.int8)

    @staticmethod
    def _validate_int8_matrix(values: np.ndarray, name: str) -> np.ndarray:
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if values.dtype != np.int8:
            raise TypeError(f"{name} must have dtype np.int8")
        if values.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix")
        return np.ascontiguousarray(values, dtype=np.int8)
