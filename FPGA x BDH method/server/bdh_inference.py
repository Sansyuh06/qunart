import asyncio
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Dict, Iterator, Optional, cast


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from bdh_tiny_llm import (  # noqa: E402
    DotProductEngine,
    TinyBDHLanguageModel,
    TinyTokenFrame,
)
from server.fpga_bridge import FPGABridge  # noqa: E402


class BDHInferenceEngine:
    def __init__(self, bridge: FPGABridge, max_tokens: int = 150) -> None:
        self.bridge = bridge
        self.max_tokens = max_tokens
        self.model = TinyBDHLanguageModel(embed_dim=128)

    async def stream_prompt(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 3,
        top_p: float = 0.9,
        offload_mode: str = "full",
        hardware_required: bool = False,
    ) -> AsyncIterator[Dict[str, object]]:
        engine: Optional[DotProductEngine] = (
            cast(DotProductEngine, self.bridge.engine)
            if self.bridge and self.bridge.status.connected
            else None
        )
        frames = self.model.iter_generate(
            prompt,
            engine=engine,
            max_tokens=self.max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            offload_mode=offload_mode,
            hardware_required=hardware_required,
        )

        for index in range(self.max_tokens):
            started = time.perf_counter()
            frame = await asyncio.to_thread(self._next_frame, frames)
            if frame is None:
                break
            yield self._payload(index, frame, started)
            await asyncio.sleep(0.02)

    @staticmethod
    def _next_frame(
        frames: Iterator[TinyTokenFrame],
    ) -> Optional[TinyTokenFrame]:
        try:
            return next(frames)
        except StopIteration:
            return None

    def _payload(
        self, token_index: int, frame: TinyTokenFrame, started: float
    ) -> Dict[str, object]:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "type": "token_frame",
            "token": frame.token,
            "token_index": token_index,
            "execution_mode": frame.execution_mode,
            "latency_ms": round(latency_ms, 3),
            "zero_sparsity_pct": frame.zero_sparsity_pct,
            "active_pct": frame.active_pct,
            "fpga_speedup": frame.fpga_speedup,
            "dsp_ops": frame.dsp_ops,
            "accumulator": frame.accumulator,
            "uart_bytes_sent": frame.uart_bytes_sent,
            "graph_nodes": frame.graph_nodes,
            "graph_edges": frame.graph_edges,
            "synaptic_matrix_16x16": frame.synaptic_matrix_16x16,
            "model": f"RealBDHLanguageModel-128d [{frame.execution_mode}]",
        }
