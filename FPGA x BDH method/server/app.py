import asyncio
import json
from pathlib import Path
from typing import Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.bdh_inference import BDHInferenceEngine
from server.fpga_bridge import FPGABridge

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"

app = FastAPI(title="Tang Nano 9K BDH Monochrome Chatbot")
bridge = FPGABridge()
inference = BDHInferenceEngine(bridge)
connections: Set[WebSocket] = set()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(status_broadcaster())


@app.on_event("shutdown")
async def shutdown() -> None:
    bridge.close()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


async def status_broadcaster() -> None:
    async for status in bridge.status_stream():
        await broadcast(status.as_payload())


async def broadcast(payload: Dict[str, object]) -> None:
    if not connections:
        return
    message = json.dumps(payload)
    dead: Set[WebSocket] = set()
    for websocket in connections:
        try:
            await websocket.send_text(message)
        except RuntimeError:
            dead.add(websocket)
    connections.difference_update(dead)


@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket) -> None:
    await websocket.accept()
    connections.add(websocket)
    await websocket.send_json(bridge.status.as_payload())
    try:
        while True:
            data = await websocket.receive_json()
            prompt = str(data.get("prompt", "")).strip()
            if not prompt:
                continue
            temperature = float(data.get("temperature", 0.7))
            top_k = int(data.get("top_k", 3))
            top_p = float(data.get("top_p", 0.9))
            offload_mode = str(data.get("offload", "full"))
            hardware_required = bool(data.get("hardware_required", False))
            await websocket.send_json({"type": "inference_start"})
            try:
                async for frame in inference.stream_prompt(
                    prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    offload_mode=offload_mode,
                    hardware_required=hardware_required,
                ):
                    await websocket.send_json(frame)
            except Exception as exc:
                await websocket.send_json(
                    {"type": "error", "message": str(exc)}
                )
            await websocket.send_json({"type": "inference_end"})
    except WebSocketDisconnect:
        connections.discard(websocket)
