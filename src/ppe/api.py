"""HTTP API (Starlette; the same handlers port 1:1 to FastAPI, which was not installable where this was built).

    PPE_MODEL=runs/best.onnx uvicorn ppe.api:app --port 8000
    curl -F image=@site.jpg localhost:8000/v1/detect
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .compliance import evaluate, load_policy
from .dataset import load_spec

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 15 * 1024 * 1024


def create_app(model_path: str | None = None, detector=None) -> Starlette:
    spec = load_spec(os.environ.get("PPE_DATA_YAML", ROOT / "config" / "dataset.yaml"))
    policy = load_policy(os.environ.get("PPE_POLICY", ROOT / "config" / "site_policy.yaml"))
    state = {"detector": detector}
    model_path = model_path or os.environ.get("PPE_MODEL")
    if state["detector"] is None and model_path and Path(model_path).exists():
        from .inference import OnnxDetector

        state["detector"] = OnnxDetector(model_path, spec.names, conf=float(os.environ.get("PPE_CONF", "0.25")))

    async def health(request: Request):
        return JSONResponse({"status": "ok", "model_loaded": state["detector"] is not None,
                             "required_ppe": list(policy.required)})

    async def detect(request: Request):
        import cv2

        if state["detector"] is None:
            return JSONResponse({"error": "no model loaded; set PPE_MODEL to a YOLOv8 ONNX export"}, status_code=503)
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "send the image as multipart field 'image'"}, status_code=422)
        data = await upload.read()
        if len(data) > MAX_BYTES:
            return JSONResponse({"error": "image larger than 15 MB"}, status_code=413)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "not a decodable image"}, status_code=422)
        dets = state["detector"](img)
        verdict = evaluate(dets, policy)
        return JSONResponse({"width": img.shape[1], "height": img.shape[0],
                             "detections": [d.to_dict() for d in dets], **verdict.to_dict()})

    return Starlette(routes=[Route("/health", health), Route("/v1/detect", detect, methods=["POST"])])


app = create_app()
