"""SAM3 text-prompt segmentation server for Huawei Ascend Atlas A2 (torch_npu).

Same HTTP API as sam3_srv.py (device npu, bf16 autocast); the /seg response additionally carries
server-side timings under "t" (clients ignore it).
  GET  /health -> {"ok": true, "device": ...}
  POST /seg {"image_b64", "prompts", "top_k", "min_score"}
       -> {"results": {prompt: [{"score", "box", "mask"}]}, "t": {"image_encoder_s", "prompts_s", "total_s"}}

SAM3 needs three small source changes to run on NPU (see docs/npu.md, "SAM3 on Ascend").
Run:  source serving/npu/sam3_npu_env.sh
      python sam3_srv_npu.py --port 8134 --ckpt /path/to/sam3.pt [--device npu|cpu]
"""
import argparse
import base64
import io
import os
import sys
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8134)
ap.add_argument("--host", default=os.environ.get("SAM3_HOST", "127.0.0.1"),
                help="bind address; use 0.0.0.0 only on a trusted network")
ap.add_argument("--device", default="npu")
ap.add_argument("--ckpt", default=os.environ.get("SAM3_CKPT"), help="path to sam3.pt (default: $SAM3_CKPT)")
A = ap.parse_args()
if not A.ckpt or not os.path.isfile(A.ckpt):
    sys.exit(f"sam3.pt not found ({A.ckpt!r}); pass --ckpt or set SAM3_CKPT")
if A.device == "npu":
    import torch_npu  # noqa: F401
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

model = build_sam3_image_model(checkpoint_path=A.ckpt, load_from_HF=False, device=A.device).to(A.device)
PROC = Sam3Processor(model, device=A.device, confidence_threshold=0.0)
LOCK = threading.Lock()
app = FastAPI()


def sync():
    if A.device == "npu":
        torch.npu.synchronize()


@app.get("/health")
def health():
    return {"ok": True, "device": A.device}


@app.post("/seg")
def seg(req: dict):
    t0 = time.perf_counter()
    im = Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGB")
    k, ms = int(req.get("top_k", 3)), float(req.get("min_score", 0.05))
    out, tt = {}, {}
    with LOCK, torch.autocast(A.device, dtype=torch.bfloat16, enabled=A.device != "cpu"):
        t1 = time.perf_counter()
        st = PROC.set_image(im)
        sync(); tt["image_encoder_s"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        for p in req["prompts"]:
            PROC.reset_all_prompts(st)
            o = PROC.set_text_prompt(state=st, prompt=p)
            sc = o["scores"].float().cpu().numpy() if o.get("scores") is not None else np.zeros(0)
            res = []
            for i in np.argsort(-sc)[:k]:
                if sc[i] < ms:
                    break
                m = o["masks"][i].float().cpu().numpy().reshape(im.size[1], im.size[0]) > 0.5
                b = io.BytesIO()
                Image.fromarray((m * 255).astype(np.uint8)).save(b, format="PNG")
                res.append({"score": float(sc[i]), "box": [float(x) for x in o["boxes"][i].float().cpu().numpy()],
                            "mask": base64.b64encode(b.getvalue()).decode()})
            out[p] = res
        tt["prompts_s"] = time.perf_counter() - t2
    tt["total_s"] = time.perf_counter() - t0
    return {"results": out, "t": tt}


uvicorn.run(app, host=A.host, port=A.port, log_level="warning")
