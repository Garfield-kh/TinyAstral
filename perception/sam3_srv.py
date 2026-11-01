"""Minimal SAM3 text-prompt segmentation server (CUDA build).

Weights: a local sam3.pt (download from the gated Hugging Face repo facebook/sam3, SAM License),
passed with --ckpt or $SAM3_CKPT; loaded offline (load_from_HF=False).

API
  GET  /health -> {"ok": true}
  POST /seg {"image_b64": png/jpg base64, "prompts": [str, ...], "top_k": 3, "min_score": 0.05}
       -> {"results": {prompt: [{"score": float, "box": [x0, y0, x1, y1], "mask": png base64}, ...]}}
  Masks are returned best-first, at most top_k per prompt, stopping at the first score < min_score.

Run:  python sam3_srv.py --port 8130 --ckpt /path/to/sam3.pt      (Huawei Ascend Atlas A2: use sam3_srv_npu.py)
"""
import argparse, base64, io, os, sys, threading
import numpy as np, torch
from PIL import Image
from fastapi import FastAPI
import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8130)
ap.add_argument("--host", default=os.environ.get("SAM3_HOST", "127.0.0.1"),
                help="bind address; use 0.0.0.0 only on a trusted network")
ap.add_argument("--ckpt", default=os.environ.get("SAM3_CKPT"), help="path to sam3.pt (default: $SAM3_CKPT)")
A = ap.parse_args()
if not A.ckpt or not os.path.isfile(A.ckpt):
    sys.exit(f"sam3.pt not found ({A.ckpt!r}); pass --ckpt or set SAM3_CKPT")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
torch.backends.cuda.matmul.allow_tf32 = True
model = build_sam3_image_model(checkpoint_path=A.ckpt, load_from_HF=False).to("cuda")
PROC = Sam3Processor(model, confidence_threshold=0.0)
LOCK = threading.Lock()
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/seg")
def seg(req: dict):
    im = Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGB")
    k, ms = int(req.get("top_k", 3)), float(req.get("min_score", 0.05))
    out = {}
    with LOCK, torch.autocast("cuda", dtype=torch.bfloat16):
        st = PROC.set_image(im)
        for p in req["prompts"]:
            PROC.reset_all_prompts(st)
            o = PROC.set_text_prompt(state=st, prompt=p)
            sc = o["scores"].float().cpu().numpy() if o.get("scores") is not None else np.zeros(0)
            res = []
            for i in np.argsort(-sc)[:k]:
                if sc[i] < ms:
                    break
                m = o["masks"][i].float().cpu().numpy().reshape(im.size[1], im.size[0]) > 0.5
                b = io.BytesIO(); Image.fromarray((m * 255).astype(np.uint8)).save(b, format="PNG")
                res.append({"score": float(sc[i]), "box": [float(x) for x in o["boxes"][i].float().cpu().numpy()],
                            "mask": base64.b64encode(b.getvalue()).decode()})
            out[p] = res
    return {"results": out}


uvicorn.run(app, host=A.host, port=A.port, log_level="warning")
