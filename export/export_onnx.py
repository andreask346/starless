# Export a trained Starless checkpoint to static-shape ONNX (512x512 tiles,
# opset 17 for the FFT ops; DirectML-friendly) and verify parity vs PyTorch.
#   .venv\Scripts\python.exe export\export_onnx.py runs\w32_v0\best.pt starless_w32.onnx

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.nafnet_star import build  # noqa: E402

TILE = 512


def main(ckpt_path, out_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ck.get("config", {})
    model = build(width=cfg.get("width", 32),
                  use_ffc=not cfg.get("no_ffc", False))
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.rand(1, 3, TILE, TILE)
    torch.onnx.export(
        model, (dummy,), out_path, opset_version=20,
        input_names=["input"], output_names=["stars"],
        dynamo=True, external_data=False)
    print(f"exported {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB, opset 17, "
          f"static {TILE}x{TILE})")

    import onnxruntime as ort
    for providers in (["DmlExecutionProvider", "CPUExecutionProvider"],
                      ["CPUExecutionProvider"]):
        try:
            sess = ort.InferenceSession(out_path, providers=providers)
            break
        except Exception as e:
            print(f"provider {providers[0]} failed: {e}")
    x = torch.rand(1, 3, TILE, TILE)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    err = float(np.abs(ref - got).max())
    print(f"parity vs PyTorch ({sess.get_providers()[0]}): "
          f"max abs err {err:.2e} -> {'OK' if err < 5e-3 else 'MISMATCH'}")
    return 0 if err < 5e-3 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
