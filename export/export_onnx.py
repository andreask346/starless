# Export a trained checkpoint to static-shape ONNX (512x512 tiles, opset 20
# for the FFT ops; DirectML-friendly) and verify parity vs PyTorch.
# Star removal (1 input):    export_onnx.py runs\w32\best.pt out.onnx
# Deconvolution (2 inputs):  export_onnx.py runs\sharp\best.pt out.onnx --sharp

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TILE = 512


def main(ckpt_path, out_path, sharp=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ck.get("config", {})
    if sharp:
        from models.nafnet_sharp import build
        model = build(width=cfg.get("width", 32),
                      use_ffc=not cfg.get("no_ffc", False))
    else:
        from models.nafnet_star import build
        model = build(width=cfg.get("width", 32),
                      use_ffc=not cfg.get("no_ffc", False))
    model.load_state_dict(ck["model"])
    model.eval()

    if sharp:
        from datagen.blur import COND_DIM
        inputs = (torch.rand(1, 3, TILE, TILE), torch.rand(1, COND_DIM))
        in_names, out_names = ["input", "cond"], ["sharp"]
    else:
        inputs = (torch.rand(1, 3, TILE, TILE),)
        in_names, out_names = ["input"], ["stars"]
    torch.onnx.export(model, inputs, out_path, opset_version=20,
                      input_names=in_names, output_names=out_names,
                      dynamo=True, external_data=False)
    print(f"exported {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB, opset 20, "
          f"static {TILE}x{TILE}, {'deconv' if sharp else 'star'})")

    import onnxruntime as ort
    sess = None
    for providers in (["DmlExecutionProvider", "CPUExecutionProvider"],
                      ["CPUExecutionProvider"]):
        try:
            sess = ort.InferenceSession(out_path, providers=providers)
            break
        except Exception as e:
            print(f"provider {providers[0]} failed: {e}")
    if sess is None:
        return 1
    test = tuple(torch.rand_like(t) for t in inputs)
    with torch.no_grad():
        ref = model(*test).numpy()
    feed = {n: t.numpy() for n, t in zip(in_names, test)}
    got = sess.run(None, feed)[0]
    err = float(np.abs(ref - got).max())
    print(f"parity vs PyTorch ({sess.get_providers()[0]}): "
          f"max abs err {err:.2e} -> {'OK' if err < 5e-3 else 'MISMATCH'}")
    return 0 if err < 5e-3 else 1


if __name__ == "__main__":
    sharp = "--sharp" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(pos[0], pos[1], sharp=sharp))
