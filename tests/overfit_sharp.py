# Fast definitive check for the deconvolution model: can the loss learn to
# SHARPEN a single frame? If output moves from the blurred input toward the
# sharp target (error drops well below the input's error), the loss is sound.
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "datagen")
import torch
import torch.nn.functional as F
from blur import make_pair
from models.nafnet_sharp import build
from train.train_sharp import loss_fn

dev = "cuda"
inp, tgt, cond, kernel, reblur = make_pair(256, 256, None, seed=55, device=dev)
m = build(width=32).to(dev).train()
opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
inp_err = float((inp - tgt).abs().mean())      # error if model does nothing
print(f"input-vs-sharp error (do-nothing baseline): {inp_err:.5f}")
for step in range(400):
    out = m(inp[None], cond[None])
    loss, _ = loss_fn(out, tgt[None], kernel[None], reblur[None])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 80 == 0 or step == 399:
        with torch.no_grad():
            e = float((m(inp[None], cond[None])[0] - tgt).abs().mean())
        print(f"step {step:3d}: out-vs-sharp err {e:.5f}  (lower than {inp_err:.5f} = sharpening)")
with torch.no_grad():
    final = float((m(inp[None], cond[None])[0] - tgt).abs().mean())
print("VERDICT:", "LOSS WORKS (moves toward sharp)" if final < 0.6 * inp_err
      else "SUSPECT (barely improves on do-nothing)")
