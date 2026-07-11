# Fast definitive check: can the loss learn to remove stars on a SINGLE frame?
# If pred rises toward the true star flux, the loss is sound. If it goes to 0,
# the loss is still mis-specified.
import sys
sys.path.insert(0, ".")
import torch
from datagen.synth import make_pair
from models.nafnet_star import build
from train.train import loss_fn

dev = "cuda"
inp, sl, st = make_pair(256, 256, None, seed=123, device=dev)
foot = st > 3e-3          # visible stars
m = build(width=32).to(dev).train()
opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
print(f"true star flux on visible-star pixels: {float(st[foot].mean()):.5f}")
for step in range(400):
    pred = m(inp[None])
    loss, _ = loss_fn(pred[0], inp, sl, st)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 80 == 0 or step == 399:
        with torch.no_grad():
            p = m(inp[None])[0]
            pm = float(p[foot].mean())
            frac = float(p[foot].sum() / st[foot].sum())
        print(f"step {step:3d}: pred_on_stars {pm:.5f}  completeness {frac:.3f}  loss {loss.item():.5f}")
print("VERDICT:", "LOSS WORKS (completeness rising)" if frac > 0.5 else "STILL BROKEN")
