import sys, os
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from datagen.synth import make_pair
from models.nafnet_star import build

dev = "cuda"
ck = torch.load(r"runs/losstest/best.pt", map_location=dev, weights_only=True)
m = build(width=32).to(dev)
m.load_state_dict(ck["model"])
m.eval()

inp, sl, st = make_pair(256, 256, None, seed=10_000_000, device=dev)
with torch.no_grad():
    pred = m(inp[None])[0]          # predicted stars

# --- what does the TARGET look like? ---
print("=== target star layer (tgt_stars) stats ===")
print(f"  max {st.max():.4f}  mean {st.mean():.6f}")
for thr in (1e-4, 1e-3, 1e-2):
    frac = float((st > thr).float().mean())
    print(f"  fraction of pixels with tgt_stars > {thr:g}: {frac*100:.2f}%")

# --- what does the MODEL predict? ---
print("=== model prediction (pred stars) stats ===")
print(f"  max {pred.max():.4f}  mean {pred.mean():.6f}")
print(f"  pred flux total / true flux total: {float(pred.sum()/st.sum()):.4f}")

# --- is it S=0? ---
print("=== is starless == input (S=0 collapse)? ===")
starless = inp - pred
print(f"  mean|pred| = {float(pred.abs().mean()):.6e}  (0 => collapse)")
print(f"  mean|starless - input| = {float((starless-inp).abs().mean()):.6e}")

# --- PSNR breakdown ---
foot = st > 1e-4
mse_model = float(((starless-sl)[foot]**2).mean())
mse_s0 = float(((inp-sl)[foot]**2).mean())            # S=0 baseline
import math
print("=== in-footprint PSNR ===")
print(f"  model:     {-10*math.log10(max(mse_model,1e-12)):.2f} dB")
print(f"  S=0 basel: {-10*math.log10(max(mse_s0,1e-12)):.2f} dB")

# --- gradient sanity: does loss push pred UP on star pixels at S=0? ---
print("=== gradient check: fresh model, one step, does pred move? ===")
m2 = build(width=32).to(dev); m2.train()
opt = torch.optim.AdamW(m2.parameters(), lr=3e-4)
w = 1.0 + 60.0*(st[None] > 1e-4).float()
before = None
for step in range(60):
    p = m2(inp[None])
    slp = inp[None]-p
    l = ((torch.sqrt((slp-sl[None])**2+1e-6)*w).sum()/w.sum()
         + (torch.sqrt((p-st[None])**2+1e-6)*w).sum()/w.sum())
    opt.zero_grad(); l.backward(); opt.step()
    if step==0: before=float(p[0][foot].mean())
after = float(m2(inp[None])[0][foot].mean())
print(f"  pred mean on star pixels: {before:.5f} -> {after:.5f} after 60 steps "
      f"(true {float(st[foot].mean()):.5f})")
print("  (if it rises toward true, the loss CAN learn on a single frame)")
