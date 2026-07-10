# Tiled full-image inference with Hann-feathered overlap blending.
# Used by the eval panel and (via ONNX) by the Siril script.

import numpy as np
import torch


def hann2d(size):
    w = np.hanning(size)
    return np.outer(w, w).astype(np.float32) + 1e-4


def infer_image(model, img, tile=512, overlap=128, device="cuda"):
    """img: (3,H,W) float32 0..1 numpy -> stars layer (3,H,W)."""
    c, h, w = img.shape
    step = tile - overlap
    pad_h = (step - (h - tile) % step) % step if h > tile else tile - h
    pad_w = (step - (w - tile) % step) % step if w > tile else tile - w
    padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape[1:]
    out = np.zeros_like(padded)
    weight = np.zeros((ph, pw), dtype=np.float32)
    win = hann2d(tile)
    model = model.to(device).eval()
    with torch.no_grad():
        for y in range(0, ph - tile + 1, step):
            for x in range(0, pw - tile + 1, step):
                t = torch.from_numpy(padded[:, y:y + tile, x:x + tile])[None]
                pred = model(t.to(device))[0].cpu().numpy()
                out[:, y:y + tile, x:x + tile] += pred * win
                weight[y:y + tile, x:x + tile] += win
    out /= weight[None]
    stars = out[:, :h, :w]
    return np.clip(np.minimum(stars, np.clip(img, 0, None)), 0, None)
