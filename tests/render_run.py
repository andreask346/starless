import sys, glob, numpy as np, tifffile
from PIL import Image
run = sys.argv[1]
out = sys.argv[2]
p = sorted(glob.glob(f"runs/{run}/panel/panel_*.tif"))
print("panels:", len(p))
if p:
    a = tifffile.imread(p[0]).astype(np.float32)
    a = a / a.max() if a.max() > 1.5 else a
    a = np.clip(a, 0, 1) ** 0.5
    Image.fromarray((a * 255).astype("uint8")).save(out, quality=88)
    print("rendered", p[0], a.shape)
