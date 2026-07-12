import glob, numpy as np, tifffile
from PIL import Image
p = sorted(glob.glob("runs/star_w32/panel/panel_*.tif"))
print("real-frame panels:", len(p))
if p:
    a = tifffile.imread(p[0]).astype(np.float32)
    a = a / a.max() if a.max() > 1.5 else a
    a = np.clip(a, 0, 1) ** 0.5
    Image.fromarray((a * 255).astype("uint8")).save("runs/_real_starless.jpg", quality=88)
    print("rendered", p[0], a.shape)
