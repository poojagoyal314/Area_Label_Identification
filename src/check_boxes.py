# check_boxes.py — draw derived YOLO boxes back onto their images to verify placement
from pathlib import Path
import matplotlib
matplotlib.use("Agg")            # save to file, no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

IMAGES = Path("data/interim")
LABELS = Path("data/annotations")
OUT    = Path("data/manifests/box_check")   # where the annotated previews land
OUT.mkdir(parents=True, exist_ok=True)

# pick a few images to check — mix of easy and multi-box; edit these names
SAMPLES = ["2.jpg", "1.jpg", "10.jpg"]      # 2.jpg has 2 boxes (good test)

for name in SAMPLES:
    img_path = IMAGES / name
    lbl_path = LABELS / (Path(name).stem + ".txt")
    if not img_path.exists() or not lbl_path.exists():
        print(f"skip {name}: missing image or label"); continue

    img = Image.open(img_path)
    W, H = img.size
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    for line in lbl_path.read_text().strip().splitlines():
        cls, xc, yc, w, h = map(float, line.split())
        # YOLO normalized center -> pixel top-left corner for drawing
        bw, bh = w * W, h * H
        x = xc * W - bw / 2
        y = yc * H - bh / 2
        ax.add_patch(patches.Rectangle((x, y), bw, bh,
                     linewidth=2, edgecolor="red", facecolor="none"))

    ax.axis("off")
    out = OUT / f"check_{Path(name).stem}.png"
    plt.savefig(out, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"wrote {out}")