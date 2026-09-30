"""
make_yolo_dataset.py

Step 4a: bridge the frozen split (data/splits/split.csv) into the layout
Ultralytics YOLO expects, WITHOUT copying images.

Ultralytics accepts, in data.yaml, a path to a .txt file listing image paths
(one per line) for train/val instead of a folder. We use that: the split stays
driven by split.csv (single source of truth), and re-splitting just regenerates
these lists. No duplicated images on disk.

YOLO finds each image's label by convention: it takes the image path, replaces
'/images/' with '/labels/' and the extension with '.txt'. Our images live in
data/interim/ and labels in data/annotations/ — those folder names don't match
YOLO's images/labels convention, so we point label discovery correctly by
giving absolute image paths and relying on YOLO's path-substitution... which
would fail for our naming. So instead we build a conventional structure using
SYMLINKS: data/yolo/images/{train,val}/ and data/yolo/labels/{train,val}/,
linking back to the real files. Symlinks (not copies) keep it lightweight.

NOTE ON 'val': for this smoke-test we use the TEST split as YOLO's 'val' set
(YOLO needs something to validate against during training; we are not doing a
separate val set this iteration — that was the train/test + CV decision).

Outputs:
  data/yolo/images/train/  -> symlinks to interim images (train)
  data/yolo/images/val/    -> symlinks to interim images (test-as-val)
  data/yolo/labels/train/  -> symlinks to annotation .txt (train)
  data/yolo/labels/val/    -> symlinks to annotation .txt (test-as-val)
  data/yolo/data.yaml      -> the dataset descriptor YOLO reads
"""

from pathlib import Path
import csv
import sys
import os

SPLIT      = Path("data/splits/split.csv")
IMAGES_SRC = Path("data/interim")
LABELS_SRC = Path("data/annotations")
YOLO_ROOT  = Path("data/yolo")

# map our 'test' split to YOLO's 'val' folder (smoke-test; see module docstring)
SPLIT_TO_YOLO = {"train": "train", "test": "val"}


def link(src: Path, dst: Path):
    """Create a symlink dst -> src, replacing any existing link."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    # absolute target so the link works regardless of CWD
    os.symlink(src.resolve(), dst)


def main():
    if not SPLIT.exists():
        sys.exit(f"ERROR: {SPLIT} not found — run make_split.py first.")

    rows = list(csv.DictReader(open(SPLIT)))
    counts = {"train": 0, "val": 0}
    missing_img, missing_lbl = [], []

    for r in rows:
        img_name = r["image"]
        yolo_split = SPLIT_TO_YOLO[r["split"]]
        stem = Path(img_name).stem

        img_src = IMAGES_SRC / img_name
        lbl_src = LABELS_SRC / (stem + ".txt")

        if not img_src.exists():
            missing_img.append(img_name); continue
        if not lbl_src.exists():
            # every image should have a label file (empty for negatives)
            missing_lbl.append(stem + ".txt"); continue

        link(img_src, YOLO_ROOT / "images" / yolo_split / img_name)
        link(lbl_src, YOLO_ROOT / "labels" / yolo_split / (stem + ".txt"))
        counts[yolo_split] += 1

    # write data.yaml
    data_yaml = YOLO_ROOT / "data.yaml"
    data_yaml.write_text(
        f"# Auto-generated from {SPLIT} — do not hand-edit.\n"
        f"path: {YOLO_ROOT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: area_label\n"
    )

    print(f"Linked  train: {counts['train']}  val(=test): {counts['val']}")
    if missing_img:
        print(f"  WARNING: {len(missing_img)} images missing from {IMAGES_SRC}: {missing_img[:5]}")
    if missing_lbl:
        print(f"  WARNING: {len(missing_lbl)} label files missing from {LABELS_SRC}: {missing_lbl[:5]}")
    print(f"Wrote {data_yaml}")
    if not missing_img and not missing_lbl:
        print("All images + labels linked cleanly.")


if __name__ == "__main__":
    main()
