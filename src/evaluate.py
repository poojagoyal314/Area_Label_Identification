"""
evaluate.py

Step 5: SLICED evaluation of a trained model. The training run reports a BLENDED
(phone + video) metric, which is dominated by the easier phone photos and does
NOT tell us how the model does on video — the deployment modality. This script
computes the video-only and phone-only numbers SEPARATELY, so the honest
question ("does it work on video?") gets an honest answer.

How it works:
  - Ultralytics model.val() evaluates against whatever a data.yaml's `val:`
    points at. We build two temporary data.yamls — one whose val list is only
    the video-frame test images, one only the phone-photo test images — and run
    val() against each. Same weights, two filtered test sets.
  - Metrics (mAP50, mAP50-95, precision, recall) are logged to MLflow under the
    SAME experiment, as a separate run tagged 'sliced_eval', referencing which
    weights + split were used.

Reads:
  - a trained weights file (best.pt)                         [--weights]
  - data/splits/split.csv  (image, split, presence, modality)
  - data/yolo/images/val/  (the symlinked test images, already built)
Writes:
  - data/yolo/val_video.txt, data/yolo/val_phone.txt  (filtered image lists)
  - data/yolo/data_video.yaml, data/yolo/data_phone.yaml
  - metrics -> MLflow (sqlite backend, same as training)
  - a short printed summary table

Usage (Colab, from repo root):
  os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////content/drive/MyDrive/area_label_mlflow/mlflow.db"
  !python src/evaluate.py --weights runs/detect/area_label_detector/nano_100ep_blended/weights/best.pt
"""

from pathlib import Path
import argparse
import csv
import os
import sys


SPLIT       = Path("data/splits/split.csv")
YOLO_ROOT   = Path("data/yolo")
IMAGES_VAL  = YOLO_ROOT / "images" / "val"
IMG_SIZE    = int(os.environ.get("IMG_SIZE", "960"))   # match training
EXPERIMENT  = os.environ.get("MLFLOW_EXPERIMENT_NAME", "area_label_detector")


def build_filtered_yaml(modality: str, image_names: list[str]) -> Path:
    """
    Build parallel images/ + labels/ folders for just this modality's test
    images, then a data.yaml pointing val at them.

    We build real sibling folders (images/val_<modality>/ and
    labels/val_<modality>/) rather than a list file, because YOLO finds each
    label by swapping '/images/' -> '/labels/' in the image path. A list of
    resolved symlink targets (e.g. data/interim/11.jpg) has no '/images/' to
    swap, so labels aren't found. Physical sibling folders make the swap work.
    """
    img_dir = YOLO_ROOT / "images" / f"val_{modality}"
    lbl_dir = YOLO_ROOT / "labels" / f"val_{modality}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for name in image_names:
        stem = Path(name).stem
        src_img = (IMAGES_VAL / name)                     # existing symlink
        src_lbl = (YOLO_ROOT / "labels" / "val" / f"{stem}.txt")
        dst_img = img_dir / name
        dst_lbl = lbl_dir / f"{stem}.txt"
        # point new symlinks at the SAME real targets the val/ symlinks use
        for dst, src in ((dst_img, src_img), (dst_lbl, src_lbl)):
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            # resolve() here is fine: we link to the real file, but the link
            # LIVES under images/<...> so YOLO's path-swap still works.
            os.symlink(src.resolve(), dst)

    yaml_path = YOLO_ROOT / f"data_{modality}.yaml"
    yaml_path.write_text(
        f"# Auto-generated filtered val set: {modality} test images only.\n"
        f"path: {YOLO_ROOT.resolve()}\n"
        f"train: images/train\n"          # unused by val, but val() wants the key present
        f"val: images/val_{modality}\n"
        f"names:\n"
        f"  0: area_label\n"
    )
    return yaml_path


def run_val(model, yaml_path: Path, tag: str):
    """Run model.val() against a filtered yaml; return the key metrics."""
    print(f"\n=== Evaluating: {tag} ({yaml_path.name}) ===")
    metrics = model.val(data=str(yaml_path), imgsz=IMG_SIZE, verbose=False)
    # Ultralytics returns a metrics object; pull the box metrics.
    box = metrics.box
    out = {
        "mAP50":    float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),   # mean precision
        "recall":    float(box.mr),   # mean recall
    }
    print(f"  mAP50={out['mAP50']:.3f}  mAP50-95={out['mAP50_95']:.3f}  "
          f"P={out['precision']:.3f}  R={out['recall']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"ERROR: weights not found at {args.weights}")
    if not SPLIT.exists():
        sys.exit(f"ERROR: {SPLIT} not found — run make_split.py first.")

    # read split, select the TEST images by modality
    rows = list(csv.DictReader(open(SPLIT)))
    test = [r for r in rows if r["split"] == "test"]
    video = [r["image"] for r in test if r["modality"] == "video_frame"]
    phone = [r["image"] for r in test if r["modality"] == "phone_photo"]

    # presence counts, so the false-positive signal's thinness is explicit
    def pres_counts(imgs):
        s = set(imgs)
        pos = sum(1 for r in test if r["image"] in s and r["presence"] == "positive")
        neg = sum(1 for r in test if r["image"] in s and r["presence"] == "negative")
        return pos, neg

    vp, vn = pres_counts(video)
    pp, pn = pres_counts(phone)
    print(f"Video test:  {len(video)} images ({vp} positive, {vn} negative)")
    print(f"Phone test:  {len(phone)} images ({pp} positive, {pn} negative)")
    print("NOTE: metrics on this few images are DIRECTIONAL, not precise — "
          "especially false-positive behaviour on the negatives.")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    results = {}
    if video:
        results["video"] = run_val(model, build_filtered_yaml("video", video), "VIDEO (deployment modality)")
    if phone:
        results["phone"] = run_val(model, build_filtered_yaml("phone", phone), "PHONE (out-of-modality)")

    # summary table
    print("\n" + "=" * 56)
    print(f"{'slice':<10}{'mAP50':>10}{'mAP50-95':>11}{'P':>9}{'R':>9}")
    print("-" * 56)
    for k, v in results.items():
        print(f"{k:<10}{v['mAP50']:>10.3f}{v['mAP50_95']:>11.3f}{v['precision']:>9.3f}{v['recall']:>9.3f}")
    print("=" * 56)
    if "video" in results and "phone" in results:
        gap = results["phone"]["mAP50"] - results["video"]["mAP50"]
        print(f"\nphone - video mAP50 gap: {gap:+.3f}  "
              f"({'phone easier, as expected' if gap > 0 else 'video not worse — notable'})")

    # log to MLflow as a separate sliced-eval run
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="sliced_eval"):
        mlflow.log_param("weights", args.weights)
        mlflow.log_param("split_file", str(SPLIT))
        mlflow.log_param("imgsz", IMG_SIZE)
        mlflow.set_tag("note", "video vs phone sliced evaluation")
        for slc, v in results.items():
            for metric, val in v.items():
                mlflow.log_metric(f"{slc}_{metric}", val)
        mlflow.log_metric("video_test_images", len(video))
        mlflow.log_metric("video_test_negatives", vn)
    print("\nLogged sliced metrics to MLflow.")


if __name__ == "__main__":
    main()
