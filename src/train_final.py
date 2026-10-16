"""
train_final.py

Train the FINAL deployable model on ALL 260 images with the config chosen by
cross-validation: yolo11n @ 960px.

WHY THIS IS SEPARATE FROM cross_validate.py:
  CV's job was to CHOOSE a config and ESTIMATE its performance (mAP50
  0.939 +/- 0.052). Its per-fold models were means to that end and are discarded.
  This script PRODUCES the single model you actually deploy, trained on every
  available image (more data than any CV fold saw, so it should be at least as
  good as the CV estimate).

APPROACH: fixed epoch budget, no held-out validation.
  You chose to use all 260 images for training. YOLO still requires a `val:`
  entry, so we point it at the SAME images (train == val). CONSEQUENCE: the val
  metrics YOLO prints during this run are measured on the training data and are
  therefore OPTIMISTICALLY INFLATED. They are NOT a performance estimate. The
  honest estimate is the cross-validation number; this run only makes the
  weights. Early stopping is effectively disabled (patience >= epochs) so the
  full fixed budget runs.

  Epoch budget defaults to 50, matching the CV training regime so the CV
  estimate faithfully predicts this model.

Outputs:
  - best.pt / last.pt under the run dir
  - BOTH copied to Drive (ephemeral Colab disk would lose them otherwise)
  - logged to MLflow as run 'final_yolo11n_960_all260'

Run (Colab, from repo root):
  os.environ["MLFLOW_TRACKING_URI"]="sqlite:////content/drive/MyDrive/area_label_mlflow/mlflow.db"
  !python src/train_final.py
"""

from pathlib import Path
import csv
import os
import shutil
import sys

MANIFEST    = Path("data/manifests/conversion_manifest.csv")
ANNOTATIONS = Path("data/annotations")
INTERIM     = Path("data/interim")
FINAL_ROOT  = Path("data/final")

IMG_SIZE     = int(os.environ.get("IMG_SIZE", "960"))
FINAL_EPOCHS = int(os.environ.get("FINAL_EPOCHS", "50"))
BASE_MODEL   = os.environ.get("BASE_MODEL", "yolo11n.pt")
EXPERIMENT   = os.environ.get("MLFLOW_EXPERIMENT_NAME", "area_label_detector")
RUN_NAME     = os.environ.get("RUN_NAME", "final_yolo11n_960_all260")
# where to durably save the final weights (Colab local disk is ephemeral)
DRIVE_OUT    = os.environ.get("FINAL_WEIGHTS_DIR",
                              "/content/drive/MyDrive/area_label_final")


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    if not MANIFEST.exists():
        sys.exit("ERROR: conversion_manifest.csv not found in data/manifests/")

    all_images = [r["saved_name"] for r in csv.DictReader(open(MANIFEST))]
    print(f"Training FINAL model on ALL {len(all_images)} images "
          f"(config: {BASE_MODEL} @ {IMG_SIZE}px, {FINAL_EPOCHS} epochs).")

    # build one folder holding every image + its label; val points here too
    for img in all_images:
        stem = Path(img).stem
        link(INTERIM / img, FINAL_ROOT / "images" / "all" / img)
        link(ANNOTATIONS / f"{stem}.txt", FINAL_ROOT / "labels" / "all" / f"{stem}.txt")

    data_yaml = FINAL_ROOT / "data.yaml"
    data_yaml.write_text(
        f"path: {FINAL_ROOT.resolve()}\n"
        f"train: images/all\n"
        f"val: images/all\n"          # train == val: satisfies YOLO; metrics inflated
        f"names:\n  0: area_label\n"
    )

    from ultralytics import YOLO, settings
    settings.update({"mlflow": True})
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    os.environ["MLFLOW_RUN"] = RUN_NAME
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU — final training on CPU will be slow.")

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=FINAL_EPOCHS,
        imgsz=IMG_SIZE,
        patience=FINAL_EPOCHS + 1,   # effectively disable early stopping
        device=device,
        project=EXPERIMENT,
        name=RUN_NAME,
        exist_ok=True,
        plots=False,
    )

    # --- durably save the final weights to Drive --------------------------
    best = Path(model.trainer.best)   # best training-fit epoch
    last = Path(model.trainer.last)   # end of the fixed budget
    out = Path(DRIVE_OUT)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for src in (best, last):
        if src.exists():
            dst = out / f"{RUN_NAME}_{src.name}"   # e.g. final_..._best.pt
            shutil.copy2(src, dst)
            saved.append(str(dst))

    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINED.")
    print("  NOTE: any val metrics above are measured on TRAINING data and are")
    print("  INFLATED. The honest estimate is the cross-validation number")
    print("  (mAP50 0.939 +/- 0.052). This run only produced the weights.")
    print("\n  Deployable weights saved to Drive:")
    for s in saved:
        print(f"    {s}")
    print("  Recommended deliverable: the *_last.pt (end of the fixed budget).")
    print("=" * 60)


if __name__ == "__main__":
    main()
