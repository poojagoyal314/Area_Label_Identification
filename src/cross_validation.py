"""
cross_validate.py

Stratified k-fold cross-validation for the area-label detector.

PURPOSE: CV is a measuring stick, not a training improvement. It gives a robust
estimate (mean +/- std over k folds) so that when we compare configs (e.g. 960
vs 1280 resolution) we can tell a real gain from fold-luck noise — essential
here because gains are small and the single test set is tiny.

USES ALL 260 IMAGES. No train/test or video/phone slicing — every image rotates
through being validation once. Modality is NOT sliced for reporting, but folds
ARE stratified on (presence x modality) so each fold has the same mix; otherwise
per-fold scores would swing on modality composition and the std would be
meaningless.

KEY CORRECTNESS POINTS:
  - FRESH model from the pretrained base every fold (no weight carry-over, or
    val data leaks into later folds).
  - Fold assignment is SEEDED and saved, so k=5@960 and k=5@1280 use IDENTICAL
    folds -> a clean apples-to-apples comparison.
  - Reduced epochs (50, patience 15): the 100-epoch run plateaued by ~epoch 30,
    so this converges without wasting compute across k runs.

Config via env vars (defaults in brackets):
  K [5], IMG_SIZE [960], EPOCHS [50], PATIENCE [15], BASE_MODEL [yolo11n.pt],
  CV_SEED [42], MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME [area_label_detector]

Run (Colab, from repo root):
  os.environ["MLFLOW_TRACKING_URI"]="sqlite:////content/drive/MyDrive/area_label_mlflow/mlflow.db"
  os.environ["K"]="3"; os.environ["IMG_SIZE"]="960"
  !python src/cross_validate.py
"""

from pathlib import Path
from collections import defaultdict, Counter
import csv
import os
import random
import statistics
import sys

MANIFEST    = Path("data/manifests/conversion_manifest.csv")
ATTRIBUTES  = Path("data/manifests/attributes.csv")
INTERIM     = Path("data/interim")
ANNOTATIONS = Path("data/annotations")
CV_ROOT     = Path("data/cv")

SEED       = int(os.environ.get("CV_SEED", "42"))
K          = int(os.environ.get("K", "5"))
IMG_SIZE   = int(os.environ.get("IMG_SIZE", "960"))
EPOCHS     = int(os.environ.get("EPOCHS", "50"))
PATIENCE   = int(os.environ.get("PATIENCE", "15"))
BASE_MODEL = os.environ.get("BASE_MODEL", "yolo11n.pt")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "area_label_detector")
CV_TAG     = os.environ.get("CV_TAG", f"cv{K}_{Path(BASE_MODEL).stem}_{IMG_SIZE}")
MODALITY_THRESHOLD = 1000


def modality(w, h):
    return "video_frame" if max(int(w), int(h)) < MODALITY_THRESHOLD else "phone_photo"


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    if not MANIFEST.exists() or not ATTRIBUTES.exists():
        sys.exit("ERROR: need conversion_manifest.csv and attributes.csv in data/manifests/")

    man = {r["saved_name"]: r for r in csv.DictReader(open(MANIFEST))}
    pos = set(r["image"] for r in csv.DictReader(open(ATTRIBUTES)))
    all_images = sorted(man.keys())

    presence = {i: ("positive" if i in pos else "negative") for i in all_images}
    mod      = {i: modality(man[i]["saved_w"], man[i]["saved_h"]) for i in all_images}

    # --- stratified fold assignment: round-robin within each stratum --------
    strata = defaultdict(list)
    for i in all_images:
        strata[(presence[i], mod[i])].append(i)
    rng = random.Random(SEED)
    fold_of = {}
    for key, imgs in strata.items():
        imgs = sorted(imgs)
        rng.shuffle(imgs)
        for idx, img in enumerate(imgs):
            fold_of[img] = idx % K

    CV_ROOT.mkdir(parents=True, exist_ok=True)
    folds_csv = CV_ROOT / f"folds_k{K}_seed{SEED}.csv"
    with open(folds_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "fold", "presence", "modality"])
        for i in all_images:
            w.writerow([i, fold_of[i], presence[i], mod[i]])

    print(f"Fold assignment (k={K}, seed={SEED}) -> {folds_csv}")
    for fi in range(K):
        imgs = [i for i in all_images if fold_of[i] == fi]
        pc = Counter((presence[i], mod[i]) for i in imgs)
        print(f"  fold {fi}: {len(imgs)} imgs  {dict(pc)}")

    # --- build per-fold symlink folders + data.yaml -------------------------
    fold_yamls = []
    for fi in range(K):
        for subset in ("train", "val"):
            want_in_fold = (subset == "val")
            for img in all_images:
                in_fold = (fold_of[img] == fi)
                if in_fold != want_in_fold:
                    continue
                stem = Path(img).stem
                link(INTERIM / img, CV_ROOT / f"fold{fi}" / "images" / subset / img)
                link(ANNOTATIONS / f"{stem}.txt", CV_ROOT / f"fold{fi}" / "labels" / subset / f"{stem}.txt")
        yaml_path = CV_ROOT / f"fold{fi}" / "data.yaml"
        yaml_path.write_text(
            f"path: {(CV_ROOT / f'fold{fi}').resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n  0: area_label\n"
        )
        fold_yamls.append(yaml_path)

    # --- train each fold FRESH, validate best, collect ----------------------
    from ultralytics import YOLO, settings
    settings.update({"mlflow": True})
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU — CV on CPU will be very slow.")

    results_csv = CV_ROOT / f"cv_results_{CV_TAG}.csv"
    rows = []
    for fi, yaml_path in enumerate(fold_yamls):
        run_name = f"{CV_TAG}_fold{fi}"
        os.environ["MLFLOW_RUN"] = run_name
        print(f"\n--- Fold {fi+1}/{K}  ({run_name}) ---")

        model = YOLO(BASE_MODEL)          # FRESH each fold — no leakage
        model.train(data=str(yaml_path), epochs=EPOCHS, imgsz=IMG_SIZE,
                    patience=PATIENCE, device=device, project=EXPERIMENT,
                    name=run_name, exist_ok=True, verbose=False, plots=False)

        best = model.trainer.best          # robust path to this fold's best.pt
        m = YOLO(str(best)).val(data=str(yaml_path), imgsz=IMG_SIZE, verbose=False)
        row = {"fold": fi,
               "mAP50":    float(m.box.map50),
               "mAP50_95": float(m.box.map),
               "P":        float(m.box.mp),
               "R":        float(m.box.mr)}
        rows.append(row)
        # incremental write — survive a Colab disconnect mid-CV
        with open(results_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"fold {fi}: mAP50={row['mAP50']:.3f} mAP50-95={row['mAP50_95']:.3f} "
              f"P={row['P']:.3f} R={row['R']:.3f}")

    # --- aggregate ----------------------------------------------------------
    def agg(key):
        vals = [r[key] for r in rows]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print("\n" + "=" * 52)
    print(f"CV SUMMARY  ({CV_TAG}, k={K}, imgsz={IMG_SIZE})")
    print("-" * 52)
    for key in ("mAP50", "mAP50_95", "P", "R"):
        mean, std = agg(key)
        print(f"  {key:10s} {mean:.3f} ± {std:.3f}")
    print("=" * 52)

    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{CV_TAG}_SUMMARY"):
        for p, v in (("k", K), ("imgsz", IMG_SIZE), ("base_model", BASE_MODEL),
                     ("epochs", EPOCHS), ("seed", SEED)):
            mlflow.log_param(p, v)
        for key in ("mAP50", "mAP50_95", "P", "R"):
            mean, std = agg(key)
            mlflow.log_metric(f"{key}_mean", mean)
            mlflow.log_metric(f"{key}_std", std)
    print(f"\nLogged CV summary to MLflow. Per-fold results: {results_csv}")


if __name__ == "__main__":
    main()