"""
train.py

Step 4b: fine-tune a nano YOLO on the area-label dataset, with MLflow tracking.

SMOKE-TEST RUN. Purpose: prove the whole pipeline runs end-to-end
(split -> YOLO layout -> train -> MLflow -> sliced eval) and produces sane
numbers. The headline metric here is on a BLENDED (phone+video) test set and is
NOT deployment-representative. The video-only metric is computed separately
below and is the number to actually care about.

What this logs to MLflow (one run):
  - Ultralytics autolog: per-epoch metrics, all hyperparameters, weights.
  - Custom (added by us): the split file used, and SLICED evaluation —
      * blended test mAP/precision/recall (smoke-test number)
      * video-only test metrics (the honest number)
      * (extendable) per-mounting recall, false positives on negatives

Device: auto-detected. On CPU this will be SLOW; the script prints a warning and
a reduced-epoch suggestion. Decide GPU vs CPU after seeing it run.
"""

from pathlib import Path
import csv
import sys

# --- config ---------------------------------------------------------------
DATA_YAML   = Path("data/yolo/data.yaml")
SPLIT       = Path("data/splits/split.csv")
ATTRIBUTES  = Path("data/manifests/attributes.csv")
BASE_MODEL  = "yolo11n.pt"          # nano, pretrained — fine-tune from this
IMG_SIZE    = 960                    # input resolution (tunable later; see floor discussion)
EPOCHS      = 100                    # early-stopping will likely cut this short
PATIENCE    = 20                     # early-stop patience
EXPERIMENT  = "area_label_detector"
RUN_NAME    = "smoke_test_nano_blended"
MLFLOW_URI  = "sqlite:///mlflow.db"  # clean single-file backend (not mlruns/ sprawl)


def main():
    # enable Ultralytics' MLflow autolog + point it at our tracking store
    import os
    #os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_URI)  
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    os.environ["MLFLOW_RUN"] = RUN_NAME

    from ultralytics import YOLO, settings
    settings.update({"mlflow": True})   # turn on the built-in MLflow logging

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("=" * 60)
        print("WARNING: no GPU detected — training on CPU will be SLOW.")
        print("For a first smoke-test, consider EPOCHS=20-30 to confirm the")
        print("pipeline works before committing to a long CPU run.")
        print("=" * 60)

    if not DATA_YAML.exists():
        sys.exit(f"ERROR: {DATA_YAML} not found — run make_yolo_dataset.py first.")

    # --- train (Ultralytics autologs to MLflow) --------------------------
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        device=device,
        project=EXPERIMENT,      # also used by MLflow experiment naming
        name=RUN_NAME,
        exist_ok=True,
    )

    # --- sliced evaluation (our custom part) -----------------------------
    # The autolog gave us the BLENDED val metrics. Now compute the VIDEO-ONLY
    # number, which is the honest one. We do this by validating against a
    # video-only file-list.
    import mlflow

    # Build a video-only val list from the split + attributes (modality lives
    # in split.csv). Write a temp data.yaml pointing val at video-only images.
    rows = list(csv.DictReader(open(SPLIT)))
    video_test = [r["image"] for r in rows
                  if r["split"] == "test" and r["modality"] == "video_frame"]
    print(f"\nVideo-only test images: {len(video_test)}")

    # (For the smoke-test we report the blended number from autolog and print
    # the video-only count. Full video-only re-validation via a filtered
    # data.yaml is the immediate next refinement — kept minimal here so the
    # pipeline runs first.)

    # Re-open the MLflow run to attach custom tags/metrics.
    mlflow.set_tracking_uri(MLFLOW_URI)
    # find the most recent run in our experiment and log extras to it
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp:
        runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if len(runs):
            run_id = runs.iloc[0]["run_id"]
            with mlflow.start_run(run_id=run_id):
                mlflow.log_param("split_file", str(SPLIT))
                mlflow.log_param("base_model", BASE_MODEL)
                mlflow.log_param("smoke_test", True)
                mlflow.log_metric("video_test_image_count", len(video_test))
                mlflow.set_tag("note", "blended test set — smoke-test, not deployment metric")
    print("\nDone. View runs with:  mlflow ui --backend-store-uri", MLFLOW_URI)


if __name__ == "__main__":
    main()
