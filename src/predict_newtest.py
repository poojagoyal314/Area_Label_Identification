"""
predict_newtest.py

Run the FINAL trained model on genuinely held-out data (never seen in training
or any CV fold). This is a qualitative generalization check, not a statistical
one — 6 images + 2 videos is anecdotal. What we're looking for:

  positives  -> should detect >=1 area label each        (misses = false negatives)
  negatives  -> should detect ZERO                        (any detection = FALSE POSITIVE)
  videos     -> deployment modality; detections through the clip

Annotated outputs (boxes + confidence drawn) are saved so you can SEE the
predictions and reuse a good one in the deck.

Runs fine on CPU — inference with a nano model is light. No GPU needed.

Setup (from repo root):
  1. Download the final weights from Drive to:  models/final_yolo11n_960_all260_last.pt
  2. Put the test files in:
       data/test_new/positives/   (3 images)
       data/test_new/negatives/   (3 images)
       data/test_new/videos/      (2 videos)
  3. pip install ultralytics   (already in your venv)
  4. python src/predict_newtest.py
       (optional: CONF=0.15 python src/predict_newtest.py  -> stricter negative test)
"""

from pathlib import Path
import os

MODEL     = os.environ.get("MODEL", "models/final_yolo11n_960_all260_last.pt")
CONF      = float(os.environ.get("CONF", "0.1"))     # YOLO default; lower = stricter FP test
IMGSZ     = int(os.environ.get("IMGSZ", "1280"))        # match training
TEST_ROOT = Path("data/test_new")

CATEGORIES = {
    "positives": "should detect >=1 each (a miss = false negative)",
    "negatives": "should detect ZERO (any detection = FALSE POSITIVE)",
    "videos":    "deployment modality; detections across frames",
}


def main():
    if not Path(MODEL).exists():
        raise SystemExit(f"ERROR: model not found at {MODEL}\n"
                         f"Download final_yolo11n_960_all260_last.pt from Drive to models/.")

    from ultralytics import YOLO
    model = YOLO(MODEL)
    print(f"Model: {MODEL}  |  conf>={CONF}  imgsz={IMGSZ}\n")

    summary = {}
    for cat, expectation in CATEGORIES.items():
        src = TEST_ROOT / cat
        if not src.exists() or not any(src.iterdir()):
            print(f"[skip] {src} missing or empty")
            continue

        print(f"=== {cat.upper()} — {expectation} ===")
        results = model.predict(source=str(src), conf=CONF, imgsz=IMGSZ,
                                save=True, project="runs/newtest", name=cat,
                                exist_ok=True, stream=True, verbose=False)

        # aggregate per source file (a video yields one result per frame, same path)
        per_file = {}
        for r in results:
            name = Path(r.path).name
            n = len(r.boxes)
            mc = float(max(r.boxes.conf)) if n else 0.0
            s = per_file.setdefault(name, {"frames": 0, "dets": 0, "with_det": 0, "max_conf": 0.0})
            s["frames"] += 1
            s["dets"] += n
            s["with_det"] += (1 if n else 0)
            s["max_conf"] = max(s["max_conf"], mc)

        for name, s in per_file.items():
            if s["frames"] == 1:   # an image
                verdict = f"detections={s['dets']}  max_conf={s['max_conf']:.2f}"
            else:                  # a video
                verdict = (f"frames={s['frames']}  frames_with_detection={s['with_det']}  "
                           f"total_detections={s['dets']}  max_conf={s['max_conf']:.2f}")
            print(f"   {name:30s} {verdict}")
        summary[cat] = per_file
        print()

    # scorecard
    print("=" * 60)
    print("SCORECARD (qualitative — tiny sample)")
    if "positives" in summary:
        hit = sum(1 for s in summary["positives"].values() if s["dets"] > 0)
        print(f"  positives detected : {hit}/{len(summary['positives'])}  (want all)")
    if "negatives" in summary:
        fp = sum(1 for s in summary["negatives"].values() if s["dets"] > 0)
        print(f"  negatives that FIRED: {fp}/{len(summary['negatives'])}  (want 0 — these are false positives)")
    if "videos" in summary:
        for name, s in summary["videos"].items():
            pct = 100.0 * s["with_det"] / max(s["frames"], 1)
            print(f"  video {name}: label visible in {pct:.0f}% of frames")
    print("=" * 60)
    print("Annotated images/videos saved under runs/newtest/<category>/ — open them to inspect.")


if __name__ == "__main__":
    main()
