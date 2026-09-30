"""
make_split.py

Step 3: partition the 260 images into train/test, stratified on
presence/absence (does the image contain >=1 area label?).

THIS RUN IS A PIPELINE SMOKE-TEST, not a reportable result. It uses a single
random split over all 260 images, so the test set is a BLEND of video-frame and
phone-photo modalities. The headline metric from this split is therefore
dominated by the easier phone photos and does NOT measure deployment (video)
performance. That is deliberate for now: the goal is to get split -> train ->
MLflow -> evaluate working end-to-end. The modality-aware split (phone -> train
only, video -> test) is the next iteration.

Design commitments:
  - Stratified on presence/absence so both train and test contain positives AND
    negatives (needed to measure recall AND false-positive behaviour).
  - SEEDED (seed=42) and FROZEN to a committed CSV, so every training run and
    every metric is computed against the identical split. The split is a
    versioned artifact, not something re-rolled each run.
  - The manifest carries per-image modality + presence, so downstream evaluation
    can slice by modality even though we didn't split on it — i.e. we can still
    REPORT the video-only metric from this blended split, we just didn't
    stratify on it.

Output: data/splits/split.csv  (columns: image, split, presence, modality)
"""

from pathlib import Path
import csv
import random
import sys
from collections import Counter, defaultdict

MANIFEST   = Path("data/manifests/conversion_manifest.csv")
ATTRIBUTES = Path("data/manifests/attributes.csv")
SPLIT_OUT  = Path("data/splits/split.csv")

SEED = 42
TEST_FRACTION = 0.20
MODALITY_LONGSIDE_THRESHOLD = 1000


def modality(w, h):
    return "video_frame" if max(int(w), int(h)) < MODALITY_LONGSIDE_THRESHOLD else "phone_photo"


def main():
    if not MANIFEST.exists() or not ATTRIBUTES.exists():
        sys.exit("ERROR: need conversion_manifest.csv and attributes.csv in data/manifests/")

    man = {r["saved_name"]: r for r in csv.DictReader(open(MANIFEST))}
    pos_images = set(r["image"] for r in csv.DictReader(open(ATTRIBUTES)))
    all_images = sorted(man.keys())

    if len(all_images) != 260:
        print(f"WARNING: expected 260 images, manifest has {len(all_images)}")

    # stratify: split positives and negatives separately, then combine
    pos_list = sorted([i for i in all_images if i in pos_images])
    neg_list = sorted([i for i in all_images if i not in pos_images])

    random.seed(SEED)
    random.shuffle(pos_list)
    random.shuffle(neg_list)

    def split(lst):
        n_test = round(len(lst) * TEST_FRACTION)
        return set(lst[n_test:]), set(lst[:n_test])   # train, test

    pos_tr, pos_te = split(pos_list)
    neg_tr, neg_te = split(neg_list)
    train = pos_tr | neg_tr
    test = pos_te | neg_te

    # sanity: no overlap, full coverage
    assert not (train & test), "train/test overlap!"
    assert train | test == set(all_images), "some image unassigned!"

    # write the frozen split manifest
    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "presence", "modality"])
        for img in all_images:
            sp = "train" if img in train else "test"
            pres = "positive" if img in pos_images else "negative"
            mod = modality(man[img]["saved_w"], man[img]["saved_h"])
            w.writerow([img, sp, pres, mod])

    # report the resulting distribution across every axis (eyeball for starvation)
    rows = list(csv.DictReader(open(SPLIT_OUT)))
    print(f"Split written to {SPLIT_OUT} (seed={SEED}, test_frac={TEST_FRACTION})\n")
    for sp in ("train", "test"):
        s = [r for r in rows if r["split"] == sp]
        pres = Counter(r["presence"] for r in s)
        mod = Counter(r["modality"] for r in s)
        print(f"{sp.upper()}: {len(s)} images | presence={dict(pres)} | modality={dict(mod)}")
    print("\nNOTE: test is a modality BLEND — this run is a smoke-test, not a")
    print("deployment-representative result. Modality column is preserved so the")
    print("video-only metric can still be reported from the blended test set.")


if __name__ == "__main__":
    main()
