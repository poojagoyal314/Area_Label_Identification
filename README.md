# Area Label Detector

A proof-of-concept object detector that locates **area labels** in warehouse
images — the signs mounted on walls, hung between bays, or fixed to racking that
mark *what a location is* (e.g. "D-01-05", a country flag for a shipping
destination). The goal is localization (find every area label, wherever it is,
however many), not reading the text on them (OCR is out of scope).

This README doubles as a **decisions log**: it records not just *what* the
project does but *why* each non-obvious choice was made, so the reasoning
survives longer than my memory of it.

---

## Headline result

Trained a fine-tuned nano YOLO detector on 260 hand-annotated images.

- **Cross-validated (5-fold, all 260 images): mAP50 0.939 ± 0.052.** This is the
  trusted headline estimate for the chosen config (yolo11n @ 960px).
- **On the deployment modality (video frames), single-split held-out: mAP50 ≈
  0.91 (precision 1.00, recall 0.90)** — stronger than on the out-of-modality
  phone photos (mAP50 ≈ 0.69). The weakness we spent the project guarding against
  (video being masked by easier phone photos in a blended score) turned out not
  to exist: video is the *strong* modality.
- **Higher input resolution (1280px) was tested and gave no meaningful gain**
  (within the cross-validation noise band), so the cheaper 960px config was kept.

Numbers are directional (small, fixed dataset); see caveats below.

---

## Problem framing

- **Task type: object detection, single class (`area_label`).** Not
  classification — the number and location of labels per image matters, and an
  image may contain zero, one, or many. Not segmentation — the targets are
  effectively rectangular signs; pixel masks would be overkill.
- **OCR is out of scope.** We detect the label as an object; we do not read it.
- **Crate labels (the labels on boxes/totes) are deliberately NOT annotated.**
  They become background / hard negatives. Rationale: we can be *complete* on
  area labels (few, prominent) but cannot be complete on crate labels (dozens
  per frame, many blurred/occluded/distant). Partial annotation of a class
  poisons it, so an un-completable class is left as background.

## The core difficulty (named, not hidden)

An area label is a **contextual/functional** category, not a visual one. Area
labels vary wildly in appearance (barcodes, QR codes, plain alphanumeric signs,
country flags) and some overlap visually with the crate labels we ignore. The
main through-line is **context**: area labels are mounted, isolated, and
elevated (on walls, hung, or on rack ends). We work under the assumption that
this context invariant holds.

Guiding principle throughout: **difficulty must not deform the task definition,
but must be instrumented so failure is diagnosable.** We define "area label" by
what it *is*, and tag each instance so that at evaluation, failure can be
attributed to a *specific* sub-population rather than hidden in a blended
average.

---

## Dataset

- **260 images total**, human-collected, no more obtainable (POC constraint).
  - **234 positives** (contain ≥1 area label) — **625 labeled boxes**,
    ~2.67 area labels per positive image.
  - **26 hard negatives** (no area label, but dense with confusable crate
    labels / signage) — included to teach "lots of barcodes ≠ area label".
    Represented as empty label files.

### Two modalities (a key structural finding)

The dataset is **bimodal**, discovered via the conversion manifest, not assumed:

- **Video frames** (~880px PNGs, screen-grabbed from warehouse video):
  lower-resolution, the intended deployment modality. ~104 images (87 positive,
  17 negative).
- **Phone photos** (large JPEG/HEIF): high-resolution, hand-framed,
  out-of-modality. ~156 images (147 positive, 9 negative).

Modality is entangled with label presence, which is a shortcut-learning risk.
Metrics are reported **per modality**, never as a single blended number, because
blending two difficulty populations produces an uninterpretable figure. The
small-object cases (labels that are a tiny fraction of the frame) are
concentrated in the *phone* photos, which is part of why phone scores lower.

---

## Annotation

Tool: **Label Studio**, run locally in Docker (off-the-shelf image, pulled not
built). Chosen specifically for **per-region attributes**, which the
instrumentation plan requires.

### Schema

- **Class (detection):** `area_label` — the only class.
- **Per-box attributes (instrumentation — NOT fed to the model):**
  - `mounting` (required): `wall` / `hung` / `rack`.
  - `content_type`: `alphabetic` / `alphanumeric` / `flag` / `unclear`.
  - `code_type`: `none` / `barcode` / `qr`.
  - `uncertain`: scope-doubt only ("unsure this is a valid area label"). <10
    boxes flagged; domain experts later judged them mostly valid; retained in
    training, flag kept as a record of annotation-time confidence.

### Box conventions

- Axis-aligned boxes, tight to the **whole physical placard**, enclosing the
  full skew even if corners catch background. Never clip the label.
- Every area label boxed (completeness). Pure-negative images: empty label file.
- Size floor applied as a normalized-size filter in code, not baked into
  annotation (so it stays tunable; not yet applied).

---

## Pipeline & repository layout

```
data/
  raw/                  # 260 originals (mixed png/jpg/jpeg/heic). READ-ONLY, gitignored.
  interim/              # all 260 converted: .jpg, EXIF baked, RGB, q95. gitignored.
  annotations/          # DERIVED YOLO label files (one .txt per image). tracked.
  annotations_source/   # the Label Studio JSON export — SOURCE OF TRUTH. tracked.
  manifests/            # conversion manifest + attributes table. tracked.
  splits/               # frozen train/test split (split.csv). tracked.
  yolo/                 # YOLO train/val symlink layout (regenerable). gitignored.
  cv/                   # k-fold symlink folders + per-fold results. gitignored.
src/
  convert_images.py     # raw -> interim (format unify, EXIF bake, manifest)
  derive_labels.py      # Label Studio JSON -> YOLO labels + attributes table
  make_split.py         # presence-stratified train/test split -> split.csv
  make_yolo_dataset.py  # split.csv -> YOLO file-list layout + data.yaml
  train.py              # fine-tune YOLO, MLflow tracking (env-parameterized)
  evaluate.py           # sliced eval: video vs phone, logged to MLflow
  cross_validate.py     # stratified k-fold CV for robust config comparison
tools/label-studio/
  compose.yaml          # Label Studio appliance (separate from any app compose)
```

- **Source vs derived is the organizing principle.** `raw/` and
  `annotations_source/` are precious and irreplaceable; everything else under
  `data/` is regenerable from source + code.
- **Git boundary:** images and regenerable layouts gitignored; code,
  annotations, source JSON, manifests, and the frozen split tracked.
- **Two-artifact model:** the Label Studio JSON is the single source of truth;
  derivation produces (a) YOLO training labels and (b) the attributes table for
  sliced evaluation. Both derive from the same source, so they cannot disagree.

### Conversion notes

- HEIC decoded via `pillow-heif`; **EXIF orientation baked into pixels** then
  stripped (19 images needed rotation — a silent annotate-sideways bug avoided).
- Non-destructive, collision-halting, with a verified manifest. The manifest is
  how the two-modality structure was discovered.

---

## Modeling & experiments

- **Model:** YOLO11n (nano), fine-tuned from pretrained weights. Nano chosen as
  the honest smallest-that-could-work baseline, and it suits the eventual edge
  (Jetson) deployment target.
- **Training:** Google Colab (T4 GPU); laptop too slow for CPU training.
  Input resolution 960; early stopping. All runs tracked in MLflow (SQLite
  backend on Google Drive, so history persists across Colab sessions).
- **Experiment tracking:** every run logged to MLflow — training curves, params,
  and custom sliced metrics. Run names encode the config (e.g.
  `cv5_yolo11n_1280_SUMMARY`) for clean comparison.

### Results

**Baseline (yolo11n, 960px, single train/test split):**

| slice | mAP50 | mAP50-95 | precision | recall | test imgs / instances |
|-------|-------|----------|-----------|--------|-----------------------|
| video (deployment) | 0.911 | 0.740 | 0.997 | 0.903 | 22 / 31 |
| phone (out-of-modality) | 0.688 | 0.414 | 0.747 | 0.565 | 30 / 92 |
| blended (not reported as headline) | 0.808 | 0.517 | 0.845 | 0.732 | 52 / 123 |

**Cross-validation (stratified on presence × modality, all 260 images):**

| config | mAP50 | mAP50-95 | precision | recall |
|--------|-------|----------|-----------|--------|
| k=3, yolo11n, 960 | 0.847 ± 0.049 | 0.546 ± 0.060 | 0.847 ± 0.041 | 0.757 ± 0.070 |
| k=5, yolo11n, 960 | **0.939 ± 0.052** | 0.660 ± 0.061 | 0.916 ± 0.051 | 0.892 ± 0.058 |
| k=5, yolo11n, 1280 | 0.947 ± 0.050 | 0.667 ± 0.068 | 0.917 ± 0.041 | 0.900 ± 0.072 |

**Reading the table:**
- **Trust k=5 over k=3.** The jump from 0.847 to 0.939 is *not* a model
  improvement — same config. Each k=5 fold trains on 4/5 of the data (~208 imgs)
  vs 2/3 for k=3 (~173), so k=5 models simply see more data. Since the final
  model trains on all 260, k=5 is the closer proxy (and 0.939 slightly
  *under*-estimates the final all-data model).
- **1280 does not beat 960.** The +0.008 mAP50 difference is far inside the
  ±0.05 noise band. Higher resolution helps small objects, but our small objects
  are in the *phone* photos (not the deployment modality), so there was little to
  gain where it counts. **Chosen config: yolo11n @ 960** — equivalent accuracy at
  ~half the compute of 1280.
- CV is the measuring stick precisely because gains are this small: without the
  std bands, 0.939 vs 0.947 looks like an upgrade; with them, it's noise.

k=5@960 and k=5@1280 use identical folds (fixed seed) for a clean comparison.

---

## Caveats (stated, because they build credibility)

- **Small, fixed dataset (260 images).** This is a POC; numbers are
  **directional**, not production guarantees.
- **Tiny in-modality test set** (~22 video images, 2 negatives). The video mAP
  is a strong directional signal, but has wide error bars; false-positive
  behaviour on video is essentially unmeasured (too few negatives).
- **Cross-validation firms up the estimate** but cannot manufacture more data.

## Status

- [x] Format unification + EXIF correction (260 images)
- [x] Annotation (234 positives / 625 boxes + 26 hard negatives), final schema
- [x] Two-artifact derivation (YOLO labels + attributes table), verified
- [x] Presence-stratified train/test split (frozen, committed)
- [x] Baseline model (yolo11n) + MLflow tracking
- [x] Sliced evaluation: video vs phone (the key finding)
- [x] Cross-validation (k=3, k=5@960, k=5@1280) — config selected: yolo11n @ 960
- [ ] Train final model on all 260 images with the chosen config; save durably
- [ ] (Optional) Export to TensorRT and deploy inference on Jetson Orin Nano

---

## Tech

Python · Label Studio (Docker) · Pillow / pillow-heif · Ultralytics YOLO11 ·
PyTorch · MLflow · Google Colab (T4) · pandas · Git / GitHub