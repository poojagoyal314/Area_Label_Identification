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
  poisons it, so an un-completable class is left as background. Warehouse scenes
  are saturated with crate labels, so the model still gets abundant
  hard-negative signal without a dedicated class.

## The core difficulty (named, not hidden)

An area label is a **contextual/functional** category, not a visual one. Area
labels vary wildly in appearance (barcodes, QR codes, plain alphanumeric signs,
country flags) and some overlap visually with the crate labels we ignore. The
main through-line the model can rely on is **context**: area labels are mounted,
isolated, and elevated (on walls, hung, or on rack ends). We work under the
assumption that this context invariant holds.

Guiding principle carried throughout: **difficulty must not deform the task
definition, but must be instrumented so failure is diagnosable.** We define
"area label" by what it *is*, not by what's easy to detect — and we tag each
instance so that, at evaluation, failure can be attributed to a *specific*
sub-population rather than hidden in a blended average.

---

## Dataset

- **260 images total**, human-collected, no more obtainable (POC constraint).
  - **234 positives** (contain >=1 area label) — **625 labeled boxes**,
    ~2.67 area labels per positive image.
  - **26 hard negatives** (no area label, but dense with confusable crate
    labels / signage) — deliberately included to teach "lots of barcodes != area
    label." Represented as empty label files.

### Two modalities (a key structural finding)

The dataset is **bimodal**, discovered via the conversion manifest, not assumed:

- **Video frames** (~880px PNGs, screen-grabbed from warehouse video):
  lower-resolution, the intended deployment modality.
- **Phone photos** (large JPEG/HEIF): high-resolution, hand-framed,
  out-of-modality (a camera in production will not see these).

Modality is entangled with label presence (phone photos skew positive; video
frames include the negatives), which is a shortcut-learning risk. **Strategy
(Option B): train on all 260, but evaluate on video frames only**, via a
modality-aware split (phone photos -> train only; video frames ->
train/val/test). The headline metric is the video-frame metric. Metrics will be
reported **per modality**, never as a single blended number, because blending
two different difficulty populations produces an uninterpretable figure.

Consequence, stated honestly: the video-only test set is small (~20 images),
so its metrics are **directional, not precise** — appropriate for a POC.

---

## Annotation

Tool: **Label Studio**, run locally in Docker (off-the-shelf image, pulled not
built). Chosen specifically because it supports **per-region attributes**, which
the instrumentation plan requires.

### Schema

- **Class (detection):** `area_label` — the only class.
- **Per-box attributes (instrumentation — NOT fed to the model):**
  - `mounting` (**required**): `wall` / `hung` / `rack`. The diagnostically
    load-bearing axis (tied to the context invariant; `rack` was added when
    rack-end signs like "702" were ruled in-scope, which softened the
    "elevated + isolated" invariant).
  - `content_type`: `alphabetic` / `alphanumeric` / `flag` / `unclear`.
    (`alphabetic` = no digits; `alphanumeric` = >=1 digit. `unclear` =
    legible-as-a-label but content unreadable.)
  - `code_type`: `none` / `barcode` / `qr`.
  - `uncertain`: scope-doubt only ("unsure this is a valid area label").

`content_type` and `code_type` are kept as **two independent axes** rather than
one flattened list, so slices ("recall on QR-bearing labels", "recall on flag
labels") don't starve each other at small counts, and new combinations slot in
for free.

### Box conventions

- Axis-aligned boxes, tight to the **whole physical placard** (including
  backing/border), enclosing the full skew even if corners catch background.
  Never clip the label to avoid background. (Rotated/oriented boxes rejected:
  more annotation cost, more data needed, labels only mildly skewed.)
- Every area label boxed (completeness). Pure-negative images: empty label file.
- Edge-clipped labels: box the visible portion only. Occluded: box if still
  identifiable. Below a sensible size floor: treat as background (floor applied
  as a normalized-size filter in code, not baked into annotation).

### Note on `uncertain`

Fewer than 10 boxes were flagged `uncertain`. Domain experts subsequently judged
these to be mostly valid area labels. Decision: **retain them in training**, and
**keep the flag unchanged** as a record of annotation-time confidence — it costs
nothing (they're trained on either way) and preserves a diagnostic thread ("were
problem cases among the originally-doubtful ones?").

---

## Data pipeline & repository layout

```
data/
  raw/                  # 260 originals (mixed png/jpg/jpeg/heic). READ-ONLY, gitignored.
  interim/              # all 260 converted: .jpg, EXIF baked, RGB, q95. gitignored.
  annotations/          # DERIVED YOLO label files (one .txt per image). tracked.
  annotations_source/   # the Label Studio JSON export — SOURCE OF TRUTH. tracked.
  manifests/            # conversion manifest + derived attributes table. tracked.
src/
  convert_images.py     # step 1: raw -> interim (format unify, EXIF bake, manifest)
tools/label-studio/
  compose.yaml          # Label Studio appliance (separate from any app compose)
```

- **Source vs derived is the organizing principle.** `raw/` (source images) and
  `annotations_source/` (source annotations, the JSON) are precious and
  irreplaceable. Everything in `interim/`, `annotations/`, and the attributes
  table is **regenerable** from source + code.
- **Git boundary:** images (`raw/`, `interim/`) are large binaries and are
  gitignored. Annotations, the JSON export, manifests, and code are small text
  and **tracked** — the JSON especially, as it is the irreplaceable product of
  the annotation effort.
- **Two-artifact model:** the JSON is the single source of truth; a derivation
  script produces (a) YOLO training labels and (b) the attributes table for
  sliced evaluation. Both derive from the same source, so they cannot disagree.

### Conversion notes

- HEIC decoded via `pillow-heif`; **EXIF orientation baked into pixels** then
  stripped (19 images in the original set needed rotation — a silent
  annotate-sideways bug avoided).
- Conversion is non-destructive (raw -> interim), collision-halting, and emits a
  manifest (per-image format + dimensions) that is verified before proceeding.
  The manifest is how the two-modality structure was discovered.

---

## Status

- [x] Format unification + EXIF correction (260 images)
- [x] Annotation of 234 positives (625 boxes) + 26 hard negatives, final schema
- [x] Count reconciliation (export 234 positives + 26 negatives = 260, verified)
- [ ] Derivation script: JSON -> 260 YOLO labels + attributes table (with
      box-count conservation + coordinate-range checks)
- [ ] Modality-aware train/val/test split (phone -> train; video -> all splits)
- [ ] Baseline model + MLflow tracking
- [ ] Evaluation: per-modality metrics, false-positive rate on negatives,
      sliced recall by mounting/content_type/code_type

## Open decisions / deferred

- Size floor threshold (normalized) — to be set with box-size data in hand.
- Whether out-of-modality (phone) training data helps video detection — a
  planned experiment (train with vs without phone photos, compare on the same
  video test set). Answers "does out-of-modality data help?" with a measurement.
- Negative ratio at training time — tunable, validated by false-positive
  measurement, now backed by 26 real hard negatives.

---

## Tech

Python · Label Studio (Docker) · Pillow / pillow-heif · (planned: YOLO-family
detector, MLflow) · scikit-learn / pandas for evaluation slicing
