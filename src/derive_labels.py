"""
derive_labels.py

Step 2: turn the Label Studio JSON (the SOURCE OF TRUTH) into
the two derived artifacts:

  1. YOLO label files  -> data/annotations/<image>.txt
       one line per box:  0 x_center y_center width height   (all normalized 0-1)
       single class (0 = area_label). Pure negatives get an EMPTY file.

  2. Attributes table  -> data/manifests/attributes.csv
       one row per box, carrying the instrumentation (mounting / content_type /
       code_type / uncertain) plus modality and the normalized box size.
       This is what evaluation is sliced on. It is NOT fed to the model.

Design commitments (argued through, not defaults):
  - Negatives are identified as "in the manifest but NOT in the JSON" — robust to
    naming (catches clip_13/clip_4, not just n*). They get empty label files.
  - Label Studio stores x,y,w,h as PERCENT (0-100) of the image, with x,y at the
    box TOP-LEFT. YOLO wants FRACTIONS (0-1) with x,y at the CENTER. Both
    transforms are applied and then range-checked.
  - Modality (video_frame / phone_photo) is derived from the manifest dimensions
    and joined per image. Used for the modality-aware split and per-modality eval.
  - uncertain boxes are INCLUDED in the YOLO labels (decision: keep scarce
    positives) but the flag is carried into the attributes table so the choice
    stays reversible.
  - NO size floor is applied here. Every box is emitted, with its normalized area
    recorded, so the floor can be decided later from the real distribution.

  - Fail LOUD: box-count must be conserved, coords must be in [0,1], every image
    must join to a manifest row. Any violation stops the run.
"""

from pathlib import Path
from urllib.parse import unquote
import json
import csv
import re
import sys

# --- configuration --------------------------------------------------------
JSON_EXPORT = Path("data/annotations_source/project-3-at-2026-09-21-11-14-f69b0533.json")
MANIFEST    = Path("data/manifests/conversion_manifest.csv")
LABELS_DIR  = Path("data/annotations")
ATTR_TABLE  = Path("data/manifests/attributes.csv")

CLASS_ID = 0                      # single class: area_label
EXPECTED_TOTAL_IMAGES = 260

# Modality heuristic: video frames are the low-res screen-grabs (~880px tall
# PNGs); phone photos are the large JPEG/HEIF. We classify by long side: below
# a threshold => video_frame, else phone_photo. (Verified against the manifest;
# the two populations are cleanly separated with a wide gap around ~900px.)
MODALITY_LONGSIDE_THRESHOLD = 1000   # long side < this -> video_frame


def image_basename(task):
    """Extract the decoded image filename from a task's data.image field."""
    for k, v in task["data"].items():
        if isinstance(v, str) and any(v.lower().endswith(e)
                                      for e in (".jpg", ".jpeg", ".png")):
            return unquote(re.split(r"[\\/=]", v)[-1])
    raise ValueError(f"No image field found in task {task.get('id')}")


def load_manifest():
    rows = {r["saved_name"]: r for r in csv.DictReader(open(MANIFEST))}
    return rows


def modality_of(manifest_row):
    w, h = int(manifest_row["saved_w"]), int(manifest_row["saved_h"])
    long_side = max(w, h)
    return "video_frame" if long_side < MODALITY_LONGSIDE_THRESHOLD else "phone_photo"


def main():
    if not JSON_EXPORT.exists():
        sys.exit(f"ERROR: JSON export not found at {JSON_EXPORT}")
    if not MANIFEST.exists():
        sys.exit(f"ERROR: manifest not found at {MANIFEST}")

    manifest = load_manifest()
    data = json.load(open(JSON_EXPORT))
    print(f"Loaded {len(data)} annotated tasks; manifest has {len(manifest)} images.")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    ATTR_TABLE.parent.mkdir(parents=True, exist_ok=True)

    annotated_images = set()
    attr_rows = []
    boxes_in_json = 0          # counted straight from JSON
    lines_written = 0          # counted as we write label files
    coord_violations = []

    # --- 1. positives: one label file per annotated task ------------------
    for task in data:
        img = image_basename(task)
        annotated_images.add(img)

        if img not in manifest:
            sys.exit(f"ERROR: annotated image '{img}' not in manifest — cannot "
                     f"resolve dimensions/modality. Halting.")
        modality = modality_of(manifest[img])

        # Take the first annotation (single annotator).
        ann = task["annotations"][0]
        results = ann.get("result", [])

        # Group result entries by region id: the box + its attribute choices.
        regions = {}
        for r in results:
            rid = r.get("id")
            regions.setdefault(rid, {})[r.get("type")] = r

        lines = []
        for rid, parts in regions.items():
            box = parts.get("rectanglelabels")
            if box is None:
                continue  # a stray choices entry with no box; skip defensively
            boxes_in_json += 1
            v = box["value"]

            # PERCENT (0-100), top-left -> FRACTION (0-1), center.
            x_frac = v["x"] / 100.0
            y_frac = v["y"] / 100.0
            w_frac = v["width"] / 100.0
            h_frac = v["height"] / 100.0
            xc = x_frac + w_frac / 2.0
            yc = y_frac + h_frac / 2.0

            # range check
            for name, val in (("xc", xc), ("yc", yc), ("w", w_frac), ("h", h_frac)):
                if not (0.0 - 1e-6 <= val <= 1.0 + 1e-6):
                    coord_violations.append((img, rid, name, val))

            lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w_frac:.6f} {h_frac:.6f}")

            # attributes: each is a 'choices' entry sharing this region's id,
            # keyed by from_name (mounting / content_type / code_type / uncertain)
            mounting = content_type = code_type = uncertain = ""
            for r in results:
                if r.get("id") == rid and r.get("type") == "choices":
                    fn = r.get("from_name")
                    ch = r["value"].get("choices", [])
                    val = ch[0] if ch else ""
                    if fn == "mounting":     mounting = val
                    elif fn == "content_type": content_type = val
                    elif fn == "code_type":  code_type = val
                    elif fn == "uncertain":  uncertain = val

            attr_rows.append({
                "image": img,
                "modality": modality,
                "mounting": mounting,
                "content_type": content_type,
                "code_type": code_type,
                "uncertain": uncertain,
                "x_center": round(xc, 6),
                "y_center": round(yc, 6),
                "width": round(w_frac, 6),
                "height": round(h_frac, 6),
                "norm_area": round(w_frac * h_frac, 8),
            })

        # write the label file (may be multi-line)
        out = LABELS_DIR / (Path(img).stem + ".txt")
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        lines_written += len(lines)

    # --- 2. negatives: manifest images NOT annotated -> empty label files -
    negatives = set(manifest.keys()) - annotated_images
    for img in sorted(negatives):
        out = LABELS_DIR / (Path(img).stem + ".txt")
        out.write_text("")   # empty = pure negative
        # negatives contribute no attribute rows (no boxes) but we could record
        # them at image level elsewhere; modality still matters for the split.

    # --- 3. write attributes table ---------------------------------------
    with open(ATTR_TABLE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(attr_rows[0].keys()))
        w.writeheader()
        w.writerows(attr_rows)

    # --- 4. verification (fail loud) -------------------------------------
    total_label_files = len(list(LABELS_DIR.glob("*.txt")))
    print("\n--- verification ---")
    print(f"Boxes counted in JSON      : {boxes_in_json}")
    print(f"Label lines written        : {lines_written}")
    print(f"Attribute rows             : {len(attr_rows)}")
    print(f"Positives (annotated)      : {len(annotated_images)}")
    print(f"Negatives (empty files)    : {len(negatives)}")
    print(f"Total label files          : {total_label_files}")

    ok = True
    if boxes_in_json != lines_written:
        print(f"  FAIL: box-count not conserved ({boxes_in_json} != {lines_written})")
        ok = False
    if len(attr_rows) != boxes_in_json:
        print(f"  FAIL: attribute rows != boxes ({len(attr_rows)} != {boxes_in_json})")
        ok = False
    if total_label_files != EXPECTED_TOTAL_IMAGES:
        print(f"  FAIL: expected {EXPECTED_TOTAL_IMAGES} label files, got {total_label_files}")
        ok = False
    if coord_violations:
        print(f"  FAIL: {len(coord_violations)} coordinate(s) outside [0,1]:")
        for v in coord_violations[:10]:
            print("     ", v)
        ok = False

    # modality breakdown (informational)
    from collections import Counter
    mod = Counter(r["modality"] for r in attr_rows)
    print(f"\nBoxes by modality: {dict(mod)}")
    neg_mod = Counter(modality_of(manifest[i]) for i in negatives)
    print(f"Negative images by modality: {dict(neg_mod)}")

    if not ok:
        sys.exit("\nVERIFICATION FAILED — do not use these outputs.")
    print("\nAll checks passed. Labels + attributes table written.")


if __name__ == "__main__":
    main()
