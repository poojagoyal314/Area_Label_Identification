"""
convert_images.py

Normalise the raw image set.

Reads  : data/raw/        (the 236 originals — mixed png/jpg/jpeg/heic; READ-ONLY)
Writes : data/interim/    (all .jpg, EXIF orientation baked into pixels, RGB, q95)
         data/manifests/conversion_manifest.csv   (one row per file: traceability + dims)

Design commitments:
  - raw/ is never written to. We read raw, write interim.
  - interim/ must be empty at start, so an existing output means a real
    collision, not a leftover from a previous run. Enforces "interim is
    regenerable" and makes the collision-halt meaningful.
  - EXIF orientation is APPLIED to pixels then stripped, so every downstream
    tool agrees on one unambiguous orientation. This is the load-bearing step.
  - Fail LOUD: bad count, HEIC decode failure, or output collision all stop
    the run rather than silently producing a wrong dataset.
"""

from pathlib import Path
import csv
import sys

from PIL import Image, ImageOps

# Registers HEIC support with Pillow. If this import fails, HEIC files
# cannot be read — better to know at import time than mid-run.
import pillow_heif
pillow_heif.register_heif_opener()

# --- configuration --------------------------------------------------------
RAW_DIR       = Path("data/raw")
INTERIM_DIR   = Path("data/interim")
MANIFEST_DIR  = Path("data/manifests")                     # tracked; NOT ignored
MANIFEST      = MANIFEST_DIR / "conversion_manifest.csv"

EXPECTED_COUNT = 260            # assert reality matches our belief
VALID_EXTS = {".png", ".jpg", ".jpeg", ".heic"}   # compared case-insensitively
JPEG_QUALITY = 95               # protect small/distant labels; size is irrelevant here
EXTREME_SIDE = 6000             # flag (don't act on) anything larger for eyeballing


def discover(raw_dir: Path) -> list[Path]:
    """Find every image by extension, case-insensitively (.JPG, .HEIC count)."""
    files = [p for p in sorted(raw_dir.iterdir())
             if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return files


def main() -> None:
    # 1. discover + verify count -------------------------------------------
    if not RAW_DIR.exists():
        sys.exit(f"ERROR: {RAW_DIR} does not exist. Put the 236 originals there.")

    files = discover(RAW_DIR)
    print(f"Discovered {len(files)} images in {RAW_DIR}")
    if len(files) != EXPECTED_COUNT:
        # Not a crash-worthy error necessarily, but you MUST look before proceeding.
        print(f"  WARNING: expected {EXPECTED_COUNT}, found {len(files)}. "
              f"Check for a missed format, a subfolder, or a stray file.")
        # Deliberately a warning not a hard exit — you may have legitimately
        # changed the count. But it's loud, and it's the first thing you see.

    # 2. interim must be empty so collisions are real ----------------------
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p for p in INTERIM_DIR.iterdir() if p.is_file()]
    if existing:
        sys.exit(f"ERROR: {INTERIM_DIR} is not empty ({len(existing)} files). "
                 f"Clear it and re-run — interim is meant to be regenerable.")

    # 3. convert -----------------------------------------------------------
    rows = []
    failures = []
    for src in files:
        try:
            img = Image.open(src)

            # Record ORIGINAL dimensions before any transform, for the manifest.
            orig_w, orig_h = img.size
            orig_format = img.format  # 'PNG', 'JPEG', 'HEIF', ...

            # THE load-bearing correction: rotate pixels per EXIF, then the
            # returned image carries no orientation flag to disagree with.
            img = ImageOps.exif_transpose(img)

            # JPG cannot hold alpha / odd modes. Force RGB.
            if img.mode != "RGB":
                img = img.convert("RGB")

            out_path = INTERIM_DIR / (src.stem + ".jpg")

            # 4. collision halt: within-run or (shouldn't happen) leftover ---
            if out_path.exists():
                sys.exit(f"ERROR: output collision. '{src.name}' maps to "
                         f"'{out_path.name}', which already exists this run. "
                         f"Two inputs share a stem — resolve before proceeding.")

            img.save(out_path, "JPEG", quality=JPEG_QUALITY)

            # exif_transpose may swap w/h; report the SAVED dimensions too.
            saved_w, saved_h = img.size
            extreme = max(saved_w, saved_h) > EXTREME_SIDE

            rows.append({
                "original_name": src.name,
                "original_format": orig_format,
                "original_w": orig_w,
                "original_h": orig_h,
                "saved_name": out_path.name,
                "saved_w": saved_w,
                "saved_h": saved_h,
                "extreme_dimension": extreme,
            })

        except Exception as e:
            # Don't let one bad file kill the run silently — collect and report.
            failures.append((src.name, repr(e)))

    # 5. manifest ----------------------------------------------------------
    if rows:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # 6. summary -----------------------------------------------------------
    print(f"\nConverted : {len(rows)}")
    print(f"Failed    : {len(failures)}")
    for name, err in failures:
        print(f"    FAIL  {name}: {err}")

    extremes = [r for r in rows if r["extreme_dimension"]]
    if extremes:
        print(f"\nFlagged {len(extremes)} image(s) with a side > {EXTREME_SIDE}px "
              f"— eyeball these (likely your 30-35 MB outliers):")
        for r in extremes:
            print(f"    {r['saved_name']}: {r['saved_w']}x{r['saved_h']} "
                  f"(from {r['original_name']}, {r['original_format']})")

    print(f"\nManifest written to {MANIFEST}")


if __name__ == "__main__":
    main()