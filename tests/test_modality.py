"""
Unit tests for the pure modality-classification logic.

This imports the REAL `modality` function from make_split.py — which works in CI
without torch/ultralytics/images, because make_split.py only imports the standard
library at module load (its heavy work lives inside main(), which import does not
run). So these tests exercise actual pipeline code, not a copy.

Run locally:  pytest -q
"""

from make_split import modality


def test_small_image_is_video_frame():
    # Long side below the 1000px threshold -> the low-res video-frame modality.
    assert modality(880, 500) == "video_frame"


def test_video_classification_is_orientation_independent():
    # Uses the LONG side, so width/height order must not matter.
    assert modality(880, 500) == modality(500, 880) == "video_frame"


def test_large_image_is_phone_photo():
    assert modality(4000, 3000) == "phone_photo"


def test_threshold_boundary():
    # Threshold is 1000 on the long side: < 1000 -> video_frame, >= 1000 -> phone_photo.
    assert modality(999, 10) == "video_frame"
    assert modality(1000, 10) == "phone_photo"


def test_accepts_string_dimensions():
    # The manifest stores dimensions as strings; modality() casts with int().
    assert modality("4000", "3000") == "phone_photo"
    assert modality("880", "500") == "video_frame"
