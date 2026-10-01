import numpy as np
import pytest

from rgba_pipeline.noise import estimate_noise


def test_estimate_noise_uses_two_sample_difference_over_sqrt_two():
    p1 = np.zeros((2, 2, 4), dtype=np.float32)
    p2 = np.full((2, 2, 4), 0.2, dtype=np.float32)
    a1 = np.full((2, 2), 0.5, dtype=np.float32)
    a2 = np.full((2, 2), 0.7, dtype=np.float32)
    report = estimate_noise(p1, p2, a1, a2)
    assert report["full_frame"]["pixel_count"] == 4
    assert report["full_frame"]["premult_rgb_sigma_rms"] == pytest.approx(0.2 / np.sqrt(2))
    assert report["full_frame"]["alpha_sigma_rms"] == pytest.approx(0.2 / np.sqrt(2))
    assert report["target_support_union"]["pixel_count"] == 4


def test_estimate_noise_reports_transparent_background_separately():
    p1 = np.zeros((2, 2, 4), dtype=np.float32)
    p2 = p1.copy()
    p2[0, 0, :3] = 0.1
    a1 = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    a2 = a1.copy()
    report = estimate_noise(p1, p2, a1, a2)
    assert report["transparent_background"]["pixel_count"] == 2
    assert report["target_support_union"]["pixel_count"] == 2
    assert report["transparent_background"]["premult_rgb_sigma_rms"] > 0


def test_estimate_noise_rejects_mismatched_or_nonfinite_inputs():
    p = np.zeros((2, 2, 4), dtype=np.float32)
    a = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="matching dimensions"):
        estimate_noise(p, np.zeros((1, 2, 4), dtype=np.float32), a, a)
    p[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        estimate_noise(p, p.copy(), a, a)
