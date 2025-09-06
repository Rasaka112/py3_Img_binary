import pytest

np = pytest.importorskip("numpy")

from imgProc_Binary_gdal import calc_threshold_otsu


def test_calc_threshold_otsu_basic():
    arr = np.hstack([np.zeros(64), np.ones(64) * 100])
    threshold = calc_threshold_otsu(arr)
    assert 40 < threshold < 60


def test_calc_threshold_otsu_ignores_nan():
    arr = np.array([0, 0, 100, 100, np.nan])
    threshold = calc_threshold_otsu(arr)
    assert 40 < threshold < 60

