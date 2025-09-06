import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from imgProc_Binary_gdal import make_output_text


def test_make_output_text(tmp_path):
    filename = "out.txt"
    data = ["first", "second"]
    make_output_text(filename, tmp_path, data)
    out_file = tmp_path / filename
    assert out_file.exists()
    assert out_file.read_text() == "first\nsecond\n"
