# py3_Img_binary

Utilities for thresholding raster images and exporting geospatial or text
outputs.  The repository contains a small collection of functions that were
originally part of a GDAL‑based binary image processing script but now rely on
`rasterio` for raster I/O.

## Features

- Extract geospatial parameters from raster datasets
- Create georeferenced output rasters from NumPy arrays
- Compute Otsu thresholds and apply them to produce binary images
- Export plain text files, e.g. threshold reports

## Optional dependencies

The core of the project relies on several heavy third‑party libraries.  They
are imported lazily so that light‑weight utilities can be used without the full
stack.  Install them as needed:

- [Rasterio](https://rasterio.readthedocs.io/) – reading and writing raster data
- [NumPy](https://numpy.org/) – array manipulation
- [scikit-image](https://scikit-image.org/) – image processing helpers

## Usage

```python
from imgProc_Binary_gdal import make_output_text

# write a CSV file containing threshold values
make_output_text("threshold.csv", "./", ["image1,42", "image2,101"])
```

Other functions in the module require the optional dependencies listed above.

## Development

Run the unit tests to verify the behaviour:

```bash
pytest -q
```

