# -*- coding: utf-8 -*-
"""
Image Processing
"""

import os

# Optional third-party dependencies ------------------------------------------------------
# The original script relies on several heavy geospatial/image processing libraries.  In
# the execution environment used for the kata these libraries might not be installed.
# Import them lazily and provide minimal fallbacks so that lightweight utility
# functions (such as ``make_output_text``) can be imported and tested without pulling
# in the full stack.

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - any ImportError subclass
    np = None

try:  # pragma: no cover - optional dependency
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
except Exception:  # pragma: no cover - any ImportError subclass
    rasterio = CRS = Affine = None

try:  # pragma: no cover - optional dependency
    from skimage import filters
    from skimage import img_as_ubyte
except Exception:  # pragma: no cover - provide fallbacks
    filters = None

    def img_as_ubyte(arr):  # type: ignore[override]
        return arr


def get_raster_parameters(inputpath):
    """Get geospatial parameters from a raster file.

    This function imports geospatial parameters from a raster file.
    Raster file should be described an absolute path.

    Args:
        inputpath: Target raster file

    Returns:
        This function returns the following parameters as a list.
        origin_x: x origin
        oritin_y: y origin
        pixelwidth: pixel width
        pixelheight: pixel height
        cols: number of columns
        ross: number of rows
        spatial_reference: spatial reference
    """
    if rasterio is None:
        raise ImportError("rasterio is required to read raster parameters")

    with rasterio.open(inputpath) as src:
        origin_x = src.transform.c
        origin_y = src.transform.f
        pixelwidth = src.transform.a
        pixelheight = src.transform.e
        cols = src.width
        rows = src.height
        spatial_reference = src.crs.to_wkt() if src.crs else None

    print(origin_x, origin_y, pixelwidth, pixelheight, cols, rows, spatial_reference)
    return origin_x, origin_y, pixelwidth, pixelheight, cols, rows, spatial_reference
def make_output_raster(array, outpath, parameters):
    """Export input array as a georeferenced raster file"""

    if rasterio is None:
        raise ImportError("rasterio is required to export raster data")

    transform = Affine(parameters[2], 0, parameters[0], 0, parameters[3], parameters[1])
    crs = CRS.from_wkt(parameters[6]) if parameters[6] else None
    profile = {
        "driver": "GTiff",
        "height": parameters[5],
        "width": parameters[4],
        "count": 1,
        "dtype": array.dtype,
        "transform": transform,
        "crs": crs,
        "NBITS": 1,
    }
    with rasterio.open(outpath, "w", **profile) as dst:
        dst.write(array, 1)


def calc_threshold_otsu(array):
    """Calculate threshold by using the Otsu-method"""

    if np is None:
        raise ImportError("NumPy is required to calculate Otsu threshold")

    data = np.asarray(array)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("Input array contains no finite values")

    if filters is not None:
        return float(filters.threshold_otsu(data))

    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = hist.sum() - weight1
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-15)
    mean2 = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / np.maximum(weight2, 1e-15)
    between = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[:-1]) ** 2
    idx = np.argmax(between)
    return float(bin_centers[:-1][idx])


def apply_threshold_to_image(array, threshold):
    """Apply threshold for an array to binarize

    This function binarize input array by using input threshold.

    Args:
        array: Target array to binarize
        threshold: Threshold to binarize input array

    Returns:
        binimg_byte: Binarized array is returned,
    """
    binimg = array < threshold
    binimg_byte = img_as_ubyte(binimg)
    return binimg_byte


def make_output_text(filename, filepath, textdata):
    """Export textdata as a textfile

    This function exports a textfile which includes input text data.

    Args:
        filename: Exported text filename
        filepath: Directory path where the text file will be exported.
        textdata: Contents of the text file.

    Returns:
        None.
    """

    outpath = os.path.join(filepath, filename)
    # Open the output file in text mode and ensure each line ends with a
    # newline.  The previous implementation attempted to write a list of
    # strings to a file opened in binary mode which raised a ``TypeError``.
    with open(outpath, "w", encoding="utf-8") as cf:
        for line in textdata:
            cf.write(f"{line}\n")


if __name__ == '__main__':
    srcdir = r""
    outdir = r""
    thresholdlist = []
    for root, dirs, files in os.walk(srcdir):
        for item in files:
            if (item.endswith(".tif")) or (item.endswith(".TIF")):
                imgpath = os.path.join(srcdir, item)
                outpath = os.path.join(outdir, item)
                with rasterio.open(imgpath) as raster:
                    rdimg = raster.read(1)

                imgparameters = get_raster_parameters(imgpath)
                # print imgparameters
                # otsu(scikit-image built-in function)
                if filters is not None:
                    print(filters.threshold_otsu(rdimg))
                # otsu = filters.threshold_otsu(rdimg)
                # binimg = rdimg < otsu

                # otsu test
                threshold_otsu = calc_threshold_otsu(rdimg)
                binary_otsu = apply_threshold_to_image(rdimg, threshold_otsu)
                make_output_raster(binary_otsu, outpath, imgparameters)
                thresholdlist.append(item+","+str(threshold_otsu))
                print("raster file : {}".format(item))
                # print "threshold(built-in otsu) = "+str(otsu)
                print("threshold(otsu) = {}".format(threshold_otsu))

    thresholdfile = "thresold.csv"
    make_output_text(thresholdfile, outdir, thresholdlist)
