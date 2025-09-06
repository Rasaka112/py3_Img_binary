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
    from numba import jit
except Exception:  # pragma: no cover - provide a dummy decorator
    def jit(func=None, **kwargs):
        return func

try:  # pragma: no cover - simply guards against missing optional dependency
    from osgeo import gdal, osr
except Exception:  # pragma: no cover - any ImportError subclass
    gdal = osr = None

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
    if gdal is None:
        raise ImportError("GDAL is required to read raster parameters")

    raster = gdal.Open(inputpath)
    geotransform = raster.GetGeoTransform()
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixelwidth = geotransform[1]
    pixelheight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize
    spatial_reference = raster.GetProjectionRef()
    print(origin_x, origin_y, pixelwidth, pixelheight, cols, rows, spatial_reference)
    return origin_x, origin_y, pixelwidth, pixelheight, cols, rows, spatial_reference

@jit
def make_output_raster(array, outpath, parameters):
    """Export input array as a georeferenced raster file

    This function exports an array as a georeferenced raster file.
    Export parameters are determined by input.

    Args:
        array: Target array to export
        outpath: Target path where the raster file is exported
        parameters: Geospatial parameters for exported raster file.

    """
    if gdal is None or osr is None:
        raise ImportError("GDAL is required to export raster data")

    driver = gdal.GetDriverByName('GTiff')
    outraster = driver.Create(outpath, parameters[4], parameters[5], 1,
                              gdal.GDT_Byte, ['NBITS=1'])
    outraster.SetGeoTransform((parameters[0], parameters[2], 0, parameters[1],
                               0, parameters[3]))
    outband = outraster.GetRasterBand(1)
    outband.WriteArray(array)
    outraster_spatialreference = osr.SpatialReference()
    outraster_spatialreference.ImportFromWkt(parameters[6])
    outraster.SetProjection(outraster_spatialreference.ExportToWkt())
    outband.FlushCache()

@jit
def calc_threshold_otsu(array):
    """Calculate threshold by using the Otsu-method

    This function calculates a threshold which divides an array.
    The Otsu-method is used.

    Args:
        array: Target array

    Returns:
        Calculated threshold from the input array.

    """
    # src1d = np.sort(src.flatten())
    # Masking the NaN element in the input array
    maskedarray = np.ma.masked_array(array, np.isnan(array))
    array1d = np.sort(maskedarray.flatten())
    # array1d = [i for i in arraysort if np.isfinite(i)]
    var = -10
    threshold_otsu = 0
    thresholdlist = np.sort(np.unique(array1d))
    count_allpixel = len(array1d)
    print(thresholdlist)
    for threshold in thresholdlist:
        index = np.argmax(array1d > threshold)
        # index = np.argmin(array1d < threshold)
        # index = np.searchsorted((array1d > =  threshold),True)
        count_black = len(array1d[:index])
        count_white = count_allpixel - count_black
        if (count_black > 0) and (count_white > 0):
            mean_black = np.mean(array1d[:index])
            mean_white = np.mean(array1d[index:])
            # print threshold, index,count_black, count_white,
            # mean_black, mean_white
            temp_variance = count_black*count_white*(mean_black-mean_white)**2
            # print "current variance = "+str(temp_variance)
            if temp_variance >= var:
                var = temp_variance
                threshold_otsu = threshold
    return threshold_otsu

@jit
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
                raster = gdal.Open(imgpath)
                rdimg = np.asarray(raster.GetRasterBand(1).ReadAsArray())

                imgparameters = get_raster_parameters(imgpath)
                # print imgparameters
                # otsu(scikit-image built-in function)
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
