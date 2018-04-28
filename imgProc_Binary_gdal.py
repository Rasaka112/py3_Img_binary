# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 18:22:17 2017
Image Processing

@author: ryuichichiba
"""

import os
import itertools
import numpy as np
from osgeo import gdal, osr
# from skimage import filters
# from skimage import img_as_ubyte


class BinarizeImage(object):
    """Image binarizing class

    """
    def __init__(self, sourceraster):
        self.threshold_otsu = -10
        self.sourceraster = sourceraster

    def __del__(self):
        del self

    def get_raster_parameters(self):
        """Get geospatial parameters from a raster file.

        This function gets geospatial parameters from a raster file.
        Raster file should be described an absolute path.

        """
        self.raster = gdal.Open(self.sourceraster)
        self.sourcearray = np.asarray(self.raster.GetRasterBand(1).ReadAsArray())

        geotransform = self.raster.GetGeoTransform()
        self.origin_x = geotransform[0]
        self.origin_y = geotransform[3]
        self.pixelwidth = geotransform[1]
        self.pixelheight = geotransform[5]
        self.cols = self.raster.RasterXSize
        self.rows = self.raster.RasterYSize
        self.spatial_reference = self.raster.GetProjectionRef()
        # print self.origin_x, self.origin_y
        # print self.pixelwidth, self.pixelheight
        # print self.cols, self.rows
        #  print self.spatial_reference

    def apply_threshold_to_image(self):
        """Apply threshold for an array to binarize

        This function binarize input array by using defined threshold.

        Returns:
            Binarized array is returned,
        """
        binimg = (np.array([i if i > self.threshold_otsu else 0 for i
            in self.sourcearray.flatten()]).reshape(len(self.sourcearray),
            len(self.sourcearray[0])))
        # return np.array([i if i > self.threshold_otsu else 0 for i
        #     in self.sourcearray.flatten()]).reshape(len(self.sourcearray),
        #     len(self.sourcearray[0]))
        return binimg

    def make_output_raster(self, outpath):
        """Export input array as a georeferenced raster file

        This function exports an array as a georeferenced raster file.
        Export parameters are determined by input.

        Args:
            outpath: Target path where the raster file is exported

        """
        driver = gdal.GetDriverByName('GTiff')
        outraster = driver.Create(outpath, self.cols, self.rows, 1,
                                  gdal.GDT_Byte, ['NBITS=1'])
        outraster.SetGeoTransform((self.origin_x, self.pixelwidth, 0,
                                   self.origin_y, 0, self.pixelheight))
        binimg = self.apply_threshold_to_image()
        outband = outraster.GetRasterBand(1)
        outband.WriteArray(binimg)
        outraster_spatialreference = osr.SpatialReference()
        outraster_spatialreference.ImportFromWkt(self.spatial_reference)
        outraster.SetProjection(outraster_spatialreference.ExportToWkt())
        outband.FlushCache()

    def calc_threshold_otsu(self):
        """Calculate threshold by using the Otsu-method

        This function calculates a threshold which divides an array.
        Threshold is calculated by using Otsu-method.

        Returns:
            Calculated threshold from the input array.

        """
        # Masking the NaN element in the input array
        maskedarray = np.ma.masked_array(self.sourcearray,
                                          np.isnan(self.sourcearray))
        array1d = np.sort(maskedarray.flatten())
        var = -10
        thresholdlist = np.sort(np.unique(array1d))
        # count_allpixel = len(array1d)
        count_allpixel = np.ma.count(array1d)
        
        # var_array = []
        # for i, n in enumerate(array1d):
        #    print len(array1d[:i])*len(array1d[i:])*(np.mean(array1d[:i])-np.mean(array1d[i:]))**2 
        #    var_array.append(len(array1d[:i])*len(array1d[i:])*(np.mean(array1d[:i])-np.mean(array1d[i:]))**2)
        # self.threshold_otsu = array1d[np.argmax[var_array]]
        #print self.threshold_otsu
        #print thresholdlist
        index_threshold = ([np.argmax(array1d > i) for i in np.unique(array1d)])
        #print index_threshold
        var_array = ([len(array1d[:i])*len(array1d[i:])*(np.mean(array1d[:i])-
                         np.mean(array1d[i:]))**2 for i in index_threshold])
        max_var = -10
        max_index = 0
        for index, var in itertools.izip(index_threshold,var_array):
            if var >= max_var:
                max_var = var
                max_index = index
        #print index, max_index
        self.threshold_otsu = array1d[max_index-1]
        print self.threshold_otsu
        # for threshold in np.unique(array1d):
            # index = np.argmax(array1d > threshold)
            # index = np.argmin(array1d < threshold)
            # index = np.searchsorted((array1d > =  threshold),True)
            # count_black = np.ma.count(array1d[:index])
            # count_black = len(array1d[:index])
            # count_white = count_allpixel - count_black
            # if (count_black > 0) and (count_white > 0):
                # mean_black = np.mean(array1d[:index])
                # mean_white = np.mean(array1d[index:])
                #print threshold, index,count_black, count_white, mean_black, mean_white
                # temp_variance = count_black*count_white*(mean_black-mean_white)**2
                # print "current variance = "+str(temp_variance)
                # if temp_variance >= var:
                    # var = temp_variance
                    # self.threshold_otsu = threshold
        return self.threshold_otsu


def make_output_text(filename, filepath, contents):
    """Export contents as a textfile

    This function exports a textfile which includes input text data.

    Args:
        filename: Exported text filename
        filepath: Directory path where the text file will be exported.
        contents: Contents will be written to the text file.
                  This argument should be list type.

    Returns:
        None.
    """

    outpath = os.path.join(filepath, filename)
    cf = open(outpath, 'wb')
    [cf.write(line+"\n") for line in contents]
    cf.close()


if __name__ == '__main__':
    srcdir = r"/Users/ryuichichiba/scriptwork/testimg/geotiff"
    outdir = r"/Users/ryuichichiba/scriptwork/testimg/out"
    thresholdlist = []
    for root, dirs, files in os.walk(srcdir):
        for item in files:
            if (item.endswith(".tif")) or (item.endswith(".TIF")):
                imgpath = os.path.join(srcdir, item)
                outpath = os.path.join(outdir, item)
                workimg = BinarizeImage(imgpath)
                workimg.get_raster_parameters()
                threshold_otsu = workimg.calc_threshold_otsu()
                workimg.make_output_raster(outpath)
                del workimg

                # print imgparameters
                # otsu(scikit-image built-in function)
                # print filters.threshold_otsu(rdimg)
                # otsu = filters.threshold_otsu(rdimg)
                # binimg = rdimg < otsu

                # otsu test
                thresholdlist.append(item+","+str(threshold_otsu))
                print "raster file : "+item
                # print "threshold(built-in otsu) = "+str(otsu)
                print "threshold(otsu) = "+str(threshold_otsu)

    thresholdfile = "thresold.csv"
    make_output_text(thresholdfile, outdir, thresholdlist)
