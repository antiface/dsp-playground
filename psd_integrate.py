#!/usr/bin/python

import sys

import Image
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.cm as cm

def integrateSubblockFreqEnergy(fname, freqrange=0, blocksize=16, logscale=True,
    fps=25, scale_by_freq=True, imgscale=None, imgscaleMethod=Image.BICUBIC):
    """ Integrates last freqrange frequencies in log_10 scale (if requested).
    @param fname: image filename
    @param freqrange: the vector will have at most this many highest frequencies
    (elements), 0 means use all frequencies
    @param blocksize: image will be split into subblocks of this size
    @param logscale: log() PSD values of subblocks before integrating them
    @param fps: frames per second; really useful just to have correct domain
    @param scale_by_freq: values will be scaled to frequency
    (see matplotlib.mlab.psd)
    @param imgscale: scale every image to resolution given by tuple (width,
    height) if not already in given resolution
    @param imgscaleMethod: interpolation method for scaling, see Image.resize()
    @return: (psdValues, completeDomain) - integrated np.array with last
    (highest) freqrange frequency energies and complete domain (before splicing
    with freqrange).
    """
    integrVector = None
    image = Image.open(fname)#.convert('YCbCr')
    if imgscale and image.size != imgscale:
        image = image.resize(imgscale, imgscaleMethod)

    imgResolution = image.size
    x = imgResolution[0]
    y = imgResolution[1]
    freq = x * y * fps

    removeNegInf = lambda l: [e != float("-inf") and e or 0 for e in l]
        
    for yy in range(0, y, blocksize):
        for xx in range(0, x, blocksize):
            block = image.crop((xx, yy, xx+blocksize, yy+blocksize))
            (values, domain) = mlab.psd(block.getdata(0), NFFT=blocksize, Fs=freq,
                scale_by_freq=scale_by_freq)
            
            values = values[-freqrange:]
            if logscale:
                values = removeNegInf(np.log10(values))
            
            if integrVector is None:
                integrVector = np.array(values)
            else:
                integrVector += values

    return (integrVector, domain)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s imagefile1 [imagefile2] ... [imagefileN]" % sys.argv[0]
        sys.exit(1)

    fnames = sys.argv[1:]
    legendTitles = list(fnames)
    fig = plt.figure()
    ax = fig.add_subplot(121)

    integrVectors = {}

    for fname in fnames:
        (integrVector, domain) = integrateSubblockFreqEnergy(fname, logscale=True)
        integrVectors[fname] = integrVector

        ax.plot(integrVector)
        print "Plotting vector %s" % integrVector

    ax.set_title("PSD of resized images vs original")
    leg = ax.legend(legendTitles,
        'upper right', shadow=True)

    #correlations
    ax = fig.add_subplot(122)
    firstVector = integrVectors[fnames[0]]
    for fname in fnames: #we plot all to preserve same colors as left graph
        integrVector = integrVectors[fname]
        corrValues = []
        for i in range(len(integrVector)):
            c = np.corrcoef(firstVector[i:], integrVector[i:])[0][1]
            corrValues.append(c)
        ax.plot(corrValues)

    ax.set_title("Correlation to PSD of first by N highest frequencies")
    leg = ax.legend(fnames,
        'lower left', shadow=True)

    plt.show()

# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
