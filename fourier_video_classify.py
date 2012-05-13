#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show, imshow

from scipy.signal import convolve2d

import math
import sys

from optparse import OptionParser

import Image

def imageToLumaMatrix(filename, crop=None):
    """ Returns luma band of image as a matrix """
    im = Image.open(filename)
    if crop:
        im = im.crop(crop)
    im2 = im#.convert("YCbCr")
    (x, y) = im2.size 
    luma = im2.getdata(0)

    return np.reshape(luma, (y, x))
    
def FFTDiff(matrix1, matrix2, logarithmic=True):
    """ Returns absolute difference of FFT(matrix1) - FFT(matrix2).
    If logarithmic is True, values are in log scale.
    """
    diff = np.abs(np.fft.fft2(matrix1) - np.fft.fft2(matrix2))
    if logarithmic:
        return np.log(diff)
    else:
        return diff
    
parser = OptionParser()
parser.add_option("-o", "--original", action="store", type="string", dest="original",
    help="Original image")
parser.add_option("-e", "--encoded", action="store", type="string", dest="encoded",
    help="Encoded image")
(opts, args) = parser.parse_args()

if not opts.original or not opts.encoded:
    parser.print_help()
    sys.exit(1)

fft_abs = lambda x, logarithmic=False: [np.log(abs(np.fft.fft2(x))), abs(np.fft.fft2(x))][not logarithmic]

##images

originalImage = imageToLumaMatrix(opts.original)
encodedImage = imageToLumaMatrix(opts.encoded)
originalFFT = fft_abs(originalImage, True)
encodedFFT = fft_abs(encodedImage, True)

imageFFTDiff = FFTDiff(originalImage, encodedImage, True)
imageAbsDiff = np.abs(encodedImage-originalImage)
FFTofDiff = np.log(np.abs(np.fft.fft2(originalImage - encodedImage)))

spatial_vmax = float(max(originalImage.max(), encodedImage.max()))
spatial_vmin = float(min(originalImage.min(), encodedImage.min()))
fft_vmax = float(max(originalFFT.max(), encodedFFT.max(), FFTofDiff.max(), imageFFTDiff.max() ))
fft_vmin = float(min(originalFFT.min(), encodedFFT.min(), FFTofDiff.min(), imageFFTDiff.min() ))

##filter kernels
laplace1 = np.matrix("-1 -1 -1; -1 8 -1; -1 -1 -1")
laplace2 = np.matrix("0 1 0; 1 -4 1; 0 1 0")
    
delta = 0.25
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
gaussKernel = mlab.bivariate_normal(X, Y, 0.5, 0.5, 0.0, 0.0)

gaussDenoise = np.matrix(
"2.0 4.0 5.0 4.0 2.0;" + \
"4.0 9.0 12.0 9.0 4.0;" + \
"5.0 12.0 15.0 12.0 5.0;" + \
"4.0 9.0 12.0 9.0 4.0;" + \
"2.0 4.0 5.0 4.0 2.0"
)*1.0/159

sobelVertical = np.matrix("1 0 -1; 0 0 0; -1 -2 -1")
sobelHorizontal = np.matrix("1 2 1; 0 0 0; -1 -2 -1")

#f - Laplace(f); this Canny is not the Canny detector as per wikipedia
cannySharpen = np.matrix("0 -1 0; -1 5 -1; 0 -1 0")
#canny = convolve2d(gaussDenoise, cannySharpen)

##filtered images

encodedCanny = convolve2d(encodedImage, cannySharpen)
encodedCannyLaplace1 = convolve2d(encodedCanny, laplace1)
encodedLaplace1 = convolve2d(encodedImage, laplace1)
encodedLaplace2 = convolve2d(encodedImage, laplace2)
encodedLaplace1Canny = convolve2d(encodedLaplace1, cannySharpen)
encodedSobelVert = convolve2d(encodedImage, sobelVertical)
encodedSobelHoriz = convolve2d(encodedImage, sobelHorizontal)
encodedSobel = convolve2d(encodedSobelVert, sobelHorizontal)

##graph

fig = figure()
#fig.subplots_adjust(hspace=0.1, wspace=0.5)
ax = fig.add_subplot(231)
ax.set_title("encoded")
imshow(encodedImage, interpolation='nearest', 
            cmap=cm.gray,
            origin='upper',
            vmax=spatial_vmax,
            vmin=spatial_vmin,
            )

ax = fig.add_subplot(232)
ax.set_title("difference (abs)")
imshow(imageAbsDiff, interpolation='nearest', 
            cmap=cm.gray,
            origin='upper',
            vmax=spatial_vmax,
            vmin=spatial_vmin,
            )

ax = fig.add_subplot(233)
ax.set_title("Laplace1(encoded)")
imshow(encodedLaplace1, interpolation='nearest', 
            cmap=cm.gray,
            origin='upper',
            vmax=spatial_vmax,
            vmin=spatial_vmin,
            )

ax = fig.add_subplot(234)
ax.set_title("fft(orig)-fft(enc) (log)")
imshow(imageFFTDiff, interpolation='nearest', 
            cmap=cm.gray,
            origin='upper',
            #vmax=fft_vmax,
            #vmin=fft_vmin,
            )

#images's PSD
ax = fig.add_subplot(235)
ax.set_title("PSD")
imgWidth=originalImage.shape[1] #np.matrix has matrix-like dimensions, height first
assert imgWidth % 2 == 0, "Image width must be even"

#frames per second == 25; however the ticmarks won't fit on X axis, using default Fs=2
freq = originalImage.size * 25
psdShape = lambda m: np.reshape(m, m.size)

plt.psd(psdShape(originalImage), NFFT=imgWidth, label="original")
plt.psd(psdShape(encodedImage), NFFT=imgWidth, label="encoded")
plt.psd(psdShape(encodedImage-originalImage), NFFT=imgWidth, label="difference")
#filtered images' PSD
plt.psd(psdShape(encodedLaplace1), NFFT=imgWidth, label='laplace1')
#plt.psd(psdShape(encodedCanny), label='canny')
#plt.psd(psdShape(encodedSobel), label='sobel')
leg = ax.legend(('original', 'encoded', 'difference', 'laplace1'),
           'upper center', shadow=True)

ax = fig.add_subplot(236)
ax.set_title("CannySharpen(Laplace1(enc))")
imshow(encodedCannyLaplace1, interpolation='nearest', 
            cmap=cm.gray,
            origin='upper',
            #vmax=spatial_vmax,
            #vmin=spatial_vmin,
            )

show()

