#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure, show, imshow

from scipy.signal import convolve2d

import math
import sys

import Image

def imageToRedMatrix(filename, crop=None):
    """ Returns red band of image as a matrix """
    #SL6 PIL aborts when coversion to YCbCr is done, but doing stuff on red
    #channel is good enough for demonstration purposes
    im = Image.open(filename)
    if crop:
        im = im.crop(crop)
    im2 = im#.convert("YCbCr")
    (x, y) = im2.size 
    luma = im2.getdata(0)

    return np.reshape(luma, (y, x))

i1 = imageToRedMatrix(sys.argv[1])

delta = 0.125
x = y = np.arange(-4.0, 4.0, delta)
X, Y = np.meshgrid(x, y)
gaussKernel = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
print gaussKernel.shape

gaussDenoise = np.matrix(
"2.0 4.0 5.0 4.0 2.0;" + \
"4.0 9.0 12.0 9.0 4.0;" + \
"5.0 12.0 15.0 12.0 5.0;" + \
"4.0 9.0 12.0 9.0 4.0;" + \
"2.0 4.0 5.0 4.0 2.0"
)*1.0/159

dumbLopass = np.matrix(
"0  0  0  0  0  0  0  0  0;" + \
"0  0  0  0  0  0  0  0  0;" + \
"0  0  1  1  1  1  1  0  0;" + \
"0  0  1  1  1  1  1  0  0;" + \
"0  0  1  1  1  1  1  0  0;" + \
"0  0  1  1  1  1  1  0  0;" + \
"0  0  1  1  1  1  1  0  0;" + \
"0  0  0  0  0  0  0  0  0;" + \
"0  0  0  0  0  0  0  0  0" 
)*1.0/25

dumbHipass = np.matrix(
"1  1  1  1  1  1  1  1  1;" + \
"1  1  1  1  1  1  1  1  1;" + \
"1  1  0  0  0  0  0  1  1;" + \
"1  1  0  0  0  0  0  1  1;" + \
"1  1  0  0  0  0  0  1  1;" + \
"1  1  0  0  0  0  0  1  1;" + \
"1  1  0  0  0  0  0  1  1;" + \
"1  1  1  1  1  1  1  1  1;" + \
"1  1  1  1  1  1  1  1  1" 
)*1.0/56

sobelHorizontal = np.matrix("1 2 1; 0 0 0; -1 -2 -1")
sobelVertical = np.matrix(
    " 1  0 -1;" + \
    " 2  0 -2;" + \
    " 1  0 -1"
)

gaussed = convolve2d(i1, gaussKernel)
gaussDenoised = convolve2d(i1, gaussDenoise)
lopassed = convolve2d(i1, dumbLopass)
hipassed = convolve2d(i1, dumbHipass)
sobeledHoriz = convolve2d(i1, sobelHorizontal)
sobeledVert = convolve2d(i1, sobelVertical)

#simple function that turns Fourier spectrum to real numbers by coefs' absolute
#value and returns it in log scale; the +1 is to prevent log() going into
#negative numbers for nicer visualization
fftLog = lambda image: np.log(1+np.abs(np.fft.fftshift(np.fft.fft2(image))))
fftImage = fftLog(i1)
#print np.min(fftImage), np.max(fftImage)

#print "gaussKernel from bivariate_normal"
#print np.log(gaussKernel)


############### graph ##################

fig = figure()

#common color map for image presentation
icmap = cm.gray

ax = fig.add_subplot(241)
ax.set_title("Sobel Vertical filter")
#imshow(i1, cmap=icmap)
imshow(sobeledVert, cmap=icmap, interpolation='nearest')

ax = fig.add_subplot(245)
ax.set_title("^ FFT(Sobel Vert) ^")
imshow(fftLog(sobeledVert), cmap=icmap, interpolation='nearest')


ax = fig.add_subplot(242)
ax.set_title("Sobel Horizontal filter")
imshow(sobeledHoriz, cmap=icmap, interpolation='nearest')

ax = fig.add_subplot(246)
ax.set_title("^ FFT(Sobel Horiz) ^")
imshow(fftLog(sobeledHoriz), cmap=icmap, interpolation='nearest')

ax = fig.add_subplot(243)
ax.set_title("Gaussian blur via GaussDenoise")
imshow(gaussDenoised, cmap=icmap, interpolation='nearest')
#imshow(hipassed, cmap=icmap)

ax = fig.add_subplot(247)
ax.set_title("^ Fourier spectrum ^")
imshow(fftLog(gaussDenoised), cmap=icmap, interpolation='nearest')

ax = fig.add_subplot(244)
ax.set_title("Gaussian kernel - bivariate normal")
imshow(gaussKernel, cmap=icmap, interpolation='nearest')
#imshow(hipassed, cmap=icmap)

ax = fig.add_subplot(248)
ax.set_title("^ FFT of Gaussian kernel ^")
imshow(fftLog(gaussKernel), cmap=icmap, interpolation='nearest')


#ax = fig.add_subplot(234)
#ax.set_title("Simple (imperfect) lopass")
#imshow(lopassed, cmap=icmap)

show()
