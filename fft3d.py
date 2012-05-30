#!/usr/bin/python
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import subplot_class_factory
from matplotlib import cm
from matplotlib.pyplot import imshow
import pylab
import numpy as np
import mpmath
import Image
import sys

mpmath.dps = 5

if len(sys.argv) < 2:
    print "Usage: fft3d.py image_file"
    sys.exit(1)

im = Image.open(sys.argv[1])#.convert("YCbCr")
x, y = im.size

fig = pylab.figure()
Subplot3D = subplot_class_factory(Axes3D)
ax = Subplot3D(fig, 121)
#ax = Axes3D(fig)
X = np.arange(-x/2, x/2)
Y = np.arange(-y/2, y/2)
X, Y = np.meshgrid(X, Y)
xn, yn = X.shape

immatrix = np.reshape(im.getdata(0), (y,x))
imfft_abs = np.abs(np.fft.fftshift(np.fft.fft2(immatrix)))
imfft_log = np.log(imfft_abs)
imfft_sqrt = np.sqrt(imfft_abs)

# can comment out one of these

ax.plot_surface(X, Y, imfft_log, rstride=x/32, cstride=y/32, cmap=cm.jet)
#ax.plot_surface(X, Y, imfft_sqrt, rstride=2, cmap=cm.gray)
#ax.plot_wireframe(X, Y, imfft_sqrt, rstride=2, cstride=2)

ax = fig.add_subplot(122)
imshow(imfft_log, cmap=cm.gray, interpolation='nearest', extent=(-x/2, x/2-1, -y/2, y/2-1))
pylab.show()
