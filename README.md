dsp-playground
==============

Few sample matplotlib/scipy scripts for experimenting with digital signal processing. Matplotlib and scipy make it very easy to add/visualize "primitives" like convolutions, operations in frequency domain and filters.

Matplotlib, numpy, scipy and Python Imaging Library must be installed.

Note: the conversion to YCbCr was commented in the sources since it causes abort due to memory corruption in some Python Imaging Library versions. Thus instead of working over luma channel it works over red channel.

