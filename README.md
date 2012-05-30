dsp-playground
==============

## About

Few sample matplotlib/scipy scripts for experimenting with digital signal processing. Matplotlib and scipy make it very easy to add/visualize "primitives" like convolutions, operations in frequency domain and filters.

Matplotlib, numpy, scipy and Python Imaging Library must be installed. The fft3d.py script requires also [mpmath](https://code.google.com/p/mpmath/).

Note: the conversion to YCbCr was commented in the sources since it causes abort due to memory corruption in some Python Imaging Library versions. Thus instead of working over luma channel it works over red channel.

## Sample invocation

Few sample images are included (I used those when measuring effects of various options and filters in video encoders):

Script showing differences in original and encoded image, plus some sample filters (edge detectors):

    python fourier_video_classify.py -o baader_original-12s-256x256.png -e baader_mencoder_fd_deint-12s-256x256.png

An analysis of frequencies in multiple videos (an ad-hoc method using linear correlation of integrated power spectral density vectors). Requires ffmpeg installed in path for frame grabbing:

    python --start=1 --time=1 source_video.mp4 encoded_video.mp4

Some screenshots saved from the matplotlib graphs are in `sample_outputs` directory.
