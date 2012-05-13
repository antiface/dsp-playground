#!/usr/bin/python
import sys
import os
import tempfile
import shutil
import subprocess
import traceback

import threading
from threading import Thread, Semaphore
import Queue

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.cm as cm

from psd_integrate import integrateSubblockFreqEnergy
from optparse import OptionParser

class FFMpegFrameGrabber(object):
    """ Grabs videoframes to directory using ffmpeg. """
    
    def __init__(self, dir=".", ffmpeg="/usr/bin/ffmpeg"):
        """ Sets up grabber.
        @param dir: stores frames into this directory
        @param ffmpeg: absolute path to ffmpeg binary
        """
        self.ffmpeg = ffmpeg
        self.dir = dir
    
    def grabFrames(self, videofname, start, timeLen, prefix):
        """ Invokes ffmpeg to grab frames. Returns list of frames.
        
        @param videofname: filename of the video to grab from
        @param start: start time, in seconds
        @param timeLen: time span, in seconds
        @param prefix: filenames of grabbed frames will have this prefix
        @return: list of frame filenames, with directory prepended
        """
        argv = [self.ffmpeg, "-i", videofname, "-ss", str(start), "-t", str(timeLen),
            "-y", "-deinterlace", os.path.join(self.dir, prefix + "%06d.png")]
        
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        res = p.wait()
        
        if res != 0:
            raise RuntimeError("FFmpeg framegrabbing failed with errorcode %s:" 
                "\nstdout:\n%s\nstderr:\n%s" % (res, stdout, stderr))
        
        return [os.path.join(self.dir, f) for f in os.listdir(self.dir)]
        

class FramelistPSDIntegrator(object):
    """ Reads given list of image filenames (frames) and integrated over their
    PSD vectors.
    """
    
    def __init__(self, frameFnames, blocksize=16, freqrange=0, logscale=True,
            imgscale=None):
        """ Sets up parameters for PSD integrator.
        
        @param frameFnames: list of image filenames
        @param blocksize: blocksize to do PSD over
        @param freqrange: resulting integrated vector will have this many
        highest frequencies, 0 means all frequencies
        @param imgscale: scale every image to resolution given by tuple (width,
        height) if not already in given resolution
        @param logscale: whether to log() values before integrating them
        """
        self.frameFnames = frameFnames
        self.blocksize = blocksize
        self.freqrange = freqrange
        self.logscale = logscale
        self.imgscale = imgscale
        
    def computePSD(self):
        """ Runs the computation. Result is stored in self.integrVector.
        """
        self.integrVector = None
        self.completeDomain = None
        
        for fname in self.frameFnames:
            #note: this will skip subblocks smaller than blocksize on edges
            (imgPsdVector, self.completeDomain) = \
                integrateSubblockFreqEnergy(fname, freqrange=self.freqrange,
                blocksize=self.blocksize, logscale=self.logscale,
                imgscale=self.imgscale)
            
            #returned vector may be shorter than freqrange and adding arrays
            #with different dimensions cannot work
            if self.integrVector is None:
                self.integrVector = imgPsdVector
            else:
                self.integrVector += imgPsdVector
                
        return (self.integrVector, self.completeDomain)
        

class MessageSink(Thread):
    """ Synchronized message sink for debug/processing messages. Use
    self.msgQueue.put() to insert messages.
    """
    
    class Finished:
        """ Putting instance of this class into queue signalizes that sink
        has to finish processing.
        """
        pass
    
    def __init__(self, out):
        """ Queue messages atop a file-like object out, print them into that
        object as they arrive.
        """
        super(MessageSink, self).__init__()
        self.msgQueue = Queue.Queue()
        self.out = out
        self.finished = False
        
    def run(self):
        """ Processes and prints messages to file-like object chosen at creation
        until an instance of MessageSink.Finished is put into queue.
        """
        while not self.finished:
            message = self.msgQueue.get()
            if isinstance(message, MessageSink.Finished):
                self.finished = True
            else:
                print >> self.out, message
            self.msgQueue.task_done()
        

class PSDIntegrateThread(Thread):
    """ Thread that processes a single videofile. Grabs frames into a temporary
    directory, then runs PSD integration over them. Semaphore, if supplied,
    serves to bound maximum number of instances to run at a given time.
    
    PSD integration result is stored in self.integrVector upon thread join.
    """
    
    def __init__ (self, videoFname, opts, semaphore=None, msgSink=None):
        """
        @param semaphore: threading.Semaphore instance set to maximum allowed
        instances of this thread to run simultaneously
        @param msgSink: MessageSink instance
        """
        super(PSDIntegrateThread, self).__init__()
        
        self.videoFname = videoFname
        self.opts = opts
        self.semaphore = semaphore
        self.errors = []
        self.validData = False
        self.integrVector = None
        self.msgSink = msgSink
        
    def run(self):
        """ Runs the computation. If semaphore was given at initialization, will
        wait on this semaphore.
        """
        #assert at most N given running thread instances
        if self.semaphore:
            self.semaphore.acquire()
            
        imgscale = None
        if opts.imgscale:
            try:
                #the double assignment ensures there are exactly two integers
                imgscale = (a, b) = tuple([int(x) for x in opts.imgscale.split("x")])
            except ValueError:
                if self.msgSink:
                    self.msgSink.msgQueue.put("Could not parse imgscale argument")
        
        tmpDir = None
        try:
            tmpDir = tempfile.mkdtemp(prefix="video_psd_", dir=self.opts.basetmp)
            
            grabber = FFMpegFrameGrabber(tmpDir, self.opts.ffmpeg)
            
            if self.msgSink:
                self.msgSink.msgQueue.put("Grabbing frames of %s to %s" % \
                    (self.videoFname, tmpDir))
                
            frameFnames = grabber.grabFrames(self.videoFname, self.opts.start,
                self.opts.time, "frame_")
            
            if self.msgSink:
                self.msgSink.msgQueue.put(
                    "Integrating PSD over frames' subblocks for %s" % self.videoFname)
            
            integrator = FramelistPSDIntegrator(frameFnames,
                self.opts.blocksize, self.opts.freqrange, self.opts.logscale,
                imgscale)
            
            (self.integrVector, self.completeDomain) = integrator.computePSD()
            
            self.validData = True
            
            if self.msgSink:
                self.msgSink.msgQueue.put("Resulting integrated vector for %s: %s" % \
                    (self.videoFname, self.integrVector))
    
        except:
            errInfo = (self.videoFname,) + sys.exc_info()[:2]
            errStr = "Error processing %s - %s: %s" % (errInfo)
            self.errors.append(errStr)
            if self.msgSink:
                self.msgSink.msgQueue.put("!!!" + errStr)
                trace = traceback.format_exc()
                self.msgSink.msgQueue.put(trace)
            
        finally:
            if self.semaphore:
                self.semaphore.release()
            #remove grabbed frames unless there is request to keep them
            if not self.opts.keeptmp and tmpDir:
                shutil.rmtree(tmpDir, True)
            else:
                if self.msgSink:
                    self.msgSink.msgQueue.put(
                        "Not cleaning grabbed frames stored in directory " + tmpDir)

if __name__ == "__main__":
    usage = "Usage: %prog -s NN -t MM [options] videofile1 [videofile2] ... [videofileN]"
    desc = "Plots the graph of integrated frequency energies in frames of video"
    
    parser = OptionParser(usage=usage, description=desc)
    parser.add_option("-s", "--start", action="store", type="int", dest="start",
        help="Start time (secs)")
    parser.add_option("-t", "--time", action="store", type="string", dest="time",
        help="Time length (secs)")
    parser.add_option("-f", "--ffmpeg", action="store", type="string", dest="ffmpeg",
        help="Absolute path to ffmpeg binary", default="/usr/bin/ffmpeg")
    parser.add_option("-T", "--basetmp", action="store", type="string", dest="basetmp",
        help="Directory where to create temp dir for frames (default /tmp)", default="/tmp")
    parser.add_option("-b", "--blocksize", action="store", type="int", dest="blocksize",
        help="Subblock size (default 16)", default=16)
    parser.add_option("-k", "--keeptmp", action="store_true",  dest="keeptmp",
        help="Do not delete temporary dir with frames", default=False)
    parser.add_option("-r", "--freqrange", action="store", type="int", dest="freqrange",
        help="Highest N frequencies to show (default: all, max: blocksize/2+1)", default=0)
    parser.add_option("-l", "--nologscale", action="store_false", dest="logscale",
        help="Do not integrate over log scaled PSD values", default=True)
    parser.add_option("-S", "--imgscale", action="store", dest="imgscale",
        help="Scale video frames to this resolution", metavar="WIDTHxHEIGHT")
    parser.add_option("-c", "--concurrency", action="store", type="int", dest="concurrency",
        help="Max number of threads to use (default 1)", default=1)
    #fps is hardwired to 25, since we'd need to parse ffmpeg output for fps info
    parser.add_option("-p", "--pixelfreq", action="store_true", dest="pixelfreq",
        help="Plot domain as 'pixel frequency' (=x*y*fps)", default=False)
    
    (opts, args) = parser.parse_args()
    if not (args and opts.time and opts.start):
        parser.print_help()
        sys.exit(1)
        
    srcFiles = args
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    
    concurrencySem = Semaphore(opts.concurrency)
    msgSink = MessageSink(sys.stdout)
    msgSink.start()
    
    psdThreads = []
    
    for srcFile in srcFiles:
        psdThread = PSDIntegrateThread(srcFile, opts,
            concurrencySem, msgSink)
        psdThreads.append(psdThread)
        psdThread.start()
        
    #msgSink.msgQueue.put("Existing threads: %s, concurrency: %d" % \
    #    (threading.enumerate(), opts.concurrency))
    
    for psdThread in psdThreads:
        psdThread.join()
        
    msgSink.msgQueue.put(MessageSink.Finished())
    msgSink.join()
    
    #computations finished, let's plot!
    validThreads = []
    for psdThread in psdThreads:
        if psdThread.validData:
            validThreads.append(psdThread)

    if validThreads: #at least one successful video
        legendStrings = []
        #there may be less freqs than opts.freqrange
        completeDomain = validThreads[0].completeDomain
        domain = completeDomain[:len(validThreads[0].integrVector)]
        print "Domain: ", domain
        
        for psdThread in validThreads:
            if opts.pixelfreq:
                ax.plot(domain, psdThread.integrVector)
            else:
                ax.plot(psdThread.integrVector)
            legendStrings.append(psdThread.videoFname)

        ax.set_title("PSD: highest %d frequencies (total %d)" % \
            (len(domain), len(completeDomain)))
        leg = ax.legend(legendStrings, 'upper right', shadow=True)

        #correlations to first video
        ax = fig.add_subplot(122)
        firstVector = np.array(validThreads[0].integrVector)

        for psdThread in validThreads:
            integrVector = psdThread.integrVector
            print "Plotting correlation for vector", integrVector
            corrValues = []
            for i in range(len(integrVector)):
                c = np.corrcoef(firstVector[i:], integrVector[i:])[0][1]
                corrValues.append(c)

            if opts.pixelfreq:
                ax.plot(domain, corrValues)
            else:
                ax.plot(corrValues)
        ax.set_title("Correlation of N highest frequencies to PSD of first video")
        leg = ax.legend(legendStrings, 'lower right', shadow=True)

        plt.show()
    else:
        print "No successful computation"

# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
