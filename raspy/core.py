from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import csv
import logging
import matplotlib
import re
import time


matplotlib.use('WXAgg')

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.optimize as opt
import scipy.signal as signal
import scipy.ndimage.filters as filters

from raspy.call import *
from raspy.utils import *

from math import floor
from numpy import convolve, nan

np.seterr(all='ignore')
from raspy.reportBuilder import ReportBuilder

VERBOSE = True
VERBOSER = False
VERBOSEST = False

logger = logging.getLogger('RaspLab')


class Rasper(object):
    """ This class will contain or wrap everything needed to load, analyze, 
        and otherwise run raspy.

        @todo neaten up some doxygen stuff
        @todo refactor Raspy to be purpose agnostic
        @todo move rat call stuff into subclass
        @todo separtate classify() into different methods for each phase
    
        Example:
            from raspy import raspy
    
            # Create a new raspy object for a single file
            test1 = raspy('C:\\Users\\cflanigan\\Desktop\\ratchat\\test1.wav', 200000)
       
            # Create a list of arrays that describe signals found in the file
            test1.isolateCalls()
      
            #pull out the second sequence of major values
            majorSignals = test1.maskedMajorValues[1]
        
            # plot the frequency of the major signal
            #   0: stdDev
            #   1: frequency
            #   2: amplitude
            plt.plot(majorSignals[:,1])
    """
    
    def __init__(self, fileName, arr, source, offset=0, fs=250000,
                 callback=logger.info):
        """ Create a new object and open the file given.
            @param [in] fileName the name of the .wav file to analyze
            @param [in] fs sampling frequency of the file  
            @param [in] yieldingCallback a function to call periodically
                throughout execution
        """
        
        if os.path.exists(os.path.splitext(fileName)[0] + '.csv'):
            os.remove(os.path.splitext(fileName)[0] + '.csv')
            
        ## Callback function to be called intermittently
        self.callback = callback
        
        # A list of calls in the file
        self.calls = None
        
        # Mask valuess from the first coarse masking
        self.coarseMaskOld = None
        
        # A rough masking of the calls based only on amplitude and standard dev
        self.coarseMaskVals = None
        
        # The data from this Raspy's file
        self.fileData = np.asanyarray(arr)  # , dtype=np.float32)
        
        # This Raspy's f = Noneile's name without the full path
        self.fileName = None
                    
        ## The name of the .wav file to analyze with the full path
        self.fullFileName = os.path.abspath(fileName)
        
        # Spectrogram frequency bins
        self.freqs = None
        
        # The sampling frequency of the file
        self.fs = fs
        
        # The spectrogram after gabor filters have been applied
        self.gaboredSignal = None
        
        # The spectrogram after gabor masking has occured
        self.gaborizedSpectrogram = None
        
        # The mask used for gabor masking
        self.gaborMask = None
        
        # The amplitudes from the most recent peak fitting
        self.majorAmplitudes = None
        
        # The frequencies from the most recent peak fitting
        self.majorFrequencies = None
        
        # The standard deviation from the most recent peak fitting
        self.majorStdDev = None
        
        # A list of the major amplitudes separated by call
        self.maskedAmplitudes = None
        
        # A list of the major frequencies separated by call
        self.maskedFrequencies = None
        
        # A list of the major standard deviations separated by call
        self.maskedStdDev = None
        
        # A list of times separated by call
        self.maskedTimes = None
        
        # The background noise of the file used in spectral subtraction
        self.noise = None
        
        # The first round of standard deviation fitting
        self.oldDev = None
        
        # A spectrogram containing only the data from peak fitting.
        self.peaks = None
        
        # The original unfiltered spectrogram
        self.rawSpectrogram = None
        
        # The major values which have not been separated from each other
        self.signal = None
        
        # The most recent step in the spectrographic analysis
        self.spectrogram = None
        
        # The phases of the original spectrogram
        self.spectroPhase = None
        
        # The spectrogram after undergoing spectral subtraction
        self.subbedSpectrogram = None
        
        # The times of the spectrogram
        self.times = None

        # The source event list
        self.source = source

        # The offset of the beginning of this file
        self.offset = offset
                
        self.times = self.freqs = self.spectrogram = self.peaks = []
        self.signal = []
            
    @property
    def length(self):
        """ Return the length of this object's calls """
        return (len(self.fileData)-1)/self.fs
            
            
    
    def _fitPeak(self, x0, y0, wholeSample=False):
        """ Fit a gaussian curve to a dataset and return the parameters of 
            the curve.
            @param [in] x the times for the data to fit to
            @param [in] y the frequencies for the data to fit to
            @param [in] wholeSample whether or not to use the entire dataset, or to
                                    fit to the peak and the adjacent data
        """
        
        idx = y0 > 0
        y = y0[idx]
        x = x0[idx]
        maxIdx = y.argmax()
        if maxIdx < 3: maxIdx = 3
        if maxIdx + 3 > len(x): maxIdx = len(x) - 3
        maxIdx0=y0.argmax()
        if maxIdx0 < 3: maxIdx0 = 3
        xPrime = x0[maxIdx0-1:maxIdx0+2]
        yPrime = y0[maxIdx0-1:maxIdx0+2]
        if wholeSample:
            mean = np.average(x, weights=y)
            dev = np.sqrt(np.average((x-mean)**2.0, weights=y))
        else:
            mean = np.average(xPrime, weights=yPrime)
            dev = np.sqrt(np.average((xPrime-mean)**2.0, weights=yPrime))
        if maxIdx < 0: 
            return (1e5, mean, y.max())
        p0=(dev, mean, y[maxIdx])
        bounds = ([0.0, 0.0, 0.0], np.inf)
        if len(y) < 5: return p0
        try:
            if wholeSample:
                popt, _ = opt.curve_fit(gauss, x, y, p0)
            else:
                popt, _ = opt.curve_fit(gauss, xPrime, yPrime, p0,ftol=0.5)
        except RuntimeError as e:
            p1 = (1e2, x[y.argmax()], y[maxIdx])
            try:
                if wholeSample:
                    popt, _ = opt.curve_fit(gauss, x, y, p1)
                else:
                    popt, _ = opt.curve_fit(gauss, xPrime, yPrime, p1)
            except RuntimeError:
                if y.max()/y.mean() > 10:
                    return list(p1)
                return list(p0)
        except ValueError as e:
            print('raspy: 113', e)
            raise e
        except Exception as e:
            print('raspy: 116', e)
            raise e
        
        return list(popt)
    
    def classify(self, caHigh=100, caLow=10, gaborMaskHigh=1000,
                       gaborMaskLow=100, gaborSize=10, majorStdMax=4000,
                       maxGapTime=25, minSig=0.1, ngabor=10, overlap=0.1,
                       syllableGap=10, wholeSample=True, windowSize=256, 
                       wings=3, startTimeOffset=0):
        """ Classify the calls within this object's file.
            @param [in] caHigh high value for the coarse amplitude filter
            @param [in] caLow low value for the coarse amplitude filter
            @param [in] gaborMaskHigh upper threshold for the gabor mask
            @param [in] gaborMaskLow lower threshold for the gabor mask
            @param [in] gaborSize size of the kernel for the gabor filter
            @param [in] majorStdMax max stdDev for the coarse filter
            @param [in] maxGapTime the maximum time between signals to consider
                                   them one call or two
            @param [in] minSig minimum signal amplitude for both sharpenPeaks calls
            @param [in] ngabor number of convulutions to use for the gabor filter
            @param [in] overlap overlap for the spectrogram
            @param [in] syllableGap the maximum time between signal before the
                                    signal is considered multiple subcalls.
            @param [in] startTimeOffset ______
            @param [in] wholeSample if true use the whole sample in sharpen peaks
            @param [in] windowSize size of the window to use in the spectrogram
            @param [in] wings size of the wings for sharpenPeaks 
        """
              
        t0 = time.time()       
        self.callback()
         
        # Process the file, and define the calls within the file.
        self.isolateCalls(caHigh=caHigh, caLow=caLow, gaborSize=gaborSize,
                gaborMaskHigh=gaborMaskHigh, gaborMaskLow=gaborMaskLow,
                majorStdMax=majorStdMax, maxGapTime=maxGapTime, minSig=minSig, 
                ngabor=ngabor, overlap=overlap, wholeSample=wholeSample, 
                windowSize=windowSize, wings=wings)
        
        self.callback(self.fullFileName)
        
        # Pull the path and file name apart, and toss aside the subfile suffix
        regMod = re.search('((.*) - (\d+))(\.wav)', self.fullFileName, flags=re.IGNORECASE)
        if regMod is None:
            self.fullFileName = self.fullFileName[:-4]
            self.pathName, self.fileName = os.path.split(self.fullFileName)
            fileIndex = None
        else:
            self.fullFileName = regMod.group(2)
            self.pathName, self.fileName = os.path.split(regMod.group(1))
            fileIndex = regMod.group(3)

        # Create a list of calls for every set of call signals that are not empty
        idx = [sum(freqs > 0) > 4 and freqs.any() for freqs in self.maskedFrequencies]
        idx = np.arange(0, len(self.maskedFrequencies))[idx]
        callArgs = [{'majorFrequencies':self.maskedFrequencies[i], 
                     'majorTimes':self.maskedTimes[i],
                     'majorAmplitudes':self.maskedAmplitudes[i],
                     'majorStdDev':self.maskedStdDev[i],
                     'parent':self, 'caLow':caLow, 'stdMax':majorStdMax} for i in idx]
        
        callMask = [np.logical_and(self.maskedFrequencies[i], 
                            self.maskedStdDev[i] < majorStdMax).sum() > 5 for i in idx]
        self.callback()
        
        self.calls = []
        for i, args in enumerate(callArgs):
            if not callMask[i]:
                continue
            self.calls.append(Call(**args))
            self.callback()
        
        # offsets to align plots and pseudocolor meshes
        reportBuilder = ReportBuilder(self.times, self.freqs, self.rawSpectrogram,
                                      self.fileData, defaultDir=self.fullFileName, 
                                      startTimeOffset=startTimeOffset)
        self.callback()

        reuseExisting = True
        if fileIndex is None or int(fileIndex) == 0 :
            reuseExisting = False
        reportBuilder.startLog('overall.csv', [('Call Num', 'Call Num'), 
                                               ('Type', 'Total Type'), 
                                               ('Start Time', 'Total StartTime'),
                                               ('Duration', 'Total Duration'), 
                                               ('RMS Volume', 'Total Volume'),
                                               ('Fit', 'Fit'), ('Avg Freq', 'Total AvgFreq'),
                                               ('Freq Bandwidth', 'Total FreqBw')],
                               reuseExisting=reuseExisting)
        self.callback()

        # if there are no calls present, plot the spectrogram, save, and exit
        if len(self.calls) == 0:
            reportBuilder.savePcolor(self.fileName + ' no calls detected.png', 
                                     self.rawSpectrogram, self.times, self.freqs,
                                     self.fileName + ' no calls detected')
            return reportBuilder
            
        self.calls = [c for c in self.calls if len(c.times) > 0]

        if self.source is not None:
            self.source.parent.subchannels[0].warningId = []

        # fit and classify every call
        for i, c in enumerate(self.calls):
            self.callback()
            if self.source is not None:
                self.source.dataset.addWarning(c.times[0] + self.offset,
                                               c.times[-1] + self.offset)
            c.classify(reportBuilder, self, syllableGap)
            
        self.calls = [c for c in self.calls if len(c.times) > 1]

        callTypes = [t.callType for t in self.calls]
        self.callback(', '.join(callTypes))
        
        s = str(self.calls)
        
        if VERBOSEST:
            filePre = self.fullFileName + '\\1 ' + self.fileName
            reportBuilder.savePcolor(filePre + ' rawSpectro.png', 
                                     np.sqrt(self.rawSpectrogram),
                       self.times, self.freqs, 'Raw Spectrogram')
            
            filePre = self.fullFileName + '\\6 ' + self.fileName
            reportBuilder.savePcolor(filePre + ' spectro.png', np.sqrt(self.spectrogram),
                       self.times, self.freqs, 'Fully Processed Spectrogram')
            
            filePre = self.fullFileName + '\\2 ' + self.fileName
            reportBuilder.savePcolor(filePre + ' peaks.png', np.sqrt(self.peaks),
                       self.times, self.freqs, 'Peaks')
            
            filePre = self.fullFileName + '\\3 ' + self.fileName
            reportBuilder.savePcolor(filePre + ' noise.png', np.sqrt(self.noise),
                       self.times, self.freqs, 'Noise')

            filePre = self.fullFileName + '\\4 ' + self.fileName
            reportBuilder.savePcolor(filePre + ' subbedSpectro.png', 
                                     np.sqrt(self.subbedSpectrogram),
                       self.times, self.freqs, 'SubbedSpectrogram')

            filePre = self.fullFileName + '\\5 ' + self.fileName
            reportBuilder.savePcolor(filePre + ' gabormask.png', np.sqrt(self.gaborMask),
                       self.times, self.freqs, 'Gabor Filter Mask')

            reportBuilder.makeScatterPlot(title='Major Frequencies', 
                                          yLabel='frequencies (kHz)',
                                          yData=self.majorFrequencies,
                                          fileName=self.fullFileName + '\\2.1 ' \
                                          + self.fileName + 'maskedFrequency.png')

            reportBuilder.makeScatterPlot(title='Major Standard Deviations', 
                                          yLabel='frequencies (kHz)',
                                          yData=self.majorStdDev,
                                          fileName=self.fullFileName + '\\2.2 ' \
                                          + self.fileName + 'majorStdDev.png')

            reportBuilder.makeScatterPlot(title='Major Amplitudes', 
                                          yLabel='amplitude (units)',
                                          yData=self.majorAmplitudes,
                                          fileName=self.fullFileName + '\\2.3 ' \
                                          + self.fileName + 'majorAmplitudes.png')
        self.callback()

        nPerSeg = windowSize*(1-overlap)

        for i, c in enumerate(self.calls):
            if len(c.times) == 0:
                self.callback('call %d contains no times, not saving' % (i + 1))
                continue
            self.callback('saving %d of %d' %(i + 1, len(self.calls)))

            c.saveReport(reportBuilder, self, i)        # Save a csv summarizing the calls in the file

        prevCalls = []
        if os.path.exists(self.fullFileName + '.csv'):
            self.callback("reading %s" % (os.path.splitext(self.fullFileName)[0] + '.csv'))
            with open(self.fullFileName + '.csv', 'r+b') as wholeFile:
                reader = csv.reader(wholeFile, delimiter=b',')
                rows = [row for row in reader][2:]
                for row in rows:
                    prevCalls += [row[0]]*int(row[1])

        # write a summary CSV, consider removing this
        self.callback('writing ' + self.fullFileName + '.csv')
        with open(self.fullFileName + '.csv', 'w+b') as wholeFile:
            wholeCsv = csv.writer(wholeFile)
            csvCalls = ['composite' if 'composite' in x else x for x in (callTypes + prevCalls)]
            wholeCsv.writerow(['Total Calls', len(csvCalls)])
            wholeCsv.writerow(['Call Type', 'Count'])
            wholeCsv.writerows([[cType, (csvCalls).count(cType)] for cType in set(csvCalls)])
            
        self.callback("file finished in %.1f minutes" % ((time.time() - t0)/60.0,))
        return reportBuilder
            
    def clearArrays(self):
        """ Delete the large spectrogram arrays to reduce memory usage """
        for key in dir(self):
            value = getattr(self, key)
            if isinstance(value, np.ndarray) and value.size > 2**20:
                setattr(self, key, np.empty((0,)*len(value.shape),
                                            dtype=value.dtype))

    def coarseMask(self, coarseAmpLow=0.1, coarseAmpHigh=0.5, 
                   majorStdDevMax=4000, maxGapTime=25):
        """ Roughly find the location of signals in a file, by checking if the 
            amplitude is not below a certain threshold, using a hysteresis-like
            process, and that the standard deviation of the signal's peak hasn't
            been too high for too long; then adds extra buffers to the windows.
            @param [in] coarseAmpLow the low threshold for hysteresis filtering 
                               for the amplitude filter
            @param [in] coarseAmpHigh the high threshold for hysteresis filtering
                                for the amplitude filter
            @param [in] majorStdDevMax the maximum allowable standard deviation
                                 for a signal to be valid
            @param [in] maxGapTime the maximum time between signals to consider
                             them one signal or two """
        
        # filter out any signal that is below the coarseAmpLow level, if it 
        # cannot be connected to a part of the signal that is above the 
        # coarseAmpHigh level
        ampFilter = hysteresis(
            signal.wiener(self.majorAmplitudes, 9),
            coarseAmpLow,
            coarseAmpHigh)
        
        # filter any signal with a standar deviation greater than the majorStdDevMax
        stdFilter = self.majorStdDev < majorStdDevMax
        # stdFilter = (signal.convolve(stdFilter, [1.0]*25, mode='same')/25) > 0.88
        
        combinedFilter = ampFilter & stdFilter

        windowTics = int((maxGapTime*1e-3) // (self.times[1]-self.times[0])) + 1
        windowBlock = np.ones(windowTics, dtype=bool)
        
        # create the complete filter
        self.coarseMaskVals = (
            np.convolve(combinedFilter[:-windowTics+1 or None], windowBlock, mode='full')
            + np.convolve(combinedFilter[windowTics-1:], windowBlock, mode='full')
        )
        
        return self.coarseMaskVals

    def convolve(self, size=10, nconv=4, mode='gabor'):
        """ Convolve a kernel or set of kernels over the spectrogram and return
            the magnitude of the activation vector.
            @param [in] size the size (squared) of the kernel to use
            @param [in] nconv the number of convolutions to use
            @param [in] mode the type of kernel to use during convolution
                        'gabor' indicates a gabor filter """
        convolvedSignals = np.zeros((self.spectrogram.shape), dtype='float32')
        size = int(size)

        for i in xrange(0, nconv):
            angle = np.pi * (0.75 * i / nconv - 0.375)
            if mode == 'gabor':
                kernel = gabor(size, size, 0.75, angle, 0, 0.5, 0.5)
            convolvedSignals[:, :] += np.power(
                np.maximum(
                    signal.convolve2d(self.rawSpectrogram, kernel, mode='same'),
                    np.zeros(self.noise.shape)),
                2)

        self.gaboredSignal = np.sqrt(convolvedSignals / nconv) * (10 ** 0.5)
    
    def createSpectrogram(self, windowSize=256, overlap=0.1):
        """ Create a spectrogram with the data. 
            @param [in] windowSize the length of the samples used to calculate the
                        FFTs
            @param [in] overlap the fraction of the window that should overlap 
                        between windows 
        """
        window = 'triang'
        noverlap = (windowSize-1) if overlap == 1 else (windowSize*overlap)
        # triang, hamming, hann, parzen
        self.freqs, self.times, self.spectrogram = signal.spectrogram(
            self.fileData.astype(np.float64),
            fs=self.fs,
            window=window,
            nperseg=windowSize,
            noverlap=noverlap,
            mode='psd',
        )
        _, _, self.spectroPhase = signal.spectrogram(
            self.fileData.astype(np.float64),
            fs=self.fs,
            window=window,
            nperseg=windowSize,
            noverlap=noverlap,
            mode='angle',
        )
        
        self.peaks = np.zeros(self.spectrogram.shape)
        self.rawSpectrogram = self.spectrogram.copy()
    
    
    def gaborize(self, size=10, nconv=10, low=5, high=25):
        """ Convolve a set of gabor filters over the spectrogram, taking the
            average response as the response vector and using hysteresis-like 
            double threshold filter to mask signals.
            @param [in]  size: width/height of the kernel to convolve
            @param [in]  nconv: the number of times to convolve the kernel
            @param [in]  low: the low threshold for the hysteresis filter
            @param [in]  high: the high threshold for the hysteresis filter  
        """
        self.gaboredSignal = compiled_convolve(self.rawSpectrogram, self.spectrogram.shape, self.noise.shape, size,
                                               nconv, True)
        # kernel = np.array([[0,0,0,0,0,0,1,0,0,0,0,0,0],
        #                    [0,0,0,0,0,1,1,1,0,0,0,0,0],
        #                    [0,0,0,1,1,1,1,1,1,1,0,0,0],
        #                    [1,1,1,1,1,1,1,1,1,1,1,1,1],
        #                    [0,0,0,1,1,1,1,1,1,1,0,0,0],
        #                    [0,0,0,0,0,1,1,1,0,0,0,0,0],
        #                    [0,0,0,0,0,0,1,0,0,0,0,0,0]])
        self.callback('applying gabor filter (hysteresis check)')
        self.gaborMask = hysteresis(self.gaboredSignal, low, high, padding=None)
        self.gaborizedSpectrogram = (self.gaborMask>0)*self.subbedSpectrogram
        self.spectrogram = self.gaborizedSpectrogram
    
    
    def isolateCalls(self, caHigh=0.1, caLow=0.01, gaborMaskHigh=1000,
                           gaborMaskLow=100, gaborSize=10, majorStdMax=4000,
                           maxGapTime=25, minSig=0.1, ngabor=10, overlap=0.1,
                           wholeSample=True, windowSize=256, wings=3):
        """ Isolate the sections of the file that contain a call. Exposes 
            keyword arguments for other methods.
            @param [in] caHigh high value for the coarse amplitude filter
            @param [in] caLow low value for the coarse amplitude filter
            @param [in] gaborMaskLow lower threshold for the gabor mask
            @param [in] gaborMaskHigh upper threshold for the gabor mask
            @param [in] gaborSize size of the kernel for the gabor filter
            @param [in] majorStdMax max stdDev for the coarse filter
            @param [in] maxGapTime the maximum time between signals to consider
                        them one signal or two
            @param [in] minSig minimum signal amplitude for both sharpenPeaks calls
            @param [in] ngabor number of convolutions to use for the gabor filter
            @param [in] overlap overlap for the spectrogram
            @param [in] wholeSample if true use the whole sample in sharpen peaks
            @param [in] windowSize size of the window to use in the spectrogram
            @param [in] wings size of the wings for sharpenPeaks 
        """

        self.callback('creating spectrogram')
        self.createSpectrogram(windowSize=windowSize, overlap=overlap)
        
        self.callback('applying wiener filter')
        self.spectrogram = signal.wiener(self.spectrogram, 3)
        
        self.callback('applying spectral subtraction')
        self.spectralSubtraction()

        self.callback('applying peak sharpening')
        self.sharpenPeaks(minSig=minSig, wings=wings, wholeSample=wholeSample)
        
        self.callback('applying coarse mask filter')
        self.coarseMask(coarseAmpLow=caLow, 
                        coarseAmpHigh=caHigh, 
                        majorStdDevMax=majorStdMax,
                        maxGapTime=maxGapTime)
        
        self.coarseMaskOld = self.coarseMaskVals
        self.oldDev = self.majorStdDev
        
        self.callback('applying gabor filter')
        self.gaborize(size=gaborSize, nconv=ngabor, low=gaborMaskLow, high=gaborMaskHigh)
        
        self.callback('reapplying peak sharpening')
        self.sharpenPeaks(minSig=minSig, wings=wings, wholeSample=wholeSample)
        
        self.callback('reapplying coarse mask filter')
        self.coarseMask(coarseAmpLow=caLow, 
                        coarseAmpHigh=caHigh, 
                        majorStdDevMax=majorStdMax)
        
        self.callback('finished first processing phase.')
        
        # create a  list of the indices of coarseMaskVals that are rising or falling edges
        indices = np.flatnonzero(np.diff(self.coarseMaskVals))
        first_slice = 0 if self.coarseMaskVals[0] else 1

        (
            self.maskedTimes, self.maskedAmplitudes, self.maskedStdDev,
            self.maskedFrequencies
        ) = tuple(np.split(array, indices)[first_slice::2] for array in (
            self.times, self.majorAmplitudes, self.majorStdDev,
            self.majorFrequencies
        ))
            
        return self.maskedTimes, self.maskedStdDev, self.maskedFrequencies, \
            self.maskedAmplitudes
        
    def sharpenPeaks(self, minSig=0.01, wholeSample=False, wings=3):
        """ Locate the primary frequency of the call, and isolate the frequency,
            amplitude, and standard deviation of the peak.  Further, isolate all
            peaks below a certain threshold. 
        """
        x = self.freqs
        # import scipy.ndimage.filters as filters
        yList = filters.gaussian_filter(self.spectrogram.copy(),2)
        # yList = self.spectrogram.copy()
        y = yList[:,0]
        self.signal = []
        self.peaks = np.zeros(self.peaks.shape)

        msgTemplate = 'sharpening peaks - {:.1%}'
        for i in xrange(len(self.times)):
            if i % 1000 == 0:
                self.callback(msgTemplate.format(i/len(self.times)))
                
            if y.max() > minSig:
                popt = self._fitPeak(x, y, wholeSample=wholeSample)
            else:
                popt = [10000,0,0]
                
            if popt[2] > 1000:
                pass
            
            popt = np.abs(popt)
                
            self.signal += [popt]
            # isErrored = False
            # j = 0
            while y.max() > minSig:# and not isErrored:
                maxIdx = max(min(len(y)-2, y.argmax()), 1)
                if y[maxIdx-1:maxIdx+2].sum() == 0:
                    # mean = 0
                    # dev = 1e4
                    continue
                # mean = np.average(x[maxIdx-1:maxIdx+2], weights=y[maxIdx-1:maxIdx+2])
                # dev = np.sqrt(np.average((x[maxIdx-1:maxIdx+2]-mean)**2.0,
                #                          weights=y[maxIdx-1:maxIdx+2]))
                self.peaks[:,i] += gauss(x, popt[0], popt[1], popt[2])
                index_starter = max(maxIdx-wings,0)
                index_ender = maxIdx+wings+1
                y[index_starter:index_ender] = np.zeros(len(y[index_starter:index_ender])) #### COULDN'T THIS JUST BE SIZE: index_ender - index_starter
            
            if i == yList.shape[1] - 1:
                continue
            
            del y
            y = yList[:,i+1]
            
        self.callback(msgTemplate.format(1))
        # print(self.signal)
        # print(self.signal[0].dtype)
        self.majorStdDev = np.empty(len(self.signal), dtype=np.float64)
        self.majorFrequencies = np.empty(len(self.signal), dtype=np.float64)
        self.majorAmplitudes = np.empty(len(self.signal), dtype=np.float64)
        for j, x in enumerate(self.signal):
            self.majorStdDev[j] = x[0]
            self.majorFrequencies[j] = x[1]
            self.majorAmplitudes[j] = x[2]


        # self.majorStdDev = np.array([x[0] for x in self.signal])
        # self.majorFrequencies = np.array([x[1] for x in self.signal])
        # self.majorAmplitudes = np.array([x[2] for x in self.signal])
        self.signal = np.array(self.signal)
        
    
    def spectralSubtraction(self):
        """ Calculate the noise as the average amplitude of each frequency across
            the whole file.  Subtract that noise from the spectrogram and return
            the max of 0 and the difference.
        """
        # define the noise as the 50th percentile of every given frequency.
        # at some point, play around with this number, 50 may not be optimal
        self.noise = np.percentile(self.spectrogram, 50, axis=1)
        
        # smooth out the noise curve. It's expected to be roughly trapezoidal, 
        # so applying smoothing should help it avoid shaving off peaks.
        self.noise = signal.wiener(self.noise)

        self.noise = np.array([self.noise]*len(self.spectrogram[1])).transpose()
        subbed = self.spectrogram - self.noise
        self.spectrogram = (subbed > 0)*subbed
        self.subbedSpectrogram = self.spectrogram
        
        
