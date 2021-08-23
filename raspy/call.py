from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import iterable
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.signal as signal

import os

from raspy.utils import makeDir


class Call(object):
    """ Lets the user run the call splitting more easily. 
        @todo convert tree into listlike
        @todo refactor Call to have a less confusing name
    """
    
    
    def __init__(self, fileName=None, majorTimes=None, majorFrequencies=None, 
                 majorStdDev=None, majorAmplitudes=None, parent=None, 
                 caLow=0.5, stdMax=1e3):
        """ @todo document Call.__init__ 
            if this is not copying from a file, copy values passed through the
            parameters, otherwise load them in from the file.
        """
        
        ## The filtered major amplitudes for this call
        self.amplitudes = None
        
        ## The type of rat call that this Call is
        self.callType = None
        
        ## The name of the file that this belongs to
        self.fileName = None
        
        ## The filtered major frequencies for this call
        self.frequencies = None
        
        ## true if this is a short call (length <6mS)
        self.isShort = None
        
        ## The major amplitudes from this Call's parent Raspy for its times
        self.majorAmplitudes = None
        
        ## The major frequencies from this Call's parent Raspy for its times
        self.majorFrequencies = None
        
        ## The major standard deviation from this Call's parent Raspy for its times
        self.majorStdDev = None
        
        ## The times for this Call's major values
        self.majorTimes = None
        
        ## This Call's parent Raspy
        self.parent = None
        
        ## The root node the call segment tree
        self.root = None
        
        ## The filtered standard deviations for this call
        self.std = None
        
        ## The filtered times for this call
        self.times = None
        
        if fileName is None:
                  
            self.majorTimes = majorTimes
            self.majorFrequencies = majorFrequencies
            self.majorAmplitudes = majorAmplitudes
            self.majorStdDev = majorStdDev
            pass
        else:
            self.fileName = fileName
            data = np.loadtxt(fileName, delimiter=',')
            self.majorTimes = data[:,0]
            self.majorStdDev = data[:,1]
            self.majorFrequencies = data[:,2]
            self.majorAmplitudes = data[:,3]
            
        if parent is None:
            pass
        
        # validate the points in regards to std dev and amplitude
        self.times, self.frequencies, self.std, self.amplitudes = \
              validPoints(majorTimes, majorFrequencies, majorStdDev, 
                          majorAmplitudes, parent.times, 
                          parent.majorFrequencies, caLow, stdMax)
        
        # return early if there are too few valid times
        # otherwise, define the root node
        if len(self.times) <= 1:
            return
        elif len(self.times) == 2:
            self.root = Line(self.times[0], self.times[-1], self.frequencies[0], 
                             self.frequencies[-1], self, True, True)
            self.parent = parent
            self.isShort = True
            return
        else:
            self.root = Line(self.times[0], self.times[-1], self.frequencies[0], 
                             self.frequencies[0], self, True, True)
            
            self.parent = parent
            self.isShort = False
            return
        
    def __iter__(self):
        """ Return this call's leaf nodes as a list. """
        return [x for x in self.root]
    
    def __repr__(self):
        """ @todo document Call.__repr__ """
        if hasattr(self, 'subcalls'):
            name = ''
            for i in range(len(self.subcalls)):
                name += 'call %d:\n' % (i+1)
                name += repr(self.subcalls[i]) + '\n'
            return name
        else:
            if hasattr(self, 'root'):
                return "Call: %s" % (self.root)
            else:
                return 'Call: []'
    
    def __call__(self, x, *args):
        """ @todo document Call.__call__ """
        if hasattr(self, 'subcalls'):
            if iterable(x):
                return np.array([self(i) for i in x])
            else:
                for s in self.subcalls:
                    if x >= s.root.startTime and x <= s.root.endTime:
                        return s(x)
                return np.NAN 
        else:
            if len(args) == 0:
                return self.root(x)
            else:
                return self.root(x, *args)
    
    def __getitem__(self, index):
        """ @todo document Call.__getitem__ """
        return self.root[index]
        
    @property
    def spectro(self):
        """ Get a portion of the spectrogram from this call's parent raspy.  
            The spectrogram it gets will be cropped around this call.
        """
        startIdx = np.abs(self.parent.times-self.times[0]).argmin()
        endIdx = np.abs(self.parent.times-self.times[-1]).argmin()
        height, width = self.parent.spectrogram[:,startIdx:endIdx].shape
        freqs = np.array([self.parent.freqs]*width)
        times = np.array([self.spectroTimes]*height)
        return times.flatten(), freqs.T.flatten(), \
            self.parent.spectrogram[:,startIdx:endIdx].flatten()
        
    @property
    def spectroTimes(self):
        """ Get the times for this call's spectrogram. """
        startIdx = np.abs(self.parent.times-self.times[0]).argmin()
        endIdx = np.abs(self.parent.times-self.times[-1]).argmin()
        return self.parent.times[startIdx:endIdx]
    
    def applyParameters(self, params, isLeft, isRight):
        """ Apply a new set of parameters to this call's lines. """
        params = np.append(self.root.startTime,
                           np.insert(params, -1, self.root.endTime))
            
        self.root.applyParameters(params, False, False)
        
    def assemble(self, fs, spectro):
        """ Determine the location of trills/quads and jumps, then create a 
            segment tree to represent it.
        """
        if self.isShort:
            self.fitFreqs()
        else:        
            startTimes = [seg.startTime for seg in self.root]
            endTimes = [seg.endTime for seg in self.root]
        
            # determine the location of trills
            xn = np.linspace(self.times[0], self.times[-1], 
                             np.round((self.times[-1]-self.times[0])*(fs)).astype(np.int))
            yn = interp.interp1d(self.times, self.frequencies, kind='linear')(xn)
            wn = interp.interp1d(self.times, self.amplitudes, kind='linear')(xn)
            
            trillidx = trillIsolate(xn, yn, wn, fs, spectro)
                
            if len(trillidx) > 0:
                trillstart = trillidx[:,0]
                trillend = trillidx[:,1]
            else:
                trillstart = np.array([])
                trillend = np.array([])
        
            # determine the location of jumps
            jumpMask = np.abs(np.diff(self.frequencies)) > 5000
            jumpidx = np.argwhere(jumpMask)
            
            for i in range(len(jumpidx)):
                jumpidx[i] = np.abs(xn-self.times[jumpidx[i]]).argmin() + 1
            
            for i in range(len(trillstart)):
                jumpidx = jumpidx[np.logical_or(jumpidx < trillstart[i], jumpidx > trillend[i])]
            
            # stitch the parts of the call together
            segments = []
            currentIdx = 0
            while len(trillstart) > 0 or len(jumpidx) > 0:
                if len(jumpidx) > 0 and jumpidx[0] >= len(yn) - 1:
                    jumpidx = []
                    continue
                # if there are no trills then only add jumps
                if len(trillstart) == 0:
                    
                    # add a line before the jump if the current index is
                    # before the jump
                    if currentIdx < jumpidx[0]:
                        st = xn[currentIdx]
                        sf = yn[currentIdx]
                        et = xn[jumpidx[0]]
                        ef = yn[jumpidx[0]]
                        nextSeg = Line(startTime=st, endTime=et, startFreq=sf, 
                                       endFreq=ef, parent=None, 
                                       isLeft=(currentIdx==0), isRight=False)
                        segments.append(nextSeg)
                        
                    st = xn[jumpidx[0]]
                    sf = yn[jumpidx[0]]
                    ef = yn[jumpidx[0]+1]
                    nextSeg = Jump(startTime=st, startFreq=sf, 
                                   endFreq=ef, parent=None)
                    segments.append(nextSeg)
                    currentIdx = jumpidx[0]
                    jumpidx = jumpidx[1:]
                    
                # if there are no jumps then only add trills
                elif len(jumpidx) == 0:
                    
                    # add a line before the quad if the current index is
                    # before the quad
                    if currentIdx < trillstart[0]:
                        st = xn[currentIdx]
                        sf = yn[currentIdx]
                        et = xn[trillstart[0]]
                        ef = yn[trillstart[0]]
                        nextSeg = Line(startTime=st, endTime=et, startFreq=sf, 
                                       endFreq=ef, parent=None, 
                                       isLeft=(currentIdx==0), isRight=False)
                        segments.append(nextSeg)
                        
                    st = xn[trillstart[0]]
                    sf = yn[trillstart[0]]
                    et = xn[trillend[0]]
                    ef = yn[trillend[0]]
                    nextSeg = Quadratic(startTime=st, endTime=et, startFreq=sf, 
                                        endFreq=ef, parent=None, isLeft=False, 
                                        isRight=(trillend[0]==len(xn)), curve=5000)
                    segments.append(nextSeg)
                    currentIdx = trillend[0]
                    trillstart = trillstart[1:]
                    trillend = trillend[1:]
                
                # otherwise add both
                else:
                    if trillstart[0] > jumpidx[0]:
                        
                        # add a line before the jump if the current index is
                        # before the jump
                        if currentIdx < jumpidx[0]:
                            st = xn[currentIdx]
                            sf = yn[currentIdx]
                            et = xn[jumpidx[0]]
                            ef = yn[jumpidx[0]]
                            nextSeg = Line(startTime=st, endTime=et, startFreq=sf, 
                                           endFreq=ef, parent=None, 
                                           isLeft=(currentIdx==0), isRight=False)
                            segments.append(nextSeg)
                            
                        st = xn[jumpidx[0]]
                        sf = yn[jumpidx[0]]
                        ef = yn[jumpidx[0]+1]
                        nextSeg = Jump(startTime=st, startFreq=sf, 
                                        endFreq=ef, parent=None)
                        segments.append(nextSeg)
                        currentIdx = jumpidx[0]
                        jumpidx = jumpidx[1:]
                    else:
                        
                        # add a line before the quad if the current index is
                        # before the quad
                        if currentIdx < trillstart[0]:
                            st = xn[currentIdx]
                            sf = yn[currentIdx]
                            et = xn[trillstart[0]]
                            ef = yn[trillstart[0]]
                            nextSeg = Line(startTime=st, endTime=et, startFreq=sf, 
                                           endFreq=ef, parent=None, 
                                           isLeft=(currentIdx==0), isRight=False)
                            segments.append(nextSeg)
                            
                        st = xn[trillstart[0]]
                        sf = yn[trillstart[0]]
                        et = xn[trillend[0]]
                        ef = yn[trillend[0]]
                        nextSeg = Quadratic(startTime=st, endTime=et, startFreq=sf, 
                                            endFreq=ef, parent=None, isLeft=False, 
                                            isRight=(trillend[0]==len(xn)), curve=5000)
                        segments.append(nextSeg)
                        currentIdx = trillend[0]
                        trillstart = trillstart[1:]
                        trillend = trillend[1:]
                        
            if currentIdx < len(xn) - 1:
                st = xn[currentIdx]
                sf = yn[currentIdx]
                et = xn[-1]
                ef = yn[-1]
                nextSeg = Line(startTime=st, endTime=et, startFreq=sf, endFreq=ef, 
                               parent=None, isLeft=(currentIdx==0), isRight=True)
                segments.append(nextSeg)
                
            for i in range(len(segments)-1):
                if type(segments[i]) is Jump and type(segments[i+1]) is Jump:
                    segments[i+1].isLeft = segments[i].isLeft
                    segments[i] = None
            
            segments = [s for s in segments if s is not None]
            
            if type(segments[0]) is Jump: 
                segments[0] = None
                segments[1].isLeft = True
                segments[1].startTime = self.times[0]
            if type(segments[-1]) is Jump:
                segments[-1] = None
                segments[-2].isRight = True
                segments[-2].endTime = self.times[-1]
            
            segments = [s for s in segments if s is not None]
            
            durs = np.array([s.duration for s in segments])
            if (durs < 0).any():
                pass
                
            while len(segments) > 1:
                newSegs = []
                for i in range(0, len(segments)-1, 2):
                    seg = Segment(segments[i], segments[i+1], 
                                  i == 0, i+2 == len(segments), None)
                    segments[i].parent = seg
                    segments[i+1].parent = seg
                    newSegs.append(seg)
                if len(segments) % 2 == 1:
                    newSegs.append(segments[-1])
                segments = newSegs
            
            self.root = segments[0]
                
            if not self.root[-1].isRight or not self.root[0].isLeft:
                pass
            self.root.parent = self 
            # fit the model to the data
            x = np.linspace(self.times[0], self.times[-1], 1000)
            self.fitFreqs()
            for seg in self.root:
                if type(seg) is Quadratic and np.abs(seg.curve) < 0.1:
                    seg = Line(startTime=seg.startTime, endTime=seg.endTime, 
                               startFreq=seg.startFreq, endFreq=seg.endFreq,
                               parent=seg.parent, isLeft=seg.isLeft, isRight=seg.isRight)
                    if seg.parent.isLeftTime(seg.times[1]):
                        seg.parent.leftChild = seg
                    else:
                        seg.parent.rightChild = seg
                        
                
                if not self.root[-1].isRight or not self.root[0].isLeft:
                    pass
            self.fitFreqs()
            
            # keep fitting and merging until no more segments can be merged together,
            # this may be too aggressive.
            unfitted = self(xn)
            plt.plot(self.times, self(self.times))
            if self.merge() and len(self.root) > 1:
                self.fitTimes()
                timeFitted = self(xn)
                self.fitFreqs()
                fullyFitted=self(xn)
                pass
            else:
                self.fitTimes()
                self.fitFreqs()
            
            if not self.root[-1].isRight or not self.root[0].isLeft:
                pass
            
        # classify the call
        self._classify()
        
    def balance(self):
        """ Remove any nodes that are None, and adjust the tree to fill the gap.
            @note this isn't balancing the tree(yet(???)).  No real need to.
        """
        if self.root is Line:
            return
        if np.array([x is None for x in self.root]).any() \
                and not isinstance(self.root, Line):
            if self.root.leftChild is None:
                self.root = self.root.rightChild
            elif self.root.rightChild is None:
                self.root = self.root.leftChild
            self.root.balance()
            for i in range(len(self.root)-1):
                self.root[i+1].startTime = self.root[i].endTime
                
            st = np.array([x.startTime for x in self.root])
            et = np.array([x.endTime for x in self.root])
            if not ((st[1:] - et[:-1] == 0).all()):
                raise ValueError('times do not line up')

    def classify(self, reportBuilder, rasper, syllableGap):
        """ Run classification on this call, do report building. """

        # scrub spikes from the signal
        spikes = []
        for j in xrange(len(self.times)):
            windowedFreqs = self.frequencies[max(0, j - 4):min(len(self.times) - 1, j + 3)].copy()
            windowedFreqs -= self.frequencies[j]
            windowedFreqs = (np.abs(windowedFreqs) < 5e3) * 1
            spikes.append(windowedFreqs.sum() > 2)

        if np.array(spikes).sum() < 3:
            self.callType = 'too few valid points'
            return

        self.times = self.times[spikes]
        self.frequencies = self.frequencies[spikes]
        self.amplitudes = self.amplitudes[spikes]
        self.std = self.std[spikes]

        if self.times[-1] - self.times[0] <= 0.006:
            self.callType = 'short'
            return

        # separate parts of calls that have more than 10mS of dead air
        # between them
        timing = np.diff(self.times)
        timing[0] = 1
        timing[-1] = 1
        gaps = np.argwhere(timing > syllableGap * 1e-3)
        if len(gaps) > 2:
            gaps[1:] += 1

            self.subcalls = []

            # create a new set of subcalls split where there are gaps
            for j in xrange(len(gaps) - 1):

                # skip subcalls that are too short to consider calls
                if gaps[j + 1] - gaps[j] < 4:
                    continue

                # create a new Call with the indices from j to j+1
                idx = np.arange(gaps[j], gaps[j + 1])
                s = Call(majorTimes=self.times[idx],
                         majorFrequencies=self.frequencies[idx],
                         majorStdDev=self.std[idx],
                         majorAmplitudes=self.amplitudes[idx],
                         parent=rasper)

                # skip this iteration if the Call s was not made correctly
                if not hasattr(s, 'root'):
                    continue

                # assemble the fit for this subcall and add it to its parent
                fs = 1 / (rasper.times[1] - rasper.times[0])
                xLims, yLims = reportBuilder.getLimits(self)  # PJS: Clear this and next line up
                s.assemble(fs, ((rasper.times[xLims[0]:xLims[1]]
                                 - reportBuilder.timePseudocolorOffset),
                                (rasper.freqs[yLims[0]:yLims[1]]
                                 - reportBuilder.freqPseudocolorOffset),
                                rasper.spectrogram[yLims[0]:yLims[1],
                                xLims[0]:xLims[1]] ** 0.5))
                self.subcalls.append(s)

            # define composite call types and report
            self.callType = 'composite - '
            for s in self.subcalls:
                self.callType += s.callType + ', '
            self.callType += ''

            reportBuilder.makeCompositeCallPlot(self, clearFirst=False)


        else:
            # report out a single call
            fs = 1 / (rasper.times[1] - rasper.times[0])
            reportBuilder.makeSingleCallPlot(self, spectrogram=rasper.spectrogram,
                                             useSqrt=True, clearFirst=False)

        rasper.callback(self.callType)

        if hasattr(self, 'subcalls'):
            for s in self.subcalls:
                plt.plot(s.times, s(s.times))
        else:
            plt.plot(self.times, self(self.times))
        plt.cla()

    def fit(self, useSpectro=False):
        """ Fit the segmented line to this call's data.  Adjusts both the time
            and frequencies of the segments.
            if using the spectrogram, gather the points to fit to from the spectrogram,
            otherwise, use the major values
        """
        if useSpectro:
            t, f, s = self.spectro
            t = t[s > 0]
            f = f[s > 0]
            s = s[s > 0]
        else:
            t = self.times
            f = self.frequencies
            s = self.amplitudes
        st = np.array([x.startTime for x in self.root])
        et = np.array([x.endTime for x in self.root])
        if not ((st[1:] - et[:-1] == 0).all()):
            raise ValueError('times do not line up')
        
        # define the parameters and bounds, and fit the call to the data
        params = self.root.getParams()
        params = np.append(params[1:-2], params[-1])
        highBounds, lowBounds = self.root.getBounds()
        lowBounds = tuple(np.append(lowBounds[1:-2], lowBounds[-1]))
        highBounds = tuple(np.append(highBounds[1:-2], highBounds[-1]))
        popt, pcov = opt.curve_fit(self, t, f, params, 
                                   sigma=1/s, 
                                   bounds=(lowBounds, highBounds))
        self.applyParameters(popt, True, True)
        if type(self.root) is Segment:
            self.root.updateTimes()
        
        st = np.array([x.startTime for x in self.root])
        et = np.array([x.endTime for x in self.root])
        if not ((st[1:] - et[:-1] == 0).all()):
            raise ValueError('times do not line up')
        return np.abs(self.frequencies - self(self.times))
    
    def fitFreqs(self, useSpectro=False):
        """ Fit the segmented line to this call's data.  Adjusts only the
            frequencies of the segments.
        """
        if useSpectro:
            t, f, s = self.spectro
            t = t[s > 0]
            f = f[s > 0]
            s = s[s > 0]
        else:
            t = self.times
            f = self.frequencies
            s = self.amplitudes
        st = np.array([x.startTime for x in self.root])
        et = np.array([x.endTime for x in self.root])
        if not ((st[1:] - et[:-1] == 0).all()):
            self.root.updateTimes()
        
        params = self.root.getFreqs()
        low = [1.5e4]*len(params)
        high = [2.5e5]*len(params)
        if len(params) > len(self.times):
            return np.inf
        popt, pcov = opt.curve_fit(self.onlyFreqs, t, f, 
                               params, ftol=0.5, xtol=0.5, gtol=0.5) #, bounds=(low, high))
        try:
            self.root.applyFrequencies(popt)
        except:
            pass
        
        st = np.array([x.startTime for x in self.root])
        et = np.array([x.endTime for x in self.root])
        if not ((st[1:] - et[:-1] == 0).all()):
            raise ValueError('times are not lining up')
        
        return np.square(self.frequencies - self.onlyFreqs(self.times))
    
    def fitTimes(self, useSpectro=False):
        """ Fit the segmented line to this call's data.  Adjusts only the
            frequencies of the segments.
        """
        try:
            if len(self.root) == 1:
                return self.frequencies - self(self.times)
            if useSpectro:
                t, f, s = self.spectro
                t = t[s > 0]
                f = f[s > 0]
                s = s[s > 0]
            else:
                t = self.times
                f = self.frequencies
                s = self.amplitudes
            st = np.array([x.startTime for x in self.root])
            et = np.array([x.endTime for x in self.root])
            if not ((st[1:] - et[:-1] == 0).all()):
                self.root.updateTimes()
            
            params = self.getTimes()
            low = [1.5e4]*len(params)
            high = [2.5e5]*len(params)
            for seg in self.root:
                if seg.endTime < seg.startTime:
                    pass
            if len(params) > len(self.times):
                return np.inf
            if np.abs(self.onlyTimes(self.times,*params).sum()) == np.inf:
                self.onlyTimes(self.times, *params)
            popt, pcov = opt.curve_fit(self.onlyTimes, t, f, params, 
                                 bounds=(self.root.startTime, self.root.endTime), 
                                 ftol=0.5, xtol=0.5, gtol=0.5)
            popt.sort()
            self.root.applyTimes(popt, isLeft=True, isRight=True)
            self.root.updateTimes()
            if params == self.root.getTimes():
                self.root.applyTimes(popt)
            
            self.root.updateTimes()
            st = np.array([x.startTime for x in self.root])
            et = np.array([x.endTime for x in self.root])
            if not ((st[1:] - et[:-1] == 0).all()):
                raise ValueError('times are not lining up')
            for seg in self.root:
                if seg.endTime < seg.startTime:
                    pass
            return np.square(self.frequencies - self(self.times))
        except:
            return np.square(self.frequencies - self(self.times))
    
    def getTimes(self):
        """ return the times of the root node """
        return self.root.getTimes()[1:-1]
    
    def merge(self):
        """ Merge adjacent segments based on 
            A) similar slopes in sequence into a single line
            B) short series of segments with a linearly changing slope
               into a quadratic/polynomial
        """
        isChanged = False
        
        # merge consecutive jumps
        i = 0
        while i < len(self.root)-1:
            if isinstance(self.root[i], Jump) and isinstance(self.root[i+1], Jump):
                self.root[i].endFreq = self.root[i+1].endFreq
                self.root[i].isRight = self.root[i+1].isRight
                self.root[i+1] = None
                self.balance()
                i = 0
                isChanged = True
            i += 1
        
        # remove jumps of less than 5kHz
        i = 0
        while i < len(self.root):
            node = self.root[i]
            if abs(node.endFreq - node.startFreq) < 5e3 and isinstance(node, Jump):
                if i == 0:
                    self.root[1].isLeft = True
                if i == len(self.root)-1:
                    self.root[-2].isRight = True
                self.root[i] = None
                self.balance()
                i -= 1
                isChanged = True
            i += 1
        
        # Merge consecutive lines with similar slopes
        i = 0
        while i < len(self.root)-1:
            slopeFactor = self.root[i].slope/self.root[i+1].slope
            if slopeFactor < 3.0 and slopeFactor > 1/3 \
                    and type(self.root[i]) is not Quadratic \
                    and type(self.root[i+1]) is not Quadratic:
                self.root[i].merge(self.root[i+1], type='line')
                self.root[i].isRight = self.root[i+1].isRight
                self.root[i+1] = None
                self.balance()
                i = 0
                isChanged = True
            i += 1
            
        # Merge short lines with the next line if the next line isn't short
        i = 0
        while i < len(self.root)-1:
            if self.root[i].duration < 0.005 and self.root[i+1].duration > 0.01 \
                    and type(self.root[i]) is not Jump \
                    and type(self.root[i]) is not Quadratic:
                self.root[i].merge(self.root[i+1], type='line')
                self.root[i].isRight = self.root[i+1].isRight
                if self.root[i].duration < self.root[i+1].duration:
                    self.root[i] = None
                else:
                    self.root[i+1] = None
                self.balance()
                i -= 1
                isChanged = True
            i += 1
            
        # Prune any short segments that are less than 3ms in length and with a
        # Frequency change less than 5kHz
        if self.root[0].duration < 0.003 \
                and np.abs(self.root[0].dFrequency) < 5000 \
                and len(self.root) > 1 \
                and not isinstance(self.root[0], Jump) \
                and type(self.root[0]) is not Quadratic:
            self.root[0].merge(self.root[1])
            self.root[1] = None
            self.balance()
            isChanged = True
            
        if self.root[0] is Jump:
            self.root[0] = None
            self.balance()
            isChanged = True
            
        if self.root[-1].duration < 0.003 \
                and np.abs(self.root[-1].dFrequency) < 5000 \
                and len(self.root) > 1 \
                and type(self.root[-1]) is not Quadratic:
            self.root[-2].merge(self.root[-1])
            self.root[-1] = None
            self.balance()
            isChanged = True
            
        if self.root[-1] is Jump:
            self.root[-1] = None
            self.balance()
            isChanged = True
            
        # prune any segments that have less than 3 points within them
        i = 0
        while i < len(self.root) - 1:
            if self.root[i].npoints <= 3 and not isinstance(self.root[i], Jump):
                self.root[i].merge(self.root[i+1])
                if type(self.root[i+1]) is not Jump: self.root[i].isRight = self.root[i+1].isRight
                self.root[i] = None
                self.balance()
                i -= 1
                isChanged = True
            i += 1
            
        if self.root[-1].npoints <= 3:
            self.root[-2].merge(self.root[-1])
            self.root[-1] = None
            self.balance()
            isChanged = True
            
        # Prune any jumps that are at the start or end of the call
        if type(self.root[0]) is Jump:
            self.root[1].isLeft = True
            self.root[0] = None
            self.balance()
            
        if type(self.root[-1]) is Jump:
            self.root[-2].isRight = True
            self.root[-1] = None
            self.balance()
            
        for i in range(len(self.root)):
            if isinstance(self.root[i], Quadratic) and self.root[i].curve < 1000:
                self.root[i] = Line(self.root[i].startTime, self.root[i].endTime, 
                                    self.root[i].startFreq, self.root[i].endFreq, 
                                    self.root[i].parent, self.root[i].isLeft, 
                                    self.root[i].isRight)
            
        return isChanged
    
    def onlyFreqs(self, x, *args):
        """ Call this object using only parameters for frequencies. """
        if len(args) == 0:
            return self(x)
        if len([i for i in self.root if i is None]) != 0:
            self.balance()
        return self(x, *self.root.paramsFromFreqs(args))
    
    def onlyTimes(self, x, *args):
        """ Call this object using only parameters for frequencies. """
        if len(args) == 0:
            return self(x)
        args = np.array(args)
        args.sort()
        args = np.array(args)
        args.sort()
        args = self.root.paramsFromTimes(args)
        if len([i for i in self.root if i is None]) != 0:
            self.balance()
        
        offsets = np.array([0]+[type(seg) is Quadratic and 3 or 2 for seg in self.root]).cumsum()
        
        for i in range(len(self.root)):
            if type(self.root[i]) is Jump:
                args[offsets[i]+2] = args[offsets[i]]
            
        args = tuple(args)
        return self(x, *args)
    
    def prune(self, test_index):
        """" SEE: Segment.prune() """
        self.root.prune()

    def saveReport(self, reportBuilder, rasper, i):
        """ Create and save a plot of the spectrogram of and around the call
            and save it to a directory named after the file
        """
        # plt.cla()
        if 'composite' in self.callType:
            trueCallType = 'composite'
        else:
            trueCallType = self.callType

        makeDir(os.path.join(rasper.fullFileName, trueCallType))
        plotTitle = rasper.fileName + ' call:%d' % (i,)
        plotFileName = os.path.join(rasper.fullFileName, rasper.fileName + '.%d.png' % (i,))
        reportBuilder.makeCallSpectrogramPlot(self, title=plotTitle, fileName=plotFileName)

        # Add annotation of the call type and parameters and a plot of the
        # line and save it to a subdirectory within the file'd directory
        # named after the call type.
        plotFileName = os.path.join(
            rasper.fullFileName,
            trueCallType,
            rasper.fileName + '.%d.png' % (i,)
        )
        reportBuilder.makeAnnotatedCallPlot(self, fileName=plotFileName, clearFirst=False)

        # Save a csv with the second image of the slope parameters
        slopeFileName = os.path.join(rasper.fullFileName, rasper.fileName + '.%d.csv' % (i,))
        slopeFileParams = [('Call Number', "%i" % (i,)), ('Fs (kHz)', "Seg StartFreq"),
                           ('Fe (kHz)', "Seg EndFreq"), ('dF/dt (kHz/ms)', "Seg Slope"),
                           ('t (ms)', "Seg Duration"), ("Call Type", "Total Type")]
        reportBuilder.makeCallReport(self, slopeFileParams, slopeFileName)

        externalParams = {'Call Num': i}
        reportBuilder.addLog(self, 'overall.csv', externalParams=externalParams)

        outputWavFileName = rasper.fileName + '.%d.wav' % (i)
        reportBuilder.makeCallWav(self, outputWavFileName)
    
    def splitAt(self, idx):
        """ Split this call at the time corresponding to the index idx. """
        currentNode = self.root
        time = (self.times[idx]+self.times[idx+1])/2
        while type(currentNode) is Segment:
            if time < currentNode.middleTime:
                currentNode = currentNode.leftChild
            else:
                currentNode = currentNode.rightChild
        segIdx = currentNode.index
        self.root[segIdx].splitAt(idx+2)
        self.root[segIdx].splitAt(idx+1)
        
        currentNode = self.root
                
        newJump = Jump(self.root[segIdx+1], self.root[segIdx+2])
        
        self.root[segIdx+1] = newJump
        self.root.updateTimes()

    def splitLeastFit(self):
        """ Split the segment with the least fitness into two segments. """
        fitness = [1/np.abs(x.frequencies - x(x.times)).mean() for x in self.root]
        fitness = [fitness[i] if len(self.root[i].times) > 5 else 1 \
                   for i in range(len(self.root))]
        fitness = np.array(fitness)
        leastFitArg = fitness.argmin()
        if (fitness == 1).all():
            return
        self.root[leastFitArg].splitLeastFit()
        
        #self.root.splitLeastFit()
        
        st = np.array([x.startTime for x in self.root])
        et = np.array([x.endTime for x in self.root])
        if not ((st[1:] - et[:-1] == 0).all()):
            raise ValueError('times do not line up')

    def _classify(self):
        """ Classify this call as whatever type of call it is."""
        segList = list(self.root)
        freqs = np.array([x.startFreq for x in segList] + [segList[-1].endFreq])
        slopes = np.array([x.slope for x in segList])

        jumpIdx = np.array([type(x) is Jump for x in segList])
        njumps = np.array([type(x) is Jump for x in segList]).sum()

        quadIdx = np.array([type(x) is Quadratic for x in segList])
        nquads = np.array([type(x) is Quadratic for x in segList]).sum()

        lineIdx = np.array([type(x) is Line for x in segList])
        nlines = np.array([type(x) is Line for x in segList]).sum()
        lineDur = np.array([x.duration for x in segList if type(x) is Line])
        lineFreq = np.array([np.abs(x.startFreq - x.endFreq) for x in segList
                             if type(x) is Line])

        # Short: duration less than 12 ms
        if self.root.endTime - self.root.startTime <= 0.006:
            self.callType = 'short'
            return self.callType

        # Trill: the only lines in the call are short in length and frequency
        # and there are no jumps
        if np.logical_and(lineDur < 0.012, lineFreq < 5e3).all() \
                and njumps == 0 and nquads/nlines > 1:
            self.callType = 'Trill'
            return self.callType

        # Trill with Jumps: trills with jumps or short and tall lines
        if (lineDur < 0.012).all() and nquads/(njumps+nlines) > 1 \
                and ((lineFreq >= 5e3).any() or njumps > 0):
            self.callType = 'Trill with Jump'
            return self.callType

        # Flat-Trill Combination: has long and fairly flat lines, and trills
        if np.logical_and(lineDur >= 0.012, lineFreq < 5e3).all() \
                and nquads/(njumps+nlines) > 1:
            self.callType = 'Flat-Trill Combination'
            return self.callType

        #after this, trills count as lines

        if njumps == 1:
            jumpIdx = jumpIdx.argmax()
            jumpSeg = segList[jumpIdx]

            # Step up: instantaneous frequency change to a higher frequency
            if jumpSeg.dFrequency > 0:
                self.callType = 'Step Up'

            #Step down: instantaneous frequency change to a lower frequency
            else:
                self.callType = 'Step Down'
            return self.callType
        elif njumps == 2:
            # Split: middle component jumps to a lower frequency and contains
            # a harmonic
            jumps = [self.root[i] for i in range(len(self.root)) \
                     if isinstance(self.root[i], Jump)]
            freqA = jumps[0].startFreq
            freqB = jumps[0].endFreq
            freqC = jumps[0].endFreq
            freqRatio = freqA/freqC
            if freqB < freqA and freqB < freqC and 0.9 < freqRatio < 1.1:
                self.callType = 'Split'
            else:
                self.callType = 'complex'
            return self.callType

        if njumps > 1:
            # Multi-step: two or more instantaneous frequency changes
            self.callType = 'Multi-step'
            return self.callType

        # 22-kHz calls: near-constant frequency calls between 20 and 25 kHz
        if np.bitwise_and(freqs > 2.0e4, freqs < 2.5e4).all():
            self.callType = '22kHz call'
            return self.callType

        # Flat: near-constant frequency greater than 30 kHz with a mean slope
        # between -0.2 and 0.2 kHz/ms
        if (np.abs(slopes) <2e5).all():
            self.callType = 'flat call'
            return self.callType

        # Downward ramp: monotonically decreasing in frequency, with a mean
        # negative slope not less than 0.2 kHz/ms
        if (slopes < 0).all():
            self.callType = 'downward ramp'
            return self.callType

        # Upward ramp: monotonically increasing in frequency, with a mean slope
        # not less than 0.2 kHz/ms
        if (slopes > 0).all():
            self.callType = 'upward ramp'
            return self.callType

        # Inverted U: a monotonic increase followed by a monotonic frequency
        # decrease, each of at least 5 kHz
        if (np.diff(slopes) < 0).all():
            self.callType = 'inverted U'
            return self.callType

        self.callType = 'indeterminate'
        return self.callType

    
class Segment(object):
    """ Contains a pair of other segments of data.  Acts as an internal node
        in a binary tree that describes the split call. 
    """
    
    def __call__(self, x, *args):
        """ Return the value of the predicted call at x.  If x is iterable, 
            return an numpy array of each element's value.  Otherwise, if the
            value of x is less than the midpoint of both children, return x at
            the left child; otherwise, return x at the right child.
            @param [in] x the value to calculate the call's fit at
            @param [in] args a list of arguments needed to describe this set of 
                             segments.  For now, follows the pattern:
                             [X1, Y1, ... Xm, Ym, ... XM+n-1, YM+n-1] 
                             where n is the length of the left Segment's args,
                             and m is the length of the right's, and the left
                             args are [1:n] and the right are [n:n+m-1]
        """
        m = len(self.leftChild.getParams())
        n = len(self.rightChild.getParams())
        
        if len(args) > 0:
            
            if self.isLeft and self.isRight:
                params = np.append(self.leftChild.startTime,
                                   np.insert(args, -1, self.rightChild.endTime))
            elif self.isLeft:
                params = np.append(self.leftChild.startTime, args)
            elif self.isRight:
                params = np.insert(args, -1, self.rightChild.endTime)
            
            if iterable(x):
                return np.array([self(i, *args) for i in x])
            
            else:
                if x < args[m-2]:
                    return self.leftChild(x, *args[:m])
                else:
                    return self.rightChild(x, *args[-n:])
        else:
            if iterable(x):
                return np.array([self(i) for i in x])
            else:
                if x < self.leftChild.endTime:
                    return self.leftChild(x)
                else:
                    return self.rightChild(x)
            
    def __getitem__(self, key):
        """ @todo document Segment.__getitem__ """
        try:
            while key < 0:
                key += len(self)
            while key >= len(self):
                key -= len(self)
            if key < len(self.leftChild):
                return self.leftChild[key]
            else:
                return self.rightChild[key-len(self.leftChild)]
        except:
            return None
    
    def __init__(self, leftChild, rightChild, isLeft, isRight, parent):
        """ @todo document Segment.__init__"""
        
        ## This segment's parent in the binary tree if not the root, otherwise
        # the Call this tree belongs to
        self.parent = parent
        
        ## This node's left child
        self.leftChild = leftChild
        
        ## this node's right child
        self.rightChild = rightChild
        
        ## this node's start time
        self.startTime = leftChild.startTime
        
        ## this node's middle time
        self.middleTime = leftChild.endTime
        
        ## this node's end time
        self.endTime = rightChild.endTime
        
        ## true if this node is on the left-most side of the tree
        self.isLeft = isLeft
        
        ## true if this node is on the right-most side of the tree
        self.isRight = isRight
        
    def __iter__(self):
        """ @todo document Segment.__iter__ """
        for seg in [self.leftChild, self.rightChild]:
            if seg is None:
                yield None
            else:
                for x in seg:
                    yield x
        
    def __len__(self):
        """ @todo document Segment.__len__ """
        if self.leftChild is None:
            return len(self.rightChild)
        elif self.rightChild is None:
            return len(self.leftChild)
        return len(self.leftChild) + len(self.rightChild)
        
    def __repr__(self):
        """ @todo document Segment.__repr__ """
        return "%s, %s" % (self.leftChild.__repr__(), self.rightChild.__repr__())
    
    def __setitem__(self, key, value):
        """ @todo document Segment.__setitem__ """
        if isinstance(self, Line):
            self = key
            return
        if key >= len(self) or key < 0:
            key = key % len(self)
        if key == 0 and isinstance(self.leftChild, Line):
            self.leftChild = value
        elif key == len(self) - 1 and isinstance(self.rightChild, Line):
            self.rightChild = value
        elif key < len(self.leftChild):
            self.leftChild[key] = value
        else:
            self.rightChild[key - len(self.leftChild)] = value  
        pass
    
    @property
    def depth(self):
        """ Return the depth of this node in the tree. """
        if type(self.parent) is Call:
            return 0
        else:
            return self.parent.depth + 1
    
    @property
    def duration(self):
        """ The period of time this segment lasts for. """
        return self.endTime - self.startTime
    
    @property
    def grandparent(self):
        """ The call this segment belongs to. """
        if isinstance(self.parent, Call):
            return self.parent
        else:
            return self.parent.grandparent
            
    @property
    def times(self):
        """ The times that this segment spans. """
        idxl = self.parent.times >= self.startTime
        idxh = self.parent.times <= self.endTime
        idx = np.bitwise_and(idxl, idxh)
        return self.parent.times[idx]
    
    @property
    def frequencies(self):
        """ The frequencies that this segment spans. """
        idxl = self.parent.times >= self.startTime
        idxh = self.parent.times <= self.endTime
        idx = np.bitwise_and(idxl, idxh)
        return self.parent.frequencies[idx]
    
    def applyFrequencies(self, params):
        """ Update the frequencies for the ends of each segment. """
        m = len(self.leftChild.getFreqs())
        n = len(self.rightChild.getFreqs())
        self.leftChild.applyFrequencies(params[:m])
        self.rightChild.applyFrequencies(params[-n:])
            
    def applyParameters(self, params, isLeft, isRight):
        """ Update the full parameters of this segment. """
        l = len(self.leftChild.getParams())
        if isLeft:
            self.leftChild.applyParameters(params[:l-1], True, False)
        else:
            self.leftChild.applyParameters(params[:l], False, False)
            
        self.startTime = self.leftChild.startTime
        self.middleTime = self.leftChild.endTime
        
        r = -1*len(self.rightChild.getParams())
        if isRight:
            self.rightChild.applyParameters(params[r+1:], False, True)
        else:
            self.rightChild.applyParameters(params[r:], False, False)
        
        self.endTime = self.rightChild.endTime
        
    def applyTimes(self, params, isLeft, isRight):
        """ Apply a new set of times to this node and to its children. """
        if isLeft:
            lParams = len(self.leftChild.getTimes())-1
        else:
            lParams = len(self.leftChild.getTimes())
            
        if isRight:
            rParams = len(self.rightChild.getTimes())-1
        else:
            rParams = len(self.rightChild.getTimes())
            
        self.leftChild.applyTimes(params[:lParams], isLeft, False)
        self.rightChild.applyTimes(params[-rParams:], False, isRight)
                     
    def argLength(self):
        """ The number of parameters for this segment. """
        return self.leftChild.argLength() + self.rightChild.argLength() - 1
    
    def balance(self):
        """ Rebalance this node and its descendants (not really balancing if 
            I'm being honest but it might be some day) 
        """
        
        # if a child is None replace this with the other child
        if self.leftChild is None:
            self = self.rightChild
        elif self.rightChild is None:
            self = self.leftChild
            
        # if any grandchild on child A is None, then replace this nodes 
        # references to A with A's other child
        if not isinstance(self.leftChild, Line):
            if self.leftChild.leftChild is None:
                self.leftChild = self.leftChild.rightChild
            elif self.leftChild.rightChild is None:
                self.leftChild = self.leftChild.leftChild
            self.leftChild.balance()
        if not isinstance(self.rightChild, Line):
            if self.rightChild.leftChild is None:
                self.rightChild = self.rightChild.rightChild
            elif self.rightChild.rightChild is None:
                self.rightChild = self.rightChild.leftChild
            self.rightChild.balance()
        
        self.updateTimes()
    
    def calcErr(self, x, y):
        """ Return a list of errors for all segments contained within this one."
        """
        return np.append(self.leftChild.calcErr(x, y), self.rightChild.calcErr(x, y))
    
    def freqParamFilter(self):
        """ Return a mask for the parameters of this node to separate 
            the frequencies. 
        """
        filter = []
        filters = [[False, True]] + [x.freqParamFilter()[2:] for x in self]
        for f in filters:
            filter += f
        return filter
    
    def fitness(self, x, y):
        """ Return the fitness of this node. 
            @todo figure out exactly what fitness is
        """
        try:
            return 1/(self.calcErr(x, y).sum()*(self.endTime-self.startTime))
        except e:
            raise e
        
    def getBounds(self):
        """ Return the min and max allowable times and frequencies for this 
            node's parameters.
        """
        leftHigh, leftLow = self.leftChild.getBounds()
        rightHigh, rightLow = self.rightChild.getBounds()
        return leftHigh + rightHigh[2:], leftLow + rightLow[2:]
    
    def getFreqs(self):
        """ Return the frequencies for this segment out of the parameters. """
        a = []
        for y in [[self[0].startFreq]] + [x.getFreqs()[1:] for x in self]:
             a += y
        return a
    
    def getParams(self):
        """ Return the parameters for this segment. """
        leftParams = self.leftChild.getParams()
        rightParams = self.rightChild.getParams()
        return np.append(leftParams, rightParams[2:])
    
    def getTimes(self):
        """ return the time parameters for this segment. """
        return self.leftChild.getTimes() + self.rightChild.getTimes()[1:]
    
    def isLeftTime(self, t):
        """ Return true if the given time t is less than/before/left of this node. 
        """
        return t < self.middleTime
    
    def paramsFromFreqs(self, freqArgs):
        """ Shuffle frequency parameters into a complete parameters list. """
        params = self.getParams()
        filter = self.freqParamFilter()
        j = 0
        try:
            for i in range(len(params)):
                if filter[i]:
                    params[i] = freqArgs[j]
                    j += 1
        except Exception as e:
            pass
        return params
    
    def paramsFromTimes(self, timeArgs):
        """ Get a complete list of parameters from the time parameters. """
        params = self.getParams()
        filter = self.freqParamFilter()
        filter[0] = True
        filter[-2] = True
        filter = [not f for f in filter]
        j = 0
        try:
            for i in range(len(params)):
                if filter[i]:
                    params[i] = timeArgs[j]
                    j += 1
        except Exception as e:
            pass
        return params

    def splitLeastFit(self):
        """ Split the least fit line of this segment into two lines. 
            @todo Probably ought to refine the fitness function at some point
        """
        
        # if a segment is very short and jumps very high, replace it with a jump
        for i in range(0, len(self) - 1):
            if self[i].duration < 0.01 and abs(self[i].dFrequency) > 1e4 \
                    and not isinstance(self[i], Jump):
                self[i] = Jump(self[i], self[i + 1])
                self.grandparent.balance()
                return
        leftFit = self.leftChild.fitness(self.parent.times, 
                                         self.parent.frequencies)
        rightFit = self.rightChild.fitness(self.parent.times, 
                                           self.parent.frequencies)
        
        if len(self.leftChild) < 5:
            leftFit = 0
        if len(self.rightChild) < 5:
            rightFit = 0
        
        if leftFit < rightFit and len(self.leftChild.times) > 5:
            self.leftChild.splitLeastFit()
        elif len(self.rightChild.times) > 5:
            self.rightChild.splitLeastFit()
            
    def updateTimes(self):
        """ Update the times of segments to match their lines. """
        nodeList = [x for x in self]
        for node in nodeList:
            if node.rightSibling is not None and type(node) is not Jump:
                node.endTime = node.rightSibling.startTime
            if node.leftSibling is not None:
                node.startTime = node.leftSibling.endTime
            parent = node.parent
            while type(parent) is not Call:
                if parent.leftChild is not None:
                    parent.startTime = parent.leftChild.startTime
                if parent.rightChild is not None:
                    parent.endTime = parent.rightChild.endTime
                    parent.middleTime = parent.rightChild.startTime
                parent = parent.parent


class Line(Segment):
    """ ## Describes a straight line segment for a fitted call. """

    def __call__(self, x, *args):
        """ Calculate the value of x on this segment.  if x is iterable, calc
            each value of x on this segment.
            @param [in] x the value or values to calculate
            @param [in] args the parameters of the line segment to use.  If empty,
                             calculate the line as defined.  If length 2, 
                             calculate the line with different start and end
                             frequencies.  If length 4, calculate with custom
                             start and end times and frequencies.  To be used with
                             scipy.optimize.curve_fit
        """
        if iterable(x):
            return np.array([self(i, *args) for i in x])
        if type(self) is Jump:
            if x > self.endTime:
                return self.endFreq
            elif x < self.startTime:
                return self.startFreq
            else:
                return (self.startFreq + self.endFreq)/2
            return self.leftSibling(x, *args)
        else:
            startTime = self.startTime
            endTime = self.endTime
            startFreq = self.startFreq
            endFreq = self.endFreq
            if len(args) == 2:
                startFreq = args[0]
                endFreq = args[1]
            elif len(args) == 4:
                startTime = args[0]
                endTime = args[2]
                startFreq = args[1]
                endFreq = args[3]
            elif len(args) != 0:
                raise Exception(("incorrect number of parameters (0, 2, or 4)," +
                                 " was %d") % (len(args),))
                
            dx = endTime - startTime
            dy = endFreq - startFreq
            
            # gently explain to curve_fit that the times can't be equal to each
            # other without breaking the algorithm
            if dx == 0:
                return -1e15
            
            m = dy/dx
            b = startFreq
            
            return m*(x - startTime) + b
        
    def __getitem__(self, key):
        """ @todo document Line.__getitem__ """
        return self
        
    def __init__(self, startTime, endTime, startFreq, endFreq, parent, isLeft, isRight):
        """ @todo document Line.__init__ """
        if startTime == endTime:
            raise ValueError
        
        ## This node's start time
        self.startTime = startTime
        ## this node's end time
        self.endTime = endTime
        
        ## this node's starting frequency
        self.startFreq = startFreq
        
        ## this node's end frequency
        self.endFreq = endFreq
        
        ## this node's parent node
        self.parent = parent
        
        ## true if this node is on the left-most side of the tree
        self.isLeft = isLeft
        
        ## tue if this node is on the right-most side of the trees
        self.isRight = isRight
        
    def __iter__(self):
        """ @todo document Line.__iter__ """
        yield self
        
    def __len__(self):
        """ @todo document Line.__len__ """
        return 1
        
    def __repr__(self):
        """ @todo document Line.__repr__ """
        dt = self.endTime - self.startTime
        dF = self.endFreq - self.startFreq
        dFdt = dF/dt
        return "[Fs: %.3f, Fe: %.3f, dF/dt: %.3f, t: %.3f]\n" %  \
            (self.startFreq/1000, self.endFreq/1000, dFdt/1e6, dt*1000)
    
    @property
    def dFrequency(self):
        """ Return the change in frequency in this line. """
        return self.endFreq - self.startFreq
    
    @property
    def grandparent(self):
        """ return the Call object that this line is a descendent of. """
        if type(self.parent) == Call:
            return self.parent
        else:
            if self.parent is None:
                pass
            return self.parent.grandparent
         
    @property
    def index(self):
        """ Return the index of this line in the root node. """
        return np.array([self == x for x in self.grandparent.root]).argmax()
    
    @property
    def leftSibling(self):
        """ Return the leaf node just before this one. """
        if self.index == 0:
            return None
        else:
            return self.grandparent.root[self.index-1]
    
    @property
    def npoints(self):
        """ return the number of points within this line. """
        return len(self.frequencies)
    
    @property
    def rightSibling(self):
        """ return the leaf node just after this line. """
        if self.index == len(self.grandparent.root) - 1:
            return None
        else:
            return self.grandparent.root[self.index+1]
        
    @property
    def slope(self):
        """ return the slope of this line. """
        try:
            return (self.endFreq-self.startFreq)/(self.endTime-self.startTime)
        except Exception as e:
            raise e
        
    def applyFrequencies(self, params):
        """ Apply a set of frequency parameters to this line. """
        self.applyParameters(params, True, True)
        
    def applyParameters(self, params, isLeft, isRight):
        """ Change this lines parameters. 
            The parameters are [start time, start freq, end time, end freq]
            the line does not have a start time if it is the leftmost line, 
            and it does not have an end time if it is the rightmost line
        """
        if isLeft and isRight:
            self.startFreq = params[0]
            self.endFreq = params[1]
        elif isRight:
            self.startTime = params[0]
            self.startFreq = params[1]
            self.endFreq = params[2]
        elif isLeft:
            self.startFreq = params[0]
            self.endTime = params[1]
            self.endFreq = params[2]
        else:
            self.startTime = params[0]
            self.startFreq = params[1]
            self.endTime = params[2]
            self.endFreq = params[3]
            
        if self.startTime == self.endTime and not isinstance(self, Jump):
            self.endTime = self.startTime + 1e-5
            self.grandparent.root.updateTimes()
        
    def applyTimes(self, params, isLeft, isRight):
        """ Set this segment's times. """
        if isLeft and isRight:
            pass
        elif isRight:
            self.startTime = params[0]
        elif isLeft:
            self.endTime = params[0]
        else:
            self.startTime = params[0]
            self.endTime = params[1]
        
    def argLength(self):
        """ return the number of parameters for this line. """
        return 4 - self.isLeft - self.isRight
    
    def balance(self):
        """ balance a one-element tree, so, do nothing. """
        pass
           
    def calcErr(self, x, y):
        """ return the error for this segment for (x, y). """
        xOut = self(x)
        idx = (xOut != 0)
        return np.abs(y[idx] - xOut[idx])
    
    def fitness(self, x, y):
        """ return the fitness of this line. """
        return Segment.fitness(self, x, y)
    
    def freqParamFilter(self):
        """ Return a mask of the frequencies in the parameters. """
        return [False, True, False, True]
    
    def getBounds(self):
        """ return the maximum and minimum frequencies and times for this line. 
        """
        tEnd = self.grandparent.times[-1]
        tStart = self.grandparent.times[0]
        return (tEnd, 2.5e5, tEnd, 2.5e5), (tStart, 1.5e4, tStart, 1.5e4)
    
    def getFreqs(self):
        """ return the frequencies of this line. """
        return [self.startFreq, self.endFreq]
    
    def getParams(self):
        """ return the parameters of this line. """
        return np.array([self.startTime, self.startFreq, self.endTime, 
                         self.endFreq])
        
    def getTimes(self):
        """ return the start and end times of this line """
        return [self.startTime, self.endTime]
        
    def jumpify(self):
        """ turn this line into a jump. """
        newSelf = Jump(self, None)
        
        if isinstance(self.parent, Call):
            self.parent.root = newSelf
        elif self.startTime < self.parent.middleTime:
            self.parent.leftChild = newSelf
        else:
            self.parent.rightChild = newSelf
            
        self = newSelf
        
    def merge(self, otherLine, type='line'):
        """ merge this line and the next line. """
        if self.duration < otherLine.duration:
            otherLine.startTime = self.startTime
            otherLine.startFreq = self.startFreq
        else:
            self.endTime = otherLine.endTime
            self.endFreq = otherLine.endFreq
    
    def split(self, newTime):
        """ Split this line into two colinear line segments at newTime.  Replace
            itself in the tree with a new Segment with a pair of Lines.
        """
        newLeft = Line(self.startTime, newTime, self.startFreq, self(newTime), 
                       None, self.isLeft, False)
        newRight = Line(newTime, self.endTime, self(newTime), self.endFreq, 
                        None, False, self.isRight)
        newSegment = Segment(newLeft, newRight, self.isLeft, self.isRight, 
                             self.parent)
        newLeft.parent = newSegment
        newRight.parent = newSegment
        if isinstance(self.parent, Call):
            self.parent.root = newSegment
        elif self.parent.isLeftTime(newTime):
            self.parent.leftChild = newSegment
        else:
            self.parent.rightChild = newSegment
            
        del self
        
    def splitAt(self, idx):
        """ split this line into two lines at the index idx. """
        newTime = (self.grandparent.times[idx-1] + self.grandparent.times[idx])/2
        self.split(newTime)
    
    def splitLeastFit(self):
        """ split this line at half of the cumulative fitness. """
        err = np.abs(self.frequencies - self(self.times))
        cErr = np.cumsum(err)
        halfIdx = np.abs(cErr - cErr.mean()).argmin()
        self.split(self.times[halfIdx])
        

class Jump(Line):
    """ Describes a sudden and drastic shift in frequency. """

    def __init__(self, line1=None, line2=None, startTime=None, startFreq=None, 
                 endFreq=None, parent=None):
        """ @todo document Jump.__init__ """
        
        ## the ending frequency of the segment
        self.endFreq = None
        
        ## Always false, Jump segments should never be on an end of a Call
        self.isLeft = False
        
        ## Always false, Jump segments should never be on an end of a Call
        self.isRight = False
        
        ## this segment's parent node
        self.parent = None
        
        ## the starting frequency of the segment
        self.startFreq = None
        
        ## the starting frequency of the segment
        self.startTime = None
        
        if line1 is None and line2 is None:
            self.startTime = startTime
            self.startFreq = startFreq
            self.endFreq = endFreq
            return
        elif line1 is None:
            line1 = line2
            line2 = line1.rightSibling
        elif line2 is None:
            line2 = line1
            line1 = line2.leftSibling
        
            
        self.startTime = line1.startTime
        line2.startTime = line1.startTime
        
        self.startFreq = line1.startFreq
        self.endFreq = line1.endFreq
        
        self.parent = line1.parent
        self.grandparent.root[self.index + 1].startTime = self.startTime
        
    def __repr__(self):
        """ @todo document Jump.__repr__ """
        try:
            dt = self.endTime - self.startTime
            return "[Fs: %.3f, Fe: %.3f, dF/dt: J, t: %.3f]\n" %  \
                (self.startFreq/1000, self.endFreq/1000, dt*1000)
        except Exception as e:
            raise e
    
    @property
    def endTime(self):
        """ Return the end time of this jump, which is the same as the start. """
        return self.startTime
    
    @property
    def slope(self):
        """ Return the slope of this segment, which is either positive or 
            negative infinity.
        """
        if self.endFreq > self.startFreq:
            return np.PINF
        else:
            return np.NINF
    
    def applyParameters(self, params, isLeft, isRight):
        """ Apply a new set of parameters to this segment. """
        if isRight:
            Line.applyParameters(self, params, isLeft, True)
        elif isLeft:
            pass
        else:
            params = params[[0,1,3]]
            Line.applyParameters(self, params, isLeft, True)
            
    def applyTimes(self, params, isLeft, isRight):
        """ Set a new start time. """
        self.startTime = params[0]
    
    def getTimes(self):
        """ get a list of this segment's times. """
        return [self.startTime]
            
    def merge(self, otherLine, type='line'):
        """ Merge this segment and the next segment. """
        return
        Line.merge(self, otherLine, type=type)
        
    def split(self, time):
        """ Split this segment into two, which should never happen to a jump. """
        raise Exception('Attempted to split a jump')


class Quadratic(Line):
    """ This functions much like a line as above, but it's a quadratic curve. """
        
    def __init__(self, startTime, endTime, startFreq, endFreq, parent, 
                 isLeft, isRight, curve):
        """ @todo document Quadratic.__init__ """
        Line.__init__(self, startTime, endTime, startFreq, endFreq, parent, 
                      isLeft, isRight)
        
        ## The height of the curve of this segment
        self.curve = curve
        
        ## the ending frequency of this segment
        self.endFreq = endFreq
        
        ## the ending time of this segment
        self.endTime = endTime
        
        ## the starting frequency of this segment
        self.startFreq = startFreq
        
        ## the starting time of this segment
        self.startTime = startTime
    
    def __call__(self, t, *args):
        """ Return the value for this segment at the time(s) t.  If args is not
            empty, then use the current parameters, otherwise substitute the 
            parameters from args for this call.
        """
        startTime = self.startTime
        startFreq = self.startFreq
        endTime = self.endTime
        endFreq = self.endFreq
        curve = self.curve
        if len(args) == 2:
            startFreq = args[0]
            endFreq = args[1]
        elif len(args) == 3:
            startFreq = args[0]
            endFreq = args[2]
            curve = args[1]
        elif len(args) == 5:
            startTime = args[0]
            endTime = args[3]
            startFreq = args[1]
            endFreq = args[4]
            curve = args[2]
        elif len(args) != 0:
            raise Exception(("incorrect number of parameters (0, 3, or 5)," +
                             " was %d") % (len(args),))
        if (t > endTime).any() or (t < startTime).any():
            pass    
        
        l = endTime - startTime
        dF = endFreq - startFreq
        x = t - startTime
        if (self._h1(x, curve, l) + self._h2(x, dF, l) + startFreq < 0).any():
            pass
        
        return self._h1(x, curve, l) + self._h2(x, dF, l) + startFreq
    
    def __repr__(self):
        """ @todo document Quadratic.__repr__ """
        dt = self.endTime - self.startTime
        dF = self.endFreq - self.startFreq
        dFdt = dF/dt
        return "[Fs: %.3f, Fe: %.3f, dF/dt: %.3f, t: %.3f, c:%.3f]\n" %  \
            (self.startFreq/1000, self.endFreq/1000, dFdt/1e6, dt*1000, self.curve/1000)        
    
    def _h1(self, x, A, l):
        """ @todo document Quadratic._h1 """
        return 4*A*((x/l) - (x/l)**2.0)
    
    def _h2(self, x, dF, l):
        """ @todo document Quadratic._h2 """
        return dF*(x/l)
        
    def applyParameters(self, params, isLeft, isRight):
        """ Set this segment's parameters to params. """
        if isLeft and isRight:
            self.startFreq = params[0]
            self.curve = params[1]
            self.endFreq = params[2]
        elif isRight:
            self.startTime = params[0]
            self.startFreq = params[1]
            self.curve = params[2]
            self.endFreq = params[3]
        elif isLeft:
            self.startFreq = params[0]
            self.curve = params[1]
            self.endTime = params[2]
            self.endFreq = params[3]
        else:
            self.startTime = params[0]
            self.startFreq = params[1]
            self.curve = params[2]
            self.endTime = params[3]
            self.endFreq = params[4]
        if self.startTime == self.endTime and not isinstance(self, Jump):
            raise ValueError
    
    def argLength(self):
        """ Return the length of this segment's parameters. """
        return 5 - self.isLeft - self.isRight
    
    def freqParamFilter(self):
        """ Return a list of booleans for which parameters are not times. """
        return [False, True, True, False, True]
    
    def getBounds(self):
        """ Return the bounds that these parameters can be fit to. """
        tEnd = self.grandparent.times[-1]
        tStart = self.grandparent.times[0]
        return (tEnd, 2.5e5, 1e4, tEnd, 2.5e5), (tStart, 1.5e4, -1e4, tStart, 1.5e4)
    
    def getFreqs(self):
        """ Return the frequency parameters for this segment. """
        return [self.startFreq, self.curve, self.endFreq]
    
    def getParams(self):
        """ Return the parameters for this segment. """
        return [self.startTime, self.startFreq, self.curve, self.endTime, self.endFreq]
    

def trillIsolate(xn, yn, wn, stepFreq, spectro, lowFreq=70, highFreq=200, curveHigh=-5e3, 
                 curveLow=-25e3, dFreqLow=3e4, dFreqHigh=0):
    """ return the indices of the beginning and end of each trill in this signal.
    """
    x = spectro[0]
    y = spectro[1]
    amp = spectro[2]
    
    # define the low and high frequencies for filtering, and the filter specs
    l = 2*(lowFreq/stepFreq)
    h = 2*(highFreq/stepFreq)
    if len(yn) < 12:
        return []
    try:
        b, a = signal.iirfilter(5, h, btype='lowPass', ftype='butter')
        out = signal.filtfilt(b,a,yn)
    except ValueError as e:
        if 'The length of the input vector x must be at least padlen' in str(e):
            b, a = signal.iirfilter(1, h, btype='lowPass', ftype='butter')
            out = signal.filtfilt(b,a,yn)
        else:
            raise e
    
    # filter the data to frequencies below the band (noHigh),
    # above the band (noLow) and within the band (out)
    
    # determine the indices of the relative extrema, and if the first max
    # is before the first min, remove the first max
    minIdx = signal.argrelmin(out, order=int(2.6e-3*stepFreq), mode='wrap')
    minIdx = minIdx[0]
    
    if not 0 in minIdx:
        minIdx = np.array([0] + list(minIdx))
    if not len(out) - 1 in minIdx:
        minIdx = np.array(list(minIdx) + [len(out) - 1])
    
    # for every min, fit a quadratic to the data between this min and the next
    # if it is very flat, ignore it
    #plt.plot(xn, yn)
    #plt.plot(xn, out + yn.mean())
    quad = lambda x, a, b, c: a*(x**2) + b*x + c
    
    params = []
    for i in range(1, len(minIdx)):
        qx = np.arange(minIdx[i-1], minIdx[i], 1.0)
        offset = qx.mean()
        qx -= offset
        if len(yn[minIdx[i-1]:minIdx[i]]) < 3:
            return np.array(params)
        
        x1 = np.array([x]*amp.shape[0])
        y1 = np.array([y]*amp.shape[1]).T
        idx = np.logical_and(x1 > xn[minIdx[i-1]], x1 < xn[minIdx[i]])
        idx2 = amp[idx] > 0
        if (idx2*1).sum() < 3:
            continue
        x2 = x1[idx][idx2]
        y2 = y1[idx][idx2]
        a2 = amp[idx][idx2]
        if x2.min() == x2.max():
            continue
        t2 = (x2-x2.min())/(x2.max()-x2.min())
        if len(t2) < 3:
            continue
        popt2, _ = opt.curve_fit(quad, t2, y2, sigma=1/(a2))
        t2.sort()
        plt.plot(t2*(x2.max()-x2.min())+x2.min(), quad(t2,*popt2),color='r')
        qx = range(minIdx[i-1], minIdx[i])
        
        # if the fit has a negative curve with a magnitude between -3 and -100
        # and the magnitude of the slope is less than 250
        # and the curve spans less than 8 kHz of signal but more than 1kHz
        if popt2[0] < curveHigh/100 and popt2[0] > curveLow*1000 \
                and np.abs(yn[0] - yn[-1]) < dFreqLow \
                and np.abs(yn[0] - yn[-1]) > dFreqHigh:
            params.append([qx[0], qx[-1]])
    return np.array(params)

def validPoints(times, majorFreqs, majorDev, majorAmp, pTimes, pFreqs, caLow, stdMax):
    """ Return the valid points for the given parameters. This function filters
        any part of the signal which has a standard deviation too high or 
        frequencies that are outside of the valid range of signal frequencies.
    """
    freqMask = np.logical_and(majorFreqs > 1.5e4, majorFreqs < 1e5)
    ampMask = majorAmp > caLow
    devMask = majorDev < 7e3
    
    mask = np.logical_and(np.logical_and(freqMask, ampMask), devMask)
    
    times = times[mask]
    majorFreqs = majorFreqs[mask]
    majorDev = majorDev[mask]
    majorAmp = majorAmp[mask]
    
    if len(times) < 5:
        return times, majorFreqs, majorDev, majorAmp
    
    mask = []
    
    for t in times[2:-2]:
        tIdx = np.abs(pTimes - t).argmin()
        tCount = (np.abs(pFreqs[max(tIdx-5, 0):tIdx+6l] - pFreqs[tIdx]) < 5000).sum()
        mask.append(tCount > 3)
    
    mask = np.array([True, True] + mask + [True, True])
    return times[mask], majorFreqs[mask], majorDev[mask], majorAmp[mask]
    
    
    
    
