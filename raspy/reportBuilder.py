"""
@file: reportBuilder.py
@brief: class to aid in generating reports based on previously separated data
Each call is taken in as a dict with the following format
key = 'Type' : Contains name of call type
key = 'A##' where A is a capital letter progressing A-Z-AA-AZ, and # is an optional 2 digit number, with a leading 0.
  Calls progress alphabetically. : Contains dict with keys 'segment' for fit line data, and 'raw' for raw data
  'segment' : {'slope', 'offset', 'duration', 'start_time'}
  'raw' : { 'fs': sample rate, 'window': window size, 'data': [[fft data],]}
  'peak' : { 'fs': sample rate, 'data': [(time, freq, amplitude,]}
This will execute comparisons of entire calls, as well as segment by segment comparisons
the _get functions will operate on a single 'unit', which can be a dict (specifying a single segment)
  or list of dicts, specifying an entire call.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
import numbers
from raspy.call import Call
import csv
import scipy.io.wavfile as wavfile
import os

#from raspy.call import plt, np, signal


class ReportBuilder(object):
    _seg_keys = ["slope", "offset", "duration"]
    _default_time_label = "Time (S)"
    _default_freq_label = "Frequency (Hz)"

    def __init__(self, times, freqs, rawSpectrogram, rawData, defaultDir="", startTimeOffset=0, sampleRate=250000):
        self.times = times
        self.freqs = freqs
        self.rawData = rawData
        self.startTimeOffset = startTimeOffset
        self.sampleRate = sampleRate
        if defaultDir:
            if not os.path.exists(defaultDir):
                os.mkdir(defaultDir)
            self.defaultDir = defaultDir
        self.rawSpectrogram = rawSpectrogram
        self.timePseudocolorOffset = (self.times[1] - self.times[0])/2
        self.freqPseudocolorOffset = (self.freqs[1] - self.freqs[0])/2

        self.logInfo = { }      # Data to be saved: logName: analysisList

        self._input_data = []
        self._processed_data = []
        self._all_segment_analysis_modes = {'frequency_average': self._get_seg_avg,
#                                       'frequency_start': self._get_seg_start,
#                                       'frequency_end': self._get_seg_end,
                                       'frequency_range': self._get_seg_range,
#                                       'slope': self._get_seg_slope,
#                                       'duration': self._get_seg_duration}
                                            }
        self._all_raw_analysis_modes = {'slice_average': self._get_raw_slice_avg,
                                  #       'frequency_center': self._get_raw_avg, 'frequency_start': self._get_raw_start,
                                  #  'frequency_end': self._get_raw_end, 'frequency_range': self._get_raw_range,
                                  #  'duration': self._get_raw_duration, 'frequency_max': self._get_raw_max,
                                  #  'frequency_min': self._get_raw_min, 'frequency_variance': self.get_raw_freq_var,
                                  # 'energy_average': self._get_raw_en_avg, 'energy_max': self._get_raw_en_max,
                                  # 'energy_min': self._get_raw_en_min,  'energy_variance': self._get_raw_en_var,
                                  # 'total_energy': self._get_raw_tot_var,
                                        }
        self._all_peak_analysis_modes = {'frequency_center': self._get_peak_avg, 'frequency_start': self._get_peak_start,
                                         'frequency_end': self._get_peak_end, 'frequency_range': self._get_peak_range,
                                         'frequency_max': self._get_peak_max, 'frequency_min': self._get_peak_min,
                                         'frequency_variance': self._get_peak_freq_var,
                                         'amplitude_average': self._get_peak_amp_avg,
                                         'amplitude_variance': self._get_peak_amp_var,
                                         'slice_average': self._get_peak_slice_avg,
                                         'mse': self._get_peak_mse}
        self._segment_analysis_modes = self._all_segment_analysis_modes.keys()
        self._raw_analysis_modes = self._all_raw_analysis_modes.keys()
        self._peak_analysis_modes = self._all_peak_analysis_modes.keys()

    def _get_call_rms(self, call, units=1, isSegment=True):
        if isSegment:
            npAudio = np.array(self._getRawData(call.times[0], call.times[-1]), dtype='float64')
            rms = np.sqrt(np.mean(npAudio**2))
            if np.isnan(rms):
                print("Got NaN in calculating volume")
                return 0
            return rms
        else:
            sum = 0
            for c in call.root:
                if c.duration > 0 and len(c.times) > 0:
                    sum += self._get_call_rms(c, isSegment=True)
            return sum

    def _get_call_avg_freq(self, call, units=1, isSegment=True):
        return sum(call.frequencies)/len(call.frequencies)

    def _get_call_freq_bandwidth(self, call, units=1, isSegment=True):
        return max(call.frequencies) - min(call.frequencies)

    def _get_call_start_time(self, call, units=1, isSegment=True):
        if isSegment:
            return call.times[0] + self.startTimeOffset
        else:
            return call.times[0] + self.startTimeOffset

    def _get_call_type(self, call, units=None, isSegment=True):
        return call.callType

    def _get_seg_avg(self, seg):
        if isinstance(seg, list):
            total_duration = 0
            total_sum = 0
            for s in seg:
                total_sum += self._get_seg_avg(s) * s['duration']
                total_duration += s['duration']
            return total_sum/total_duration
        return seg['offset'] + seg['slope'] * seg['duration']/2

    def _get_call_start_freq(self, call, units=1e3, isSegment=True):
        if isSegment:
            total = call.startFreq
        else:
            total = call.root[0].startFreq
        return total/units

    def _get_call_end_freq(self, call, units=1e3, isSegment=True):
        if isSegment:
            total = call.endFreq
        else:
            total = call.root[-1].endFreq
        return total/units

    def _get_seg_max(self, seg):
        if isinstance(seg, list):
            total_max = 0
            for s in seg:
                this_max = self._get_seg_max(s)
                total_max = max(this_max, total_max)
            return total_max
        if seg['slope'] > 0:
            return seg['offset'] + seg['slope'] * seg['duration']
        return seg['offset']

    def _get_seg_min(self, seg):
        if isinstance(seg, list):
            total_min = 0
            for s in seg:
                this_min = self._get_seg_min(s)
                total_min = min(this_min, total_min)
            return total_min
        if seg['slope'] > 0:
            return seg['offset']
        return seg['offset'] + seg['slope'] * seg['duration']

    def _get_seg_range(self, seg):
        return self._get_seg_max(seg) - self._get_seg_min(seg)

    def _get_call_slope(self, call, units=1e6, isSegment=True):
        if isSegment:
            total = call.slope
        else:
            total = (self._get_call_end_freq(call, units=1, isSegment=False) - self._get_call_start_freq(call, units=1, isSegment=False))
            total = total / self._get_call_duration(call, units=1, isSegment=False)
        return total/units

    def _get_call_duration(self, call, units=1e3, isSegment=True):
        if isSegment:
            total = call.duration
        else:
            total = sum([c.duration for c in call.root])
        return total*units

    def _get_raw_slice_avg(self, raw):
        if isinstance(raw, list):
            total_length = 0
            slice_list = []
            for r in raw:
                slice_list.append([sum(x) for x in zip(*raw['data'])])
                total_length += len(raw['data'][0])
            slice_avg = [sum(x)/total_length for x in slice_list]
            return slice_avg
        length = len(raw['data'][0])
        slice_avg = [sum(x)/length for x in zip(*raw['data'])]
        return slice_avg

    def _get_peak_avg(self, peak):
        return sum(zip(*peak['data'])[1])/(len(peak['data']))

    def _get_peak_start(self, peak):
        return peak['data'][0][1]

    def _get_peak_end(self, peak):
        return peak['data'][-1][1]

    def _get_peak_range(self, peak):
        return self._get_peak_max(peak) - self._get_peak_min(peak)

    def _get_peak_max(self, peak):
        frequencies = zip(*peak['data'])[1]
        return max(frequencies)

    def _get_peak_min(self, peak):
        frequencies = zip(*peak['data'])[1]
        return min(frequencies)

    def _get_peak_freq_var(self, peak):
        frequencies = zip(*peak['data'])[1]
        return np.var(frequencies)

    def _get_peak_amp_avg(self, peak):
        amps = zip(*peak['data'])[2]
        return sum(amps)/len(amps)

    def _get_peak_amp_var(self, peak):
        amps = zip(*peak['data'])[2]
        return np.std(amps)

    def _get_peak_mse(self, peak):
        pass

    def _get_peak_slice_avg(self, peak):
        pass

    def analyzeSegment(self, segments):
        """
        :param segments: A list of either calls or segments of calls to compare. Analysis will be run on each item
          in the list
        :return:
        """
        allAnalyzedSegments = []
        for seg in segments:
            analyzedSegment = {key: self._all_segment_analysis_modes[key](seg) for key in self._segment_analysis_modes}
            for segmentKey in seg.keys():
                if segmentKey not in self._seg_keys:
                    analyzedSegment[segmentKey] = seg[segmentKey]
            allAnalyzedSegments.append(analyzedSegment)
        return allAnalyzedSegments

    def computePca(self, analysis):
        """
        Compute mean and covariance of input data across all analysis components
        :param analysis:
        :return:
        """
        analysisKeys = analysis[0].keys()
        npa = np.array([ [analysisData[analysisKey] for analysisData in analysis] for analysisKey in analysisKeys])   # Each row is a separate type of analysis
        means = np.mean(npa, axis=1)    # mean of each row
        covariance = np.cov(npa)
        eigenVal, eigenVec = np.linalg.eig(covariance)
        eigenPairs = zip(np.abs(eigenVal), eigenVec)
        eigenPairs.sort(key=lambda x: x[0], reverse=True)
        size = len(analysisKeys)
        transformMatrix = np.hstack((eigenPairs[0][1].reshape(size, 1), eigenPairs[1][1].reshape(size, 1)))
        transformed = transformMatrix.T.dot(npa)
        id = np.identity(size)
        mix = transformMatrix.T.dot(id)
        # for a in range(size):
        #     for b in range(a+1, size):
        #         print("x: %s, y: %s" % (keys[a], keys[b]))
        #         plt.scatter(npa[a, :], npa[b, :])
        #         plt.show()
        print("Transformed:")
        print("Components in x: %s" % (zip(analysisKeys, mix[0])))
        print("Components in y: %s" % (zip(analysisKeys, mix[1])))
        plt.scatter(transformed[0, :], transformed[1, :])
        plt.show()
        return means

    def getClusterDistance(self, a, b, ax=1):
        return np.linalg.norm(a-b, axis=ax)

    def getClusterCenters(self, analysis, clusters):
        keys = analysis[0].keys()
        npa = np.array([[x[k] for x in analysis] for k in keys])   # Each row is a separate type of analysis, eg start frequency or average frequency
        axisSize, dataSize = npa.shape
        maxNpa = np.max(npa, axis=1)
        minNpa = np.min(npa, axis=1)
        centers = np.array([np.random.uniform(low=minNpa[i], high=maxNpa[i], size=clusters) for i in range(axisSize)]) #centers[a][b] is the center of cluster b on axis a
        centers = centers.T             # Now centers[a][b] is the center of cluster a on axis b
        oldCenters = np.zeros(centers.shape)
        dataLabel = np.zeros(dataSize)
        centerMovement = self.getClusterDistance(centers, oldCenters, ax=None)
        while centerMovement != 0:
            # assign each point to its closest center
            for i in range(dataSize):
                distances = self.getClusterDistance(npa[:, i], centers)
                cluster = np.argmin(distances)
                dataLabel[i] = cluster
            oldCenters = deepcopy(centers)
            # find the new centers by taking the average value
            for i in range(clusters):
                points = [npa[:, j] for j in range(dataSize) if dataLabel[j] == i]
                centers[i] = np.mean(points, axis=0)
            centerMovement = self.getClusterDistance(centers, oldCenters, ax=None)
        print(keys)
        print(centers)
        colors = ['r', 'g', 'b']
        fig, ax = plt.subplots()
        for i in range(clusters):
            points = np.array([npa[:, j] for j in range(dataSize) if dataLabel[j] == i])
            ax.scatter(points[:, 2], points[:, 3], s=7, c=colors[i])
        ax.scatter(centers[:, 2], centers[:, 3], marker='*', s=200, c='#050505')
        ax.set_xlabel(keys[2])
        ax.set_ylabel(keys[3])
        plt.show()
        return centers

    def mkseg(self, l=None, o=0, s=0, d=1, t=0):
        if l is not None:
            return {'offset': l[0], 'slope': l[1], 'duration': l[2], 'start_time': l[3]}
        return {'offset': o, 'slope': s, 'duration': d, 'start_time': t}

    def testSegments(self):
        """
        Generate fake data to test the k-means clustering
        :return:
        """
        samples = 100
        # offsets = [np.random.normal(10000, 800) for x in np.linspace(22000, 23000, samples)]
        offsets = list(np.random.normal(20000, 800, size=samples//2))
        offsets += list(np.random.normal(25000, 800, size=samples//2))
        slopes = [np.random.normal(10000, 0.0002) for x in range(samples)]
        durations = [x+np.random.normal(0.005, 0.002) for x in np.linspace(0.015, 0.05, samples)]
        times = [abs(np.random.normal(0.005, 0.002))]
        for (i, dur) in enumerate(durations[:-1]):
            times.append(times[-1] + dur + abs(np.random.normal(0.005, 0.002)))
#        times = [times[-1]+x+abs(np.random.normal(0.005, 0.002)) for x in durations[:-1]]
        full = zip(offsets, slopes, durations, times)
        segments = [self.mkseg(l=x) for x in full]
        for i in range(len(segments)):
            segments[i]['rand'] = np.random.normal(0, 100)
        self._segment_analysis_modes = ['slope', 'duration', 'frequency_start']    #
        analysis = self.analyzeSegment(segments)
        print(self.getClusterCenters(analysis, 2))
#        self.compute_pca(analysis)
#        self.show_segment_analysis(analysis, x='start_time', y='slope', z='duration')

    def _getValidFileName(self, fileName):
        if not fileName:
            return fileName
        fullFileName = fileName
        if self.defaultDir:
            fullFileName = os.path.join(self.defaultDir, fullFileName)
        dirName = os.path.dirname(fullFileName)
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        return fullFileName

    def _savePlot(self, fileName):
        fullFileName = self._getValidFileName(fileName)
        if fullFileName:
            plt.savefig(fullFileName)


    def savePcolor(self, fileName, data, times, freqs, title, skipOffset=False, clearFirst=True):
        """
        Save a graph to a file
        :param fileName:
        :param data:
        :param times:
        :param freqs:
        :param title:
        :return: void
        """
        try:
            fullFileName = fileName
            if self.defaultDir:
                fullFileName = os.path.join(self.defaultDir, fullFileName)
            if clearFirst:
                plt.cla()
            plt.xlabel(self._default_time_label)
            plt.ylabel(self._default_freq_label)
            if skipOffset:
                plt.pcolormesh((times), (freqs), np.sqrt(data))
            else:
                plt.pcolormesh((times - self.timePseudocolorOffset), (freqs - self.freqPseudocolorOffset), np.sqrt(data))
            plt.title(title)
            # plt.get_current_fig_manager().window.state('zoomed')
            print(fullFileName)
            plt.savefig(fullFileName)
            plt.cla()
        except MemoryError as e:
            data = signal.decimate(data, 10, axis=1)
            times = signal.decimate(times, 10)
            self.savePcolor(fullFileName, data, times, freqs, title)
    #
    # def makeCallScatterPlot(self, argumentDict):
    #     marker = argumentDict.get('marker', 'x')
    #     marker = argumentDict.get('color', 'r')
    #     plt.scatter(argumentDict['call'].times, argumentDict['call'].frequencies, marker=marker, color=color)
    #
    # def makeCallPlot(self, argumentDict):
    #     plt.plot(argumentDict['call'].times, argumentDict['call'](argumentDict['call'].times))
    #
    # def multiPlot(self, chartTypeList, display=False, skipOffset=False, fileName='',
    #               xLabel='Time (ms)', yLabel='Frequency (kHz)' ):
    #     pass

    def getLimits(self, call, xMargin = 0.05, yMargin=8e3):
        if not (isinstance(xMargin, list) or isinstance(xMargin, tuple)):
            xMargin = (xMargin, xMargin)
        if not (isinstance(yMargin, list) or isinstance(yMargin, tuple)):
            yMargin = (yMargin, yMargin)
        xLims = (call.times[0] - xMargin[0], call.times[-1] + xMargin[1])
        xLims = [np.abs(self.times - q).argmin() for q in xLims]
        #yLims = (call.frequencies.min() - yMargin[0], call.frequencies.max() + yMargin[1])
        yLims = (15e3, 100e3)
        yLims = [np.abs(self.freqs - r).argmin() for r in yLims]
        return xLims, yLims

    def makeCompositeCallPlot(self, call, spectrogram=None, clearFirst=True):
        if spectrogram is None:
            spectrogram = self.rawSpectrogram
        xLims, yLims = self.getLimits(call)
        if clearFirst:
           plt.cla()
        plt.pcolormesh((self.times[xLims[0]:xLims[1]] - self.timePseudocolorOffset),
                       (self.freqs[yLims[0]:yLims[1]] - self.freqPseudocolorOffset),
                       spectrogram[yLims[0]:yLims[1], xLims[0]:xLims[1]])
        plt.scatter(call.times, call.frequencies, marker='x', color='r')
        plt.plot(call.times, call(call.times))

    def makeSingleCallPlot(self, call, spectrogram=None, useSqrt=False, clearFirst=True):
        if spectrogram is None:
            spectrogram = self.rawSpectrogram
        fs = 1 / (self.times[1] - self.times[0])
        xLims, yLims = self.getLimits(call)

        plotSpectro = spectrogram[yLims[0]:yLims[1], xLims[0]:xLims[1]]
        if useSqrt:
            plotSpectro = plotSpectro ** 0.5

        plt.pcolormesh((self.times[xLims[0]:xLims[1]] - self.timePseudocolorOffset),
                       (self.freqs[yLims[0]:yLims[1]] - self.freqPseudocolorOffset),
                       plotSpectro)

        plt.scatter(call.times, call.frequencies, marker='x', color='r')
        call.assemble(fs, ((self.times[xLims[0]:xLims[1]]-self.timePseudocolorOffset),
                           (self.freqs[yLims[0]:yLims[1]]-self.freqPseudocolorOffset),
                           spectrogram[yLims[0]:yLims[1], xLims[0]:xLims[1]]))
        x = np.linspace(call.times[0], call.times[-1], 1000)
        plt.plot(x, call(x))

    # def makeCallMeshScatterPlot(self, call, clearFirst=True):
    #     xLims, yLims = self.getLimits(call)
    #
    #     plt.pcolormesh((self.times[xLims[0]:xLims[1]] - self.timePseudocolorOffset),
    #                    (self.freqs[yLims[0]:yLims[1]] - self.freqPseudocolorOffset),
    #                    self.rawSpectrogram[yLims[0]:yLims[1], xLims[0]:xLims[1]])
    #     plt.scatter(call.times, call.frequencies, marker='x', color='r')
    #
    # def makeCallBasePlot(self, call, clearFirst=True):
    #     plt.plot(call.times, call(call.times))

    def makeScatterPlot(self, title='', xLabel=None, xData=None,
                        yLabel=None, yData=None, fileName=None, clearFirst=True):
        if xData is None:
            xData = self.times
        if yData is None:
            yData = self.freqs
        if xLabel is None:
            xLabel = self._default_time_label
        if yLabel is None:
            yLabel = self._default_freq_label
        if clearFirst:
            plt.cla()
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.scatter(xData, yData)
        self._savePlot(fileName)

    def makeCallSpectrogramPlot(self, call, title='call', fileName=None, clearFirst=True):
        xLims, yLims = self.getLimits(call, xMargin=0.1, yMargin=(20000, 10000))
        if clearFirst:
            plt.cla()
        plt.pcolormesh((self.times[xLims[0]:xLims[1]] - self.timePseudocolorOffset),
                       (self.freqs[yLims[0]:yLims[1]] - self.freqPseudocolorOffset),
                       self.rawSpectrogram[yLims[0]:yLims[1], xLims[0]:xLims[1]])
        plt.title(title)
        plt.xlabel(self._default_time_label)
        plt.ylabel(self._default_freq_label)
        self._savePlot(fileName)

    def makeAnnotatedCallPlot(self, call, fileName=None, clearFirst=True):
        if clearFirst:
            plt.cla()
        plt.text(call.times[0] - 50, call.frequencies.mean() - 20, call.callType + '\n' + call.root.__repr__(), color='w')
        x1 = np.arange(call.times[0], call.times[-1], 1e-4)
        plt.plot(x1, call(x1), color='r')
        self._savePlot(fileName)

    def makeCallReport(self, call, parameterList, fileName):
        """
        Write specified call parameters into the file
        :param call:
        :param parameterList: List of tuples of (Row Name, Value) if Value is in parameterToFunction,
                                value comes from the specified function
        :param fileName:
        :return:
        """
        segmentUsed = False
        segmentCount = len(call.root)
        columns = []
        values = [ [] for x in range(segmentCount)]

        for parameter in parameterList:
            columns.append(parameter[0])
            valueParam = parameter[1]

            valueFunction, isSegment = self._getAnalysisFunction(valueParam)
            if valueFunction:
                if isSegment:
                    segmentUsed = True
                    for i, segment in enumerate(call.root):
                        values[i].append(valueFunction(segment, isSegment=True))
                else:
                    value = valueFunction(call, isSegment=False)
                    for i in range(segmentCount):
                        values[i].append(value)
            else:
                for i in range(segmentCount):
                    values[i].append(parameter[1])

        fullFileName = self._getValidFileName(fileName)
        with open(fullFileName, 'w+b') as slopeFile:
            slopeCsv = csv.writer(slopeFile)
            slopeCsv.writerow(columns)
            if not segmentUsed:
                values = values[0]
            slopeCsv.writerows(values)
        del slopeCsv    # Not sure this is needed

    def startLog(self, fileName, parameterList, reuseExisting=False):
        fullFileName = self._getValidFileName(fileName)
        header = [x[0] for x in parameterList]
        analysisList = [x[1] for x in parameterList]
        self.logInfo[fileName] = analysisList[:]
        if not reuseExisting:
            with open(fullFileName, 'w+b') as logFile:
                logCsv = csv.writer(logFile)
                logCsv.writerow(header)
            del logCsv

    def _getAnalysisFunction(self, keyWord):
        parameterToFunction = { "StartFreq": self._get_call_start_freq, "EndFreq": self._get_call_end_freq,
                                "Slope": self._get_call_slope, "Duration": self._get_call_duration,
                                "Type": self._get_call_type, "StartTime": self._get_call_start_time,
                                "AvgFreq": self._get_call_avg_freq, "FreqBw": self._get_call_freq_bandwidth,
                                "Volume": self._get_call_rms, }
        segKeyWord = "Seg "
        totalKeyWord = "Total "

        if keyWord.startswith(segKeyWord):
            segment = True
            funcKey = keyWord[len(segKeyWord):]
        elif keyWord.startswith(totalKeyWord):
            segment = False
            funcKey = keyWord[len(totalKeyWord):]
        else:
            # print("Potential Error! Got parameter of '%s', don't know what that means. reportBuilder._getAnalysisFunction", keyWord)
            return None, False
        if funcKey not in parameterToFunction:
            print("Potential Error! Got parameter of '%s' without a valid keyword. reportBuilder._getAnalysisFunction", keyWord)
            return None, segment
        return parameterToFunction[funcKey], segment

    def _applyAnalysisFunction(self, call, analysisName):
        returnList = []
        segmentCount = len(call.root)
        analysisFunction, isSegment = self._getAnalysisFunction(analysisName)
        if analysisFunction is None:
            for i in range(segmentCount):
                returnList.append("Invalid Function: %s" % analysisFunction)
            return returnList
        if isSegment:
            segmentUsed = True
            for i, segment in enumerate(call.root):
                returnList.append(analysisFunction(segment, isSegment=True))
            return returnList
        else:
            for i in range(segmentCount):
                returnList.append(analysisFunction(call, isSegment=False))
            return returnList

    def _timeToRawIndex(self, time):
        index = int(time * self.sampleRate)
        index = max(index, 0)
        return index

    def _getRawData(self, startTime, endTime):
        [startIndex, endIndex] = [self._timeToRawIndex(time) for time in [startTime, endTime]]
        return self.rawData[startIndex:endIndex+1]

    def makeCallWav(self, call, outFileName):
        wavSlice = self._getRawData(call.times[0] - 0.1, call.times[-1] + 0.1)
        fullFileName = self._getValidFileName(outFileName)
        wavfile.write(fullFileName, 11025, wavSlice)
        del wavSlice    # Unnecessary?

    def addLog(self, call, logName, externalParams={}):
        keyWordList = self.logInfo[logName]
        segmentUsed = False
        segmentCount = len(call.root)
        values = [ [] for x in range(segmentCount)]
        for analysisName in keyWordList:
            if analysisName in externalParams:
                for i in range(segmentCount):
                    values[i].append(externalParams[analysisName])
            else:
                transposedValues = self._applyAnalysisFunction(call, analysisName)
                for i, val in enumerate(transposedValues):
                    values[i].append(val)

        fullFileName = self._getValidFileName(logName)
        with open(fullFileName, 'ab') as logFile:
            logCsv = csv.writer(logFile)
            if not segmentUsed:
                values = values[0]
            logCsv.writerow(values)
        del logCsv
        
    def summarizeCalls(self, callList, fileName, fileIndex, fileStart):
                    
        callSummary = []
        for i in range(len(callList)):
            c = callList[i]
            callSummary.append([os.path.split(fileName)[1], fileIndex, i, 
                                fileStart+c.times[0], fileStart+c.times[-1], 
                                c.times[0], c.times[-1], c.times[-1]-c.times[0],
                                c.callType, fileStart])
            
        return callSummary

if __name__ == "__main__":
    r = ReportBuilder()
    r.testSegments()
