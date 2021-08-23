from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import itertools
import os
import shutil
import sys
import json
import time

import scipy
import scipy.io.wavfile as wav
import numpy as np
from raspy.argEstimator import estimateArgs
from raspy.core import Rasper
from raspy.progressTrackers import ProgressPrinter

maxFileSize = 2**23


def splitLargeFile(fileName, arr, args, callback=ProgressPrinter(), sampling_frequency=250000):
    """ Split large files, checking for signals near the end. """
    bigFileData = arr
    # fs, bigFileData = scipy.io.wavfile.read(fileName, True)
    fileList = []
    minFileLen = 5000
    maxFileLen = maxFileSize // bigFileData.itemsize
    lastl = 0
    l = maxFileLen

    with callback.context('partitioning file data', subtotal=len(bigFileData),
                          uniform=True):
        # subtract noise out and check thresholding for basic call avoidance
        _, __, s = scipy.signal.spectrogram(
            bigFileData[:l], fs=sampling_frequency, window='triang', nperseg=512,
            noverlap=384, mode='psd'
        )
        noise = np.array([s.mean(1)]*20).T

        del _, __, s

        # keep breaking the file into chunks while there is still file left
        i = 0
        while lastl < len(bigFileData):
            callback(count=lastl)
            nsignal = 100

            if l + minFileLen < len(bigFileData):
                # inch the end of the chunk back until it's not on a call
                while nsignal > 15:
                    if l - lastl < minFileLen:
                        l = lastl + maxFileLen // 2
                        args['minSig'] *= 10.0**0.5
                    else:
                        l -= 1280
                    _, __, s = scipy.signal.spectrogram(
                        # bigFileData[l-(23*128):l], fs=250000, window='triang',
                        bigFileData[l - (23 * 128):l], fs=sampling_frequency, window='triang',
                        nperseg=512, noverlap=384, mode='psd'
                    )
                    nsignal = ((s-noise)>args['minSig']).sum()
            else:
                l = len(bigFileData)

            # save this chunk as a file
            path, fname = os.path.split(os.path.abspath(fileName))
            newFileName = fileName[:-4] + '\\' + fname[:-4] + ' - ' + str(i) + '.wav'
            try:
                os.makedirs(path + '\\' + fname[:-4])
            except WindowsError:
                pass
            # wav.write(newFileName, fs, bigFileData[lastl:l].copy())
            fileList.append((bigFileData[lastl:l].copy(), lastl, newFileName))

            # move to the next chunk
            lastl = l
            i += 1
            l += maxFileSize//(2*bigFileData.itemsize) + 1280

            callback("broke file into %d subfiles" % (i,), count=len(bigFileData))

    return fileList


def runFiles(sourceFileNames, fileArrays=(), args=(), callback=ProgressPrinter()):
    with callback.context(subtotal=len(sourceFileNames), uniform=True):
        t1 = time.time()
        for file, array, arg in itertools.izip_longest(sourceFileNames, fileArrays, args):
            run(file, array, arg, callback=callback)
            callback(increment=1)

        callback("completed all files in %.1f minutes"
                 % ((time.time() - t1)/60.0))


def run(fileName, array=None, args=None, source=None, callback=ProgressPrinter()):
    callback()

    if array is None:
        _, array = wav.read(fileName)
        array = array.astype(np.float32)
    else:
        array = np.asanyarray(array, dtype=np.float32)

    with callback.context(
        'analyzing {}'.format(fileName),
        '{} analysis complete'.format(fileName),
        subtotal=2, uniform=False
    ):
        absFileName = os.path.abspath(fileName)
        resultsFolderName = os.path.splitext(absFileName)[0]
        coreName = os.path.basename(resultsFolderName)

        argsFileName = resultsFolderName + '.json'

        # make results folder
        if os.path.isdir(resultsFolderName):
            for i in itertools.count(1):
                newFolderName = '{} ({})'.format(resultsFolderName, i)
                if not os.path.isdir(newFolderName):
                    break
            os.rename(resultsFolderName, newFolderName)
        os.mkdir(resultsFolderName)

        if args is None:
            # get the parameters
            if not os.path.exists(argsFileName):
                # generate parameters from data
                fs, data = scipy.io.wavfile.read(absFileName, True)
                args = estimateArgs(data, fs, callback)
                # save to file
                with open(argsFileName, 'w') as argsFile:
                    json.dump(args, argsFile, indent=4)
            else:
                # load these files args
                with open(argsFileName, 'r') as argsFile:
                    args = json.load(argsFile)
            # copy parameters file into results folder
            argsFileCopyName = os.path.join(resultsFolderName, 'parameters.json')
            shutil.copy(argsFileName, argsFileCopyName)

        # if the file is too large, split it (avoiding calls) into several
        # smaller files, and process each one.
        if os.path.getsize(absFileName) > maxFileSize + 128:
            fileList = splitLargeFile(absFileName, array, args, callback)

            fileList = zip(*zip(*fileList)[:-1])#######THIS IS TO ACCOUNT FOR THE ADDITION OF THE FILENAME RETURNING (by removing that info)########
        else:
            fileList = [(array, 0)]

        ################################################################################################################################################

        callback(count=1)
        with callback.context("processing file data", subtotal=len(fileList),
                              uniform=True):
            classifiedChunks = []
            prevEnd = 0.0
            for i, (fileChunk, idx) in enumerate(fileList):
                with callback.context("processing subfile %s of %s"
                                      % (i, len(fileList))):
                    r = Rasper('{1} - {0}{2}'.format(i, *os.path.splitext(fileName)),
                              fileChunk, source, offset=idx, callback=callback)
                    rBuilder = r.classify(**args)
                    rTable = rBuilder.summarizeCalls(r.calls, fileName, i, prevEnd)

                classifiedChunks.append(rTable)
                # r.clearArrays()
                prevEnd += r.length
                callback(count=i+1)

        # HACK: moves generated call count .csv file to results folder
        # TODO change `classify` to put it in the right place from the start
        csvName = resultsFolderName + '.csv'
        if os.path.isfile(csvName):
            newCsvName = os.path.join(resultsFolderName, 'calltype totals.csv')
            os.rename(csvName, newCsvName)

        summaryFileName = os.path.join(resultsFolderName, 'summary.csv')
        with open(summaryFileName, 'w+b') as summaryFile:
            writer = csv.writer(summaryFile, delimiter=b',')
            writer.writerow(['File', 'Sub File Index', 'Call Index',
                             'Abs. Starting Time', 'Abs. End Time',
                             'Rel. Start Time', 'Rel. End Time', 'Call Length',
                             'Call Type', 'File Start Time'])
            for chunk in classifiedChunks:
                writer.writerows(chunk)
        callback(count=2)


def main():
    import glob
    import datetime
    import time

    # Create a list of the files to process
    if len(sys.argv) == 2:
        files = glob.glob(sys.argv[1])

        if not files:
            print("No files matching pattern, %s" % (sys.argv[1],))
            return
        print("found %d files matching input pattern:" % len(files))
        for file in files:
            print('\t' + file)
    else:
        files = sys.argv[1:]
        invalidFiles = [f for f in files if not os.path.isfile(f)]
        if len(invalidFiles) > 0:
            print('file(s) %s not found on the disk' % (repr(invalidFiles),))
            return

    runFiles(files)


if __name__ == '__main__':
    main()
