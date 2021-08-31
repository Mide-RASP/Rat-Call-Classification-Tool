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

    with callback.context('partitioning file data', subtotal=len(bigFileData), uniform=True):
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
            wav.write(newFileName, sampling_frequency, bigFileData[lastl:l].copy())
            fileList.append((bigFileData[lastl:l].copy(), lastl, newFileName))

            # move to the next chunk
            lastl = l
            i += 1
            l += maxFileSize//(2*bigFileData.itemsize) + 1280

            callback("broke file into %d subfiles" % (i,), count=len(bigFileData))

    return fileList
