from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import scipy.signal as sig
import numpy as np

from raspy.progressTrackers import ProgressPrinter

def estimateArgs(data, fs, callback=ProgressPrinter()):
    if not data.ndim == 1:
        raise ValueError('files must have precisely one audio channel')

    rsm = []
    chunkSize = 2**20
    # TODO remove manual percentage calculations
    #   -> move to `ProgressTrackerBase` subclass
    msgTemplate = 'analyzing file - {:.1%}'
    with callback.context('calculating default arguments', subtotal=len(data),
                          uniform=True):
        for i in xrange(0, len(data), chunkSize):
            completeness = i / len(data)
            callback(msgTemplate.format(completeness), count=i)

            _, _, s = sig.spectrogram(data[i:i+chunkSize], fs=250000,
                                      window='triang', nperseg=512,
                                      noverlap=512*0.75, mode='psd')
            rsm.append(s.mean())
        callback(msgTemplate.format(1), count=len(data))

    del _, s
    rsm = float(np.mean(rsm))

    recArgs = {
        'caHigh': 0.5*rsm,
        'caLow': 0.05*rsm,
        'gaborMaskHigh': 2e2*rsm,
        'gaborMaskLow': 1e2*rsm,
        'gaborSize': 20,
        'majorStdMax': 10000,
        'maxGapTime': 25,
        'minSig': rsm,
        'ngabor': 8,
        'overlap': 0.75,
        'syllableGap': 10,
        'wholeSample': True,
        'windowSize': 512,
        'wings': 3,
    }

    print(
        'Recommended minimum signal: %f, \r\n'
        'Recommended coarse filter low bound: %f \r\n'
        'Recommended coarse filter high bound: %f \r\n'
        'Recommended gabor mask low bound: %f \r\n'
        'Recommended gabor mask high bound: %f'
        % tuple(recArgs[arg] for arg in
                'minSig caLow caHigh gaborMaskLow gaborMaskHigh'.split())
    )

    return recArgs


def main():
    import json
    import os
    import sys
    import scipy.io.wavfile

    filein = sys.argv[1]
    fs, data = scipy.io.wavfile.read(filein, True)

    args = estimateArgs(data, fs)

    fileout = os.path.splitext(filein)[0] + '.json'
    with open(fileout, 'w') as fileHandle:
        json.dump(args, fileHandle, indent=4)


if __name__ == '__main__':
    main()
