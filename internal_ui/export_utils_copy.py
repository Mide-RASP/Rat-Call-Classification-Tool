from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import abc
import collections
import datetime
import fractions

import wx
import scipy.io.wavfile

import raspy.progressTrackers


# ==============================================================================
#
# ==============================================================================

SECONDS_PER_PIXEL = 2e-5


def calculate_processing_time_est(wav_len, windowSize, overlap):
    # TODO take as parameters the wave data, sampling frequency & classify
    #   parameters -> no future changes to estimate would require a function
    #   signature change
    #   - what to do with batch processing, which does NOT use arg-estimator?
    noverlap = (windowSize-1) if overlap == 1 else int(windowSize*overlap)
    noffset = windowSize - noverlap

    # wav_len = scipy.io.wavfile.read(file, mmap=True)[1].size
    spect_timelen = (wav_len-1) // noffset + 1
    spect_freqlen = windowSize

    return SECONDS_PER_PIXEL * spect_timelen * spect_freqlen


# ==============================================================================
#
# ==============================================================================

class StopExecution(Exception):
    pass


class ProgressTrackerBase(raspy.progressTrackers.ProgressTrackerBase):
    __metaclass__ = abc.ABCMeta

    TaskDescriptor = collections.namedtuple('TaskDescriptor', (
        raspy.progressTrackers.ProgressTrackerBase.TaskDescriptor._fields
        + ('entryMessage', 'exitMessage')
    ))

    def __init__(self, progressBar):
        super(ProgressTrackerBase, self).__init__()
        self.progressBar = progressBar

    @abc.abstractmethod
    def _updateProgressBar(self, message=''):
        pass

    def __call__(self, message='', count=None, increment=None):
        wx.Yield()
        result = super(ProgressTrackerBase, self).__call__(count, increment)
        self._updateProgressBar(message)

        return result

    def _enter(self, entryMessage='', exitMessage='', subtotal=None,
                  uniform=False):
        wx.Yield()
        result = super(ProgressTrackerBase, self)._enter(
            subtotal, uniform,
            entryMessage=entryMessage,
            exitMessage=exitMessage
        )
        self._updateProgressBar(entryMessage)

        return result

    def _exit(self, type, value, traceback):
        wx.Yield()
        self._updateProgressBar(self._task_stack[-1].exitMessage)
        result = super(ProgressTrackerBase, self)._exit(type, value, traceback)

        return result


class RaspyArgsProgressTracker(ProgressTrackerBase):
    def _updateProgressBar(self, message=''):
        percentage = self._getPercentageEstimate()
        partsPerMax = int(percentage*self.progressBar.Range)
        # progress bar will hang once the bar reaches 100%
        # -> artificially ensure bar is incomplete until all events
        #    are processed
        if len(self._task_stack) > 1:
            partsPerMax = min(partsPerMax, self.progressBar.Range-1)

        keepGoing, skipped = self.progressBar.Update(partsPerMax, message)

        # Handle "cancel" button
        if not keepGoing:
            # bubbles up through process; should be caught by original caller
            raise StopExecution


class RaspyProgressTracker(ProgressTrackerBase):
    def _updateProgressBar(self, message=''):
        progTasks = self._getProgressIndicativeTasks()
        if len(progTasks) >= 1:
            count, total = progTasks[0].subcount, progTasks[0].subtotal
            if len(progTasks) >= 2:
                subcount, subtotal = progTasks[1].subcount, progTasks[1].subtotal
            else:
                subcount, subtotal = None, None
        else:
            count, total = None, None

        msg = ''
        # If the total number of files is not given, we cannot calculate a
        #   percentage completion
        if not total:
            if count:
                msg += "Exported {} files".format(count)
            else:
                msg += "Processing..."
            msg += "\n\n" + message
            keepGoing, skipped = self.progressBar.Pulse(msg)

        # If the total number of files *is* given, we can calculate a
        #   percentage completion
        else:
            msg += "Exported {} of {} files".format(count, total)

            if subcount and subtotal:
                msg += "; {} of {} subfiles".format(subcount, subtotal)

            percentage = self._getPercentageEstimate(progTasks)
            msg += " ({:.0%})".format(float(percentage))
            partsPerMax = int(percentage*self.progressBar.Range)
            # progress bar will hang once the bar reaches 100%
            # -> artificially ensure bar is incomplete until all events
            #    are processed
            if len(self._task_stack) > 1:
                partsPerMax = min(partsPerMax, self.progressBar.Range-1)

            if msg:
                msg += '\n\n'
            msg += message

            keepGoing, skipped = self.progressBar.Update(partsPerMax, msg)

        # Handle "cancel" button
        if not keepGoing:
            # bubbles up through process; should be caught by original caller
            raise StopExecution
