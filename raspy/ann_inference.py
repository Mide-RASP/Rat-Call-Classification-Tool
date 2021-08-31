from __future__ import division
import sys
import os

from tempfile import NamedTemporaryFile

import numpy as np
import scipy

# To appease pyinstaller  (is this still needed here?)
import scipy.special
import scipy.special.cython_special

try:
    import tensorflow as tf
    import tensorflow.python
    import tensorflow.python.keras
    import tensorflow.python.keras.engine
    import tensorflow.python.keras.engine.base_layer_v1
except ImportError:
    print("No TensorFlow Imported!")
    

from raspy.deep_learning import load_examples_and_labels_from_waveform, get_segment_classification_from_ann_output






def classify_calls_transfer_learning(wav_file_or_waveform, frequency=250000, tf_model=None, return_tf_model=False, **kwargs):
    """
    :param wav_file_or_waveform: Either a string, in which case the path for the waveform to
     analyze, or a ndarray of floats containing the waveform data to be analyzed.
    :param frequency: The frequnecy of the waveform data being analyzed.  The default value
     is 250000 Hz.
    :param tf_model: The TensorFlow model to be loaded and used to classify
    :param return_tf_model: If the TensorFlow model used should be returned along with 
     the results of the classification
    :return: tuple, the first element of which is a 2d ndarray of shape [2,N] containing
     the start and end times of the call segments like this:
     [[start_time1, end_time1], ..., [start_timeN, end_timeN]]
     The second element of the tuple will be an array (or list) of length N which
     contains strings, each corresponding to the same index in the returned segment data,
     containing the label for that segment, or None if no label was given.
    """
    desired_freq = 500000 if 'desired_freq' not in kwargs else kwargs['desired_freq']
    samples_per_example = 39124 if 'samples_per_example' not in kwargs else kwargs['samples_per_example']
    window_stride = samples_per_example // 10 if 'window_stride' not in kwargs else kwargs['window_stride']

    wav_file_or_waveform = wav_file_or_waveform.astype(np.float64) / 2**15

    test_wav_examples, _ = load_examples_and_labels_from_waveform(
        waveform=wav_file_or_waveform,
        frequency=frequency,
        call_segments=None,
        samples_per_example=samples_per_example,
        sliding_window_jump_samples=window_stride,
        desired_freq=desired_freq,
        ratio_of_audio_to_check_for_call_in=.5,
    )

    if tf_model is None:
        tf_model = tf.keras.models.load_model(kwargs['model_path'])

    tf_model_result = tf_model.predict(test_wav_examples)

    result = get_segment_classification_from_ann_output(tf_model_result)
    
    if return_tf_model:
        return result, tf_model
    return result

def cli_interface(cli_args):
    """

    :param cli_args: List of command line arguments given when this file is run, this function is here rather than
     within this file's '__main__' program purely for orginizational purposes.  The first index of this list is the
     string of the path to this file (as it is with any Python CLI).  The second value is the path the
     classification model's TensorFlow SavedModel (pretty sure it's of that class).   The Third value is the path to
     the numpy saved array (.npy file) containing the waveform data to analize.  The fourth parameter is the frequency
     of the wav file (defaults to 250000)
    :param output_dir:
    :return:
    """

    run_dir = os.environ['USERPROFILE']#"%USERPROFILE%"#os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    assert all(isinstance(arg, str) for arg in cli_args), 'The cli arguments were not strings, instead the given tpes were: %s'%np.unique(type(arg) for arg in cli_args)


    model_path = cli_args[1]
    waveform = np.load(cli_args[2])

    try:
        frequency = np.float64(cli_args[3])
    except TypeError as E:
        raise TypeError(
            "The fourth value in the given supposed cli arguments could not be converted from the given string form to np.float64.  The given value was: %s" %
            cli_args[3])

    assert frequency > 0, 'the frequency given was %d, which is less than or equal to 0!' % frequency

    output_dir = os.path.join(os.environ['USERPROFILE'], 'Mide', "cli_output.npz") if len(cli_args) < 5 else cli_args[4]

    segments, classifications = classify_calls_transfer_learning(
        waveform,
        frequency,
        model_path=model_path,
        window_stride=39124//10
    )

    np.savez(output_dir, segments=segments, classifications=classifications)



if __name__ == "__main__":
    cli_interface(sys.argv)