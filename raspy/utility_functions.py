# from __future__ import division

try:
    import tensorflow as tf
except ImportError as E:
    print("TensorFlow not imported!")

try:
    from tensorboard.plugins import projector
except ImportError as E:
    print("tensorboard.plugins not imported!")

import os
import os.path

import numpy as np

import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler




def get_files_and_segments(seg_info_csv, get_files_without_segments=True):
    # # A dictionary of filenames mapping to a list of call information of the form:
    # #  (time of call start, time of call end, class of call)
    
    if get_files_without_segments:
        # Adds all the wav files in the directory with with a mapping to an empty list.  The files which do have
        #  classified segments will be filled in with that data within the loop below
        classification_data_dir = os.path.dirname(seg_info_csv)
        answer = {os.path.normpath(os.path.join(classification_data_dir, f)): [] for f in os.listdir(classification_data_dir) if f.endswith('.wav')}
    else:
        answer = {}

    with open(seg_info_csv, 'r') as f:
        f.readline()
        for line in f:
            elements = line.split(',')

            filename = os.path.normpath(elements[0])
            call_start = elements[6]
            call_end = elements[7]
            call_type = elements[9]

            if filename not in answer or answer[filename] is None:
                answer[filename] = [(call_start, call_end, call_type)]
            else:
                answer[filename].append((call_start, call_end, call_type))

    return answer



def create_tf_embedding_output(log_dir, examples, labels):
    # Set up a logs directory, so Tensorboard knows where to look for files
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for label in labels:
            f.write("{}\n".format(label))

    # Save the weights we want to analyse as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, so
    # we will remove that value.
    weights = tf.Variable(examples)
    
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def normalize_the_examples(examples):
    example_ary = np.array([ex for ex in examples if len(ex) != 0])
    
    if len(example_ary) == 0:
        return []
    return StandardScaler().fit_transform(example_ary)

def save_examples_as_npy(log_dir, examples, labels, append_to_filename=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    np.save(os.path.join(log_dir, "examples" + append_to_filename), examples)
    np.save(os.path.join(log_dir, "labels" + append_to_filename), labels)


def create_matplotlib_figure():
    ax = plt.gca()
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False)  # this removes the ticks and numbers for y axis
    plt.show()

