from __future__ import division

import os

import numpy as np
import scipy
import matplotlib.pyplot as plt

import tempfile


RANDOM_SEED = 3

try:
    import tensorflow as tf
    import tensorflow_hub as hub

    import os
    import datetime
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    from tensorboard_embedding_visualization import create_matplotlib_figure, get_files_and_segments,\
        create_tf_embedding_output, normalize_the_examples

    tf.random.set_seed(RANDOM_SEED)  # For reproducibility

    tf.config.set_visible_devices([], 'GPU')
except ImportError as E:
    print("Issue occurred during imports of tf_hub_testing.py")







call_types = [
    "Complex", "Upward Ramp", "Downward Ramp", "Flat", "Short", "Split", "Step Up", "Step Down", "Multi-Step",
    "Trill", "Flat/Trill Combination", "Trill with Jumps", "Inverted U","Composite", "22-kHz Call", "None"]#, "Other/Unknown"]

call_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 0, 1, 5, 10, 11]

num_call_features = len(np.unique(call_indices))


call_type_to_index_map = dict(zip(call_types, call_indices))
inverse_call_type_to_index_map = {}

for call_type in sorted(call_type_to_index_map):
    call_index = call_type_to_index_map[call_type]
    if call_index in inverse_call_type_to_index_map:
        inverse_call_type_to_index_map[call_index] += "/%s"%call_type
    else:
        inverse_call_type_to_index_map[call_index] = call_type



def rolling_window(a, window_size):
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def visualize_predictions(spectrogram,
                          scores,
                          time_bins=None,
                          freq_vec=None,
                          segments_to_highlight=[],
                          highlight_on_spectrogram_too=True,
                          window_duration=3.92,
                          context_step_secs=1.,
                          highlight_confidence_threshold=.5):
    fig, (spec_axis, score_axis) = plt.subplots(2, gridspec_kw={'height_ratios': [4, 1]})

    if time_bins is None:
        time_bins = np.arange(spectrogram.shape[1])
    else:
        score_axis.set_xlabel("Time (s)")

    if freq_vec is None:
        freq_vec = np.arange(spectrogram.shape[0])
    else:
        spec_axis.set_ylabel("Frequency (Hz)")
        
    spec_axis.pcolormesh(time_bins, freq_vec, spectrogram)
    for j, score in enumerate(scores):
        if score > highlight_confidence_threshold:
            seg_start = j*context_step_secs
            seg_end = j*context_step_secs + window_duration

            score_axis.axvspan(seg_start, seg_end, facecolor=(1, 0, 0), alpha=.5)
            if highlight_on_spectrogram_too:
                spec_axis.axvspan(seg_start*250000, seg_end*250000, facecolor=(1, 0, 0), alpha=.2)

    time_bin_span = time_bins[-1] - time_bins[0]
    if len(scores) == spectrogram.shape[1]:
        score_x_values = np.arange(len(scores))
    else:
        score_x_values = window_duration / 2 + context_step_secs * np.arange(len(scores))

    score_axis.plot(score_x_values, scores)

    score_axis.set_ylim(0, 1)
    score_axis.set_ylabel("'Score'")

    create_matplotlib_figure()


def load_examples_and_labels_from_waveform(waveform,
                                           frequency,
                                           call_segments,
                                           samples_per_example=4096,
                                           sliding_window_jump_samples=1024,
                                           waveform_rescaler_factor=1,
                                           desired_freq=500000,
                                           ratio_of_audio_to_check_for_call_in=.5):
    """
    :param call_segments: If none, they're ignored and this function acts purely as pre-processing, if not it has the
     form: [[start_time_0, end_time_0], [start_time_M, end_time_N]
    """
    # The check for None shouldn't be needed anymore, but didn't want to remove it in case it was used in a place I didn't remember
    has_segments = call_segments is not None and len(call_segments) != 0

    if desired_freq != frequency:
        # This should keep all existing points and fill each space between 2 points with waveform_rescaler_factor
        # number of other points
        waveform_rescaler_factor = desired_freq / frequency

        waveform = scipy.signal.resample(waveform, int(waveform_rescaler_factor * len(waveform) - 1))
        frequency *= waveform_rescaler_factor

    if len(waveform) < samples_per_example:
        return None

    sliding_window_transform = lambda ary: rolling_window(ary, samples_per_example)[::sliding_window_jump_samples]
    
    assert len(waveform.shape) == 1, f"'waveform' should be a 1 dimentional array, but has shape {waveform.shape}"
    
    if has_segments:
        # Boolean array of same shape as waveform (NOT ANY MORE BUT NO TIME TO EXPLAIN :P)
        is_call = np.zeros((len(waveform), num_call_features), dtype=np.bool_)

        call_scaled_times = np.asarray(call_segments[:, :-1], dtype=np.float32)
        
        for (start_index, end_index), seg_label in zip((frequency * call_scaled_times).astype(np.int32), call_segments[:, -1]):
            is_call[start_index: end_index, call_type_to_index_map[seg_label]] = True

        is_call_reshaped = sliding_window_transform(is_call)
    
        samples_not_classifying_on_each_side = int((1-ratio_of_audio_to_check_for_call_in) * is_call_reshaped.shape[1] / 2)
        relevant_is_call_window_for_is_call = is_call_reshaped[:, samples_not_classifying_on_each_side:is_call_reshaped.shape[1]-samples_not_classifying_on_each_side]
        
        example_contains_call = np.any(relevant_is_call_window_for_is_call, axis=1)

    else:
        example_contains_call = None

    waveform_examples = sliding_window_transform(waveform)

    return waveform_examples, example_contains_call


def put_entire_dataset_into_single_numpy_arrays(segment_info_csv_from_tufts_golden_files,
                                                samples_per_example=4096,
                                                sliding_window_jump_samples=1024,
                                                frequency_override=None,
                                                desired_freq=500000,
                                                get_files_without_segments=True,
                                                ratio_of_audio_to_check_for_call_in=.5):
    example_arrays = []
    label_arrays = []
    for segment_info_csv in segment_info_csv_from_tufts_golden_files:
        for (wav_file, segments_with_classifications) in get_files_and_segments(segment_info_csv, get_files_without_segments=get_files_without_segments).items():
            freq, waveform = scipy.io.wavfile.read(wav_file)
            
            waveform = waveform / (2 ** 15)
            
            cur_waveform_examples, cur_labels = load_examples_and_labels_from_waveform(
                waveform=waveform,
                frequency=freq if frequency_override is None else frequency_override,
                call_segments=np.asarray(segments_with_classifications) if len(segments_with_classifications) else np.array([]),
                samples_per_example=samples_per_example,
                sliding_window_jump_samples=sliding_window_jump_samples,
                desired_freq=desired_freq,
                ratio_of_audio_to_check_for_call_in=ratio_of_audio_to_check_for_call_in,
            )
            
            if np.max(cur_waveform_examples) <= 1:
                if cur_labels is None:
                    cur_labels = np.zeros((len(cur_waveform_examples), num_call_features), dtype=np.bool_)

                example_arrays.append(cur_waveform_examples)
                label_arrays.append(cur_labels)

    examples = np.concatenate(example_arrays)
    labels = np.concatenate(label_arrays)

    shuffle_indicies = np.arange(len(labels), dtype=np.int32)
    np.random.shuffle(shuffle_indicies)

    examples, labels = examples[shuffle_indicies], labels[shuffle_indicies]

    return examples, labels


def get_segment_classification_from_ann_output(ann_output, smoothing_window_length=5, pad_ratio_to_remove=0):
    assert smoothing_window_length % 2 == 1

    # TODO: I Should probably explain this magic number...
    time_step_for_examples = 0.0078248

    ann_output = rolling_window(ann_output, smoothing_window_length).mean(axis=1)

    smoothed_catagory_indices = class_catagory_indecies = np.argmax(ann_output, axis=-1)

    smoothed_is_no_call = smoothed_catagory_indices == call_type_to_index_map['None']
    
    call_start_or_stop_indecies = np.nonzero(np.diff(smoothed_is_no_call))[0]
    call_start_or_stop_indecies = call_start_or_stop_indecies.astype(np.float64)

    if smoothed_catagory_indices[0] != call_type_to_index_map['None']:
        call_start_or_stop_indecies = np.r_[-1e-5, call_start_or_stop_indecies]
    if smoothed_catagory_indices[-1] != call_type_to_index_map['None']:
        call_start_or_stop_indecies = np.r_[call_start_or_stop_indecies, len(smoothed_catagory_indices)-1-1e-5]


    num_segments = len(call_start_or_stop_indecies) // 2

    call_indices = np.array([smoothed_catagory_indices[call_start_or_stop_indecies[2*j].astype(np.int64) + 1] for j in range(num_segments)])
    call_labels = np.array(list(map(inverse_call_type_to_index_map.get, call_indices)))

    call_start_or_stop_indecies = call_start_or_stop_indecies.reshape((-1, 2))

    padding_to_remove = pad_ratio_to_remove * (smoothing_window_length - 1) / 2

    call_start_or_stop_indecies += (smoothing_window_length-1) / 2
    #call_start_or_stop_indecies[:, 0] += (smoothing_window_length-1) / 2
    #call_start_or_stop_indecies[:, 1] -= (smoothing_window_length-1) / 2

    TEMP_DIFF_PADDINGS = (call_start_or_stop_indecies[:, 1] - call_start_or_stop_indecies[:, 0])/4
    #call_start_or_stop_indecies[:,0] += TEMP_DIFF_PADDINGS
    #call_start_or_stop_indecies[:,1] -= TEMP_DIFF_PADDINGS
    call_start_or_stop_indecies += TEMP_DIFF_PADDINGS[:, np.newaxis]

    call_start_or_stop_indecies[:, 0] += padding_to_remove
    call_start_or_stop_indecies[:, 1] -= padding_to_remove
    
    
    call_start_or_stop_indecies *= time_step_for_examples
    
    space_between_calls = call_start_or_stop_indecies[1:, 0] - call_start_or_stop_indecies[:-1, 1]

    overlapping_call_mask = space_between_calls < 0
    
    
    # There's certainly cleaner ways of merging segments of different and the same class, but this works well enough for now.
    # There is a situation where this will actually cause issues.  Specifically when there are more than 2 consecutively
    # overlapping clusters (e.g. [[1,5],[2,6],[4,10]) all of the same class prediction, it will merge the first two and lose
    # track of any additional segments alltogether.  This is something that should really be addressed.
    while True:
        same_call_consecutive = np.diff(call_indices) == 0
        overlapping_same_call = overlapping_call_mask & same_call_consecutive
        if np.any(overlapping_same_call):
            to_remove_mask = np.r_[False, overlapping_same_call]
        
            call_start_or_stop_indecies[np.r_[overlapping_same_call ,False], 1] = call_start_or_stop_indecies[to_remove_mask, 1] 
            
            call_start_or_stop_indecies = call_start_or_stop_indecies[~ to_remove_mask]
            call_indices = call_indices[~ to_remove_mask]
            call_labels = call_labels[~ to_remove_mask]
            
        elif np.any(overlapping_call_mask):
            print("we did it!")
            
            abs_distance_between_calls = np.abs(space_between_calls)
            
            call_start_or_stop_indecies[np.r_[False, overlapping_call_mask],0] += abs_distance_between_calls[overlapping_call_mask]/2
            
            call_start_or_stop_indecies[np.r_[overlapping_call_mask, False],1] -= abs_distance_between_calls[overlapping_call_mask]/2
            
            should_remove = np.squeeze(np.diff(call_start_or_stop_indecies, axis=1)) <= 0
            print(should_remove)
            print(should_remove.shape)
            if np.any(should_remove):
                call_start_or_stop_indecies = call_start_or_stop_indecies[~ should_remove, :]
                call_indices = call_indices[~ should_remove]
                call_labels = call_labels[~ should_remove]
        
        assert all(x[0] < x[1] for x in call_start_or_stop_indecies), str(call_start_or_stop_indecies) + '\n\n' + str(call_labels)
        space_between_calls = call_start_or_stop_indecies[1:, 0] - call_start_or_stop_indecies[:-1, 1]

        overlapping_call_mask = space_between_calls < 0
        if not np.any(overlapping_call_mask):
            break
            
    return call_start_or_stop_indecies, call_labels


def save_features_for_frozen_network(save_dir,
                                     hub_model_location,
                                     golden_file_csv_file,
                                     samples_per_example=4096,
                                     sliding_window_jump_samples=1024,
                                     data_name_addition="",
                                     frequency=250000,
                                     desired_freq=500000,
                                     get_files_without_segments=True,
                                     ratio_of_audio_to_check_for_call_in=.5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hub_model = hub.load(hub_model_location)

    wavform_examples, labels = put_entire_dataset_into_single_numpy_arrays(
        [golden_file_csv_file],
        samples_per_example=samples_per_example,
        sliding_window_jump_samples=sliding_window_jump_samples,
        frequency_override=frequency,
        desired_freq=desired_freq,
        get_files_without_segments=get_files_without_segments,
        ratio_of_audio_to_check_for_call_in=ratio_of_audio_to_check_for_call_in
    )

    spectrogram_examples = hub_model.front_end(wavform_examples[..., np.newaxis])

    # assert tf.reduce_all(spectrogram_examples.shape[-2:] == [128, 64]), "Spectrogram's being produced by front-end are shape %s, not the expected [128, 64]" % spectrogram_examples.shape[-2:]

    frozen_feature_examples = hub_model.features(spectrogram_examples)

    np.save(os.path.join(save_dir, "raw_waveform_examples" + data_name_addition), wavform_examples)
    np.save(os.path.join(save_dir, "labels" + data_name_addition), labels)
    np.save(os.path.join(save_dir, "frozen_feature_examples" + data_name_addition), frozen_feature_examples)

    print(frozen_feature_examples.shape)
    print(labels.shape)
    print(wavform_examples.shape)

    # I have no idea when someone would want this returned, but it feels like allowing chainability is always positive
    return frozen_feature_examples




def train_logits_on_frozen_features(saved_data_dir,
                                    training_parameters,
                                    test_wav_filename=None,
                                    save_model_path=None,
                                    run_version_suffix="",
                                    num_output_classes=1,
                                    frequency=250000):
                                    
    raw_waveform_examples = np.load(os.path.join(saved_data_dir, "raw_waveform_examples%s.npy"%run_version_suffix))
    frozen_feature_examples = np.load(os.path.join(saved_data_dir, "frozen_feature_examples%s.npy"%run_version_suffix))
    labels = np.load(os.path.join(saved_data_dir, "labels%s.npy"%run_version_suffix))
    
    if num_output_classes == 1:
        labels = np.any(labels, axis=-1)[..., np.newaxis]
    else:
        no_call_mask = np.sum(labels, axis=(-1)) == 0

        # This is removing the multi-class examples
        not_multi_class_exapmle_mask = np.sum(labels, axis=(-1)) <= 1

        frozen_feature_examples = frozen_feature_examples[not_multi_class_exapmle_mask]
        raw_waveform_examples = raw_waveform_examples[not_multi_class_exapmle_mask]
        labels = labels[not_multi_class_exapmle_mask]
        
        labels[:, -1] = no_call_mask[not_multi_class_exapmle_mask]

        assert labels.shape[1] == num_call_features, "Labels are shape: %s"%str(labels.shape)

        PLOT_BAR_GRAPH_OF_CLASS_REPRESENTATION = False    
        if PLOT_BAR_GRAPH_OF_CLASS_REPRESENTATION:
            call_type_occurance_counts = np.sum(labels, axis=0)
            plt.bar(np.arange(num_call_features), call_type_occurance_counts)
            for j in range(num_call_features):
                plt.annotate(str(call_type_occurance_counts[j]), xy=(j, call_type_occurance_counts[j]), ha='center', va='bottom')
            plt.show()

    train_wavs, test_wavs, train_features, test_features, train_labels, test_labels = train_test_split(
        raw_waveform_examples[..., np.newaxis], frozen_feature_examples, labels,
        test_size=training_parameters['validation_ratio'], random_state=RANDOM_SEED,
    )

    tensorboard_log_dir = os.path.join(save_model_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=100)

    training_metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]

    if num_output_classes == 1:
        loss_function = tf.keras.losses.BinaryCrossentropy()
    else:
        loss_function = tf.keras.losses.CategoricalCrossentropy()

    logit_layer = tf.keras.layers.Dense(
        num_output_classes,
        activation='sigmoid' if num_output_classes == 1 else 'softmax',
    )

    model_head_trainer = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=train_features.shape[1:]),
        tf.keras.layers.Dropout(training_parameters['logit_dropout_rate']),
        logit_layer
    ])

    model_head_trainer.compile(
        optimizer=tf.keras.optimizers.Adam(training_parameters['head_training_learning_rate']),
        loss=loss_function,
        metrics=training_metrics,
    )

    model_head_trainer.fit(
        train_features,
        train_labels,
        epochs=training_parameters['final_head_trainer_epoch'],
        callbacks=[tensorboard_callback],
        validation_data=(test_features, test_labels),
        batch_size=training_parameters['batch_size'],
    )


    hub_model_location = "https://tfhub.dev/google/humpback_whale/1"
    hub_model = hub.load(hub_model_location)

    model_fine_tuner = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=train_wavs.shape[1:]),
        hub.KerasLayer(hub_model.front_end, trainable=True),
        hub.KerasLayer(hub_model.features, trainable=True),
        tf.keras.layers.Dropout(training_parameters['logit_dropout_rate']),
        logit_layer,
    ])

    model_fine_tuner.compile(
        optimizer=tf.keras.optimizers.Adam(training_parameters['fine_tuning_learning_rate']),
        loss=loss_function,
        metrics=training_metrics,
    )

    RUN_DATA_VERIFICATION = False
    if RUN_DATA_VERIFICATION:
        # Made to ensure the waveform examples fed through the frozen parts of the pre-trained models will equal the
        # data saved for those examples as features
        NUM_SAMPLES_TO_CHECK = 100
        spec_for_verification = hub_model.front_end(train_wavs[:NUM_SAMPLES_TO_CHECK])
        features_for_verification = hub_model.features(spec_for_verification)

        features_saved_as_data = train_features[:NUM_SAMPLES_TO_CHECK]

        assert np.allclose(features_for_verification.numpy(), features_saved_as_data), "Features produced from waveforms are not equal to saved features!  Maximum magnitude of difference in feature value is: %f"%np.max(np.abs(features_for_verification - train_features[:NUM_SAMPLES_TO_CHECK]))


        calculated_probabilities = logit_layer(features_saved_as_data)

        model_head_predictions = model_head_trainer.predict(features_saved_as_data)


        assert np.allclose(calculated_probabilities.numpy(), model_head_predictions), "PROBABILITIES SHOULD BE EQUAL.  Max absolute difference is: %f" % np.max(np.abs(calculated_probabilities.numpy() - model_head_predictions))

    model_fine_tuner.fit(
        train_wavs,
        train_labels,
        epochs=training_parameters['final_fine_tuner_epoch'],
        callbacks=[tensorboard_callback],
        validation_data=(test_wavs, test_labels),
        batch_size=training_parameters['batch_size'],
        initial_epoch=training_parameters['final_head_trainer_epoch'],
    )
    
    model_fine_tuner.save(save_model_path)
    
    
    # Run the trained model on example audio and plot the results
    if test_wav_filename is not None:
        sample_rate, og_waveform = scipy.io.wavfile.read(test_wav_filename)
        
        og_waveform = og_waveform.astype(np.float32)
        waveform = og_waveform / 2**15

        test_wav_examples, _ = load_examples_and_labels_from_waveform(
            waveform=waveform,
            frequency=frequency,
            call_segments=None,
            samples_per_example=39124,
            sliding_window_jump_samples=39124//10,
            desired_freq=training_parameters['desired_frequency'],
            ratio_of_audio_to_check_for_call_in=training_parameters['ratio_of_audio_being_classified']
        )

        test_wav_examples = test_wav_examples.astype(np.float32)
        pcen_spectrogram = hub_model.front_end(test_wav_examples[..., np.newaxis])

        predictions = model_fine_tuner.predict(test_wav_examples[..., np.newaxis])

        VERIFY_RELOADED_MODEL = False
        if VERIFY_RELOADED_MODEL:
            reloaded = tf.keras.models.load_model(OUTPUT_PATH)

            reloaded_result_batch = reloaded.predict(test_wav_examples)

            assert abs(reloaded_result_batch - predictions).max() == 0, "Saved and reloaded model did not create the same predictions as the one just trained!"

        scores = tf.reduce_max(predictions[:, :-1], axis=-1)
        choose_no_call = (scores < predictions[:, -1]).numpy()

        scores = scores.numpy()
        scores[choose_no_call] = 0

        visualize_predictions(
            spectrogram=scipy.signal.spectrogram(waveform, fs=250000, mode='magnitude')[2],
            scores=np.squeeze(scores),
            highlight_on_spectrogram_too=True,
        )



def final_train_logits_on_frozen_features(saved_data_dir, training_parameters, save_model_path=None,run_version_suffix="", num_output_classes=1, frequency=250000):

    raw_waveform_examples = np.load(os.path.join(saved_data_dir, "raw_waveform_examples%s.npy"%run_version_suffix))
    frozen_feature_examples = np.load(os.path.join(saved_data_dir, "frozen_feature_examples%s.npy"%run_version_suffix))
    labels = np.load(os.path.join(saved_data_dir, "labels%s.npy"%run_version_suffix))




    # call_type_occurance_counts = np.sum(labels, axis=0)
    # plt.bar(np.arange(num_call_features), call_type_occurance_counts)#call_types)
    # for j in range(num_call_features):
    #     plt.annotate(str(call_type_occurance_counts[j]), xy=(j, call_type_occurance_counts[j]), ha='center', va='bottom')
    # plt.show()

    print(raw_waveform_examples.shape, frozen_feature_examples.shape, labels.shape)

    if num_output_classes == 1:
        labels = np.any(labels, axis=-1)[..., np.newaxis]
    else:

        print("DA RATIO WITH MORE THAN 1 CALL IN SEGMENT PRE REMOVAL", np.mean(np.sum(labels, axis=(-1)) > 1))
        no_call_mask = np.sum(labels, axis=(-1)) == 0

        # labels[no_call_mask, call_type_to_index_map["None"]] = 1

        # This is removing the multi-class examples
        not_multi_class_exapmle_mask = np.sum(labels, axis=(-1)) <= 1


        # labels[:, -1] = no_call_mask

        frozen_feature_examples = frozen_feature_examples[not_multi_class_exapmle_mask]
        raw_waveform_examples = raw_waveform_examples[not_multi_class_exapmle_mask]
        labels = labels[not_multi_class_exapmle_mask]
        print("LABEL TYPE", labels.dtype)
        labels[:, -1] = no_call_mask[not_multi_class_exapmle_mask]

        assert labels.shape[1] == num_call_features, "Labels are shape: %s"%str(labels.shape)

        # assert labels.shape[1] + 1 == num_call_features, "Labels are shape: %s"%str(labels.shape)
        #
        # # Should be np.empty for speed but reversing this to ensure things are working properly
        # temp_labels = np.zeros((len(labels), num_call_features), dtype=np.bool_)
        # temp_labels[:, :-1] = labels
        # temp_labels[:, -1] = no_call_mask[not_multi_class_exapmle_mask]
        # labels = temp_labels



        print("label shape", labels.shape)

    # with tf.distribute.MirroredStrategy().scope():
    # strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # with strategy.scope()
    train_wavs, test_wavs, train_features, test_features, train_labels, test_labels = train_test_split(
        raw_waveform_examples[..., np.newaxis], frozen_feature_examples, labels,
        test_size=training_parameters['validation_ratio'], random_state=RANDOM_SEED,
    )

    print("DATA SHAPES", train_wavs.shape, train_features.shape, train_labels.shape)


    # # input_layer = tf.keras.Input(shape=)
    # # input_dropout = tf.keras.layers.Dropout(logit_dropout_rate)(input_layer)
    # # # logits = tf.keras.layers.Dense(1, activation=)(input_dropout)
    # # logits = tf.keras.layers.Dense(
    # #     num_output_classes,
    # #     activation='sigmoid' if num_output_classes == 1 else 'softmax',
    # #     # kernel_regularizer=tf.keras.regularizers.l2(0.001),
    # # )(input_dropout)
    #
    # # logits = tf.nn.sigmoid(dense_layer)
    # print(logits.shape, labels.shape)
    # model = tf.keras.Model(inputs=input_layer, outputs=logits, name='logit_trainer_model')

    tensorboard_log_dir = os.path.join(save_model_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=100)

    training_metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]

    if num_output_classes == 1:
        loss_function = tf.keras.losses.BinaryCrossentropy()
    else:
        loss_function = tf.keras.losses.CategoricalCrossentropy()

    logit_layer = tf.keras.layers.Dense(
        num_output_classes,
        activation='sigmoid' if num_output_classes == 1 else 'softmax',
        #kernel_regularizer=tf.keras.regularizers.l2(.0 1),
    )

    # # with tf.device('/cpu:0'):
    # model_head_trainer = tf.keras.Sequential([
        # tf.keras.layers.InputLayer(input_shape=train_features.shape[1:]),
        # #tf.keras.layers.Dropout(training_parameters['logit_dropout_rate']),
        # #tf.keras.layers.Dense(256, activation='relu',), #kernel_regularizer=tf.keras.regularizers.l2(.0 1),
        # tf.keras.layers.Dropout(training_parameters['logit_dropout_rate']),
    # #)

        # logit_layer
    # ])

    # model_head_trainer.compile(
        # optimizer=tf.keras.optimizers.Adam(training_parameters['head_training_learning_rate']),
        # loss=loss_function,
        # metrics=training_metrics,
    # )
    # print(train_features[:10])
    # print(train_labels[:10])
    # print(train_labels.dtype)
    # print(model_head_trainer.summary())

    # print("TRAIN WAV SHAPE", train_wavs.shape)
    # print(np.max(train_wavs), np.min(train_wavs))

    # model_head_trainer.fit(
        # train_features,
        # train_labels,
        # epochs=training_parameters['final_head_trainer_epoch'],
        # callbacks=[tensorboard_callback],
        # validation_data=(test_features, test_labels),
        # batch_size=training_parameters['batch_size'],
    # )


    # _, pretrained_weights = tempfile.mkstemp('.tf')
    # model_head_trainer.save_weights(pretrained_weights)

    hub_model_location = "https://tfhub.dev/google/humpback_whale/1"
    hub_model = hub.load(hub_model_location)




    model_fine_tuner = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=train_wavs.shape[1:]),
        hub.KerasLayer(hub_model.front_end, trainable=False),
        hub.KerasLayer(hub_model.features, trainable=False),
        tf.keras.layers.Dropout(training_parameters['logit_dropout_rate']),
        logit_layer,
    ])

    # model_fine_tuner.load_weights(pretrained_weights)

    model_fine_tuner.compile(
        optimizer=tf.keras.optimizers.Adam(training_parameters['fine_tuning_learning_rate']),
        loss=loss_function,
        metrics=training_metrics,
    )

    RUN_DATA_VERIFICATION = False
    if RUN_DATA_VERIFICATION:
        # Made to ensure the waveform examples fed through the frozen parts of the pre-trained models will equal the
        # data saved for those examples as features
        NUM_SAMPLES_TO_CHECK = 100
        spec_for_verification = hub_model.front_end(train_wavs[:NUM_SAMPLES_TO_CHECK])
        features_for_verification = hub_model.features(spec_for_verification)

        features_saved_as_data = train_features[:NUM_SAMPLES_TO_CHECK]
        print(features_saved_as_data)

        print(features_for_verification)
        # have_same_values = features_for_verification == features_saved_as_data
        # print("HAVE SAME VALUE SHAPE", have_same_values.shape)

        assert np.allclose(features_for_verification.numpy(), features_saved_as_data), "Features produced from waveforms are not equal to saved features!  Maximum magnitude of difference in feature value is: %f"%np.max(np.abs(features_for_verification - train_features[:NUM_SAMPLES_TO_CHECK]))


        calculated_probabilities = logit_layer(features_saved_as_data)

        model_head_predictions = model_head_trainer.predict(features_saved_as_data)


        assert np.allclose(calculated_probabilities.numpy(), model_head_predictions), "PROBABILITIES SHOULD BE EQUAL.  Max absolute difference is: %f" % np.max(np.abs(calculated_probabilities.numpy() - model_head_predictions))

    # model_fine_tuner.get_weights()
    # model_fine_tuner.set_weights(logit_layer.get_weights())
    # model_fine_tuner.get_weights()

    model_fine_tuner.fit(
        train_wavs,
        train_labels,
        epochs=training_parameters['final_fine_tuner_epoch'],
        callbacks=[tensorboard_callback],
        validation_data=(test_wavs, test_labels),
        batch_size=training_parameters['batch_size'],
        initial_epoch=training_parameters['final_head_trainer_epoch'],
    )


    # create_tf_embedding_output(
    #     os.path.join(saved_data_dir, 'pre_trained_feature_embedding'),
    #     frozen_feature_examples,
    #     labels,
    # )


    # print("RATIO EXAMPLES HAVE CALLS:", np.mean(labels))


    # model.summary()



    # tf.keras.models.save_model(model_head_trainer, OUTPUT_PATH)

    # model_fine_tuner.save("AHHHHHHHHHHHH-ann_model")
    model_fine_tuner.save(save_model_path)

    # if save_model_path is not None:
    #     model.save(save_model_path)
    
        # scores = model.predict(test_wav_examples)
    # with tf.device('/cpu:0'):
    TESTING_WAV_FILENAME = "C:\\Users\\samra\\Desktop\\ann_running_4-1-2021\\ann_training\\878Pre2 - 0.wav"#"C:\\Users\\Sam\\PycharmProjects\\Raspy\\raspy\\878Pre2 - 0.wav"
    # TESTING_WAV_FILENAME = "F:\\mide\\rat_chat\\data\\play_files_TO_REMOVE\\879Fight1.wav"
    # TESTING_WAV_FILENAME = "F:\\mide\\rat_chat\\data\\play_files_TO_REMOVE\\878Post4.wav"

    sample_rate, og_waveform = scipy.io.wavfile.read(TESTING_WAV_FILENAME)
    # og_waveform = og_waveform[24 * 250000: 28* 250000]#og_waveform[50 * 250000:54 * 250000]
    # og_waveform = og_waveform[50 * 250000:54 * 250000]
    og_waveform = og_waveform.astype(np.float32)
    waveform = og_waveform / 2**15

    test_wav_examples, _ = load_examples_and_labels_from_waveform(
        waveform=waveform,
        frequency=frequency,
        call_segments=None,
        samples_per_example=39124,
        sliding_window_jump_samples=39124//10,
        desired_freq=training_parameters['desired_frequency'],
        ratio_of_audio_to_check_for_call_in=training_parameters['ratio_of_audio_being_classified']
    )

    ahhhhhhh, _ = load_examples_and_labels_from_waveform(
        waveform=waveform,
        frequency=frequency,
        call_segments=None,
        samples_per_example=39124,
        sliding_window_jump_samples=39124,
        desired_freq=training_parameters['desired_frequency'],
        ratio_of_audio_to_check_for_call_in=training_parameters['ratio_of_audio_being_classified'],
    )
    print(ahhhhhhh.shape)
    #ahhhhhhh = ahhhhhhh.reshape((-1, 39124, 1)).astype(np.float32)
    ahhhhhhh = ahhhhhhh[..., np.newaxis].astype(np.float32)#.reshape((-1, 39124, 1)).astype(np.float32)

    print(test_wav_examples)
    test_wav_examples = test_wav_examples.astype(np.float32)
    pcen_spectrogram = hub_model.front_end(test_wav_examples[..., np.newaxis])


    predictions = model_fine_tuner.predict(test_wav_examples[..., np.newaxis])

    # print(get_segment_classification_from_ann_output(predictions))


    VERIFY_RELOADED_MODEL = False
    if VERIFY_RELOADED_MODEL:
        reloaded = tf.keras.models.load_model(OUTPUT_PATH)

        reloaded_result_batch = reloaded.predict(test_wav_examples)

        assert abs(reloaded_result_batch - predictions).max() == 0, "Saved and reloaded model did not create the same predictions as the one just trained!"

    scores = tf.reduce_max(predictions[:, :-1], axis=-1)
    choose_no_call = (scores < predictions[:, -1]).numpy()

    scores = scores.numpy()
    scores[choose_no_call] = 0

    plt.plot(np.argmax(predictions, axis=-1))
    plt.show()

    # pcen_spectrogram = pcen_spectrogram.numpy()[0].Td
#     # print(pcen_spectrogram.shape)
#     #
#     # score_fn = hub_model.signatures['score']
#     # scores = score_fn(waveform=waveform, context_step_samples=context_step_samples)['scores']
#     # print(scores['scores'].shape)
#     # print(waveform.shape[1] / sample_rate)
#
#
    plt.pcolormesh(pcen_spectrogram.numpy()[0].T)
    plt.show()
    # print(ahhhhhhh.numpy()[0].T.shape)
    visualize_predictions(
        spectrogram=pcen_spectrogram.numpy()[0].T,#hub_model.front_end(ahhhhhhh).numpy()[0].T,
        # time_bins=np.linspace(0, waveform.shape[1] / sample_rate, pcen_spectrogram.shape[1]),
        #time_bins=np.linspace(0, test_wav_examples.shape[1] / sample_rate, pcen_spectrogram.shape[1]),
        scores=np.squeeze(scores),  # np.random.rand(pcen_spectrogram.shape[1]),
        # segments_to_highlight=[[30, 100], [400, 555], [666, 750], [1600, 2100]],
        highlight_on_spectrogram_too=True,
        # context_step_secs=context_step_samples/1000,
    )


def run_cli_executable(waveform, frequency=250000, **kwargs):
    run_dir = os.environ['USERPROFILE']

    run_resource_dir = os.path.join(run_dir, 'mide', 'raspy_resources')
    if not os.path.exists(run_resource_dir):
        if not os.path.exists(os.path.join(run_dir, 'mide')):
            os.mkdir(os.path.join(run_dir, 'mide'))
        os.mkdir(run_resource_dir)

    output_ary_file_path = os.path.join(run_resource_dir, 'cli_output.npz')
    cli_executable_path = os.path.join(run_resource_dir, 'tensorflow_CLI_model_runner.exe')

    if 'model_path' in kwargs:
        model_path = kwargs['model_path']
    else:
        model_path = os.path.join(run_resource_dir, 'tensorflow_model_files')

    temp_wav_ary = os.path.join(run_resource_dir, 'np_waveform')
    np.save(temp_wav_ary, waveform)

    command_str = 'cd %s && "%s" "%s" "%s" %d "%s"' % (
        run_dir, cli_executable_path, model_path, temp_wav_ary + '.npy', frequency, output_ary_file_path)

    os.system(command_str)

    with np.load(output_ary_file_path) as loaded_results:
        call_segments = loaded_results['segments']
        call_classifications = loaded_results['classifications'].astype(str)

    return call_segments, call_classifications


def create_tf_embeddings_from_saved_feature_data(saved_data_dir, output_dir="", run_version_suffix=""):
    # raw_waveform_examples np.load(os.path.join(saved_data_dir, "raw_waveform_examples.npy"))
    frozen_feature_examples = np.load(os.path.join(saved_data_dir, "frozen_feature_examples%s.npy"%run_version_suffix))
    mask = np.load(os.path.join(saved_data_dir, "labels%s.npy"%run_version_suffix))

    # standardized_features = normalize_the_examples(frozen_feature_examples)
    labels = 1 + np.argmax(mask, axis=-1)
    labels[np.all(np.logical_not(mask), axis=-1)] = 0

    create_tf_embedding_output(output_dir, frozen_feature_examples, labels)


if __name__ == '__main__':
    """
    This is one of the places I've used when exploring ways of representing the data and exploring different
    deep learning architectures, as such it's a bit messy, but I figure it's better to leave it visible
    than hide use-cases of the code.
    
    The model training at the bottom of this file is what's used in this program.
    """

    hub_model_loc = "https://tfhub.dev/google/humpback_whale/1"

    #SAVE_FROZEN_DATA_DIR = "ann_training/connors_classification_try_4_blank_subfiles_not_omitted_no_overlap/"
    SAVE_FROZEN_DATA_DIR = "ann_training/more_of_my_data_only_center_half_calls/"
    #
    session_filenames_part = [f'partially_classified/partial - {j}' for j in range(1,3)] + [f'fully_classified/full - {j}' for j in range(1,7)]
    
    files_fully_classified = 2 * [False] + 6 * [True]
    # session_file_num = 5
    # session_filename = session_filenames_part[session_file_num]
    # print("Session filename:", session_filename)
    run_version_suffix = "first_try"
    # suffix_addition = "_%s%s" % (session_filename, run_version_suffix)

    ###############################################################################################################################
    
    # create_tf_embeddings_from_saved_feature_data(
    #     saved_data_dir=SAVE_FROZEN_DATA_DIR,
    #     output_dir=os.path.join(SAVE_FROZEN_DATA_DIR, "embeddings" + suffix_addition),
    #     run_version_suffix=suffix_addition,
    # )


    ####################################################################################################################################
    # for session_file_num, get_files_without_segments in enumerate(files_fully_classified):
        # session_filename = session_filenames_part[session_file_num]
        #print("Session filename:", session_filename)
        # suffix_addition = "_%s%s" % (session_filename, run_version_suffix)
        # save_features_for_frozen_network(
            # save_dir=SAVE_FROZEN_DATA_DIR,
            # hub_model_location=hub_model_loc,
            #golden_file_csv_file="C:/work/mide/rat_chat/golden_files/Sessions 2 and 6/weird_combo/segments_info.csv",
            #golden_file_csv_file="C:/work/mide/rat_chat/golden_files/Sessions 2 and 6/%s/segments_info.csv"%session_filename,
            #golden_file_csv_file="F:\\mide\\rat_chat\\data\\connors_manual_segmentation\\%s\\segments_info.csv"%session_filename,
            # golden_file_csv_file="F:/mide/rat_chat/data/sams_secret_spectral_stash/round_1/COLLECTION_FOR_FIRST_USE_COPIES/%s/segments_info.csv"%session_filename,
            # samples_per_example=39124,#2*8192,
            # sliding_window_jump_samples=39124,#4096,
            # data_name_addition=suffix_addition,
            # frequency=250000,
            # desired_freq=500000,
            # get_files_without_segments=get_files_without_segments,
            # ratio_of_audio_to_check_for_call_in=1,
        # )

    ########################################################################################################################################
    #
    # features = []
    # labels = []
    # waveforms = []
    # # # session_filenames_part = ["session002", "session006"] + [("_%d"%j) + deepsqueak_suffix for j in range(1,5)]
    # # # session_filenames_part = ["session002"] + [("_part_%d" % j) + deepsqueak_suffix for j in range(1, 5)]
    # # # is_deep_squeak = [False] + (len(session_filenames_part) - 1) * [True]
    # # # session_filenames_part = [("_part_%d" % j) + deepsqueak_suffix for j in range(1, 4)]
    # # # is_deep_squeak = len(session_filenames_part) * [True]
    # # session_filenames_part = ["session002", "session006"]
    # # is_deep_squeak = 2 * [False]

    # is_deep_squeak = len(session_filenames_part) * [False]

    # for part, squeaker in zip(session_filenames_part, is_deep_squeak):
        # # saved_data_dir = saved_data_dir_fn(part)
        # if not squeaker:
            # suffix_addition = "_%s%s" % (part, run_version_suffix)
        # else:
            # suffix_addition = part
        # features.append(np.load(os.path.join(SAVE_FROZEN_DATA_DIR, "frozen_feature_examples%s.npy" % suffix_addition)))
        # labels.append(np.load(os.path.join(SAVE_FROZEN_DATA_DIR, "labels%s.npy" % suffix_addition)))
        # waveforms.append(np.load(os.path.join(SAVE_FROZEN_DATA_DIR, "raw_waveform_examples%s.npy" % suffix_addition)))

    # combined_features = np.concatenate(features)
    # combined_labels = np.concatenate(labels)
    # combined_waveforms = np.concatenate(waveforms)

    # shuffle_indicies = np.arange(len(combined_labels), dtype=np.int32)
    # np.random.shuffle(shuffle_indicies)
    # #
    # combined_features, combined_labels, combined_waveforms = combined_features[shuffle_indicies], combined_labels[shuffle_indicies], combined_waveforms[shuffle_indicies]

    # print(combined_waveforms.shape, combined_labels.shape, combined_features.shape)

    # np.save(os.path.join(SAVE_FROZEN_DATA_DIR, "frozen_feature_examples_combined"+run_version_suffix), combined_features)
    # np.save(os.path.join(SAVE_FROZEN_DATA_DIR, "labels_combined"+run_version_suffix), combined_labels)
    # np.save(os.path.join(SAVE_FROZEN_DATA_DIR, "raw_waveform_examples_combined"+run_version_suffix), combined_waveforms)

    #####################################################################################################################################


    # validation_examples = np.load("C:\\Users\\Sam\\PycharmProjects\\Raspy\\raspy\\ann_training\\second_try_logits_trained_on_frozen_features\\frozen_feature_examples_part_4_TEST_deepsqueak.npy")
    # validation_labels = np.load("C:\\Users\\Sam\\PycharmProjects\\Raspy\\raspy\\ann_training\\second_try_logits_trained_on_frozen_features\\labels_part_4_TEST_deepsqueak.npy")
    # labels.append(np.load(os.path.join(SAVE_FROZEN_DATA_DIR, "labels%s.npy" % suffix_addition)))

    #############################################################################
    
    
    train_logits_on_frozen_features(
        saved_data_dir=SAVE_FROZEN_DATA_DIR,
        save_model_path="ann_training/TESTING/50-0-5e5-all_at_once-just_logits",
        test_wav_filename="C:\\Users\\samra\\Desktop\\ann_running_4-1-2021\\ann_training\\878Pre2 - 0.wav",
        training_parameters={
            'final_head_trainer_epoch': 0,
            'final_fine_tuner_epoch': 50,
            'head_training_learning_rate': 5e-5,
            'fine_tuning_learning_rate': 1e-5,
            'batch_size': 32,
            'logit_dropout_rate': 0.2,
            'validation_ratio': 0.2,
            'desired_frequency': 500000,
            'ratio_of_audio_being_classified': .5,
        },
        run_version_suffix="_combined" + run_version_suffix,
        num_output_classes=num_call_features,
        frequency=250000,
    )

